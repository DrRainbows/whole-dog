"""The physics dog run online by a router (run in WSL): falling is a regime, and the dog gets back up.

  python online_dog.py RUN_DIR [--seconds 40] [--anchor 0.1] [--push 10:1.5:0.2,...] [--tag NAME] [--no-recover]

The loop the game needs, in miniature. Every control step (50 Hz):
  1. the router (here a fixed plan: stand, then walk a 4 m square of waypoints) asks for a regime and a heading;
  2. the motion matcher (motion_match.py) turns the ask into capture frames, kept 0.4 s ahead of now;
  3. those frames are anchored to where the physics dog actually is, so it never chases a drifting clip
     (--anchor = share of the gap closed per step; 0 = a fixed path, as in training);
  4. the trained tracking policy follows them with its motors, in MuJoCo physics;
  5. a fall detector watches the body. When the trunk tips 0.8 rad off the reference, or the hips drop 0.2 m
     below it, the body is off the plan, and the body says why:
     - FALLING, if the thorax has rolled past 60 deg (onto its side or back) or the hips dropped 0.2 m within
       0.5 s: the scramble_up reflex takes the body (--reflex), or the matcher is re-synced to the capture's
       nearest lying frame starting exactly from the fallen pose (inertialization), the dog lies a moment, is
       asked to stand, and the plan resumes;
     - LAGGING otherwise (it can't keep up with a fast rise or a jump): not a fall; the plan re-syncs to the
       captured frame nearest the body and asks again from there.
Writes RUN_DIR/online{tag}.npz (physics and reference poses, what the router said, per step) for
render_rollout.py, and RUN_DIR/online{tag}.json (falls, recoveries, distance walked).
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "--cpu" in sys.argv:  # the dog runs on the CPU (dog_cpu.py); keep JAX's few remaining calls off the busy GPU
    os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from dog_track import DogTrack, default_config
from mujoco import mjx

HERE = os.path.dirname(os.path.abspath(__file__))
BODY = os.path.dirname(HERE)
sys.path.insert(0, BODY)


def run_conf(run):
    try:
        return json.load(open(os.path.join(run, "run.json")))
    except (OSError, ValueError):
        return {}


# a run trained on the anatomical body (anat.py) plays from the capture in anatomical joints
ANAT = "dog_anat" in run_conf(sys.argv[1] if len(sys.argv) > 1 else "").get("xml", "")
if ANAT:
    os.environ.setdefault("DOG_MM_FLAT", os.environ.get("DOG_WORK", "runs") + "/refs_anat/flat")
    os.environ.setdefault("DOG_MM_CACHE", os.environ.get("DOG_WORK", "runs") + "/refs_anat/mm_db.npz")
import motion_match as mmod  # noqa: E402

XML = os.path.join(BODY, "dog_mocap_mjx.xml")
W = 20  # reference frames held: now, the next (the motors' target), and the policy's look-ahead (16 on)
DT = 0.02
LIE_S, RISE_S, GRACE_S = 0.8, 3.0, 1.0  # lie after a fall; then asked to stand; no new fall call right after a re-sync
TILT, DROP = 0.8, 0.2  # fall: trunk tipped this far (rad) from the reference, or hips this far (m) below it
LAG = 15  # control steps of reference history a lagging body may match (0.3 s)
ROLL = 0.79  # a real fall: the thorax rolled this far (45 deg) about the long axis (in the capture: no stand, walk,
#   trot, run, jump or sit frame and 0.1% of down frames roll this far; 60 deg called a fall in progress a lag)
SLUMP, SLUMP_S = 0.2, 0.5  # or the hips dropped this far (m) within this long (s)
LYING = [mmod.SUPPORT.index("down"), mmod.SUPPORT.index("side")]


def load(run):
    cfg = default_config()
    rc = json.load(open(os.path.join(run, "run.json")))
    global XML
    XML = rc.get("xml") or XML  # the body the policy was trained on, with its muscles
    cfg.muscles, cfg.obs_drive = rc.get("muscles", ""), rc.get("obs_drive", False)
    cfg.action_scale = rc.get("action_scale", cfg.action_scale)
    cfg.lookahead2, cfg.obs_paws, cfg.w_rootvel = rc.get("lookahead2", 0), rc.get("obs_paws", False), rc.get("w_rootvel", 0.0)
    cfg.rfc_force, cfg.rfc_torque, cfg.w_rfc = rc.get("rfc_force", 0.0), rc.get("rfc_torque", 0.0), rc.get("w_rfc", 0.0)
    cfg.restart_info = False  # the loop runs its own clock
    if "--cpu" in sys.argv:
        cfg.impl = "jax"  # only built for its constants; the CPU dog does the physics
    dummy = sorted(glob.glob(os.path.join(os.environ["DOG_MM_FLAT"] if ANAT else os.path.join(BODY, "refs", "walk"),
                                          "*.npz")))[:1]  # replaced by the live window
    env = DogTrack(XML, dummy, config=cfg, max_dogs=8)
    params = pickle.load(open(os.path.join(run, "params.pkl"), "rb"))
    net_cfg = json.load(open(os.path.join(run, "net.json")))
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize, **net_cfg)
    policy = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
    return env, policy


def live_fns(env):
    """The environment's own step and observation, run against a live reference window instead of its clips."""
    def with_ref(bq, bv, bp, f):
        saved = env._ref_q, env._ref_v, env._ref_paws
        env._ref_q, env._ref_v, env._ref_paws = bq, bv, bp
        try:
            return f()
        finally:
            env._ref_q, env._ref_v, env._ref_paws = saved

    step = jax.jit(lambda bq, bv, bp, st, a: with_ref(bq, bv, bp, lambda: env.step(st, a)))
    obs = jax.jit(lambda bq, bv, bp, d, aim: with_ref(bq, bv, bp, lambda: jp.nan_to_num(
        env._obs(d, jp.int32(0), jp.int32(1000), aim=aim), nan=0.0, posinf=0.0, neginf=0.0)))
    reset = jax.jit(lambda bq, bv, bp: with_ref(bq, bv, bp, lambda: env.reset_at(jax.random.PRNGKey(0), jp.int32(0))))
    return step, obs, reset


def load_reflex(run, env):
    """The scramble_up reflex (train_track.py --task right): its policy, observation, physics step, and its tests
    for 'on its feet' and 'righted'. Same body and muscles as the tracker it hands back to."""
    from dog_right import DogRight, right_config
    rc = json.load(open(os.path.join(run, "run.json")))
    if rc.get("xml") != env._xml_path or rc.get("muscles", "") != env._config.muscles:
        print(f"WARNING: the reflex was trained on {rc.get('xml')} {rc.get('muscles')}, the tracker on "
              f"{env._xml_path} {env._config.muscles}", flush=True)
    cfg = right_config(default_config())
    cfg.muscles = env._config.muscles
    refs = sorted(glob.glob(os.path.join(os.environ.get("DOG_MM_FLAT", ""), "*.npz")))  # its standing pose and height
    er = DogRight(env._xml_path, refs, config=cfg, max_dogs=8)
    params = pickle.load(open(os.path.join(run, "params.pkl"), "rb"))
    net_cfg = json.load(open(os.path.join(run, "net.json")))
    nets = ppo_networks.make_ppo_networks(er.observation_size, er.action_size,
                                          preprocess_observations_fn=running_statistics.normalize, **net_cfg)

    def step(data, mus, act, cap):
        target = jp.clip(cfg.right_scale * act[:er._mj.nu], er._lo, er._hi)
        if er._mus:
            data, mus, _ = er.muscle_steps(data, target, mus, cap)
            return data, mus
        data = jax.lax.fori_loop(0, er._n_sub, lambda _, d: mjx.step(er._mx, d), data.replace(ctrl=target))
        return data, target

    return {"policy": jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True)),
            "obs": jax.jit(er._robs), "step": jax.jit(step), "is_up": jax.jit(er.is_up),
            "is_sternal": jax.jit(er.is_sternal)}


class Kin:
    """CPU kinematics of one pose: hips, neck, paw pads."""

    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.hips = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Hips")
        self.neck = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Neck")
        self.pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in mmod.PADS]

    def __call__(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        return (self.d.xpos[self.hips].copy(), self.d.xpos[self.neck].copy(),
                np.array([self.d.geom_xpos[g] for g in self.pads]))


def heading(hips, neck):
    v = neck[:2] - hips[:2]
    return float(np.arctan2(v[1], v[0]))


def pose_feature(hips, neck, pads):
    """The matcher's pose feature (motion_match.features): paws around the hips in the dog's heading, heights."""
    rel = (pads[:, :2] - hips[:2]) @ mmod.rot2(-heading(hips, neck)).T
    return np.concatenate([rel.ravel(), pads[:, 2], [hips[2]]])


def up_dir(Q):
    """World up seen from the trunk (R^T e_z) for one quaternion (4,) or many (N, 4): which way the dog lies."""
    Q = np.asarray(Q, float)
    w, x, y, z = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]
    return np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)


def up_err(qa, qb):
    """Angle between the gravity directions seen from two trunk orientations (ignores heading)."""
    return float(np.arccos(np.clip(np.dot(up_dir(qa), up_dir(qb)), -1.0, 1.0)))


class Window:
    """The reference just ahead of the physics, in the matcher's own (kinematic) world."""

    def __init__(self, m, kin):
        self.m, self.kin = m, kin
        self.Q, self.V, self.P, self.H, self.X = [], [], [], [], []

    def clear(self):
        self.__init__(self.m, self.kin)

    def push(self, q):
        hips, neck, pads = self.kin(q)
        v = np.zeros(self.m.nv)
        if self.Q:  # forward differences, as in the training references (make_reference.finish)
            mujoco.mj_differentiatePos(self.m, v, DT, self.Q[-1], q)
            self.V[-1] = v.copy()
        for lst, x in ((self.Q, q), (self.V, v), (self.P, pads), (self.H, heading(hips, neck)), (self.X, hips[:2])):
            lst.append(np.array(x, float) if not np.isscalar(x) else x)
            if len(lst) > W:
                lst.pop(0)

    def placed(self, th, t):
        """Window rotated by th about the vertical and shifted by t: kinematic world -> physics world."""
        R = mmod.rot2(th)
        Q, V, P = np.array(self.Q), np.array(self.V), np.array(self.P)
        Q[:, 0:2] = Q[:, 0:2] @ R.T + t
        yq = mmod.yaw_quat(th)
        Q[:, 3:7] = np.array([mmod.qmul(yq, q) for q in Q[:, 3:7]])
        V[:, 0:2] = V[:, 0:2] @ R.T
        P[:, :, 0:2] = P[:, :, 0:2] @ R.T + t
        return Q, V, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--anchor", type=float, default=0.1)
    ap.add_argument("--push", default="", help="pushes: T:F:DUR,... (start s, force in body weights, seconds)")
    ap.add_argument("--side", type=float, default=4.0, help="side of the square walked (m)")
    ap.add_argument("--no-recover", action="store_true", help="stop 2 s after the first fall")
    ap.add_argument("--tag", default="")
    ap.add_argument("--brain", default="", choices=["", "lab", "mal"], help="the dog's brain is the router (dog/brain.py)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--calls", default="20,26,32", help="with --brain: when the person calls 'come' (s)")
    ap.add_argument("--person", default="4,2", help="with --brain: where the person stands (m)")
    ap.add_argument("--say", default="", help="with --brain: words the person says, 'T:words[:urgency]|...' "
                                              "(urgency 0-2: said again, insisting); replaces --calls")
    ap.add_argument("--present", action="store_true", help="with --brain: a person is there (social mode)")
    ap.add_argument("--where", default="-3,1", help="with --say: where a named object or place is (m)")
    ap.add_argument("--start", default="", help="with --brain: the regime the dog starts in")
    ap.add_argument("--cpu", action="store_true", help="run the dog on the CPU (dog_cpu.py: native MuJoCo, numpy "
                    "policies; needs export_policy.py run on the tracker and on the reflex)")
    ap.add_argument("--reflex", default="", help="a scramble_up run (train_track.py --task right): when the dog falls,"
                    " this reflex takes the body until its feet are back under it")
    a = ap.parse_args()
    brain = None
    if a.brain:
        sys.path.insert(0, os.path.join(os.path.dirname(BODY), "dog"))
        from brain import LAB, MAL, Brain, Router

        person = np.array([float(x) for x in a.person.split(",")])
        grammar = None
        says = []
        if a.say or a.present:
            from grammar import Grammar

            grammar = Grammar()
            for item in a.say.split("|"):
                if item.strip():
                    parts = item.split(":")
                    says.append((float(parts[0]), parts[1], float(parts[2]) if len(parts) > 2 else 0.0))
        brain = Brain({"lab": LAB, "mal": MAL}[a.brain], Router(), np.random.default_rng(a.seed),
                      **({"regime": a.start} if a.start else {}), grammar=grammar,
                      person=person if (a.present or a.say) else None)
        calls = [] if a.say else [float(x) for x in a.calls.split(",") if x]
        says.sort(key=lambda s: s[0])
        where = np.array([float(x) for x in a.where.split(",")])
        spine = None  # breathing is laid over these two spine hinges (found once the model is loaded)

    if a.cpu:  # the same body, muscles and policies on native MuJoCo and numpy (dog_cpu.py): real time on one core
        from dog_cpu import online_backend
        global XML
        XML = run_conf(a.run).get("xml") or XML
        env, reset_fn, obs_fn, step_fn, policy, reflex = online_backend(a.run, a.reflex or None, seed=a.seed)
    else:
        env, policy = load(a.run)
        step_fn, obs_fn, reset_fn = live_fns(env)
        reflex = load_reflex(a.reflex, env) if a.reflex else None
    m = mujoco.MjModel.from_xml_path(XML)
    kin = Kin(m)
    t0 = time.time()
    mm = mmod.Matcher(m)
    print(f"matcher ready ({time.time() - t0:.0f} s, {len(mm.cand)} frames)", flush=True)
    lying = mm.cand[np.isin(mm.db["regime"][mm.cand], LYING)]
    F_pose, s_pose = mm.F["pose"], mm.scale["pose"]
    up_lying = up_dir(mm.db["qpos"][lying, 3:7])  # how each lying frame's trunk sits against gravity
    # standing, as the capture stands: hips height and how the trunk sits against gravity (for "is it up?")
    standing = mm.cand[mm.db["regime"][mm.cand] == mmod.SUPPORT.index("stand")]
    stand_up = np.median(up_dir(mm.db["qpos"][standing, 3:7]), 0)
    stand_up /= np.linalg.norm(stand_up)
    stand_z = float(np.median(mm.db["hips"][standing, 2]))

    # the thorax's standing up and lateral axes in its own frame, for "has it rolled?" (calibrated on the capture: a
    # roll past 60 deg is in no stand, walk, trot, run, jump, sit or down frame; a sitting pelvis tips 97 deg, so
    # the pelvis can't say this)
    thx_id = m.body("Spine1").id
    dk = mujoco.MjData(m)
    U, Fw = [], []
    for j in standing[:: max(1, len(standing) // 400)]:
        dk.qpos[:] = mm.db["qpos"][j]
        mujoco.mj_kinematics(m, dk)
        Rt = dk.xmat[thx_id].reshape(3, 3)
        fw = dk.xpos[m.body("Neck").id] - dk.xpos[m.body("Hips").id]
        U.append(Rt.T @ np.array([0.0, 0.0, 1.0]))
        Fw.append(Rt.T @ (fw / np.linalg.norm(fw)))
    thx_up = np.median(U, 0) / np.linalg.norm(np.median(U, 0))
    fw = np.median(Fw, 0)
    fw = fw - (fw @ thx_up) * thx_up
    thx_side = np.cross(thx_up, fw / np.linalg.norm(fw))
    every = mm.cand
    up_every = up_dir(mm.db["qpos"][every, 3:7])

    def thorax_roll(xmat):
        u = np.array(xmat[thx_id], float).reshape(3, 3)[2]  # world up, seen in the thorax's frame
        return float(np.arctan2(abs(u @ thx_side), u @ thx_up))

    def is_up(qp, xp):
        tilt_ok = np.arccos(np.clip(np.dot(up_dir(qp[3:7]), stand_up), -1, 1)) < 0.6
        return bool(tilt_ok and xp[hips_id][2] > 0.75 * stand_z)

    downs = mm.cand[mm.db["regime"][mm.cand] == mmod.SUPPORT.index("down")]
    up_stand, up_down = up_dir(mm.db["qpos"][standing, 3:7]), up_dir(mm.db["qpos"][downs, 3:7])

    def nearest(pool, up_pool, qp, xp, gx):
        """The capture's frame in pool nearest the body: paws around the hips, heights, the trunk against gravity."""
        f_now = pose_feature(xp[hips_id], xp[neck_id], gx[pads_idx])
        d = (((F_pose[pool] - f_now) / s_pose) ** 2).sum(1) + 10.0 * ((up_pool - up_dir(qp[3:7])) ** 2).sum(1)
        return int(pool[int(np.argmin(d))])

    def resync_frame(qp, xp, gx):
        """Nearest lying frame of the capture to the fallen body: paws around the hips, heights, and the way
        the trunk lies (on the belly, on the left or right side). Orientation weighs most."""
        f_now = pose_feature(xp[hips_id], xp[neck_id], gx[pads_idx])
        d = (((F_pose[lying] - f_now) / s_pose) ** 2).sum(1) + 10.0 * ((up_lying - up_dir(qp[3:7])) ** 2).sum(1)
        j = int(np.argmin(d))
        return int(lying[j]), float(np.degrees(np.arccos(np.clip(np.dot(up_lying[j], up_dir(qp[3:7])), -1, 1))))
    nu, n_rfc = env._mj.nu, env._n_rfc
    pushes = [tuple(float(x) for x in p.split(":")) for p in a.push.split(",") if p]
    wps = np.array([[a.side, 0], [a.side, a.side], [0, a.side], [0, 0]], float)

    start_reg = brain.reg if brain else "stand"
    if brain:
        spine = [m.jnt_qposadr[m.joint(n).id] for n in (("Spine_a1", "Spine1_a1") if ANAT else ("Spine_x", "Spine1_x"))]
    player = mmod.Player(mm, start_reg)
    win = Window(m, kin)
    for _ in range(W):
        win.push(player.step(start_reg)[0])
    th, tr = 0.0, np.zeros(2)  # kinematic world -> physics world

    def buffers():
        Q, V, P = win.placed(th, tr)
        return jp.array(Q, jp.float32), jp.array(V, jp.float32), jp.array(P, jp.float32)

    bq, bv, bp = buffers()
    zero_push = jp.zeros(m.nv, jp.float32)
    st = reset_fn(bq, bv, bp)
    st = st.replace(info=dict(st.info, k=jp.int32(0), end=jp.int32(1000), push=zero_push))
    hips_id, neck_id, pads_idx = kin.hips, kin.neck, np.array(kin.pads)

    mode, t_mode, grace, wp, ask_reg = "plan", 0.0, 0.0, 0, "stand"
    recent = []  # the last LAG reference frames, placed in the physics world
    hips_z = []  # the body's own hips height over the last SLUMP_S
    lie_reg = "down"
    log = {"falls": [], "rises": [], "waypoints": [], "scrambles": []}
    sim_q, ref_q, labels, assist = [], [], [], []
    walked, n_steps, moving_steps = 0.0, int(a.seconds / DT), 0
    prev_xy = None
    t_loop = time.time()
    i_scr = up_n = st_n = 0
    mm_frames = []  # the capture frame the matcher is playing, per step (which clips lead into falls)
    for i in range(n_steps):
        tnow = (i + 1) * DT
        mm_frames.append(int(player.c))
        if mode == "scramble":  # the reflex has the body: no reference, no tracker
            data = st.pipeline_state
            mus = st.info["mus"] if "mus" in st.info else data.ctrl
            o_r = reflex["obs"](data, mus, jp.int32(min(i - i_scr, 149)))
            act_r, _ = reflex["policy"](o_r, jax.random.PRNGKey(i))
            if brain and "cap" in st.info:
                st = st.replace(info=dict(st.info, cap=jp.float32(brain.muscle_capacity())))
            data, mus = reflex["step"](data, mus, act_r, st.info.get("cap", jp.float32(1.0)))
            st = st.replace(pipeline_state=data, info=dict(st.info, mus=mus) if "mus" in st.info else st.info)
            qp, xp = np.array(data.qpos, float), np.array(data.xpos, float)
            hp = heading(xp[hips_id], xp[neck_id])
            if not np.all(np.isfinite(qp)):
                print("physics blew up at", tnow, flush=True)
                break
            up_n = up_n + 1 if bool(reflex["is_up"](data)) else 0
            st_n = st_n + 1 if bool(reflex["is_sternal"](data)) else 0
            if brain:
                brain.tick(tnow, DT, qp[0:2])
            back = "stand" if up_n >= 15 else ("down" if st_n >= 50 and tnow - t_mode > 2.0 else None)
            if back:  # feet back under it (or righted on its belly): the matcher picks up from the body as it is
                gx = np.array(data.geom_xpos, float)
                j = nearest(standing, up_stand, qp, xp, gx) if back == "stand" else nearest(downs, up_down, qp, xp, gx)
                player.resync(j, qp, hp, back, blend=25)
                win.clear()
                win.push(qp)
                for _ in range(W - 2):
                    win.push(player.step(back)[0])
                th, tr = 0.0, np.zeros(2)
                recent.clear()
                bq, bv, bp = buffers()
                log["scrambles"].append({"fell": round(t_mode, 2), "until": round(tnow, 2), "to": back})
                if back == "stand":
                    mode, t_mode, grace = "plan", tnow, tnow + 0.5
                    log["rises"].append(round(tnow, 2))
                    if brain:
                        brain.got_up(tnow)
                else:  # righted on its belly: the planned rise (down -> stand) from here
                    mode, t_mode = "rise", tnow
            sim_q.append(qp)
            ref_q.append(qp)
            lab = f"reflex: scramble_up ({tnow - t_mode:.1f} s)" if mode == "scramble" else f"reflex done -> {back}"
            if brain:
                lab = f"{a.brain.upper()} {lab}  breath {brain.breath_rate():.0f}/min  arousal {brain.A:.2f}"
            labels.append(f"{lab}   falls {len(log['falls'])}")
            assist.append([0.0, 0.0])
            continue
        if brain and "cap" in st.info:  # blood and breath: the muscles have the strength the brain's state allows
            st = st.replace(info=dict(st.info, cap=jp.float32(brain.muscle_capacity())))
        aim = st.info["mus"] if "mus" in st.info else st.pipeline_state.ctrl  # where the muscles aim now
        obs = obs_fn(bq, bv, bp, st.pipeline_state, aim)
        act, _ = policy(obs, jax.random.PRNGKey(i))
        if n_rfc and mode == "fallen":
            act = act.at[nu:].set(0.0)  # falling: no hand of god
        push = zero_push
        for (p0, f, dur) in pushes:
            if p0 <= tnow < p0 + dur and prev_xy is not None:
                side = hp + np.pi / 2
                push = zero_push.at[0:2].set(jp.array([np.cos(side), np.sin(side)]) * f * env._mg)
        st = st.replace(info=dict(st.info, k=jp.int32(0), push=push))
        st = step_fn(bq, bv, bp, st, act)
        qp = np.array(st.pipeline_state.qpos, float)
        xp = np.array(st.pipeline_state.xpos, float)
        hp = heading(xp[hips_id], xp[neck_id])
        if not np.all(np.isfinite(qp)):
            print("physics blew up at", tnow, flush=True)
            break

        # the router watches the body against the reference for this moment (window frame 1: physics just got there),
        # forgiving lag: the body only counts as off if it matches none of the last 0.3 s of reference frames
        q_ref = win.placed(th, tr)[0][1]
        recent.append(q_ref.copy())
        del recent[:-LAG]
        tilt = min(up_err(qp[3:7], r[3:7]) for r in recent)
        drop = min(r[2] - qp[2] for r in recent)
        hips_z.append(float(xp[hips_id][2]))
        del hips_z[:-int(SLUMP_S / DT)]
        event = None
        if mode == "plan" and tnow >= grace and (tilt > TILT or drop > DROP):
            # lying already: the body just lies differently from the reference; follow the body, no get-up
            if ask_reg in ("down", "side"):
                event = "relie"
            else:
                roll = thorax_roll(st.pipeline_state.xmat)
                slump = max(hips_z) - hips_z[-1]
                # a real fall rolls the thorax or drops the hips; anything else is a body that can't keep up
                event = "fall" if (roll > ROLL or slump > SLUMP) else "lag"
                if event == "lag":
                    log.setdefault("lags", []).append({"t": round(tnow, 2), "during": ask_reg, "tilt": round(tilt, 2),
                                                       "drop": round(float(drop), 2), "roll": round(roll, 2)})
        elif mode == "fallen" and tnow - t_mode >= LIE_S:
            mode, t_mode = "rise", tnow
        elif mode == "rise" and tnow - t_mode >= RISE_S:
            if is_up(qp, xp):  # on its feet, as the capture stands: back to the plan
                mode, t_mode, grace = "plan", tnow, tnow + 0.5
                log["rises"].append(round(tnow, 2))
            else:
                event = "failed rise"
        if brain and mode == "plan" and log["rises"] and log["rises"][-1] == round(tnow, 2):
            brain.got_up(tnow)
        if event == "lag":  # the plan waits for the body: re-sync to the captured frame nearest it, keep the ask
            j = nearest(every, up_every, qp, xp, np.array(st.pipeline_state.geom_xpos, float))
            player.resync(j, qp, hp, mmod.SUPPORT[int(mm.db["regime"][j])], blend=15)
            win.clear()
            win.push(qp)
            for _ in range(W - 2):
                win.push(player.step(ask_reg)[0])
            th, tr = 0.0, np.zeros(2)
            recent.clear()
            grace, event = tnow + 0.5, None
        if event:
            if event != "relie":
                log["falls"].append({"t": round(tnow, 2), "what": event, "during": ask_reg,
                                     "tilt": round(tilt, 2), "drop": round(float(drop), 2)})
                if brain and event == "fall":
                    brain.fell(tnow)
            if a.no_recover and event != "relie":
                n_steps, grace, event = min(n_steps, i + int(2.0 / DT)), 1e9, None
            elif reflex and event == "fall":  # FALLING: the reflex takes the body ("scramble up", in the lexicon)
                mode, t_mode, i_scr, up_n, st_n = "scramble", tnow, i, 0, 0
                log["falls"][-1].update(reflex="scramble_up")
                event = None
            else:
                # the nearest lying frame of the capture, starting exactly from the body as it lies
                j, deg = resync_frame(qp, xp, np.array(st.pipeline_state.geom_xpos, float))
                lie_reg = mmod.SUPPORT[int(mm.db["regime"][j])]
                player.resync(j, qp, hp, lie_reg, blend=25)
                win.clear()
                win.push(qp)
                for _ in range(W - 2):
                    win.push(player.step(lie_reg)[0])
                th, tr = 0.0, np.zeros(2)
                recent.clear()
                if event == "relie":  # already lying: follow the body quietly, no get-up
                    log["relies"] = log.get("relies", 0) + 1
                    grace = tnow + 1.0
                else:  # FALLING
                    log["falls"][-1].update(matched=lie_reg, trunk_mismatch_deg=round(deg))
                    mode, t_mode = "fallen", tnow
        want = None
        if brain:  # the brain keeps living through a fall: its clock, breath and arousal run on
            if calls and tnow >= calls[0]:
                calls.pop(0)
                if brain.goal is None and not any(e[1] == "arrives" for e in brain.events):
                    brain.request(tnow, "come", person)
                    brain.events.append((tnow, "person: 'come!'"))
            while says and tnow >= says[0][0]:
                _, text, urg = says.pop(0)
                brain.hear(tnow, text, where=where, urgency=urg)
                brain.events.append((tnow, f"person: '{text}'" + (" (insisting)" if urg else "")))
            b_reg, _, b_want = brain.tick(tnow, DT, qp[0:2])
            # over-commanded: the change is taken straight across, with no route through the postures between
            player.plan = player.same_regime = not (brain.forced and mode == "plan")
        if mode == "plan" and brain:
            ask_reg, want = b_reg, b_want
        elif mode == "plan":
            if tnow < 1.5:
                ask_reg = "stand"
            else:
                ask_reg = "walk"
                if np.linalg.norm(wps[wp] - qp[0:2]) < 0.7:
                    log["waypoints"].append(round(tnow, 2))
                    wp = (wp + 1) % len(wps)
                d = wps[wp] - qp[0:2]
                want = float(np.arctan2(d[1], d[0]))
        elif mode == "fallen":
            ask_reg = lie_reg
        else:
            ask_reg = "stand"
        q_new, played = player.step(ask_reg, want_yaw=None if want is None else want - th)
        if player.blend_t == 0:
            player.blend = 10
        if brain:  # breathing, laid over the spine of the reference the motors follow
            q_new = q_new.copy()
            q_new[spine[0]] += brain.breath_offset()
            q_new[spine[1]] -= 0.5 * brain.breath_offset()
        win.push(q_new)

        if a.anchor > 0 and mode != "fallen" and not event:
            # anchor the reference to the physics dog: turn it about its own root, then shift it
            h0, p0 = win.H[0], win.X[0]
            e_yaw = mmod.wrap(hp - (h0 + th))
            here = mmod.rot2(th) @ p0 + tr
            th = th + a.anchor * e_yaw
            tr = here - mmod.rot2(th) @ p0
            tr = tr + a.anchor * (qp[0:2] - (mmod.rot2(th) @ p0 + tr))
        bq, bv, bp = buffers()

        moving = mode == "plan" and ask_reg in ("walk", "trot", "run")
        if moving and prev_xy is not None:
            walked += float(np.linalg.norm(qp[0:2] - prev_xy))
        moving_steps += moving
        prev_xy = qp[0:2].copy()
        sim_q.append(qp)
        ref_q.append(np.array(bq[0]))
        lab = {"plan": f"router: {ask_reg}", "fallen": f"router: FALLEN -> {lie_reg}", "rise": "router: get up", "scramble": "reflex: scramble_up"}[mode]
        if brain:
            ev = [f"{te:.1f}s {msg}" for te, msg in brain.events[-2:]]
            lab = (f"{a.brain.upper()} {lab}  breath {brain.breath_rate():.0f}/min  arousal {brain.A:.2f}"
                   f"{'  ' + ' | '.join(ev) if ev else ''}")
        labels.append(f"{lab}   falls {len(log['falls'])}   matcher plays: {played}")
        assist.append([float(st.metrics["assist_f"]), float(st.metrics["assist_t"])])
        if i % 250 == 0:
            print(f"t {tnow:5.1f}s  {lab:28s} falls {len(log['falls'])}  walked {walked:5.1f} m  "
                  f"({(time.time() - t_loop) / (i + 1) * 1000:.0f} ms/step)", flush=True)

    T = len(sim_q)
    walk_s = moving_steps * DT
    res = {"seconds": round(T * DT, 1), "anchor": a.anchor, "pushes": pushes, "brain": a.brain,
           "falls": len(log["falls"]), "fall_log": log["falls"], "rises_completed": len(log["rises"]),
           "lying_resyncs": log.get("relies", 0), "lags": log.get("lags", []), "scrambles": log["scrambles"],
           "brain_events": [f"{t:.1f}s {msg}" for t, msg in brain.events] if brain else [],
           "heard": brain.heard if brain else [],
           "first_fall_s": log["falls"][0]["t"] if log["falls"] else None,
           "walking_seconds": round(walk_s, 1), "walked_m": round(walked, 1),
           "falls_per_walking_minute": round(len(log["falls"]) / max(walk_s / 60, 1e-6), 2),
           "waypoints_reached": len(log["waypoints"]),
           "mean_assist_force_share_of_weight": round(float(np.mean([x[0] for x in assist])), 4) if n_rfc else 0.0,
           "ms_per_step": round((time.time() - t_loop) / max(T, 1) * 1000, 1)}
    print("ONLINE " + json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(a.run, f"online{a.tag}.json"), "w"), indent=1)
    np.savez(os.path.join(a.run, f"online{a.tag}.npz"), starts=np.array([0]), sim_0=np.array(sim_q),
             ref_0=np.array(ref_q), fail_0=-1, label_0=np.array(labels), assist_0=np.array(assist),
             mm_frame_0=np.array(mm_frames, np.int32))


if __name__ == "__main__":
    main()
