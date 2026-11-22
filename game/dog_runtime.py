"""The dog at run time, with no JAX: one call to step() is one 20 ms control step of the whole dog.

This is body/train/online_dog.py's loop (with --cpu) as a class a game can drive:
  brain (dog/brain.py: words, modes, physiology)
    -> motion matcher (body/motion_match.py: the capture, as frames just ahead of now)
    -> the physics dog (body/train/dog_cpu.py: native MuJoCo, a dog's muscles, the trained tracker in numpy)
    -> the fall detector: a real fall (the thorax rolled past 45 deg, or the hips dropped 0.2 m in 0.5 s) hands
       the body to the scramble_up reflex; anything else is lag, and the plan re-syncs to the body.

    rt = DogRuntime(TRACKER_DIR, REFLEX_DIR, dog="mal", present=True)
    rt.say("platz")          # heard at the next step
    for _ in range(50):
        rt.step()            # 20 ms
    rt.data                  # mujoco.MjData of the physics dog: draw it
    rt.label                 # what the dog is doing, for a HUD

It is kept to numpy, MuJoCo and plain Python data so the same boundary can later move to a native (C/C++ or Rust)
stack: the brain, the matcher and the policies are all data plus small loops.
"""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BODY = os.path.join(ROOT, "body")
for p in (os.path.dirname(os.path.abspath(__file__)), BODY, os.path.join(BODY, "train"), os.path.join(ROOT, "dog")):
    if p not in sys.path:
        sys.path.insert(0, p)
# the matcher plays from the capture in anatomical joints (anat.py)
os.environ.setdefault("DOG_MM_FLAT", "runs/refs_anat/flat")
os.environ.setdefault("DOG_MM_CACHE", "runs/refs_anat/mm_db.npz")
import motion_match as mmod  # noqa: E402
from dog_cpu import NumpyPolicy  # noqa: E402
from world import World, win_path  # noqa: E402

W = 20  # reference frames held: now, the next (the motors' target), and the policy's look-ahead
DT = 0.02
LIE_S, RISE_S = 0.8, 3.0  # lie after a fall (no reflex); then asked to stand
TILT, DROP = 0.8, 0.2  # off the plan: trunk tipped this far (rad) from the reference, or hips this far (m) below it
LAG = 15  # control steps of reference history a lagging body may match (0.3 s)
ROLL = 0.79  # a real fall: the thorax rolled 45 deg about the long axis (no upright capture frame rolls this far)
SLUMP, SLUMP_S = 0.2, 0.5  # or the hips dropped this far (m) within this long (s)
LYING = [mmod.SUPPORT.index("down"), mmod.SUPPORT.index("side")]


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
    rel = (pads[:, :2] - hips[:2]) @ mmod.rot2(-heading(hips, neck)).T
    return np.concatenate([rel.ravel(), pads[:, 2], [hips[2]]])


def up_dir(Q):
    Q = np.asarray(Q, float)
    w, x, y, z = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]
    return np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)


def up_err(qa, qb):
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
        if self.Q:  # forward differences, as in the training references
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


class DataView:
    """One dog of the world as the runtime and a renderer read it: its own (trained) single-dog layout."""

    def __init__(self, body):
        self.b = body

    qpos = property(lambda s: s.b.q())
    qvel = property(lambda s: s.b.v())
    xpos = property(lambda s: s.b.xpos())
    xmat = property(lambda s: s.b.xmat())
    xquat = property(lambda s: s.b.xquat())
    geom_xpos = property(lambda s: s.b.geom_xpos())
    geom_xmat = property(lambda s: s.b.world.d.geom_xmat[s.b.gmap].copy())


class DogRuntime:
    """One dog: its brain, matcher and reference window, and its body in a world (its own, or shared with others).
    Alone: step(). With other dogs in one world: decide() for every dog, world.physics(), after() for every dog."""

    def __init__(self, tracker, reflex, dog="lab", seed=3, present=True, start=None, person=(4.0, 2.0),
                 where=(-3.0, 1.0), anchor=0.1, grammar=True, world=None, slot=0, place=None):
        from brain import LAB, MAL, Brain, Router
        self.person = np.array(person, float)
        self.where = np.array(where, float)
        self.anchor = anchor
        g = None
        if grammar:
            from grammar import Grammar
            g = Grammar()
        self.brain = Brain({"lab": LAB, "mal": MAL}[dog], Router(), np.random.default_rng(seed),
                           **({"regime": start} if start else {}), grammar=g,
                           person=self.person if present else None)
        self.dog_name = dog
        own = world is None
        if own:  # a dog alone: a world of one
            world = World([{"prefix": f"{dog}/", "tracker": tracker, "reflex": reflex}])
            slot = 0
        self.world, self.body, self.own_world = world, world.bodies[slot], own
        self.policy = NumpyPolicy(os.path.join(tracker, "policy_np.npz"))
        self.rpolicy = NumpyPolicy(os.path.join(reflex, "policy_np.npz")) if reflex else None
        self.data = DataView(self.body)  # what a renderer draws, in the single-dog layout of self.model
        self.model = self.body.single
        self.push_force = self.body.push  # an outside shove on the trunk this step (force xyz, torque xyz)
        m = mujoco.MjModel.from_xml_path(win_path(self.body.c["xml"]))
        self.m = m
        self.kin = Kin(m)
        mm = self.mm = mmod.Matcher(m)
        self.lying = mm.cand[np.isin(mm.db["regime"][mm.cand], LYING)]
        self.F_pose, self.s_pose = mm.F["pose"], mm.scale["pose"]
        self.up_lying = up_dir(mm.db["qpos"][self.lying, 3:7])
        self.standing = mm.cand[mm.db["regime"][mm.cand] == mmod.SUPPORT.index("stand")]
        su = np.median(up_dir(mm.db["qpos"][self.standing, 3:7]), 0)
        self.stand_up = su / np.linalg.norm(su)
        self.stand_z = float(np.median(mm.db["hips"][self.standing, 2]))
        self._calibrate_thorax()
        self.every = mm.cand
        self.up_every = up_dir(mm.db["qpos"][self.every, 3:7])
        self.downs = mm.cand[mm.db["regime"][mm.cand] == mmod.SUPPORT.index("down")]
        self.up_stand = up_dir(mm.db["qpos"][self.standing, 3:7])
        self.up_down = up_dir(mm.db["qpos"][self.downs, 3:7])
        self.hips_id, self.neck_id, self.pads_idx = self.kin.hips, self.kin.neck, np.array(self.kin.pads)
        self.spine = [m.jnt_qposadr[m.joint(n).id] for n in ("Spine_a1", "Spine1_a1")]
        start_reg = self.brain.reg
        self.player = mmod.Player(mm, start_reg)
        self.win = Window(m, self.kin)
        for _ in range(W):
            self.win.push(self.player.step(start_reg)[0])
        self.th, self.tr = 0.0, np.zeros(2)
        if place is not None:  # start this dog's plan where it should stand in the world
            self.tr = np.asarray(place, float) - np.asarray(self.win.Q[0][0:2])
        self.bq, self.bv, self.bp = self.win.placed(self.th, self.tr)
        rng = np.random.default_rng(seed)
        q0 = np.array(self.bq[0], float)
        noise = 0.03 * rng.normal(size=q0.shape)
        noise[0:7] = 0.0
        q = q0 + noise
        b = self.body
        q[b.act_q] = np.clip(q[b.act_q], b.lo, b.hi)  # start inside the ranges
        b.set_state(q, np.array(self.bv[0], float))
        self.mus = q0[b.act_q].copy()  # the muscles aim at the first frame
        self.cap = 1.0
        if own:
            world.forward()
        self.mode, self.t_mode, self.grace, self.ask_reg = "plan", 0.0, 0.0, "stand"
        self.recent, self.hips_z = [], []
        self.lie_reg = "down"
        self.log = {"falls": [], "rises": [], "scrambles": [], "lags": [], "relies": 0}
        self.i = 0
        self.i_scr = self.up_n = self.st_n = 0
        self.says = []  # (text, urgency) heard at the next step
        self.played = start_reg
        self.label = ""
        self.hp = 0.0

    def _calibrate_thorax(self):
        m, mm = self.m, self.mm
        self.thx_id = m.body("Spine1").id
        dk = mujoco.MjData(m)
        U, Fw = [], []
        for j in self.standing[:: max(1, len(self.standing) // 400)]:
            dk.qpos[:] = mm.db["qpos"][j]
            mujoco.mj_kinematics(m, dk)
            Rt = dk.xmat[self.thx_id].reshape(3, 3)
            fw = dk.xpos[m.body("Neck").id] - dk.xpos[m.body("Hips").id]
            U.append(Rt.T @ np.array([0.0, 0.0, 1.0]))
            Fw.append(Rt.T @ (fw / np.linalg.norm(fw)))
        self.thx_up = np.median(U, 0) / np.linalg.norm(np.median(U, 0))
        fw = np.median(Fw, 0)
        fw = fw - (fw @ self.thx_up) * self.thx_up
        self.thx_side = np.cross(self.thx_up, fw / np.linalg.norm(fw))

    # -- small helpers ------------------------------------------------------------------------------------------
    def thorax_roll(self, xm):
        u = xm[self.thx_id].reshape(3, 3)[2]
        return float(np.arctan2(abs(u @ self.thx_side), u @ self.thx_up))

    def is_up(self, qp, xp):
        tilt_ok = np.arccos(np.clip(np.dot(up_dir(qp[3:7]), self.stand_up), -1, 1)) < 0.6
        return bool(tilt_ok and xp[self.hips_id][2] > 0.75 * self.stand_z)

    def nearest(self, pool, up_pool, qp, xp, gx):
        f_now = pose_feature(xp[self.hips_id], xp[self.neck_id], gx[self.pads_idx])
        d = (((self.F_pose[pool] - f_now) / self.s_pose) ** 2).sum(1) + 10.0 * ((up_pool - up_dir(qp[3:7])) ** 2).sum(1)
        return int(pool[int(np.argmin(d))])

    def resync_frame(self, qp, xp, gx):
        f_now = pose_feature(xp[self.hips_id], xp[self.neck_id], gx[self.pads_idx])
        d = (((self.F_pose[self.lying] - f_now) / self.s_pose) ** 2).sum(1) + \
            10.0 * ((self.up_lying - up_dir(qp[3:7])) ** 2).sum(1)
        j = int(np.argmin(d))
        return int(self.lying[j]), float(np.degrees(np.arccos(np.clip(np.dot(self.up_lying[j], up_dir(qp[3:7])), -1, 1))))

    def _restart_window(self, j, qp, regime, ask, blend):
        self.player.resync(j, qp, self.hp, regime, blend=blend)
        self.win.clear()
        self.win.push(qp)
        for _ in range(W - 2):
            self.win.push(self.player.step(ask)[0])
        self.th, self.tr = 0.0, np.zeros(2)
        self.recent.clear()

    @property
    def t(self):
        return self.i * DT

    @property
    def mg(self):
        return self.body.mg

    def say(self, text, urgency=0.0):
        """The person says something; the brain hears it at the next step."""
        self.says.append((text, float(urgency)))

    # -- one control step ---------------------------------------------------------------------------------------
    def step(self):
        """A dog alone in its world: decide, physics, after."""
        self.decide()
        self.world.physics()
        self.after()

    def decide(self):
        """Before the physics: the muscles' strength from blood and breath, and where they aim (tracker or reflex)."""
        self.cap = self.brain.muscle_capacity()
        b = self.body
        if self.mode == "scramble":  # the reflex has the body: no reference, no tracker
            o = b.reflex_obs(self.mus, min(self.i - self.i_scr, 149))
            target = b.reflex_target(self.rpolicy(o))
        else:
            o = b.track_obs(self.bq, self.bv, self.bp, self.mus)
            target = b.track_target(self.bq, self.policy(o))
        b.begin(target, self.mus, self.cap)

    def after(self):
        """After the physics: watch the body, let the brain and the matcher move on."""
        self.mus = self.body.s.copy()
        i = self.i
        tnow = (i + 1) * DT
        if self.mode == "scramble":
            self._after_scramble(tnow)
            self.i += 1
            return
        brain = self.brain
        qp, xp, xm = self.data.qpos, self.data.xpos, self.data.xmat
        self.hp = heading(xp[self.hips_id], xp[self.neck_id])
        if not np.all(np.isfinite(qp)):
            raise FloatingPointError(f"physics blew up at {tnow:.2f} s")
        q_ref = self.win.placed(self.th, self.tr)[0][1]
        self.recent.append(q_ref.copy())
        del self.recent[:-LAG]
        tilt = min(up_err(qp[3:7], r[3:7]) for r in self.recent)
        drop = min(r[2] - qp[2] for r in self.recent)
        self.hips_z.append(float(xp[self.hips_id][2]))
        del self.hips_z[:-int(SLUMP_S / DT)]
        event = None
        if self.mode == "plan" and tnow >= self.grace and (tilt > TILT or drop > DROP):
            if self.ask_reg in ("down", "side"):
                event = "relie"
            else:
                roll = self.thorax_roll(xm)
                slump = max(self.hips_z) - self.hips_z[-1]
                event = "fall" if (roll > ROLL or slump > SLUMP) else "lag"
                if event == "lag":
                    self.log["lags"].append({"t": round(tnow, 2), "during": self.ask_reg, "tilt": round(tilt, 2),
                                             "drop": round(float(drop), 2), "roll": round(roll, 2)})
        elif self.mode == "fallen" and tnow - self.t_mode >= LIE_S:
            self.mode, self.t_mode = "rise", tnow
        elif self.mode == "rise" and tnow - self.t_mode >= RISE_S:
            if self.is_up(qp, xp):
                self.mode, self.t_mode, self.grace = "plan", tnow, tnow + 0.5
                self.log["rises"].append(round(tnow, 2))
                brain.got_up(tnow)
            else:
                event = "failed rise"
        gx = self.data.geom_xpos
        if event == "lag":  # the plan waits for the body: re-sync to the captured frame nearest it, keep the ask
            j = self.nearest(self.every, self.up_every, qp, xp, gx)
            self._restart_window(j, qp, mmod.SUPPORT[int(self.mm.db["regime"][j])], self.ask_reg, 15)
            self.grace, event = tnow + 0.5, None
        if event:
            if event != "relie":
                self.log["falls"].append({"t": round(tnow, 2), "what": event, "during": self.ask_reg,
                                          "tilt": round(tilt, 2), "drop": round(float(drop), 2)})
                if event == "fall":
                    brain.fell(tnow)
            if self.rpolicy is not None and event == "fall":  # FALLING: the reflex takes the body
                self.mode, self.t_mode, self.i_scr, self.up_n, self.st_n = "scramble", tnow, i, 0, 0
                self.log["falls"][-1].update(reflex="scramble_up")
                event = None
            else:
                j, deg = self.resync_frame(qp, xp, gx)
                self.lie_reg = mmod.SUPPORT[int(self.mm.db["regime"][j])]
                self._restart_window(j, qp, self.lie_reg, self.lie_reg, 25)
                if event == "relie":
                    self.log["relies"] += 1
                    self.grace = tnow + 1.0
                else:
                    self.mode, self.t_mode = "fallen", tnow
        # the brain keeps living through a fall: its clock, breath and arousal run on
        while self.says:
            text, urg = self.says.pop(0)
            brain.hear(tnow, text, where=self.where, urgency=urg)
            brain.events.append((tnow, f"person: '{text}'" + (" (insisting)" if urg else "")))
        b_reg, _, b_want = brain.tick(tnow, DT, qp[0:2])
        self.player.plan = self.player.same_regime = not (brain.forced and self.mode == "plan")
        want = None
        if self.mode == "plan":
            self.ask_reg, want = b_reg, b_want
        elif self.mode == "fallen":
            self.ask_reg = self.lie_reg
        else:
            self.ask_reg = "stand"
        q_new, self.played = self.player.step(self.ask_reg, want_yaw=None if want is None else want - self.th)
        if self.player.blend_t == 0:
            self.player.blend = 10
        q_new = q_new.copy()  # breathing, laid over the spine of the reference the motors follow
        q_new[self.spine[0]] += brain.breath_offset()
        q_new[self.spine[1]] -= 0.5 * brain.breath_offset()
        self.win.push(q_new)
        if self.anchor > 0 and self.mode != "fallen" and not event:
            h0, p0 = self.win.H[0], self.win.X[0]
            e_yaw = mmod.wrap(self.hp - (h0 + self.th))
            here = mmod.rot2(self.th) @ p0 + self.tr
            self.th = self.th + self.anchor * e_yaw
            self.tr = here - mmod.rot2(self.th) @ p0
            self.tr = self.tr + self.anchor * (qp[0:2] - (mmod.rot2(self.th) @ p0 + self.tr))
        self.bq, self.bv, self.bp = self.win.placed(self.th, self.tr)
        self._set_label()
        self.i += 1

    def _after_scramble(self, tnow):
        brain, b = self.brain, self.body
        qp, xp = self.data.qpos, self.data.xpos
        self.hp = heading(xp[self.hips_id], xp[self.neck_id])
        if not np.all(np.isfinite(qp)):
            raise FloatingPointError(f"physics blew up at {tnow:.2f} s")
        self.up_n = self.up_n + 1 if b.is_up() else 0
        self.st_n = self.st_n + 1 if b.is_sternal() else 0
        brain.tick(tnow, DT, qp[0:2])
        back = "stand" if self.up_n >= 15 else ("down" if self.st_n >= 50 and tnow - self.t_mode > 2.0 else None)
        if back:  # feet back under it (or righted on its belly): the matcher picks up from the body as it is
            gx = self.data.geom_xpos
            j = (self.nearest(self.standing, self.up_stand, qp, xp, gx) if back == "stand"
                 else self.nearest(self.downs, self.up_down, qp, xp, gx))
            self._restart_window(j, qp, back, back, 25)
            self.bq, self.bv, self.bp = self.win.placed(self.th, self.tr)
            self.log["scrambles"].append({"fell": round(self.t_mode, 2), "until": round(tnow, 2), "to": back})
            if back == "stand":
                self.mode, self.t_mode, self.grace = "plan", tnow, tnow + 0.5
                self.log["rises"].append(round(tnow, 2))
                brain.got_up(tnow)
            else:
                self.mode, self.t_mode = "rise", tnow
        self._set_label(reflex=True)

    def _set_label(self, reflex=False):
        b = self.brain
        if reflex:
            what = f"reflex: scramble_up ({self.t - self.t_mode:.1f} s)" if self.mode == "scramble" else "reflex done"
        else:
            what = {"plan": f"{self.ask_reg}", "fallen": f"FALLEN -> {self.lie_reg}", "rise": "getting up"}.get(
                self.mode, self.mode)
        self.label = (f"{self.dog_name.upper()}  {what}   breath {b.breath_rate():.0f}/min   arousal {b.A:.2f}   "
                      f"falls {len([f for f in self.log['falls'] if f['what'] == 'fall'])}")

    def recent_events(self, n=4):
        return [f"{t:.1f}s {msg}" for t, msg in self.brain.events[-n:]]


class Pack:
    """Several dogs in one world: each decides, the world steps once, each takes in what happened."""

    def __init__(self, dogs, tracker, reflex, person=(4.0, 2.0), present=True, seed=3, spacing=1.5, **kw):
        """dogs: e.g. ["lab", "mal"]; one shared world, each dog with its own brain and plan."""
        self.world = World([{"prefix": f"{d}/", "tracker": tracker, "reflex": reflex} for d in dogs])
        self.person = np.array(person, float)  # shared: every brain reads the same array
        self.dogs = [DogRuntime(tracker, reflex, dog=d, seed=seed + 17 * k, present=present, person=self.person,
                                world=self.world, slot=k, place=(0.0, spacing * k), **kw) for k, d in enumerate(dogs)]
        for rt in self.dogs:  # one shared person: the runtimes must not keep private copies
            rt.person = self.person
            if rt.brain.person is not None:
                rt.brain.person = self.person
        self.world.forward()
        # dog with dog: each brain sees where the next dog is (kept up to date every step) and can invite it to play
        self.pos = [rt.data.xpos[rt.hips_id][:2].copy() for rt in self.dogs]
        n = len(self.dogs)
        if n > 1:
            for k, rt in enumerate(self.dogs):
                rt.brain.meet(self.pos[(k + 1) % n], self.dogs[(k + 1) % n].brain)

    @property
    def t(self):
        return self.dogs[0].t

    def step(self):
        for rt in self.dogs:
            rt.decide()
        self.world.physics()
        for k, rt in enumerate(self.dogs):  # where each dog is now, before the brains think
            self.pos[k][:] = rt.data.xpos[rt.hips_id][:2]
        for rt in self.dogs:
            rt.after()
