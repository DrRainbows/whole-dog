"""Physics dog that learns to perform a captured motion (DeepMimic-style tracking) in MJX, as a Brax env.

The policy outputs corrections added to the captured joint angles of the next control step;
position actuators then drive the joints. Reward compares the simulated dog with the capture:
joint angles, joint speeds, paw positions, and the trunk's position and orientation.
Episodes start at random moments of the clip and end when the dog strays too far from it.
"""
import json

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from ml_collections import config_dict
from mujoco import mjx

PADS = ("LeftHand_pad", "RightHand_pad", "LeftFoot_pad", "RightFoot_pad")


def default_config():
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=300,
        action_scale=0.3,
        w_pose=0.5, w_vel=0.05, w_paws=0.2, w_root=0.15, w_rot=0.1,
        k_pose=0.5, k_vel=0.025, k_paws=40.0, k_root=10.0, k_rot=2.0,
        fail_root_dist=0.35, fail_rot=0.9, fail_drop=0.2,
        init_noise=0.03,
        impl="warp",  # MuJoCo Warp: about 13x the JAX backend's physics throughput on this model
        contacts_per_dog=48,  # warp's contact buffer is global: sized dogs x this
        njmax=400,  # constraint rows per dog
        solver_iterations=10,
        min_left=60,  # an episode starts at least this many control steps before its clip ends
        lookahead=5,  # second reference frame shown to the policy, in control steps (0.1 s)
        lookahead2=0,  # optional third reference frame (e.g. 15 = 0.3 s), to anticipate footfalls; 0 = off
        obs_paws=False,  # show the policy its paw positions and the capture's, relative to its trunk
        w_rootvel=0.0, k_rootvel=2.0,  # optional reward for matching the capture's trunk velocity (body frame)
        # Residual force control (Yuan & Kitani 2020): six extra actions push and twist the trunk directly, in the
        # trunk's frame. rfc_force = largest push per axis as a share of body weight; rfc_torque = largest twist per
        # axis in units of body weight x 0.3 m. 0 and 0 = no assist (the policy has only its joint motors).
        # w_rfc charges for using it: reward -= w_rfc * mean(assist action ** 2).
        rfc_force=0.0, rfc_torque=0.0, w_rfc=0.0,
        # Anchoring: each step the reference is turned and shifted (about the vertical) toward where the physics
        # dog actually is, closing this share of the gap, so drift never builds up (0 = fixed clip, as trained).
        anchor=0.0,
        # When an episode ends: "track" = strays from the capture (position, orientation, height), as trained;
        # "fall" = only a real fall (trunk tipped fall_tilt rad off the reference's tilt, or hips fail_drop below).
        term="track", fall_tilt=0.8,
        # "topple" = the thorax rolled more than topple_roll about the body's long axis from standing (onto its side
        # or back), whatever the reference does: an absolute fall, so lagging behind a fast rise is not one, and
        # sitting or bowing (pitch) is not either
        topple_roll=1.05,
        # Brax's auto-reset puts back only the body and the observation of this env's FIRST start when an episode
        # ends (and its time limit ends episodes too). The env must put its own clock back to match, or every
        # later episode follows the wrong frames. Off for rollouts that keep simulating past a failure.
        restart_info=True,
        # Random shoves to the trunk (as legged robots are trained to recover from): on average one every
        # push_every seconds (0 = none), sideways or any horizontal direction, up to push_force x body weight,
        # lasting push_dur seconds. The policy isn't told; it has to feel it.
        push_every=0.0, push_force=0.0, push_dur=0.15,
        # how often episodes start in each regime, relative to its share of frames: practise what is hard
        # (moving) more than what the motors already do (lying, sitting). Index = regime id; -1 (unknown) -> 1.
        start_weights=(1.0, 2.0, 4.0, 4.0, 3.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        # extra weight for starts within transition_window control steps before the regime changes (rises, rolls,
        # sitting, lying down): rare in the capture, so practise them more. 1 = off.
        transition_boost=1.0, transition_window=50,
        # changes of pace (starting off, arriving, speeding up): the transition boost treats stand/walk/trot/run as one
        # posture, so these are never boosted; online, half the falls come where the reference walks below 0.3 m/s
        pace_boost=1.0, pace_dv=0.4, pace_slow=0.3,
        # A dog lying on the floor can't fall: while the reference's hips are below lying_z (m), a trunk that lags
        # the reference (a late roll) doesn't end the episode; only straying in position does. 0 = off.
        lying_z=0.0,
        # Muscles (empty = the XML's position motors, instant and as strong as their forcerange). A JSON spec turns
        # every motor into a muscle, computed here at each physics step: the policy's target reaches the muscle
        # through an activation lag (the brain's signal, act_tau seconds); around that lagged target the limb's own
        # stiffness (kp, kv: the local reflex) acts at once; the torque is capped at the joint's strength tau_max,
        # which falls as the joint turns faster in the pull's direction (Hill: zero at w_max, curvature hill_k) and
        # rises up to ecc times when the joint is forced the other way. What the muscles don't hold, the joint
        # ranges in the XML (the ligaments) take.
        muscles="",
        # Show the policy its own drive: where each motor is aiming now (the lagged target, relative to the joint)
        # and how hard it pulls (share of its strength). The user's notes: "actuation is itself a form of knowing
        # where it is ... signal emit act and return".
        obs_drive=False,
        # Physiology as a determiner of strength: every muscle's strength is scaled by info["cap"] (the brain's
        # muscle_capacity(): fatigue, being winded, arousal). Training may draw it per episode from this range.
        capacity_range=(1.0, 1.0),
    )


def quat_inv_apply(q_wxyz, v):
    """Rotate v by the conjugate of unit quaternion q = (w, x, y, z): world vector -> body frame."""
    w, u = q_wxyz[0], -q_wxyz[1:4]
    t = 2.0 * jp.cross(u, v)
    return v + w * t + jp.cross(u, t)


def quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rel_rot6(q_body, q_ref):
    """The reference orientation seen from the body: first two columns of R_body^T R_ref. Continuous and free of
    the quaternion sign ambiguity (q and -q are one rotation; the capture's quaternions flip sign near 180 deg)."""
    return (quat_to_mat(q_body).T @ quat_to_mat(q_ref))[:, :2].reshape(-1)


def qmul(a, b):
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    return jp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rot2(th):
    c, s = jp.cos(th), jp.sin(th)
    return jp.array([[c, -s], [s, c]])


def wrap(a):
    return (a + jp.pi) % (2 * jp.pi) - jp.pi


def up_err(q, r):
    """Angle between the gravity directions seen from two trunk orientations (heading ignored)."""
    ez = jp.array([0.0, 0.0, 1.0])
    return jp.arccos(jp.clip(jp.dot(quat_inv_apply(q, ez), quat_inv_apply(r, ez)), -1.0, 1.0))


def quat_err(q, r):
    """Rotation angle between two unit quaternions (radians)."""
    d = jp.abs(jp.sum(q * r))
    return 2.0 * jp.arccos(jp.clip(d, 0.0, 1.0))


class DogTrack(Env):
    def __init__(self, xml_path, ref_path, config=None, max_dogs=4096):
        self._config = config or default_config()
        self._xml_path = xml_path
        self._mj = mujoco.MjModel.from_xml_path(xml_path)
        self._mj.opt.timestep = self._config.sim_dt
        self._mj.opt.iterations = self._config.solver_iterations
        self._mj.opt.ls_iterations = self._config.solver_iterations
        self._impl = self._config.impl
        self._naconmax = int(max_dogs * self._config.contacts_per_dog)
        self._mus = None
        self._strength = jp.array(np.abs(self._mj.actuator_forcerange[:, 1]) + 1e-6, jp.float32)
        if self._config.muscles:
            spec = json.load(open(self._config.muscles))
            names = [self._mj.joint(self._mj.actuator_trnid[i, 0]).name for i in range(self._mj.nu)]
            kp = self._mj.actuator_gainprm[:, 0].copy()
            kv = -self._mj.actuator_biasprm[:, 2].copy()
            # every position motor becomes a plain torque motor (force = ctrl); the muscle torque is computed below
            self._mj.actuator_gainprm[:, 0] = 1.0
            self._mj.actuator_biasprm[:, :3] = 0.0
            self._mj.actuator_forcelimited[:] = 0
            self._mj.actuator_ctrllimited[:] = 0
            self._mus = {"kp": jp.array(kp, jp.float32), "kv": jp.array(kv, jp.float32),
                         "tmax": jp.array([spec["tau_max"][n] for n in names], jp.float32),
                         "wmax": jp.array([spec["w_max"][n] for n in names], jp.float32),
                         "alpha": float(1.0 - np.exp(-self._config.sim_dt / spec["act_tau"])),
                         "k": float(spec.get("hill_k", 0.25)), "ecc": float(spec.get("ecc", 1.5))}
            self._strength = self._mus["tmax"] + 1e-6
        self._mx = mjx.put_model(self._mj, impl=self._impl)
        # one clip or many: frames are concatenated, and every frame knows where its own clip ends
        paths = [ref_path] if isinstance(ref_path, str) else list(ref_path)
        qs, vs, ps, ends, starts, regs = [], [], [], [], [], []
        off = 0
        for p in paths:
            ref = np.load(p)
            n = int(ref["qpos"].shape[0])
            qs.append(ref["qpos"]), vs.append(ref["qvel"]), ps.append(ref["paws"])
            regs.append(ref["regime"].astype(np.int32) if "regime" in ref.files else np.full(n, -1, np.int32))
            ends.append(np.full(n, off + n - 1, np.int32))
            starts.append(np.arange(off, off + max(1, n - self._config.min_left)))
            off += n
        self._ref_q = jp.array(np.concatenate(qs), dtype=jp.float32)
        self._ref_v = jp.array(np.concatenate(vs), dtype=jp.float32)
        self._ref_paws = jp.array(np.concatenate(ps), dtype=jp.float32)
        self._end = jp.array(np.concatenate(ends))
        self._starts = jp.array(np.concatenate(starts), dtype=jp.int32)
        self._T = off
        self._n_clips = len(paths)
        self._regime = np.concatenate(regs)  # per-frame regime label (numpy), -1 if unknown
        if self._config.term == "topple":  # the thorax's standing up and lateral axes, in its own frame
            self._thx_id = self._mj.body("Spine1").id
            dk = mujoco.MjData(self._mj)
            qs = np.array(self._ref_q)[self._regime == 0] if (self._regime == 0).any() else np.array(self._ref_q)
            U, F = [], []
            for qq in qs[:: max(1, len(qs) // 500)]:
                dk.qpos[:] = qq
                mujoco.mj_kinematics(self._mj, dk)
                Rt = dk.xmat[self._thx_id].reshape(3, 3)
                fw = dk.xpos[self._mj.body("Neck").id] - dk.xpos[self._mj.body("Hips").id]
                U.append(Rt.T @ np.array([0.0, 0.0, 1.0]))
                F.append(Rt.T @ (fw / np.linalg.norm(fw)))
            ub = np.median(U, 0)
            ub /= np.linalg.norm(ub)
            fw = np.median(F, 0)
            fw -= (fw @ ub) * ub
            fw /= np.linalg.norm(fw)
            self._thx_up, self._thx_side = jp.array(ub, jp.float32), jp.array(np.cross(ub, fw), jp.float32)
        sw = np.array(self._config.start_weights, np.float64)
        r0 = self._regime[np.array(self._starts)]
        w = np.where(r0 >= 0, sw[np.clip(r0, 0, len(sw) - 1)], 1.0)
        if self._config.transition_boost != 1.0:
            reg_all, end_all = self._regime, np.concatenate(ends)
            st = np.array(self._starts)
            win = self._config.transition_window
            ahead = np.minimum(st[:, None] + np.arange(1, win + 1)[None, :], end_all[st][:, None])
            # postures, not gaits: stand/walk/trot/run (ids 0-3) count as one, since they flicker into each other
            post = np.where((reg_all >= 0) & (reg_all <= 3), 0, reg_all)
            soon = (post[ahead] != post[st][:, None]).any(1) & (reg_all[st] >= 0)
            w = w * np.where(soon, self._config.transition_boost, 1.0)
            self._soon_share = float(soon.mean())
        if self._config.pace_boost != 1.0:
            v_all = np.linalg.norm(np.concatenate(vs)[:, 0:2], axis=1)  # the reference trunk's ground speed
            st = np.array(self._starts)
            end_all = np.concatenate(ends)
            win = self._config.transition_window
            ahead = np.minimum(st[:, None] + np.arange(1, win + 1)[None, :], end_all[st][:, None])
            va = v_all[ahead]
            pace = ((np.abs(va - v_all[st][:, None]).max(1) > self._config.pace_dv) |  # a big change of speed ahead
                    ((va.min(1) < self._config.pace_slow) & (va.max(1) > self._config.pace_slow)))  # or crossing slow
            w = w * np.where(pace, self._config.pace_boost, 1.0)
            self._pace_share = float(pace.mean())
        self._start_p = jp.array(w / w.sum(), dtype=jp.float32)
        self._pads = jp.array([mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_GEOM, n) for n in PADS])
        # actuator i drives hinge qpos index act_q[i]; its dof index is act_q[i] - 1 (free joint: 7 qpos, 6 dof)
        self._act_q = jp.array([self._mj.jnt_qposadr[self._mj.actuator_trnid[i, 0]] for i in range(self._mj.nu)])
        self._act_v = self._act_q - 1
        jids = [int(self._mj.actuator_trnid[i, 0]) for i in range(self._mj.nu)]
        lim = np.array([bool(self._mj.jnt_limited[j]) for j in jids])
        self._lo = jp.array(np.where(lim, self._mj.jnt_range[jids, 0], -np.inf), jp.float32)
        self._hi = jp.array(np.where(lim, self._mj.jnt_range[jids, 1], np.inf), jp.float32)
        self._n_sub = int(round(self._config.ctrl_dt / self._config.sim_dt))
        self._obs_size = None
        self._mg = float(self._mj.body_mass.sum()) * 9.81  # body weight (N)
        self._n_rfc = 6 if (self._config.rfc_force or self._config.rfc_torque) else 0
        # the trunk's forward axis in the root's own frame (hips toward neck, averaged over the capture): its
        # direction in the world is the dog's heading, for physics and reference alike
        d = mujoco.MjData(self._mj)
        hips = mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_BODY, "Hips")
        neck = mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_BODY, "Neck")
        allq = np.concatenate(qs)
        f = np.zeros(3)
        for qq in allq[:: max(1, len(allq) // 400)]:
            d.qpos[:] = qq
            mujoco.mj_kinematics(self._mj, d)
            v = d.xpos[neck] - d.xpos[hips]
            f += d.xmat[hips].reshape(3, 3).T @ (v / (np.linalg.norm(v) + 1e-9))
        self._fwd = jp.array(f / np.linalg.norm(f), dtype=jp.float32)

    def _yaw(self, quat):
        f = quat_to_mat(quat) @ self._fwd
        return jp.arctan2(f[1], f[0])

    @staticmethod
    def _place(q, th, tr):
        """A reference pose (clip world) placed in the physics world: turned th about the vertical, shifted tr."""
        xy = rot2(th) @ q[0:2] + tr
        quat = qmul(jp.array([jp.cos(th / 2), 0.0, 0.0, jp.sin(th / 2)]), q[3:7])
        return q.at[0:2].set(xy).at[3:7].set(quat)

    def _obs(self, data, k, end, th=0.0, tr=None, aim=None):
        """Proprioception plus the reference just ahead (the next step and 0.1 s on): the reference is the
        command, so one policy can follow any clip. th, tr place the clip in the physics world (anchoring)."""
        tr = jp.zeros(2) if tr is None else tr
        q, v = data.qpos, data.qvel
        nxt = self._place(self._ref_q[jp.minimum(k + 1, end)], th, tr)
        far = self._place(self._ref_q[jp.minimum(k + self._config.lookahead, end)], th, tr)
        nxt_v = self._ref_v[jp.minimum(k + 1, end)]
        nxt_v = nxt_v.at[0:2].set(rot2(th) @ nxt_v[0:2])
        root_quat = q[3:7]
        lin = quat_inv_apply(root_quat, v[0:3])
        ang = v[3:6]  # free-joint angular velocity is already in the body frame
        grav = quat_inv_apply(root_quat, jp.array([0.0, 0.0, -1.0]))
        parts = [
            q[2:3], grav, lin, ang * 0.25,
            q[self._act_q], v[self._act_v] * 0.05,
            nxt[self._act_q] - q[self._act_q],
            far[self._act_q] - q[self._act_q],
            quat_inv_apply(root_quat, nxt[0:3] - q[0:3]), rel_rot6(root_quat, nxt[3:7]),
            quat_inv_apply(root_quat, far[0:3] - q[0:3]), rel_rot6(root_quat, far[3:7]),
            quat_inv_apply(root_quat, nxt_v[0:3]),
            jp.stack([jp.minimum(end - k, 100) / 100.0]),
        ]
        c = self._config
        if c.lookahead2:
            far2 = self._place(self._ref_q[jp.minimum(k + c.lookahead2, end)], th, tr)
            parts += [far2[self._act_q] - q[self._act_q], quat_inv_apply(root_quat, far2[0:3] - q[0:3]),
                      rel_rot6(root_quat, far2[3:7])]
        if c.obs_paws:
            paws = data.geom_xpos[self._pads] - q[None, 0:3]
            raw = self._ref_q[jp.minimum(k + 1, end)]
            ref_paws = self._ref_paws[jp.minimum(k + 1, end)] - raw[None, 0:3]  # around the reference's root
            ref_paws = ref_paws.at[:, 0:2].set(ref_paws[:, 0:2] @ rot2(th).T)
            parts += [jax.vmap(lambda p: quat_inv_apply(root_quat, p))(paws).reshape(-1),
                      jax.vmap(lambda p: quat_inv_apply(root_quat, p))(ref_paws).reshape(-1),
                      data.geom_xpos[self._pads][:, 2]]
        if c.obs_drive:
            parts += [aim - q[self._act_q], data.actuator_force / self._strength]
        return jp.concatenate(parts)

    def reset(self, rng):
        rng, r1 = jax.random.split(rng)
        return self.reset_at(rng, jax.random.choice(r1, self._starts, p=self._start_p))

    def reset_at(self, rng, k):
        """Start on the capture at frame k (evaluation uses this to start at chosen moments)."""
        _, r2 = jax.random.split(rng)
        end = self._end[k]
        noise = self._config.init_noise * jax.random.normal(r2, (self._mj.nq,))
        noise = noise.at[0:7].set(0.0)
        if self._impl == "warp":
            data = mjx.make_data(self._mj, impl="warp", naconmax=self._naconmax, njmax=self._config.njmax)
        else:
            data = mjx.make_data(self._mx)
        qpos = self._ref_q[k] + noise
        qpos = qpos.at[self._act_q].set(jp.clip(qpos[self._act_q], self._lo, self._hi))  # start inside the ranges
        tgt = self._ref_q[k][self._act_q]
        data = data.replace(qpos=qpos, qvel=self._ref_v[k], ctrl=jp.zeros_like(tgt) if self._mus else tgt)
        data = mjx.forward(self._mx, data)
        info = {"k": k, "end": end, "k0": k, "end0": end, "th": jp.zeros(()), "tr": jp.zeros(2)}
        if self._mus:
            info.update(mus=tgt, mus0=tgt, sat_v=jp.zeros(self._mj.nu))  # lagged target; per-muscle limit share
            lo_c, hi_c = self._config.capacity_range
            info["cap"] = jax.random.uniform(jax.random.fold_in(rng, 3), minval=lo_c, maxval=hi_c) if hi_c > lo_c                 else jp.float32(lo_c)
        if self._config.push_every:
            info.update(rng=jax.random.fold_in(rng, 7), push_left=jp.int32(0), push_vec=jp.zeros(3))
        metrics = {"r_pose": jp.zeros(()), "r_paws": jp.zeros(()), "r_root": jp.zeros(()),
                   "assist_f": jp.zeros(()), "assist_t": jp.zeros(()), "lig": jp.zeros(()), "sat": jp.zeros(())}
        return State(data, self._obs(data, k, end, aim=tgt), jp.zeros(()), jp.zeros(()), metrics, info)

    def muscle_steps(self, data, target, mus, cap=1.0):
        """One control step of physics with muscles pulling toward target (see config.muscles). Returns the new
        data, the muscles' lagged target, and per muscle the share of substeps held at its strength limit."""
        M = self._mus

        def sub(_, carry):
            d, s, sat = carry
            s = s + M["alpha"] * (target - s)  # activation lag
            qd = d.qvel[self._act_v]
            tau = M["kp"] * (s - d.qpos[self._act_q]) - M["kv"] * qd  # the limb's stiffness about the target
            x = qd * jp.sign(tau) / M["wmax"]  # > 0: turning the way it pulls (shortening)
            fv = jp.where(x >= 0, jp.maximum(0.0, (1.0 - x) / (1.0 + x / M["k"])),
                          1.0 + (M["ecc"] - 1.0) * (-x) / (-x + 0.1))
            lim = cap * M["tmax"] * fv
            sat = sat + (jp.abs(tau) > lim).astype(jp.float32) / self._n_sub
            return mjx.step(self._mx, d.replace(ctrl=jp.clip(tau, -lim, lim))), s, sat

        return jax.lax.fori_loop(0, self._n_sub, sub, (data, mus, jp.zeros(self._mj.nu)))

    def step(self, state, action):
        k = state.info["k"] + 1
        end = state.info["end"]
        k_ref = jp.minimum(k, end)
        c = self._config
        nu = self._mj.nu
        target = self._ref_q[k_ref][self._act_q] + c.action_scale * action[:nu]
        data = state.pipeline_state.replace(ctrl=target)
        assist_f = assist_t = jp.zeros(())
        push = state.info.get("push")  # optional outside force on the trunk (generalized, nv), e.g. a bump
        shove = {}
        if c.push_every:  # random shoves, decided here so every simulated dog gets its own
            rng, r1, r2, r3 = jax.random.split(state.info["rng"], 4)
            start = (state.info["push_left"] <= 0) & (jax.random.uniform(r1) < c.ctrl_dt / c.push_every)
            ang = jax.random.uniform(r2, minval=-jp.pi, maxval=jp.pi)
            mag = jax.random.uniform(r3) * c.push_force * self._mg
            vec = jp.where(start, jp.array([jp.cos(ang), jp.sin(ang), 0.0]) * mag, state.info["push_vec"])
            left = jp.where(start, jp.int32(round(c.push_dur / c.ctrl_dt)), state.info["push_left"] - 1)
            base = jp.zeros(self._mj.nv) if push is None else push
            push = base.at[0:3].add(jp.where(left > 0, vec, 0.0))
            shove = {"rng": rng, "push_left": left, "push_vec": vec}
        if self._n_rfc or push is not None:
            qfrc = jp.zeros(self._mj.nv) if push is None else push
            if self._n_rfc:
                # free joint: generalized force = world-frame force at the trunk origin, then torque in trunk frame
                a = action[nu:nu + 6]
                f_body = c.rfc_force * self._mg * a[0:3]
                tau = c.rfc_torque * self._mg * 0.3 * a[3:6]
                qfrc = qfrc.at[0:3].add(quat_to_mat(data.qpos[3:7]) @ f_body).at[3:6].add(tau)
                assist_f = jp.linalg.norm(f_body) / self._mg  # share of body weight
                assist_t = jp.linalg.norm(tau) / (self._mg * 0.3)
            data = data.replace(qfrc_applied=qfrc)
        mus, sat, sat_v = None, jp.zeros(()), None
        if self._mus:
            data, mus, sat_v = self.muscle_steps(data, target, state.info["mus"], state.info.get("cap", 1.0))
            sat = jp.mean(sat_v)
        else:
            data = jax.lax.fori_loop(0, self._n_sub, lambda _, d: mjx.step(self._mx, d), data)
        q, v = data.qpos, data.qvel
        th, tr = state.info["th"], state.info["tr"]
        rq_raw, rv = self._ref_q[k_ref], self._ref_v[k_ref]
        rq = self._place(rq_raw, th, tr)
        rv = rv.at[0:2].set(rot2(th) @ rv[0:2])
        e_pose = jp.sum((q[self._act_q] - rq[self._act_q]) ** 2)
        e_vel = jp.sum((v[self._act_v] - rv[self._act_v]) ** 2)
        rel = data.geom_xpos[self._pads] - q[None, 0:3]
        rel_ref = self._ref_paws[k_ref] - rq_raw[None, 0:3]
        rel_ref = rel_ref.at[:, 0:2].set(rel_ref[:, 0:2] @ rot2(th).T)
        e_paws = jp.sum((rel - rel_ref) ** 2)
        e_root = jp.sum((q[0:3] - rq[0:3]) ** 2)
        e_rot = quat_err(q[3:7], rq[3:7])
        r_pose, r_vel = jp.exp(-c.k_pose * e_pose), jp.exp(-c.k_vel * e_vel)
        r_paws, r_root, r_rot = jp.exp(-c.k_paws * e_paws), jp.exp(-c.k_root * e_root), jp.exp(-c.k_rot * e_rot ** 2)
        reward = c.w_pose * r_pose + c.w_vel * r_vel + c.w_paws * r_paws + c.w_root * r_root + c.w_rot * r_rot
        if c.w_rootvel:  # trunk velocity in the trunk's own frame, against the capture's in its own frame
            e_rv = jp.sum((quat_inv_apply(q[3:7], v[0:3]) - quat_inv_apply(rq[3:7], rv[0:3])) ** 2)
            reward = reward + c.w_rootvel * jp.exp(-c.k_rootvel * e_rv)
        if self._n_rfc:
            reward = reward - c.w_rfc * jp.mean(action[nu:nu + 6] ** 2)
        qa = q[self._act_q]  # ligament strain: how far the joints are pushed past their ranges (radians, summed)
        lig = jp.sum(jp.maximum(0.0, qa - self._hi) + jp.maximum(0.0, self._lo - qa))
        bad = ~(jp.all(jp.isfinite(q)) & jp.all(jp.isfinite(v)))  # rare float32 blow-up after long falls
        if c.term == "fall":
            fail = (up_err(q[3:7], rq[3:7]) > c.fall_tilt) | (q[2] < rq[2] - c.fail_drop) | bad
        elif c.term == "topple":
            u = data.xmat[self._thx_id][2]  # world up, seen in the thorax's frame
            roll = jp.arctan2(jp.abs(jp.dot(u, self._thx_side)), jp.dot(u, self._thx_up))
            fail = (roll > c.topple_roll) | bad
        else:
            rot_bad = e_rot > c.fail_rot
            if c.lying_z:
                rot_bad = rot_bad & (rq[2] >= c.lying_z)
            fail = (jp.sqrt(e_root) > c.fail_root_dist) | rot_bad | (q[2] < rq[2] - c.fail_drop) | bad
        done = (fail | (k >= end)).astype(jp.float32)
        if c.anchor:
            # pull the reference toward the dog: turn it about its own root by a share of the heading gap,
            # then shift it by the same share of the position gap
            e_yaw = wrap(self._yaw(q[3:7]) - self._yaw(rq[3:7]))
            here = rq[0:2]
            th = th + c.anchor * e_yaw
            tr = here - rot2(th) @ rq_raw[0:2]
            tr = tr + c.anchor * (q[0:2] - here)
        k_out, end_out = k, end
        if c.restart_info:
            restart = done > 0
            if "steps" in state.info:  # Brax's EpisodeWrapper ends the episode after this step (time limit)
                restart = restart | (state.info["steps"] + 1 >= c.episode_length)
            k_out = jp.where(restart, state.info["k0"], k)
            end_out = jp.where(restart, state.info["end0"], end)
            th, tr = jp.where(restart, 0.0, th), jp.where(restart, jp.zeros(2), tr)
            if shove:
                shove["push_left"] = jp.where(restart, 0, shove["push_left"])
        if mus is not None:
            shove["mus"] = jp.where(done > 0, state.info["mus0"], mus) if c.restart_info else mus
            shove["sat_v"] = sat_v
        info = dict(state.info, k=k_out, end=end_out, th=th, tr=tr, **shove)
        # update in place: Brax wrappers add their own keys (e.g. "reward") and the pytree must keep its shape
        metrics = dict(state.metrics, r_pose=jp.nan_to_num(r_pose), r_paws=jp.nan_to_num(r_paws),
                       r_root=jp.nan_to_num(r_root), assist_f=jp.nan_to_num(assist_f), assist_t=jp.nan_to_num(assist_t),
                       lig=jp.nan_to_num(lig), sat=jp.nan_to_num(sat))
        obs = jp.nan_to_num(self._obs(data, k_ref, end, th, tr, aim=target if mus is None else mus),
                            nan=0.0, posinf=0.0, neginf=0.0)
        return state.replace(pipeline_state=data, obs=obs, reward=jp.where(bad, 0.0, jp.nan_to_num(reward)),
                             done=done, metrics=metrics, info=info)

    @property
    def observation_size(self):
        if self._obs_size is None:
            self._obs_size = int(jax.jit(self.reset)(jax.random.PRNGKey(0)).obs.shape[-1])
        return self._obs_size

    @property
    def action_size(self):
        return self._mj.nu + self._n_rfc

    @property
    def backend(self):
        return "mjx"
