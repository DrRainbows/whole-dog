"""The physics dog on the CPU, with no JAX: native MuJoCo, the muscles and the trained policies in numpy.

A game needs one dog at 50 Hz well inside 20 ms per control step. The JAX/GPU loop (online_dog.py) runs one dog at
about 107 ms per step, because the GPU is built for thousands of dogs at once. Here the same body, muscles,
observation and policy run on the CPU:
  - CpuDog(consts):   the body (dog_anat*.xml) set up exactly as dog_track.DogTrack sets it up for training (time
                      step, solver iterations, position motors turned into torque motors when it has muscles);
  - .obs(...):        dog_track.DogTrack._obs, line for line, for the online loop's live reference window (k = 0);
  - .step(...):       the muscles (activation lag, the limb's stiffness, Hill force-velocity, strength scaled by the
                      brain's capacity) over 5 physics steps of 4 ms;
  - NumpyPolicy:      the Brax policy (normalizer, dense layers with swish, tanh of the location).
The constants come from the training environment itself (export_policy.py writes RUN_DIR/cpu_body.json beside
policy_np.npz), so the two can't drift apart. parity_cpu.py checks this file against the JAX environment.
"""
import json
import os

import mujoco
import numpy as np


def swish(x):
    return x / (1.0 + np.exp(-x))


class NumpyPolicy:
    """A Brax PPO policy, deterministic: tanh of the location the network outputs."""

    def __init__(self, path):
        z = np.load(path)
        self.mean, self.std = z["mean"].astype(np.float64), z["std"].astype(np.float64)
        n = int(z["n_layers"])
        self.W = [z[f"W{i}"].astype(np.float64) for i in range(n)]
        self.b = [z[f"b{i}"].astype(np.float64) for i in range(n)]

    def __call__(self, obs):
        x = (np.asarray(obs, np.float64) - self.mean) / self.std
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = swish(x @ W + b)
        out = x @ self.W[-1] + self.b[-1]
        return np.tanh(out[: out.shape[0] // 2])


def quat_inv_apply(q, v):
    """Rotate v by the conjugate of unit quaternion q = (w, x, y, z): world vector -> body frame."""
    w, u = q[0], -np.asarray(q[1:4])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rel_rot6(q_body, q_ref):
    return (quat_to_mat(q_body).T @ quat_to_mat(q_ref))[:, :2].reshape(-1)


class CpuDog:
    """One physics dog on the CPU, set up from the training environment's own constants (cpu_body.json)."""

    def __init__(self, consts, xml=None):
        c = json.load(open(consts)) if isinstance(consts, str) else consts
        self.c = c
        xml = xml or c["xml"]
        if os.name == "nt" and xml.startswith("/") and len(xml) > 3 and xml[2] == "/":  # WSL /c/... -> C:/...
            xml = f"{xml[1].upper()}:{xml[2:]}"
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.m.opt.timestep = c["sim_dt"]
        self.m.opt.iterations = c["solver_iterations"]
        self.m.opt.ls_iterations = c["solver_iterations"]
        self.mus = c.get("muscles_spec") is not None
        if self.mus:  # as in training: every position motor becomes a plain torque motor (force = ctrl)
            self.m.actuator_gainprm[:, 0] = 1.0
            self.m.actuator_biasprm[:, :3] = 0.0
            self.m.actuator_forcelimited[:] = 0
            self.m.actuator_ctrllimited[:] = 0
            M = c["muscles_spec"]
            self.kp, self.kv = np.array(M["kp"]), np.array(M["kv"])
            self.tmax, self.wmax = np.array(M["tmax"]), np.array(M["wmax"])
            self.alpha, self.hk, self.ecc = M["alpha"], M["k"], M["ecc"]
        self.d = mujoco.MjData(self.m)
        self.act_q, self.act_v = np.array(c["act_q"]), np.array(c["act_v"])
        self.lo, self.hi = np.array(c["lo"]), np.array(c["hi"])
        self.strength = np.array(c["strength"])
        self.pads = np.array(c["pads"])
        self.fwd = np.array(c["fwd"])
        self.n_sub, self.nu = int(c["n_sub"]), self.m.nu
        self.action_scale = c["action_scale"]
        self.look, self.look2 = int(c["lookahead"]), int(c["lookahead2"])
        self.obs_paws, self.obs_drive = bool(c["obs_paws"]), bool(c["obs_drive"])
        # what online_dog.py reads from its environment object
        self._mj, self._n_rfc, self._mg = self.m, 0, float(self.m.body_mass.sum()) * 9.81

    # -- the tracker's observation, for the online loop's live window (k = 0, the window already placed) ------------
    def obs(self, Q, V, P, aim, k=0, end=1000):
        d = self.d
        q, v = d.qpos.copy(), d.qvel.copy()
        nxt, far = Q[min(k + 1, end)], Q[min(k + self.look, end)]
        nxt_v = V[min(k + 1, end)]
        rq = q[3:7]
        parts = [q[2:3], quat_inv_apply(rq, np.array([0.0, 0.0, -1.0])), quat_inv_apply(rq, v[0:3]), v[3:6] * 0.25,
                 q[self.act_q], v[self.act_v] * 0.05,
                 nxt[self.act_q] - q[self.act_q], far[self.act_q] - q[self.act_q],
                 quat_inv_apply(rq, nxt[0:3] - q[0:3]), rel_rot6(rq, nxt[3:7]),
                 quat_inv_apply(rq, far[0:3] - q[0:3]), rel_rot6(rq, far[3:7]),
                 quat_inv_apply(rq, nxt_v[0:3]), np.array([min(end - k, 100) / 100.0])]
        if self.look2:
            far2 = Q[min(k + self.look2, end)]
            parts += [far2[self.act_q] - q[self.act_q], quat_inv_apply(rq, far2[0:3] - q[0:3]), rel_rot6(rq, far2[3:7])]
        if self.obs_paws:
            paws = d.geom_xpos[self.pads] - q[None, 0:3]
            raw = Q[min(k + 1, end)]
            ref_paws = P[min(k + 1, end)] - raw[None, 0:3]
            parts += [np.stack([quat_inv_apply(rq, p) for p in paws]).reshape(-1),
                      np.stack([quat_inv_apply(rq, p) for p in ref_paws]).reshape(-1), d.geom_xpos[self.pads][:, 2]]
        if self.obs_drive:
            parts += [aim - q[self.act_q], d.actuator_force / self.strength]
        o = np.concatenate([np.ravel(p) for p in parts])
        return np.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)

    # -- one 20 ms control step: the muscles over n_sub physics steps --------------------------------------------
    def muscle_steps(self, target, mus, cap=1.0):
        d, m = self.d, self.m
        s = np.array(mus, float)
        sat = np.zeros(self.nu)
        for _ in range(self.n_sub):
            s = s + self.alpha * (target - s)  # activation lag
            qd = d.qvel[self.act_v]
            tau = self.kp * (s - d.qpos[self.act_q]) - self.kv * qd  # the limb's stiffness about the target
            x = qd * np.sign(tau) / self.wmax  # > 0: turning the way it pulls (shortening)
            with np.errstate(divide="ignore", invalid="ignore"):
                fv = np.where(x >= 0, np.maximum(0.0, (1.0 - x) / (1.0 + x / self.hk)),
                              1.0 + (self.ecc - 1.0) * (-x) / (-x + 0.1))
            lim = cap * self.tmax * fv
            sat += (np.abs(tau) > lim) / self.n_sub
            d.ctrl[:] = np.clip(tau, -lim, lim)
            mujoco.mj_step(m, d)
        return s, sat

    def step(self, Q, action, mus, cap=1.0, k=0, end=1000):
        """The tracker's step: aim at the next reference frame plus the policy's offset."""
        target = Q[min(k + 1, end)][self.act_q] + self.action_scale * np.asarray(action)[: self.nu]
        if self.mus:
            mus, sat = self.muscle_steps(target, mus, cap)
            return mus, sat
        self.d.ctrl[:] = target
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)
        return target, np.zeros(self.nu)


class CpuReflex:
    """The scramble_up reflex on the same CPU body (dog_right.DogRight): its observation, its step (aims at offsets
    from standing, which is every angle at zero), and its tests for 'on its feet' and 'righted'."""

    def __init__(self, dog, consts):
        c = json.load(open(consts)) if isinstance(consts, str) else consts
        self.dog, self.c = dog, c
        self.right_scale, self.ep_len = c["right_scale"], c["episode_length"]
        self.u_hips, self.u_thx = np.array(c["u_hips"]), np.array(c["u_thx"])
        self.z_hips, self.z_withers = c["z_hips"], c["z_withers"]
        self.hips_id, self.thx_id, self.withers_id = c["hips_id"], c["thx_id"], c["withers_id"]

    def obs(self, aim, t):
        dg = self.dog
        d = dg.d
        q, v = d.qpos.copy(), d.qvel.copy()
        rq = q[3:7]
        paws = d.geom_xpos[dg.pads] - q[None, 0:3]
        parts = [q[dg.act_q], v[dg.act_v] * 0.05, quat_inv_apply(rq, np.array([0.0, 0.0, -1.0])), v[3:6] * 0.25,
                 quat_inv_apply(rq, v[0:3]), q[2:3], np.stack([quat_inv_apply(rq, p) for p in paws]).reshape(-1),
                 d.geom_xpos[dg.pads][:, 2], aim - q[dg.act_q], d.actuator_force / dg.strength,
                 np.array([t / self.ep_len])]
        return np.nan_to_num(np.concatenate([np.ravel(p) for p in parts]), nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action, mus, cap=1.0):
        dg = self.dog
        target = np.clip(self.right_scale * np.asarray(action)[: dg.nu], dg.lo, dg.hi)  # offsets from standing
        if dg.mus:
            mus, _ = dg.muscle_steps(target, mus, cap)
            return mus
        dg.d.ctrl[:] = target
        for _ in range(dg.n_sub):
            mujoco.mj_step(dg.m, dg.d)
        return target

    def _standing(self):
        d = self.dog.d
        al_h = (d.xmat[self.hips_id].reshape(3, 3) @ self.u_hips)[2]
        al_t = (d.xmat[self.thx_id].reshape(3, 3) @ self.u_thx)[2]
        return al_h, al_t, d.xpos[self.hips_id][2] / self.z_hips, d.xpos[self.withers_id][2] / self.z_withers

    def is_up(self):
        al_h, al_t, h_h, h_w = self._standing()
        paws_down = int(np.sum(self.dog.d.geom_xpos[self.dog.pads][:, 2] < 0.06))
        return bool(al_h > np.cos(0.6) and al_t > np.cos(0.6) and h_h > 0.75 and h_w > 0.75 and paws_down >= 3)

    def is_sternal(self):
        al_h, al_t, _, _ = self._standing()
        return bool(al_h > np.cos(0.6) and al_t > np.cos(0.6))


def consts_from_reflex(er):
    """The constants of a dog_right.DogRight environment the CPU reflex needs."""
    c = er._config
    return {"right_scale": c.right_scale, "episode_length": c.episode_length,
            "u_hips": np.asarray(er._u_hips).tolist(), "u_thx": np.asarray(er._u_thx).tolist(),
            "z_hips": float(er._z_hips), "z_withers": float(er._z_withers),
            "hips_id": int(er._hips_id), "thx_id": int(er._thx_id), "withers_id": int(er._withers_id)}


class _Ps:
    """The online loop's view of the physics state (what it reads from a JAX pipeline_state), on a MuJoCo MjData."""

    def __init__(self, d):
        self.d = d

    qpos = property(lambda s: s.d.qpos)
    qvel = property(lambda s: s.d.qvel)
    xpos = property(lambda s: s.d.xpos)
    xmat = property(lambda s: s.d.xmat.reshape(-1, 3, 3))
    geom_xpos = property(lambda s: s.d.geom_xpos)
    ctrl = property(lambda s: s.d.ctrl)
    actuator_force = property(lambda s: s.d.actuator_force)


class CpuState:
    """A stand-in for the JAX env state the online loop passes around: pipeline_state, info, metrics, replace()."""

    def __init__(self, ps, info, metrics):
        self.pipeline_state, self.info, self.metrics = ps, info, metrics

    def replace(self, **kw):
        return CpuState(kw.get("pipeline_state", self.pipeline_state), kw.get("info", self.info),
                        kw.get("metrics", self.metrics))


def online_backend(run, reflex_run=None, init_noise=0.03, seed=0):
    """reset_fn, obs_fn, step_fn and policy for online_dog.py (same calls as its JAX live_fns), and the reflex dict
    (same keys as its load_reflex), all on one CPU body. Needs RUN_DIR/cpu_body.json and policy_np.npz (export_policy
    .py), and REFLEX_DIR/cpu_reflex.json and policy_np.npz (export_policy.py --reflex)."""
    dog = CpuDog(os.path.join(run, "cpu_body.json"))
    pol = NumpyPolicy(os.path.join(run, "policy_np.npz"))
    rng = np.random.default_rng(seed)

    def reset_fn(bq, bv, bp):
        d = dog.d
        mujoco.mj_resetData(dog.m, d)
        q0 = np.array(bq[0], float)
        noise = init_noise * rng.normal(size=dog.m.nq)
        noise[0:7] = 0.0
        q = q0 + noise
        q[dog.act_q] = np.clip(q[dog.act_q], dog.lo, dog.hi)  # start inside the ranges
        d.qpos[:], d.qvel[:] = q, np.array(bv[0], float)
        tgt = q0[dog.act_q]
        d.ctrl[:] = 0.0 if dog.mus else tgt
        mujoco.mj_forward(dog.m, d)
        info = {"k": 0, "end": 1000, "cap": 1.0}
        if dog.mus:
            info["mus"] = tgt.copy()
        return CpuState(_Ps(d), info, {"assist_f": 0.0, "assist_t": 0.0})

    def obs_fn(bq, bv, bp, ps, aim):
        return dog.obs(np.asarray(bq), np.asarray(bv), np.asarray(bp), np.asarray(aim, float))

    def step_fn(bq, bv, bp, st, act):
        push = st.info.get("push")
        dog.d.qfrc_applied[:] = 0.0 if push is None else np.asarray(push, float)
        mus, _ = dog.step(np.asarray(bq), np.asarray(act, float), st.info.get("mus"), float(st.info.get("cap", 1.0)))
        info = dict(st.info, mus=mus) if dog.mus else st.info
        return st.replace(info=info)

    def policy(obs, _key=None):
        return pol(np.asarray(obs)), None

    reflex = None
    if reflex_run:
        rx = CpuReflex(dog, os.path.join(reflex_run, "cpu_reflex.json"))
        rpol = NumpyPolicy(os.path.join(reflex_run, "policy_np.npz"))

        def r_step(data, mus, act, cap):
            dog.d.qfrc_applied[:] = 0.0
            return data, rx.step(np.asarray(act, float), np.asarray(mus, float), float(cap))

        reflex = {"policy": lambda o, _k=None: (rpol(np.asarray(o)), None),
                  "obs": lambda data, mus, t: rx.obs(np.asarray(mus, float), int(t)),
                  "step": r_step, "is_up": lambda data: rx.is_up(), "is_sternal": lambda data: rx.is_sternal()}
    return dog, reset_fn, obs_fn, step_fn, policy, reflex


def consts_from_env(env, xml=None):
    """The constants of a dog_track.DogTrack environment that the CPU dog needs (called in WSL by export_policy)."""
    c = env._config
    out = {"xml": os.path.abspath(xml or env._xml_path), "sim_dt": c.sim_dt, "solver_iterations": c.solver_iterations,
           "act_q": np.asarray(env._act_q).tolist(), "act_v": np.asarray(env._act_v).tolist(),
           "lo": np.asarray(env._lo).tolist(), "hi": np.asarray(env._hi).tolist(),
           "strength": np.asarray(env._strength).tolist(), "pads": np.asarray(env._pads).tolist(),
           "fwd": np.asarray(env._fwd).tolist(), "n_sub": env._n_sub, "action_scale": c.action_scale,
           "lookahead": c.lookahead, "lookahead2": c.lookahead2, "obs_paws": bool(c.obs_paws),
           "obs_drive": bool(c.obs_drive), "muscles_spec": None}
    if env._mus is not None:
        M = env._mus
        out["muscles_spec"] = {"kp": np.asarray(M["kp"]).tolist(), "kv": np.asarray(M["kv"]).tolist(),
                               "tmax": np.asarray(M["tmax"]).tolist(), "wmax": np.asarray(M["wmax"]).tolist(),
                               "alpha": float(M["alpha"]), "k": float(M["k"]), "ecc": float(M["ecc"])}
    return out
