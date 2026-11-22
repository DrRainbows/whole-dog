"""Several physics dogs in one MuJoCo world (on the CPU, no JAX).

Each dog was trained alone, in the single-dog layout of dog_anat*.xml. Here the world is composed from copies of that
body (MjSpec: one copy per dog, its names prefixed "lab/", "mal/", ...), and every dog sees and drives the world through
index maps from its own single-dog layout (joints, dofs, actuators, bodies, geoms) to the world's. So each dog's
observation, muscles, policy and reflex are exactly the ones it was trained with, and the dogs share one floor and can
bump into each other: each dog's capsules collide with the floor and with the other dogs, never with themselves.

    world = World([{"prefix": "lab/", "tracker": T, "reflex": R}, {"prefix": "mal/", "tracker": T, "reflex": R}])
    body = world.bodies[0]      # the Lab's view: body.q(), body.xpos(), body.obs(...), body.begin(...)
    world.physics()             # one 20 ms control step: every dog's muscles, then the shared physics, 5 x 4 ms
"""
import json
import os

import mujoco
import numpy as np

QN = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4, mujoco.mjtJoint.mjJNT_SLIDE: 1,
      mujoco.mjtJoint.mjJNT_HINGE: 1}
VN = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3, mujoco.mjtJoint.mjJNT_SLIDE: 1,
      mujoco.mjtJoint.mjJNT_HINGE: 1}


def win_path(x):
    if os.name == "nt" and x.startswith("/") and len(x) > 3 and x[2] == "/":  # WSL /c/... -> C:/...
        return f"{x[1].upper()}:{x[2:]}"
    return x


def quat_inv_apply(q, v):
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


class WorldBody:
    """One dog's view of the shared world, in its own (trained) single-dog layout."""

    def __init__(self, world, index, prefix, tracker, reflex=None):
        self.world, self.index, self.prefix = world, index, prefix
        c = json.load(open(os.path.join(tracker, "cpu_body.json")))
        self.c = c
        single = mujoco.MjModel.from_xml_path(win_path(c["xml"]))
        self.single = single
        m = world.m
        self.qmap = np.zeros(single.nq, int)
        self.vmap = np.zeros(single.nv, int)
        for j in range(single.njnt):
            J = m.joint(prefix + single.joint(j).name).id
            nq, nv = QN[single.jnt_type[j]], VN[single.jnt_type[j]]
            self.qmap[single.jnt_qposadr[j]:single.jnt_qposadr[j] + nq] = m.jnt_qposadr[J] + np.arange(nq)
            self.vmap[single.jnt_dofadr[j]:single.jnt_dofadr[j] + nv] = m.jnt_dofadr[J] + np.arange(nv)
        self.amap = np.array([m.actuator(prefix + single.actuator(i).name).id for i in range(single.nu)])
        self.bmap = np.array([0] + [m.body(prefix + single.body(b).name).id for b in range(1, single.nbody)])
        self.gmap = np.zeros(single.ngeom, int)
        floor = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE][0]
        for b in range(single.nbody):
            sg = [g for g in range(single.ngeom) if single.geom_bodyid[g] == b]
            if b == 0:  # the floor
                for g in sg:
                    self.gmap[g] = floor
                continue
            wg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.bmap[b]]
            assert len(sg) == len(wg), (single.body(b).name, len(sg), len(wg))
            for g1, g2 in zip(sg, wg):
                self.gmap[g1] = g2
        # the trained constants (single-dog layout)
        self.act_q, self.act_v = np.array(c["act_q"]), np.array(c["act_v"])
        self.lo, self.hi = np.array(c["lo"]), np.array(c["hi"])
        self.strength = np.array(c["strength"])
        self.pads = np.array(c["pads"])
        self.n_sub, self.nu = int(c["n_sub"]), single.nu
        self.action_scale = c["action_scale"]
        self.look, self.look2 = int(c["lookahead"]), int(c["lookahead2"])
        self.obs_paws, self.obs_drive = bool(c["obs_paws"]), bool(c["obs_drive"])
        M = c["muscles_spec"]
        assert M is not None, "the world runs dogs with muscles"
        self.kp, self.kv = np.array(M["kp"]), np.array(M["kv"])
        self.tmax, self.wmax = np.array(M["tmax"]), np.array(M["wmax"])
        self.alpha, self.hk, self.ecc = M["alpha"], M["k"], M["ecc"]
        self.mg = float(single.body_mass.sum()) * 9.81
        self.r = None
        if reflex:
            r = json.load(open(os.path.join(reflex, "cpu_reflex.json")))
            self.r = r
            self.u_hips, self.u_thx = np.array(r["u_hips"]), np.array(r["u_thx"])
        self.target = np.zeros(self.nu)
        self.s = np.zeros(self.nu)
        self.cap = 1.0
        self.sat = np.zeros(self.nu)
        self.push = np.zeros(6)  # an outside force and torque on the trunk this control step

    # -- the dog's own state, in its trained layout (copies) ------------------------------------------------------
    def q(self):
        return self.world.d.qpos[self.qmap].copy()

    def v(self):
        return self.world.d.qvel[self.vmap].copy()

    def xpos(self):
        return self.world.d.xpos[self.bmap].copy()

    def xmat(self):
        return self.world.d.xmat[self.bmap].copy()

    def xquat(self):
        return self.world.d.xquat[self.bmap].copy()

    def geom_xpos(self):
        return self.world.d.geom_xpos[self.gmap].copy()

    def actuator_force(self):
        return self.world.d.actuator_force[self.amap].copy()

    def set_state(self, qpos, qvel):
        self.world.d.qpos[self.qmap] = qpos
        self.world.d.qvel[self.vmap] = qvel
        self.world.d.ctrl[self.amap] = 0.0

    # -- the tracker (dog_track.DogTrack._obs for a live window, k = 0) -------------------------------------------
    def track_obs(self, Q, V, P, aim, k=0, end=1000):
        q, v = self.q(), self.v()
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
        gx = self.geom_xpos()
        if self.obs_paws:
            paws = gx[self.pads] - q[None, 0:3]
            raw = Q[min(k + 1, end)]
            ref_paws = P[min(k + 1, end)] - raw[None, 0:3]
            parts += [np.stack([quat_inv_apply(rq, p) for p in paws]).reshape(-1),
                      np.stack([quat_inv_apply(rq, p) for p in ref_paws]).reshape(-1), gx[self.pads][:, 2]]
        if self.obs_drive:
            parts += [aim - q[self.act_q], self.actuator_force() / self.strength]
        return np.nan_to_num(np.concatenate([np.ravel(p) for p in parts]), nan=0.0, posinf=0.0, neginf=0.0)

    def track_target(self, Q, action, k=0, end=1000):
        return Q[min(k + 1, end)][self.act_q] + self.action_scale * np.asarray(action)[: self.nu]

    # -- the reflex (dog_right.DogRight) --------------------------------------------------------------------------
    def reflex_obs(self, aim, t):
        q, v = self.q(), self.v()
        rq = q[3:7]
        gx = self.geom_xpos()
        paws = gx[self.pads] - q[None, 0:3]
        parts = [q[self.act_q], v[self.act_v] * 0.05, quat_inv_apply(rq, np.array([0.0, 0.0, -1.0])), v[3:6] * 0.25,
                 quat_inv_apply(rq, v[0:3]), q[2:3], np.stack([quat_inv_apply(rq, p) for p in paws]).reshape(-1),
                 gx[self.pads][:, 2], aim - q[self.act_q], self.actuator_force() / self.strength,
                 np.array([t / self.r["episode_length"]])]
        return np.nan_to_num(np.concatenate([np.ravel(p) for p in parts]), nan=0.0, posinf=0.0, neginf=0.0)

    def reflex_target(self, action):
        return np.clip(self.r["right_scale"] * np.asarray(action)[: self.nu], self.lo, self.hi)

    def _standing(self):
        xm, xp = self.xmat(), self.xpos()
        r = self.r
        al_h = (xm[r["hips_id"]].reshape(3, 3) @ self.u_hips)[2]
        al_t = (xm[r["thx_id"]].reshape(3, 3) @ self.u_thx)[2]
        return al_h, al_t, xp[r["hips_id"]][2] / r["z_hips"], xp[r["withers_id"]][2] / r["z_withers"]

    def is_up(self):
        al_h, al_t, h_h, h_w = self._standing()
        paws_down = int(np.sum(self.geom_xpos()[self.pads][:, 2] < 0.06))
        return bool(al_h > np.cos(0.6) and al_t > np.cos(0.6) and h_h > 0.75 and h_w > 0.75 and paws_down >= 3)

    def is_sternal(self):
        al_h, al_t, _, _ = self._standing()
        return bool(al_h > np.cos(0.6) and al_t > np.cos(0.6))

    # -- the muscles, one 4 ms physics step at a time (the world steps every dog together) ------------------------
    def begin(self, target, mus, cap=1.0):
        self.target, self.s, self.cap = np.asarray(target, float), np.array(mus, float), float(cap)
        self.sat = np.zeros(self.nu)

    def substep_torque(self):
        d = self.world.d
        self.s = self.s + self.alpha * (self.target - self.s)  # activation lag
        qd = d.qvel[self.vmap][self.act_v]
        tau = self.kp * (self.s - d.qpos[self.qmap][self.act_q]) - self.kv * qd  # the limb's stiffness about the aim
        x = qd * np.sign(tau) / self.wmax  # > 0: turning the way it pulls (shortening)
        with np.errstate(divide="ignore", invalid="ignore"):
            fv = np.where(x >= 0, np.maximum(0.0, (1.0 - x) / (1.0 + x / self.hk)),
                          1.0 + (self.ecc - 1.0) * (-x) / (-x + 0.1))
        lim = self.cap * self.tmax * fv
        self.sat += (np.abs(tau) > lim) / self.n_sub
        return np.clip(tau, -lim, lim)


class World:
    def __init__(self, dogs, spread=1.5):
        """dogs: [{"prefix": "lab/", "tracker": RUN_DIR, "reflex": RUN_DIR or None}, ...] (same body for all)."""
        c = json.load(open(os.path.join(dogs[0]["tracker"], "cpu_body.json")))
        xml = win_path(c["xml"])
        base = mujoco.MjSpec.from_file(xml)
        base.delete(base.body("Hips"))
        for i, dg in enumerate(dogs):
            src = mujoco.MjSpec.from_file(win_path(json.load(open(os.path.join(dg["tracker"], "cpu_body.json")))["xml"]))
            fr = base.worldbody.add_frame(pos=[0.0, spread * i, 0.0])
            fr.attach_body(src.body("Hips"), dg["prefix"], "")
        m = base.compile()
        m.opt.timestep = c["sim_dt"]
        m.opt.iterations = c["solver_iterations"]
        m.opt.ls_iterations = c["solver_iterations"]
        # as in training: every position motor becomes a plain torque motor (force = ctrl); the muscles are computed
        m.actuator_gainprm[:, 0] = 1.0
        m.actuator_biasprm[:, :3] = 0.0
        m.actuator_forcelimited[:] = 0
        m.actuator_ctrllimited[:] = 0
        # contacts: each dog with the floor (bit 2, as trained) and with every other dog, never with itself
        bits = [1 << (3 + i) for i in range(len(dogs))]
        owner = {}
        for i, dg in enumerate(dogs):
            for b in range(m.nbody):
                if m.body(b).name.startswith(dg["prefix"]):
                    owner[b] = i
        for g in range(m.ngeom):
            i = owner.get(int(m.geom_bodyid[g]))
            if i is None:
                continue
            m.geom_contype[g] = 2 | bits[i]
            m.geom_conaffinity[g] = sum(bits) - bits[i]
        self.m, self.d = m, mujoco.MjData(m)
        self.bodies = [WorldBody(self, i, dg["prefix"], dg["tracker"], dg.get("reflex")) for i, dg in enumerate(dogs)]
        self.n_sub = self.bodies[0].n_sub

    def forward(self):
        mujoco.mj_forward(self.m, self.d)

    def physics(self):
        """One 20 ms control step for every dog: each one's muscles, then the shared physics, n_sub x 4 ms."""
        d = self.d
        d.qfrc_applied[:] = 0.0
        for b in self.bodies:
            d.qfrc_applied[b.vmap[0:6]] = b.push
        for _ in range(self.n_sub):
            for b in self.bodies:
                d.ctrl[b.amap] = b.substep_torque()
            mujoco.mj_step(self.m, d)
        for b in self.bodies:
            b.push[:] = 0.0
