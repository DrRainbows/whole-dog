"""scramble_up: the righting reflex as its own small network, on the same body and muscles as the tracker.

The lexicon (dog-lexicon/03-locomotion.md, written 19 Nov from text, before any data) already names it:
  scramble_up | Getting the feet back fast after a fall, often followed by shake_off_dry | A | RFX
class R (reflexive: triggered, never chosen), written as trunk.twist + move.go, fast; and the posture graph's
fall -> stand edge reads "scramble up". The capture shows a dog rising from its side only five times (all side ->
down), so a tracker can't learn getting up from it. A reflex needs no reference: in the anatomical joints
(body/anat.py) every angle at zero is the dog's standing pose, so the reflex aims its muscles at offsets from
standing and learns the scramble.

Episodes start with the dog dropped in a fallen orientation (on a side, on its back, half over, or only stumbling)
with its limbs in a captured pose. Reward per step:
  upright x height  (pelvis and thorax each: its up against standing, times its height against standing; a hunched
                     dog with only its hips up doesn't count)
  paws              (on the floor)
  dog-like          (how close the joints are to the nearest captured pose: poses a dog has)
  - ligament strain (the muscles protect the ligaments), - effort
and the sooner it is up the more it collects: "getting the feet back fast".
"""
import jax
import jax.numpy as jp
import numpy as np
from brax.envs.base import State
from mujoco import mjx

from dog_track import DogTrack, quat_inv_apply


def right_config(cfg):
    cfg.episode_length = 150  # 3 s
    cfg.right_scale = 1.0  # radians of target offset from standing per unit action
    cfg.w_up, cfg.w_paws, cfg.w_prior, cfg.w_lig, cfg.w_effort = 1.0, 0.2, 0.5, 0.5, 0.02
    cfg.prior_s2 = 1.5  # squared joint distance (rad^2, summed) at which the dog-like reward falls to 1/e
    cfg.upright_share = 0.2  # share of starts only stumbling (tilted < 30 deg) rather than fallen over
    return cfg


class DogRight(DogTrack):
    """refs: the capture (anatomical joints), for starting poses and the dog-like prior."""

    def __init__(self, xml_path, ref_path, config=None, max_dogs=4096):
        super().__init__(xml_path, ref_path, config=config, max_dogs=max_dogs)
        c = self._config
        reg = self._regime
        q = np.array(self._ref_q)
        stand = reg == 0
        # the trunk's up direction when standing, in the trunk's own frame; and the standing hips height
        w, x, y, z = (q[stand][:, 3 + i] for i in range(4))
        ups = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)  # R^T e_z
        u = np.median(ups, 0)
        self._up_b = jp.array(u / np.linalg.norm(u), jp.float32)
        self._stand_z = float(np.median(q[stand, 2]))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(q), size=min(2048, len(q)), replace=False)
        self._prior = jp.array(q[idx][:, np.array(self._act_q)], jp.float32)  # captured poses (actuated joints)
        self._pool = jp.arange(len(q), dtype=jp.int32)
        # standing, as the capture stands, for pelvis (Hips) and thorax (Spine1): each one's up direction in its own
        # frame, and its height (the thorax measured at the withers, where the neck starts)
        import mujoco
        m, d = self._mj, mujoco.MjData(self._mj)
        self._hips_id, self._thx_id = m.body("Hips").id, m.body("Spine1").id
        self._withers_id = m.body("Neck").id
        uh, ut, zh, zw = [], [], [], []
        for qq in q[stand][:: max(1, int(stand.sum()) // 500)]:
            d.qpos[:] = qq
            mujoco.mj_kinematics(m, d)
            uh.append(d.xmat[self._hips_id].reshape(3, 3).T @ np.array([0, 0, 1.0]))
            ut.append(d.xmat[self._thx_id].reshape(3, 3).T @ np.array([0, 0, 1.0]))
            zh.append(d.xpos[self._hips_id][2])
            zw.append(d.xpos[self._withers_id][2])
        uh, ut = np.median(uh, 0), np.median(ut, 0)
        self._u_hips = jp.array(uh / np.linalg.norm(uh), jp.float32)
        self._u_thx = jp.array(ut / np.linalg.norm(ut), jp.float32)
        self._z_hips, self._z_withers = float(np.median(zh)), float(np.median(zw))
        # every geom but the floor, for placing a start just above the floor: radius and capsule half-length
        m = self._mj
        body_geom = np.array([m.geom_type[g] != 0 for g in range(m.ngeom)])  # 0 = plane
        self._g_on = jp.array(body_geom)
        self._g_r = jp.array(m.geom_size[:, 0], jp.float32)
        self._g_h = jp.array(np.where(m.geom_type == 3, m.geom_size[:, 1], 0.0), jp.float32)  # 3 = capsule
        self._c = c

    def reset(self, rng):
        c = self._c
        rng, r1, r2, r3, r4, r5 = jax.random.split(rng, 6)
        k = jax.random.choice(r1, self._pool)
        base = self._ref_q[k]
        qa = jp.clip(base[self._act_q] + 0.1 * jax.random.normal(r2, (self._mj.nu,)), self._lo, self._hi)
        # tip the trunk over about a random horizontal axis: fallen (60-180 deg) or only stumbling (< 30 deg)
        az = jax.random.uniform(r3, minval=-jp.pi, maxval=jp.pi)
        fallen = jax.random.uniform(r4) > c.upright_share
        ang = jp.where(fallen, jax.random.uniform(r5, minval=1.05, maxval=jp.pi), jax.random.uniform(r5, maxval=0.5))
        axis = jp.array([jp.cos(az), jp.sin(az), 0.0])
        tip = jp.concatenate([jp.cos(ang / 2)[None], jp.sin(ang / 2) * axis])
        w1, x1, y1, z1 = tip
        w2, x2, y2, z2 = base[3], base[4], base[5], base[6]
        quat = jp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
        qpos = base.at[0:3].set(jp.array([0.0, 0.0, 0.45])).at[3:7].set(quat / jp.linalg.norm(quat))
        qpos = qpos.at[self._act_q].set(qa)
        if self._impl == "warp":
            data = mjx.make_data(self._mj, impl="warp", naconmax=self._naconmax, njmax=self._config.njmax)
        else:
            data = mjx.make_data(self._mx)
        data = data.replace(qpos=qpos, qvel=jp.zeros(self._mj.nv), ctrl=jp.zeros(self._mj.nu))
        data = mjx.forward(self._mx, data)
        # lift (or lower) the whole dog so its lowest point starts 3 cm above the floor
        ax_z = data.geom_xmat[:, 2, 2]
        bottom = data.geom_xpos[:, 2] - self._g_h * jp.abs(ax_z) - self._g_r
        low = jp.min(jp.where(self._g_on, bottom, jp.inf))
        data = mjx.forward(self._mx, data.replace(qpos=qpos.at[2].add(0.03 - low)))
        info = {"t": jp.int32(0), "mus": qa, "mus0": qa}
        lo_c, hi_c = self._config.capacity_range  # the brain's muscle capacity (fatigue, winded, arousal)
        info["cap"] = (jax.random.uniform(jax.random.fold_in(rng, 11), minval=lo_c, maxval=hi_c) if hi_c > lo_c
                       else jp.float32(lo_c))
        metrics = {"r_up": jp.zeros(()), "r_prior": jp.zeros(()), "up": jp.zeros(()), "lig": jp.zeros(()),
                   "sat": jp.zeros(())}
        return State(data, self._robs(data, qa, jp.int32(0)), jp.zeros(()), jp.zeros(()), metrics, info)

    def _robs(self, data, aim, t):
        q, v = data.qpos, data.qvel
        rq = q[3:7]
        grav = quat_inv_apply(rq, jp.array([0.0, 0.0, -1.0]))
        paws = data.geom_xpos[self._pads] - q[None, 0:3]
        parts = [q[self._act_q], v[self._act_v] * 0.05, grav, v[3:6] * 0.25, quat_inv_apply(rq, v[0:3]),
                 q[2:3], jax.vmap(lambda p: quat_inv_apply(rq, p))(paws).reshape(-1), data.geom_xpos[self._pads][:, 2],
                 aim - q[self._act_q], data.actuator_force / self._strength,
                 jp.stack([t / self._c.episode_length])]
        return jp.nan_to_num(jp.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, state, action):
        c = self._c
        nu = self._mj.nu
        target = jp.clip(c.right_scale * action[:nu], self._lo, self._hi)  # offsets from standing (all zeros)
        data = state.pipeline_state
        if self._mus:
            data, mus, sat_v = self.muscle_steps(data, target, state.info["mus"], state.info.get("cap", 1.0))
            sat = jp.mean(sat_v)
        else:
            data = jax.lax.fori_loop(0, self._n_sub, lambda _, d: mjx.step(self._mx, d), data.replace(ctrl=target))
            mus, sat = target, jp.zeros(())
        q = data.qpos
        al_h, al_t, h_h, h_w = self._standing(data)
        up = al_h
        r_up = 0.25 * ((al_h + 1.0) * jp.clip(h_h, 0.0, 1.2) + (al_t + 1.0) * jp.clip(h_w, 0.0, 1.2))
        paw_z = data.geom_xpos[self._pads][:, 2]
        r_paws = jp.mean(jp.exp(-jp.maximum(0.0, paw_z - 0.03) / 0.03))
        qa = q[self._act_q]
        d2 = jp.min(jp.sum((self._prior - qa[None]) ** 2, axis=1))
        r_prior = jp.exp(-d2 / c.prior_s2)
        lig = jp.sum(jp.maximum(0.0, qa - self._hi) + jp.maximum(0.0, self._lo - qa))
        effort = jp.mean((data.actuator_force / self._strength) ** 2)
        reward = c.w_up * r_up + c.w_paws * r_paws + c.w_prior * r_prior - c.w_lig * lig - c.w_effort * effort
        bad = ~(jp.all(jp.isfinite(q)) & jp.all(jp.isfinite(data.qvel)))
        done = bad.astype(jp.float32)
        t = state.info["t"] + 1
        restart = done > 0
        if "steps" in state.info:  # Brax's EpisodeWrapper ends the episode after this step (time limit)
            restart = restart | (state.info["steps"] + 1 >= c.episode_length)
        info = dict(state.info, t=jp.where(restart, 0, t), mus=jp.where(restart, state.info["mus0"], mus))
        metrics = dict(state.metrics, r_up=jp.nan_to_num(r_up), r_prior=jp.nan_to_num(r_prior), up=jp.nan_to_num(up),
                       lig=jp.nan_to_num(lig), sat=jp.nan_to_num(sat))
        return state.replace(pipeline_state=data, obs=self._robs(data, mus, t), reward=jp.where(bad, 0.0, jp.nan_to_num(reward)),
                             done=done, metrics=metrics, info=info)

    def _standing(self, data):
        """Pelvis and thorax against standing: each one's up (1 = as when standing) and height (1 = standing height:
        hips for the pelvis, withers for the thorax)."""
        ez = jp.array([0.0, 0.0, 1.0])
        al_h = jp.dot(data.xmat[self._hips_id] @ self._u_hips, ez)
        al_t = jp.dot(data.xmat[self._thx_id] @ self._u_thx, ez)
        return al_h, al_t, data.xpos[self._hips_id][2] / self._z_hips, data.xpos[self._withers_id][2] / self._z_withers

    def is_up(self, data):
        """On its feet as the capture stands: pelvis and thorax each within 35 deg of standing, hips and withers each
        at 75% of standing height or more, and at least three paws on the floor."""
        al_h, al_t, h_h, h_w = self._standing(data)
        paws_down = jp.sum(data.geom_xpos[self._pads][:, 2] < 0.06)
        return (al_h > jp.cos(0.6)) & (al_t > jp.cos(0.6)) & (h_h > 0.75) & (h_w > 0.75) & (paws_down >= 3)

    def is_sternal(self, data):
        """Righted but down: pelvis and thorax each within 35 deg of standing (on its belly, feet under it)."""
        al_h, al_t, _, _ = self._standing(data)
        return (al_h > jp.cos(0.6)) & (al_t > jp.cos(0.6))
