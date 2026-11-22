"""Does the CPU dog (dog_cpu.py) match the JAX environment it was trained in? (run in WSL)

  python parity_cpu.py RUN_DIR [--clip PATTERN] [--steps 150] [--n 8]

1. Observation and action: the same state in both, the same reference window: largest difference in the
   observation and in the policy's action.
2. Physics: the same action sequence applied to both from the same start: how far the bodies drift apart.
3. On its own: the CPU dog (numpy policy, native MuJoCo) tracks the clip by itself, from several starts, ending
   only on a real fall (thorax rolled past 60 deg); how long it lasts, against the JAX dog from the same starts.
"""
import argparse
import glob
import json
import os
import pickle
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from dog_cpu import CpuDog, NumpyPolicy
from dog_track import DogTrack, default_config

W = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--clip", default=os.environ.get("DOG_WORK", "runs") + "/refs_anat/flat/*.npz")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    rc = json.load(open(os.path.join(a.run, "run.json")))
    cfg = default_config()
    cfg.muscles, cfg.obs_drive = rc.get("muscles", ""), rc.get("obs_drive", False)
    cfg.action_scale = rc.get("action_scale", cfg.action_scale)
    cfg.lookahead2, cfg.obs_paws = rc.get("lookahead2", 0), rc.get("obs_paws", False)
    cfg.init_noise = 0.0
    cfg.term = "topple"  # both end only on a real fall (thorax rolled past 60 deg)
    clips = sorted(glob.glob(a.clip))
    clip = max(clips, key=lambda p: np.load(p)["qpos"].shape[0])  # the longest clip: room for long runs
    env = DogTrack(rc["xml"], [clip], config=cfg, max_dogs=8)
    cpu = CpuDog(os.path.join(a.run, "cpu_body.json"))
    npol = NumpyPolicy(os.path.join(a.run, "policy_np.npz"))
    params = pickle.load(open(os.path.join(a.run, "params.pkl"), "rb"))
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize,
                                          **json.load(open(os.path.join(a.run, "net.json"))))
    jpol = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
    jstep = jax.jit(env.step)
    jobs = jax.jit(lambda d, k, end, aim: env._obs(d, k, end, aim=aim))
    RQ, RV, RP = np.array(env._ref_q), np.array(env._ref_v), np.array(env._ref_paws)
    T = RQ.shape[0]
    print(f"clip {os.path.basename(clip)}: {T} frames; obs {env.observation_size}", flush=True)

    def window(k):
        idx = np.minimum(np.arange(k, k + W), T - 1)
        return RQ[idx], RV[idx], RP[idx]

    def cpu_set(st):
        d = cpu.d
        d.qpos[:], d.qvel[:] = np.array(st.pipeline_state.qpos), np.array(st.pipeline_state.qvel)
        d.ctrl[:] = np.array(st.pipeline_state.ctrl)
        d.act[:] = 0
        mujoco.mj_forward(cpu.m, d)
        d.actuator_force[:] = np.array(st.pipeline_state.actuator_force)

    # 1 + 2: lockstep from one start
    k0 = 10
    st = env.reset_at(jax.random.PRNGKey(0), jp.int32(k0))
    cpu_set(st)
    mus = np.array(st.info["mus"]) if "mus" in st.info else None
    worst_obs = worst_act = 0.0
    drift = []
    for t in range(min(a.steps, T - k0 - W - 2)):
        k = k0 + t
        aim_j = st.info["mus"] if "mus" in st.info else st.pipeline_state.ctrl
        o_j = np.array(jobs(st.pipeline_state, jp.int32(k), jp.int32(T - 1), aim_j))
        Q, V, P = window(k)
        aim_c = mus if mus is not None else np.array(cpu.d.ctrl)
        o_c = cpu.obs(Q, V, P, aim_c, k=0, end=T - 1 - k)
        if t == 0:
            worst_obs = float(np.abs(o_j - o_c).max())
            parts = np.argsort(-np.abs(o_j - o_c))[:5]
            print("step 0: largest observation differences at", parts.tolist(), np.abs(o_j - o_c)[parts].round(6).tolist())
        act_j = np.array(jpol(jp.array(o_j, jp.float32), jax.random.PRNGKey(0))[0])
        act_c = npol(o_j)
        worst_act = max(worst_act, float(np.abs(act_j - act_c).max()))
        st = jstep(st, jp.array(act_j))
        mus, _ = cpu.step(Q, act_j, mus) if mus is not None else cpu.step(Q, act_j, None)
        drift.append(float(np.abs(np.array(st.pipeline_state.qpos)[:3] - cpu.d.qpos[:3]).max()))
        if bool(st.done):
            break
    print(f"PARITY observation (same state) {worst_obs:.2e}; action numpy vs JAX {worst_act:.2e}; "
          f"root drift after 1, 10, 50 steps: {[round(drift[i], 4) for i in (0, 9, 49) if i < len(drift)]} m", flush=True)

    # 3: the CPU dog on its own, against the JAX dog, from the same starts; ends only on a real fall (roll)
    thx = cpu.m.body("Spine1").id
    thx_up, thx_side = np.array(env._thx_up), np.array(env._thx_side)
    starts = np.linspace(10, max(11, T - W - a.steps - 2), a.n).astype(int)
    res = []
    t_cpu = 0.0
    n_cpu = 0
    for k0 in starts:
        st = env.reset_at(jax.random.PRNGKey(0), jp.int32(k0))
        st0 = st
        cpu_set(st)
        mus = np.array(st.info["mus"]) if "mus" in st.info else None
        lasted_c = a.steps
        for t in range(a.steps):
            Q, V, P = window(k0 + t)
            t1 = time.perf_counter()
            o = cpu.obs(Q, V, P, mus if mus is not None else np.array(cpu.d.ctrl), k=0, end=T - 1 - k0 - t)
            act = npol(o)
            mus, _ = cpu.step(Q, act, mus)
            t_cpu += time.perf_counter() - t1
            n_cpu += 1
            u = cpu.d.xmat[thx].reshape(3, 3)[2]
            if np.arctan2(abs(u @ thx_side), u @ thx_up) > env._config.topple_roll or not np.all(np.isfinite(cpu.d.qpos)):
                lasted_c = t
                break
        st = st0
        lasted_j = a.steps
        for t in range(a.steps):
            aim_j = st.info["mus"] if "mus" in st.info else st.pipeline_state.ctrl
            o = jobs(st.pipeline_state, jp.int32(k0 + t), jp.int32(T - 1), aim_j)
            st = jstep(st, jpol(o, jax.random.PRNGKey(0))[0])
            if bool(st.done) and int(st.info["k"]) < T - 2:
                lasted_j = t
                break
        res.append((int(k0), lasted_c, lasted_j))
    print("ALONE start: CPU steps / JAX steps (of %d): %s" % (a.steps, res), flush=True)
    print(f"CPU speed: {1000 * t_cpu / max(n_cpu, 1):.2f} ms per 20 ms control step (observation + policy + 5 physics "
          f"steps)", flush=True)


if __name__ == "__main__":
    main()
