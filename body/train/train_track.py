"""Train the tracking policy with Brax PPO on the GPU (run inside WSL2).

  python train_track.py REF.npz [--steps 60000000] [--envs 4096] [--out runs/NAME]

Writes the policy parameters, the learning curve, and an evaluation: how long the
simulated dog stays on the capture from fresh starts, and its tracking errors.
"""
import argparse
import functools
import json
import os
import pickle
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")
import jax

jax.config.update("jax_default_matmul_precision", "tensorfloat32")
import jax.numpy as jp
import numpy as np

if not hasattr(jax, "device_put_replicated"):
    # Brax 0.14.2's PPO still calls jax.device_put_replicated, which JAX 0.12 removed. Same layout as before:
    # one copy per device stacked on a new leading axis, sharded across the devices.
    def _device_put_replicated(x, devices):
        mesh = jax.sharding.Mesh(np.array(devices), ("d",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
        return jax.tree.map(lambda a: jax.device_put(jp.stack([jp.asarray(a)] * len(devices)), sharding), x)

    jax.device_put_replicated = _device_put_replicated
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from dog_track import DogTrack, default_config

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(os.path.dirname(HERE), "dog_mocap_mjx.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", nargs="+", help="one or more reference .npz files (globs allowed)")
    ap.add_argument("--steps", type=int, default=60_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--minibatches", type=int, default=32)
    ap.add_argument("--evals", type=int, default=12)
    ap.add_argument("--eval-envs", type=int, default=128)
    ap.add_argument("--dist", default="tanh_normal", choices=["tanh_normal", "normal"])
    ap.add_argument("--noise", type=float, default=1.0, help="initial exploration std of the policy")
    ap.add_argument("--impl", default="warp", choices=["warp", "jax"], help="physics backend")
    ap.add_argument("--gamma", type=float, default=0.95, help="discount (0.95 at 50 Hz looks about 0.4 s ahead)")
    ap.add_argument("--action-scale", type=float, default=None, help="radians of target offset per unit action")
    ap.add_argument("--lookahead2", type=int, default=0, help="third reference frame, control steps ahead (0 = off)")
    ap.add_argument("--paws-obs", action="store_true", help="show the policy its paws and the capture's")
    ap.add_argument("--w-rootvel", type=float, default=0.0, help="weight of a trunk-velocity reward")
    ap.add_argument("--net", default="512,256,128", help="hidden layer sizes of the policy and value networks")
    ap.add_argument("--rfc-force", type=float, default=0.0, help="trunk assist: largest push per axis, share of body weight")
    ap.add_argument("--rfc-torque", type=float, default=0.0, help="trunk assist: largest twist per axis, body weight x 0.3 m")
    ap.add_argument("--push-every", type=float, default=0.0, help="random shoves: mean seconds between (0 = none)")
    ap.add_argument("--push-force", type=float, default=0.0, help="random shoves: largest, in body weights")
    ap.add_argument("--restore", default="", help="start from this run directory's params.pkl (same network and action size)")
    ap.add_argument("--transition-boost", type=float, default=1.0, help="weight on starts just before a regime change")
    ap.add_argument("--lying-z", type=float, default=0.0, help="no orientation failure while the reference hips are below this (m)")
    ap.add_argument("--resets", type=int, default=5, help="full resets of every env per evaluation period")
    ap.add_argument("--w-rfc", type=float, default=0.0, help="penalty weight on using the trunk assist")
    ap.add_argument("--xml", default=XML, help="the body (e.g. ../dog_anat.xml: anatomical joints with ligament ranges)")
    ap.add_argument("--muscles", default="", help="muscle spec (JSON): dog-strength torque, force-velocity, activation lag")
    ap.add_argument("--obs-drive", action="store_true", help="show the policy its own drive (aim and pull)")
    ap.add_argument("--pace-boost", type=float, default=1.0, help="weight on starts just before a change of pace "
                    "(speed changes 0.4 m/s, or crosses 0.3 m/s, within the transition window)")
    ap.add_argument("--capacity", default="1,1", help="LO,HI: each episode's muscle capacity drawn from this range "
                    "(fatigue, being winded, arousal: brain.muscle_capacity() gives 0.5-1.1); needs --muscles")
    ap.add_argument("--task", default="track", choices=["track", "right"],
                    help="track: follow references (DogTrack); right: the scramble_up reflex (DogRight, no reference)")
    ap.add_argument("--keep-checkpoints", action="store_true", help="also keep the policy of every evaluation (params_STEP.pkl)")
    a = ap.parse_args()
    import glob

    refs = sorted(p for r in a.ref for p in (glob.glob(r) if any(c in r for c in "*?[") else [r]))
    name = os.path.splitext(os.path.basename(refs[0]))[0] if len(refs) == 1 else f"{len(refs)}clips"
    out = a.out or os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    hidden = tuple(int(x) for x in a.net.split(","))
    net_cfg = {"policy_hidden_layer_sizes": hidden, "value_hidden_layer_sizes": hidden,
               "distribution_type": a.dist, "noise_std_type": "log" if a.dist == "normal" else "scalar",
               "init_noise_std": a.noise}
    json.dump(net_cfg, open(os.path.join(out, "net.json"), "w"), indent=1)  # rollout.py rebuilds the same network
    cfg = default_config()
    cfg.impl = a.impl
    if a.action_scale is not None:
        cfg.action_scale = a.action_scale
    cfg.lookahead2, cfg.obs_paws, cfg.w_rootvel = a.lookahead2, bool(a.paws_obs), a.w_rootvel
    cfg.rfc_force, cfg.rfc_torque, cfg.w_rfc = a.rfc_force, a.rfc_torque, a.w_rfc
    cfg.push_every, cfg.push_force = a.push_every, a.push_force
    cfg.transition_boost = a.transition_boost
    cfg.lying_z = a.lying_z
    cfg.muscles = a.muscles
    cfg.obs_drive = bool(a.obs_drive)
    cfg.capacity_range = tuple(float(x) for x in a.capacity.split(","))
    cfg.pace_boost = a.pace_boost
    assert len(cfg.capacity_range) == 2 and (cfg.capacity_range == (1.0, 1.0) or a.muscles), "--capacity needs --muscles"
    json.dump({"gamma": a.gamma, "action_scale": float(cfg.action_scale), "steps": a.steps, "envs": a.envs,
               "refs": len(refs), "lookahead2": a.lookahead2, "obs_paws": bool(a.paws_obs), "w_rootvel": a.w_rootvel,
               "rfc_force": a.rfc_force, "rfc_torque": a.rfc_torque, "w_rfc": a.w_rfc,
               "push_every": a.push_every, "push_force": a.push_force, "restore": a.restore, "transition_boost": a.transition_boost, "lying_z": a.lying_z,
               "xml": os.path.abspath(a.xml), "muscles": os.path.abspath(a.muscles) if a.muscles else "",
               "obs_drive": bool(a.obs_drive), "capacity_range": list(cfg.capacity_range), "pace_boost": a.pace_boost},
              open(os.path.join(out, "run.json"), "w"), indent=1)
    Env = DogTrack
    if a.task == "right":
        from dog_right import DogRight, right_config
        Env, cfg = DogRight, right_config(cfg)
    env = Env(a.xml, refs, config=cfg, max_dogs=a.envs)
    print(f"{len(refs)} clip(s), {env._T} frames ({env._T * 0.02 / 60:.1f} min), obs {env.observation_size}, "
          f"actions {env.action_size}" + (f", starts near a transition {env._soon_share:.0%}" if hasattr(env, "_soon_share") else ""), flush=True)
    curve = []
    t0 = time.time()

    def checkpoint(step, make_policy, params):
        """Save the policy at every evaluation, so a stopped run keeps what it learned."""
        with open(os.path.join(out, "params.pkl"), "wb") as f:
            pickle.dump(params, f)
        if a.keep_checkpoints:
            with open(os.path.join(out, f"params_{int(step)}.pkl"), "wb") as f:
                pickle.dump(params, f)

    def progress(step, m):
        row = {"step": int(step), "minutes": round((time.time() - t0) / 60, 2),
               "reward": float(m.get("eval/episode_reward", np.nan)),
               "length": float(m.get("eval/avg_episode_length", np.nan)),
               "r_pose": float(m.get("eval/episode_r_pose", np.nan)),
               "lig": float(m.get("eval/episode_lig", np.nan)), "sat": float(m.get("eval/episode_sat", np.nan))}
        curve.append(row)
        print(json.dumps(row), flush=True)
        json.dump(curve, open(os.path.join(out, "curve.json"), "w"), indent=1)

    train_fn = functools.partial(
        ppo.train, num_timesteps=a.steps, num_evals=a.evals, num_eval_envs=a.eval_envs, reward_scaling=1.0,
        episode_length=env._config.episode_length, normalize_observations=True, action_repeat=1,
        unroll_length=20, num_minibatches=a.minibatches, num_updates_per_batch=4, discounting=a.gamma,
        learning_rate=3e-4, entropy_cost=5e-3, num_envs=a.envs, batch_size=a.batch, seed=0,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **net_cfg),
        progress_fn=progress, num_resets_per_eval=a.resets, policy_params_fn=checkpoint)
    restore = None
    if a.restore:  # a curriculum stage: carry on from an earlier run (e.g. with less trunk assist)
        restore = pickle.load(open(os.path.join(a.restore, "params.pkl"), "rb"))
        print(f"restoring {a.restore} ({len(restore)} parts)", flush=True)
    make_inference, params, _ = train_fn(environment=env, eval_env=Env(a.xml, refs, config=cfg, max_dogs=max(a.eval_envs, 256)),
                                         restore_params=restore)
    with open(os.path.join(out, "params.pkl"), "wb") as f:
        pickle.dump(params, f)

    policy = make_inference(params, deterministic=True)
    if a.task == "right":
        evaluate_right(env, policy, out, t0)
        return
    # evaluation from many starts in the clip at once, deterministic policy, all on the GPU
    starts = env._starts[::25]
    horizon = env._config.episode_length

    def eval_one(s):
        st0 = env.reset_at(jax.random.PRNGKey(s), s)

        def body(carry, i):
            st, alive, steps, pose_sum, af, at = carry
            act, _ = policy(st.obs, jax.random.PRNGKey(i))
            nxt = env.step(st, act)
            pose_sum = pose_sum + alive * nxt.metrics["r_pose"]
            af, at = af + alive * nxt.metrics["assist_f"], at + alive * nxt.metrics["assist_t"]
            steps = steps + alive
            alive = alive * (1.0 - nxt.done)
            return (nxt, alive, steps, pose_sum, af, at), None

        z = jp.float32(0.0)
        (_, _, steps, pose_sum, af, at), _ = jax.lax.scan(body, (st0, jp.float32(1.0), z, z, z, z), jp.arange(horizon))
        return steps, pose_sum, af, at

    steps, pose_sums, afs, ats = (np.array(x) for x in jax.jit(jax.vmap(eval_one))(starts))
    reachable = np.minimum(horizon, np.array(env._end)[np.array(starts)] - np.array(starts))
    full = steps >= reachable - 1
    ev = {"episodes": int(len(steps)), "mean_steps_on_capture": float(steps.mean()),
          "full_length_share": float(np.mean(full)),
          "mean_pose_reward": float(pose_sums.sum() / max(steps.sum(), 1)),
          "train_minutes": round((time.time() - t0) / 60, 1)}
    if env._n_rfc:  # how hard the trunk assist works while the dog is on the capture
        ev["assist_force_share_of_weight"] = round(float(afs.sum() / max(steps.sum(), 1)), 4)
        ev["assist_torque_share_of_weight_x_0.3m"] = round(float(ats.sum() / max(steps.sum(), 1)), 4)
    names = ["stand", "walk", "trot", "run", "jump", "sit", "down", "side", "back", "crouch", "rear_up", "swim", "wade"]
    reg = env._regime[np.array(starts)]
    by = {}
    for r in sorted(set(reg.tolist())):
        if r >= 0:
            sel = reg == r
            by[names[r]] = {"starts": int(sel.sum()), "mean_steps": round(float(steps[sel].mean()), 1),
                            "full_length_share": round(float(full[sel].mean()), 3)}
    if by:
        ev["by_regime_at_start"] = by
    json.dump(ev, open(os.path.join(out, "eval.json"), "w"), indent=1)
    print("EVAL", json.dumps(ev), flush=True)


def evaluate_right(env, policy, out, t0, n=1024):
    """From n fresh falls: how often, and how fast, the reflex gets the dog upright (and on its feet)."""
    horizon = env._config.episode_length

    def one(key):
        st0 = env.reset(key)

        def body(carry, i):
            st, t_up, t_feet = carry
            act, _ = policy(st.obs, jax.random.PRNGKey(i))
            nxt = env.step(st, act)
            t_up = jp.where((t_up < 0) & env.is_sternal(nxt.pipeline_state), i + 1, t_up)
            t_feet = jp.where((t_feet < 0) & env.is_up(nxt.pipeline_state), i + 1, t_feet)
            return (nxt, t_up, t_feet), None

        (st, t_up, t_feet), _ = jax.lax.scan(body, (st0, jp.int32(-1), jp.int32(-1)), jp.arange(horizon))
        return t_up, t_feet, env.is_up(st.pipeline_state), env.is_sternal(st.pipeline_state)

    keys = jax.random.split(jax.random.PRNGKey(123), n)
    t_up, t_feet, end_up, end_st = (np.array(x) for x in jax.jit(jax.vmap(one))(keys))
    dt = env._config.ctrl_dt
    ev = {"falls": n, "righted_share": round(float(np.mean(t_up >= 0)), 3),
          "on_feet_share": round(float(np.mean(t_feet >= 0)), 3),
          "median_s_to_righted": round(float(np.median(t_up[t_up >= 0]) * dt), 2) if (t_up >= 0).any() else None,
          "median_s_to_feet": round(float(np.median(t_feet[t_feet >= 0]) * dt), 2) if (t_feet >= 0).any() else None,
          "on_feet_at_end": round(float(np.mean(end_up)), 3), "righted_at_end": round(float(np.mean(end_st)), 3),
          "train_minutes": round((time.time() - t0) / 60, 1)}
    json.dump(ev, open(os.path.join(out, "eval.json"), "w"), indent=1)
    print("EVAL", json.dumps(ev), flush=True)


if __name__ == "__main__":
    main()
