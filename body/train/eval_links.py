"""Can the physics dog do what each reference asks? (run in WSL)

  python eval_links.py RUN_DIR REF_DIR [--anchor 0.1] [--term fall] [--out NAME]

Starts every reference at its first frame with the trained policy and runs it to the end; records how many
control steps the dog lasted, so a fall can be placed before or after the reference's `link` (the frame where the
ask changed). --term fall: the episode ends only on a real fall (trunk tipped 0.8 rad off the reference's tilt,
or hips 0.2 m below it); --anchor 0.1 pulls the reference toward the dog each step, as the online loop does.
Writes RUN_DIR/links_NAME.json: one row per reference.
"""
import argparse
import glob
import json
import os
import pickle

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from dog_track import DogTrack, default_config

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(os.path.dirname(HERE), "dog_mocap_mjx.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("refs")
    ap.add_argument("--anchor", type=float, default=0.1)
    ap.add_argument("--term", default="fall", choices=["track", "fall", "topple"])
    ap.add_argument("--out", default="")
    ap.add_argument("--xml", default="", help="the body to test (default: the one the run was trained on)")
    ap.add_argument("--muscles", default=None, help="muscle spec to test with (default: the run's own)")
    ap.add_argument("--params", default="params.pkl", help="which saved policy in the run")
    ap.add_argument("--capacity", type=float, default=None, help="hold the muscles' capacity at this (fatigue: "
                    "brain.muscle_capacity() gives 0.5-1.1); default 1.0")
    ap.add_argument("--seeds", default="0", help="comma-separated repeats; each draws its own start noise. Seed 0 "
                    "writes links_NAME.json, seed k writes links_NAME_sK.json (the physics on the GPU is not exactly "
                    "reproducible, so a single pass is one sample of each reference's outcome)")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.refs, "*.npz")))
    cfg = default_config()
    rc = json.load(open(os.path.join(a.run, "run.json")))
    cfg.action_scale = rc.get("action_scale", cfg.action_scale)
    cfg.lookahead2, cfg.obs_paws, cfg.w_rootvel = rc.get("lookahead2", 0), rc.get("obs_paws", False), rc.get("w_rootvel", 0.0)
    cfg.rfc_force, cfg.rfc_torque = rc.get("rfc_force", 0.0), rc.get("rfc_torque", 0.0)
    cfg.anchor, cfg.term = a.anchor, a.term
    cfg.muscles = rc.get("muscles", "") if a.muscles is None else a.muscles
    cfg.obs_drive = rc.get("obs_drive", False)
    if a.capacity is not None:
        assert cfg.muscles, "--capacity needs a run with muscles"
        cfg.capacity_range = (a.capacity, a.capacity)
    a.xml = a.xml or rc.get("xml") or XML
    cfg.min_left = 1
    env = DogTrack(a.xml, paths, config=cfg, max_dogs=max(64, len(paths)))
    lens, meta = [], []
    for p in paths:
        z = np.load(p)
        lens.append(int(z["qpos"].shape[0]))
        meta.append({k: (z[k].item() if z[k].shape == () else None) for k in ("a", "b", "mode", "link") if k in z.files})
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int32)
    params = pickle.load(open(os.path.join(a.run, a.params), "rb"))
    net_cfg = json.load(open(os.path.join(a.run, "net.json")))
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize, **net_cfg)
    policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    horizon = int(max(lens))

    def eval_one(s, seed):
        k0 = jax.random.PRNGKey(s)
        st0 = env.reset_at(jp.where(seed == 0, k0, jax.random.fold_in(k0, seed)), s)

        def body(carry, i):
            st, alive, steps = carry
            act, _ = policy(st.obs, jax.random.PRNGKey(i))
            nxt = env.step(st, act)
            steps = steps + alive
            lig, sat = alive * nxt.metrics["lig"], alive * nxt.metrics["sat"]
            pose = alive * nxt.metrics["r_pose"]  # how closely the body follows the reference's pose (1 = exactly)
            alive = alive * (1.0 - nxt.done)
            return (nxt, alive, steps), (lig, sat, pose)

        (_, _, steps), (lig, sat, pose) = jax.lax.scan(body, (st0, jp.float32(1.0), jp.float32(0.0)), jp.arange(horizon))
        return steps, lig, sat, pose

    run = jax.jit(jax.vmap(eval_one, in_axes=(0, None)))
    for seed in (int(x) for x in a.seeds.split(",")):
        steps, lig, sat, pose = (np.array(x) for x in run(jp.array(offs), jp.int32(seed)))
        write(a, cfg, paths, lens, meta, steps, lig, sat, pose, seed)


def write(a, cfg, paths, lens, meta, steps, lig, sat, pose, seed):
    rows = []
    for i, (p, n, s, md) in enumerate(zip(paths, lens, steps, meta)):
        k = int(md.get("link") or 0)
        after = lig[i, k: int(s)] if s > k else np.zeros(1)
        # ligament strain (radians past the joint ranges, summed over joints): mean per step before and after
        # the ask changed, and the largest after; and how often the muscles were at their limit after it
        rows.append({"file": os.path.basename(p), "n": n, "steps": int(s), "fell": bool(s < n - 2), **md,
                     "lig_before": round(float(lig[i, :max(k, 1)].mean()), 5), "lig_after": round(float(after.mean()), 5),
                     "lig_after_max": round(float(after.max()), 4),
                     "sat_after": round(float(sat[i, k: int(s)].mean()) if s > k else 0.0, 4),
                     # followed the ask? pose match before the change, and over the last second it was up after it
                     "pose_before": round(float(pose[i, :max(k, 1)].mean()), 4),
                     "pose_after": round(float(pose[i, k: int(s)].mean()) if s > k else 0.0, 4),
                     "pose_last": round(float(pose[i, max(k, int(s) - 50): int(s)].mean()) if s > k else 0.0, 4)})
    name = a.out or os.path.basename(a.refs.rstrip("/"))
    path = os.path.join(a.run, f"links_{name}.json" if seed == 0 else f"links_{name}_s{seed}.json")
    json.dump({"run": a.run, "refs": a.refs, "xml": a.xml, "muscles": cfg.muscles, "capacity": list(cfg.capacity_range), "anchor": a.anchor, "term": a.term,
               "seed": seed, "params": a.params, "rows": rows}, open(path, "w"), indent=0)
    fell = np.mean([r["fell"] for r in rows])
    print(f"LINKS {len(rows)} references, {fell:.0%} ended in a fall; wrote {path}", flush=True)


if __name__ == "__main__":
    main()
