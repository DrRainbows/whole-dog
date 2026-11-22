"""Evaluate a trained tracking policy under different rules, without retraining (run in WSL).

  python eval_track.py RUN_DIR REF... [--anchor 0.1] [--term fall] [--every 25] [--horizon 300] [--no-assist]

Starts at every 25th frame of the references, runs the deterministic policy, and reports control steps until the
episode ends, by regime at the start. --term track ends when the dog strays from the capture (as in training);
--term fall ends only on a real fall (trunk tipped 0.8 rad off the reference's tilt, or hips 0.2 m below it).
--anchor pulls the reference toward the dog each step (turned and shifted), so drift cannot build up.
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
NAMES = ["stand", "walk", "trot", "run", "jump", "sit", "down", "side", "back", "crouch", "rear_up", "swim", "wade"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ref", nargs="+")
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--term", default="track", choices=["track", "fall", "topple"])
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--no-assist", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--params", default="params.pkl", help="which saved policy in the run (e.g. params_300000000.pkl)")
    ap.add_argument("--capacity", type=float, default=None, help="hold the muscles' capacity at this (fatigue: "
                    "brain.muscle_capacity() gives 0.5-1.1); default 1.0")
    ap.add_argument("--push-every", type=float, default=0.0, help="random shoves during the test: mean seconds between")
    ap.add_argument("--push-force", type=float, default=0.0, help="random shoves during the test: largest, body weights")
    a = ap.parse_args()
    refs = sorted(p for r in a.ref for p in (glob.glob(r) if any(c in r for c in "*?[") else [r]))
    cfg = default_config()
    rc = json.load(open(os.path.join(a.run, "run.json")))
    cfg.action_scale = rc.get("action_scale", cfg.action_scale)
    cfg.lookahead2, cfg.obs_paws, cfg.w_rootvel = rc.get("lookahead2", 0), rc.get("obs_paws", False), rc.get("w_rootvel", 0.0)
    cfg.rfc_force, cfg.rfc_torque = rc.get("rfc_force", 0.0), rc.get("rfc_torque", 0.0)
    if a.no_assist and (cfg.rfc_force or cfg.rfc_torque):
        cfg.rfc_force, cfg.rfc_torque = 1e-9, 1e-9
    cfg.anchor, cfg.term = a.anchor, a.term
    cfg.push_every, cfg.push_force = a.push_every, a.push_force
    cfg.muscles = rc.get("muscles", "")  # the body the policy was trained on
    cfg.obs_drive = rc.get("obs_drive", False)
    if a.capacity is not None:
        assert cfg.muscles, "--capacity needs a run with muscles"
        cfg.capacity_range = (a.capacity, a.capacity)
    env = DogTrack(rc.get("xml") or XML, refs, config=cfg, max_dogs=8192)
    params = pickle.load(open(os.path.join(a.run, a.params), "rb"))
    net_cfg = json.load(open(os.path.join(a.run, "net.json")))
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize, **net_cfg)
    policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    starts = env._starts[:: a.every]
    horizon = a.horizon

    def eval_one(s):
        st0 = env.reset_at(jax.random.PRNGKey(s), s)

        def body(carry, i):
            st, alive, steps, af, at, lig, sat = carry
            act, _ = policy(st.obs, jax.random.PRNGKey(i))
            nxt = env.step(st, act)
            af = af + alive * nxt.metrics["assist_f"]
            at = at + alive * nxt.metrics["assist_t"]
            lig, sat = lig + alive * nxt.metrics["lig"], sat + alive * nxt.metrics["sat"]
            steps = steps + alive
            alive = alive * (1.0 - nxt.done)
            return (nxt, alive, steps, af, at, lig, sat), None

        z = jp.float32(0.0)
        (_, _, steps, af, at, lig, sat), _ = jax.lax.scan(body, (st0, jp.float32(1.0), z, z, z, z, z), jp.arange(horizon))
        return steps, af, at, lig, sat

    steps, afs, ats, ligs, sats = (np.array(x) for x in jax.jit(jax.vmap(eval_one))(starts))
    reachable = np.minimum(horizon, np.array(env._end)[np.array(starts)] - np.array(starts))
    full = steps >= reachable - 1
    ev = {"run": os.path.basename(a.run.rstrip("/")), "tag": a.tag, "params": a.params, "refs": a.ref, "capacity": list(cfg.capacity_range),
          "ligament_strain_per_step": round(float(ligs.sum() / max(steps.sum(), 1)), 5),
          "muscles_at_limit_share": round(float(sats.sum() / max(steps.sum(), 1)), 4), "anchor": a.anchor, "term": a.term, "no_assist": a.no_assist, "push_every": a.push_every, "push_force": a.push_force,
          "episodes": int(len(steps)), "mean_steps": round(float(steps.mean()), 1),
          "full_length_share": round(float(np.mean(full)), 3),
          "assist_force_share_of_weight": round(float(afs.sum() / max(steps.sum(), 1)), 4),
          "assist_torque_share_of_weight_x_0.3m": round(float(ats.sum() / max(steps.sum(), 1)), 4)}
    reg = env._regime[np.array(starts)]
    by = {}
    for r in sorted(set(reg.tolist())):
        if r >= 0:
            sel = reg == r
            by[NAMES[r]] = {"starts": int(sel.sum()), "mean_steps": round(float(steps[sel].mean()), 1),
                            "full": round(float(full[sel].mean()), 3)}
    ev["by_regime_at_start"] = by
    print("EVAL " + json.dumps(ev), flush=True)
    with open(os.path.join(a.run, "evals.jsonl"), "a") as f:
        f.write(json.dumps(ev) + "\n")


if __name__ == "__main__":
    main()
