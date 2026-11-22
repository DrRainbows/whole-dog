"""Which muscles work at their limit while a trained policy follows references (run in WSL).

  python diag_muscles.py RUN_DIR "REF_GLOB" [--every 25] [--params params.pkl]

Per joint group and hinge: the share of physics steps its muscle is held at its strength (torque cap), while the
dog is still up. A body too weak for what a dog did shows up here first.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("refs")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--params", default="params.pkl")
    a = ap.parse_args()
    rc = json.load(open(os.path.join(a.run, "run.json")))
    cfg = default_config()
    cfg.action_scale = rc.get("action_scale", cfg.action_scale)
    cfg.lookahead2, cfg.obs_paws, cfg.w_rootvel = rc.get("lookahead2", 0), rc.get("obs_paws", False), rc.get("w_rootvel", 0.0)
    cfg.muscles, cfg.obs_drive = rc.get("muscles", ""), rc.get("obs_drive", False)
    cfg.anchor, cfg.term = 0.1, "fall"
    assert cfg.muscles, "this run has no muscles"
    env = DogTrack(rc["xml"], sorted(glob.glob(a.refs)), config=cfg, max_dogs=8192)
    params = pickle.load(open(os.path.join(a.run, a.params), "rb"))
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize,
                                          **json.load(open(os.path.join(a.run, "net.json"))))
    policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    starts = env._starts[:: a.every]

    def one(s):
        st0 = env.reset_at(jax.random.PRNGKey(s), s)

        def body(c, i):
            st, alive, acc, n = c
            act, _ = policy(st.obs, jax.random.PRNGKey(i))
            nxt = env.step(st, act)
            acc, n = acc + alive * nxt.info["sat_v"], n + alive
            return (nxt, alive * (1.0 - nxt.done), acc, n), None

        (_, _, acc, n), _ = jax.lax.scan(body, (st0, jp.float32(1.0), jp.zeros(env._mj.nu), jp.float32(0.0)), jp.arange(300))
        return acc, n

    acc, n = (np.array(x) for x in jax.jit(jax.vmap(one))(starts))
    share = acc.sum(0) / max(n.sum(), 1)
    names = [env._mj.joint(env._mj.actuator_trnid[i, 0]).name for i in range(env._mj.nu)]
    order = np.argsort(-share)
    print("muscles most often at their limit (share of physics steps while up):")
    for i in order[:15]:
        print(f"  {names[i]:18s} {share[i]:.3f}")
    json.dump({"run": a.run, "refs": a.refs, "share_at_limit": dict(zip(names, share.round(4).tolist()))},
              open(os.path.join(a.run, "muscles_at_limit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
