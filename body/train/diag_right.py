"""How the scramble_up reflex does by how far over the dog starts (run in WSL).

  python diag_right.py RUN_DIR [--n 2048] [--params params.pkl]

Falls are binned by the trunk's tilt from standing at the start: stumbling (< 45 deg), on its side (45-135),
on its back (> 135). Per bin: share righted (trunk within 35 deg of standing) and on its feet (and hips at 75% of
standing height) within 3 s, and the median time to each.
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

from dog_right import DogRight, right_config
from dog_track import default_config, quat_to_mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--capacity", type=float, default=1.0, help="the muscles' capacity (a tired dog: 0.65)")
    a = ap.parse_args()
    rc = json.load(open(os.path.join(a.run, "run.json")))
    cfg = right_config(default_config())
    cfg.muscles = rc.get("muscles", "")
    cfg.capacity_range = (a.capacity, a.capacity)
    env = DogRight(rc["xml"], sorted(glob.glob(os.environ.get("DOG_WORK", "runs") + "/refs_anat/flat/*.npz")), config=cfg, max_dogs=a.n)
    params = pickle.load(open(os.path.join(a.run, a.params), "rb"))
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize,
                                          **json.load(open(os.path.join(a.run, "net.json"))))
    policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)

    def one(key):
        st0 = env.reset(key)
        tilt0 = jp.arccos(jp.clip(jp.dot(quat_to_mat(st0.pipeline_state.qpos[3:7]) @ env._up_b, jp.array([0.0, 0, 1])), -1, 1))

        def body(c, i):
            st, tr, tf, n_up = c
            act, _ = policy(st.obs, jax.random.PRNGKey(i))
            nxt = env.step(st, act)
            up = env.is_up(nxt.pipeline_state)
            tr = jp.where((tr < 0) & env.is_sternal(nxt.pipeline_state), i + 1, tr)
            tf = jp.where((tf < 0) & up, i + 1, tf)
            n_up = n_up + ((tf >= 0) & up).astype(jp.int32)  # steps on its feet since it first got there
            return (nxt, tr, tf, n_up), None

        (st, tr, tf, n_up), _ = jax.lax.scan(body, (st0, jp.int32(-1), jp.int32(-1), jp.int32(0)), jp.arange(150))
        return tilt0, tr, tf, n_up, env.is_up(st.pipeline_state)

    keys = jax.random.split(jax.random.PRNGKey(7), a.n)
    tilt, tr, tf, n_up, end_up = (np.array(x) for x in jax.jit(jax.vmap(one))(keys))
    deg = np.degrees(tilt)
    out = {"run": os.path.basename(a.run.rstrip("/")), "params": a.params, "capacity": a.capacity}
    for name, sel in (("stumbling <45", deg < 45), ("side 45-135", (deg >= 45) & (deg <= 135)), ("back >135", deg > 135)):
        r, f, nu, eu = tr[sel], tf[sel], n_up[sel], end_up[sel]
        got = f >= 0
        out[name] = {"falls": int(sel.sum()), "righted": round(float(np.mean(r >= 0)), 3),
                     "on_feet": round(float(np.mean(f >= 0)), 3),
                     "median_s_righted": round(float(np.median(r[r >= 0]) * 0.02), 2) if (r >= 0).any() else None,
                     "median_s_feet": round(float(np.median(f[f >= 0]) * 0.02), 2) if (f >= 0).any() else None,
                     # does it stay there? share of the steps after first getting up that it is still up, and up at 3 s
                     "stays_up_share": round(float(np.mean(nu[got] / np.maximum(150 - f[got] + 1, 1))), 3) if got.any() else None,
                     "up_at_3s": round(float(np.mean(eu)), 3)}
    print("RIGHT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
