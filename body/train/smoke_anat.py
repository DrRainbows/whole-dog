"""Hold the reference with no policy at all (action 0: the motors aim at the captured pose) and see what the body
does: how long it stays up, whether anything blows up, how hard it leans on its ligaments and how often the muscles
are at their limit. A check that a body and its muscle code behave before a policy is trained on them (run in WSL).

  python smoke_anat.py XML "REF_GLOB" [--muscles SPEC] [--n 512] [--steps 150]
"""
import argparse
import glob
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np

from dog_track import DogTrack, default_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("refs")
    ap.add_argument("--muscles", default="")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    refs = sorted(glob.glob(a.refs))
    cfg = default_config()
    cfg.lookahead2, cfg.obs_paws, cfg.w_rootvel = 15, True, 0.1
    cfg.term, cfg.anchor, cfg.muscles, cfg.min_left = "fall", 0.1, a.muscles, a.steps
    cfg.obs_drive = True
    env = DogTrack(a.xml, refs, config=cfg, max_dogs=a.n)
    starts = env._starts[:: max(1, len(env._starts) // a.n)][: a.n]
    zero = jp.zeros(env.action_size)

    def one(s):
        st0 = env.reset_at(jax.random.PRNGKey(s), s)

        def body(c, i):
            st, alive, steps, lig, sat = c
            nxt = env.step(st, zero)
            steps, lig, sat = steps + alive, lig + alive * nxt.metrics["lig"], sat + alive * nxt.metrics["sat"]
            alive = alive * (1.0 - nxt.done)
            return (nxt, alive, steps, lig, sat), None

        z = jp.float32(0.0)
        (st, _, steps, lig, sat), _ = jax.lax.scan(body, (st0, jp.float32(1.0), z, z, z), jp.arange(a.steps))
        finite = jp.all(jp.isfinite(st.pipeline_state.qpos))
        return steps, lig, sat, finite

    f = jax.jit(jax.vmap(one))
    t0 = time.time()
    out = [np.array(x) for x in f(jp.array(starts))]
    t1 = time.time()
    out = [np.array(x) for x in f(jp.array(starts))]
    t2 = time.time()
    steps, lig, sat, fin = out
    res = {"xml": os.path.basename(a.xml), "muscles": os.path.basename(a.muscles) or "-", "starts": int(len(starts)),
           "mean_steps": round(float(steps.mean()), 1), "full": round(float(np.mean(steps >= a.steps - 1)), 3),
           "ligament_strain_rad_per_step": round(float(lig.sum() / max(steps.sum(), 1)), 5),
           "muscles_at_limit_share": round(float(sat.sum() / max(steps.sum(), 1)), 4),
           "non_finite": int((~fin.astype(bool)).sum()), "compile_s": round(t1 - t0 - (t2 - t1), 1),
           "ms_per_control_step_all_dogs": round((t2 - t1) / a.steps * 1000, 2)}
    print("SMOKE", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
