"""Record the scramble_up reflex of several runs from the same falls, for render_right.py (run in WSL).

  python record_right.py OUT.npz RUN_DIR [RUN_DIR ...] [--side 2] [--back 2]
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
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--side", type=int, default=2)
    ap.add_argument("--back", type=int, default=2)
    a = ap.parse_args()
    refs = sorted(glob.glob(os.environ.get("DOG_WORK", "runs") + "/refs_anat/flat/*.npz"))
    rec = {}
    keys = None
    for ri, run in enumerate(a.runs):
        rc = json.load(open(os.path.join(run, "run.json")))
        cfg = right_config(default_config())
        cfg.muscles = rc.get("muscles", "")
        env = DogRight(rc["xml"], refs, config=cfg, max_dogs=64)
        if keys is None:  # the same falls for every run: some onto the side, some onto the back
            cand = jax.random.split(jax.random.PRNGKey(11), 64)
            tilt = np.degrees(np.array(jax.vmap(lambda k: jp.arccos(jp.clip(jp.dot(
                quat_to_mat(env.reset(k).pipeline_state.qpos[3:7]) @ env._up_b, jp.array([0.0, 0, 1])), -1, 1)))(cand)))
            side = [i for i in range(64) if 70 < tilt[i] < 110][: a.side]
            back = [i for i in range(64) if tilt[i] > 150][: a.back]
            pick = side + back
            keys = cand[np.array(pick)]
            rec["tilt"] = tilt[pick]
        params = pickle.load(open(os.path.join(run, "params.pkl"), "rb"))
        nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                              preprocess_observations_fn=running_statistics.normalize,
                                              **json.load(open(os.path.join(run, "net.json"))))
        policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)

        def one(key):
            st0 = env.reset(key)

            def body(st, i):
                act, _ = policy(st.obs, jax.random.PRNGKey(i))
                nxt = env.step(st, act)
                return nxt, (nxt.pipeline_state.qpos, env.is_up(nxt.pipeline_state), env.is_sternal(nxt.pipeline_state))

            _, (q, up, stn) = jax.lax.scan(body, st0, jp.arange(150))
            return jp.concatenate([st0.pipeline_state.qpos[None], q]), up, stn

        q, up, stn = (np.array(x) for x in jax.jit(jax.vmap(one))(keys))
        rec[f"qpos_{ri}"], rec[f"up_{ri}"], rec[f"sternal_{ri}"] = q, up, stn
        rec[f"name_{ri}"] = np.array(os.path.basename(run.rstrip("/")) + (" (dog muscles)" if cfg.muscles else " (old motors)"))
    np.savez(a.out, n_runs=len(a.runs), **rec)
    print("RECORDED", a.out, {k: v.shape for k, v in rec.items() if hasattr(v, "shape")})


if __name__ == "__main__":
    main()
