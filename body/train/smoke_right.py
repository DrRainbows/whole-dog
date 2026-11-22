"""Smoke test of the scramble_up environment (run in WSL): zero policy (all muscles aim at standing) from fresh falls."""
import glob
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np

from dog_right import DogRight, right_config
from dog_track import default_config

cfg = right_config(default_config())
cfg.muscles = sys.argv[1] if len(sys.argv) > 1 else ""
env = DogRight("../dog_anat.xml", sorted(glob.glob(os.environ.get("DOG_WORK", "runs") + "/refs_anat/flat/*.npz")), config=cfg, max_dogs=64)
print("obs", env.observation_size, "actions", env.action_size, "stand z", round(env._stand_z, 3))
zero = jp.zeros(env.action_size)


def one(key):
    st = env.reset(key)
    st0_up = env.is_sternal(st.pipeline_state)

    def body(c, i):
        st, r, ups = c
        nxt = env.step(st, zero)
        return (nxt, r + nxt.reward, ups + env.is_up(nxt.pipeline_state)), None

    (st, r, ups), _ = jax.lax.scan(body, (st, jp.float32(0), jp.float32(0)), jp.arange(150))
    return r, ups, env.is_up(st.pipeline_state), env.is_sternal(st.pipeline_state), jp.all(jp.isfinite(st.pipeline_state.qpos)), st0_up


def start_up(key):
    return env.is_sternal(env.reset(key).pipeline_state)


r, ups, up, stn, fin, s0 = (np.array(x) for x in jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(0), 64)))
print("SMOKE_RIGHT", json.dumps({"muscles": os.path.basename(cfg.muscles) or "-", "mean_return": round(float(r.mean()), 1),
      "on_feet_at_end": float(up.mean()), "righted_at_end": float(stn.mean()), "righted_at_start": float(s0.mean()), "non_finite": int((~fin).sum())}))
