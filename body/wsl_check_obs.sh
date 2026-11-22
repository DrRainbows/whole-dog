#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
pgrep -af train_track.py || echo "no trainer running"
JAX_PLATFORMS=cpu python - <<'PY'
import jax, jax.numpy as jp, numpy as np
from dog_track import rel_rot6, quat_err
q = jp.array([0.1, 0.7, -0.3, 0.64]); q = q / jp.linalg.norm(q)
print("same:", np.round(np.array(rel_rot6(q, q)), 4))
print("sign-flipped ref:", np.round(np.array(rel_rot6(q, -q)), 4))
yaw90 = jp.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
print("world-yaw 90 of identity:", np.round(np.array(rel_rot6(jp.array([1.,0,0,0]), yaw90)), 4))
PY
