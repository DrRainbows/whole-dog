#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
python - <<'PY'
import inspect, importlib
import mujoco
from mujoco import mjx
print("mujoco", mujoco.__version__)
sig = inspect.signature(mjx.put_model)
print("put_model", sig)
for mod in ("mujoco_warp", "warp"):
    try:
        m = importlib.import_module(mod); print(mod, getattr(m, "__version__", "?"))
    except Exception as e:
        print(mod, "not installed:", type(e).__name__)
PY
uv pip list 2>/dev/null | grep -i -E "warp|mujoco|jax|brax|playground" || pip list 2>/dev/null | grep -i -E "warp|mujoco|jax|brax"
