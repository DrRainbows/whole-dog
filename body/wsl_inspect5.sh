#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
python - <<'PY'
import inspect
from brax.envs import base
from brax.training.agents.ppo import train as ppo
print("brax.envs.base:", [n for n in dir(base) if not n.startswith("_")])
print("State fields:", list(base.State.__dataclass_fields__.keys()) if hasattr(base.State, "__dataclass_fields__") else base.State)
sig = inspect.signature(ppo.train)
print("ppo.train params:", [p for p in sig.parameters][:60])
from mujoco import mjx
print("mjx has:", [n for n in ("put_model", "make_data", "step", "forward", "Data") if hasattr(mjx, n)])
PY
