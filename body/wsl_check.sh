#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
python - <<'PY'
import jax, mujoco, importlib.metadata as md
print("jax", jax.__version__, jax.devices())
print("mujoco", mujoco.__version__, "| mjx", md.version("mujoco-mjx"), "| brax", md.version("brax"), "| playground", md.version("playground"))
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from brax.training.agents.ppo import train as ppo
print("playground locomotion envs:", sorted(registry.locomotion.ALL_ENVS)[:8], "...")
print("MjxEnv ok:", hasattr(mjx_env, "MjxEnv"), "| mjx_env.step:", hasattr(mjx_env, "step"), "| mjx_env.init:", hasattr(mjx_env, "init"))
PY
