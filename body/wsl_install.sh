#!/usr/bin/env bash
# Training stack inside WSL2 Ubuntu: uv, then JAX with CUDA, MuJoCo, MJX, MuJoCo Playground.
set -e
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
fi
mkdir -p "$HOME/dogsim" && cd "$HOME/dogsim"
[ -d .venv ] || uv venv .venv --python 3.12 -q
. .venv/bin/activate
uv pip install "jax[cuda12]" mujoco mujoco-mjx playground 2>&1 | tail -3
python -c "import jax, mujoco; print('jax', jax.__version__, jax.devices()); print('mujoco', mujoco.__version__)"
python -c "import mujoco_playground as mp; print('playground ok:', sorted(mp.registry.locomotion.ALL_ENVS)[:40])"
