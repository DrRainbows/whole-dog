#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd "$HOME/dogsim" && . .venv/bin/activate
uv pip install brax flax optax orbax-checkpoint ml_collections mediapy 2>&1 | tail -4
