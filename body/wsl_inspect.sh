#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
P=$(python -c "import mujoco_playground, os; print(os.path.dirname(mujoco_playground.__file__))")
echo "pkg: $P"; ls "$P"; echo "--- _src:"; ls "$P/_src" 2>/dev/null | head -30
echo "--- __init__:"; head -40 "$P/__init__.py"
echo "--- wrapper grep:"; grep -rl "def wrap_for_brax_training" "$P" | head
echo "--- MjxEnv grep:"; grep -rn "^class MjxEnv\|^class .*Env(.*abc" "$P" | head -5
echo "--- step/init helpers:"; grep -rn "^def step\|^def init" "$P/_src/mjx_env.py" 2>/dev/null | head
