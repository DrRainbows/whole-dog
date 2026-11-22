#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
M=$(python -c "import mujoco.mjx,os;print(os.path.dirname(mujoco.mjx.__file__))")
sed -n '/^def make_data(/,/^def /p' "$M/_src/io.py" | grep -n "naconmax\|nconmax\|njmax\|warp\|nworld\|batch" | head -40
echo ----
grep -n "naconmax\|nworld" "$M/warp/types.py" | head -30
