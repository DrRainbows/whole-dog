#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
P=$(python -c "import mujoco_playground,os;print(os.path.dirname(mujoco_playground.__file__))" 2>/dev/null)
echo "playground at $P"
grep -rn "impl=\|nconmax\|njmax\|naconmax\|impl ==\|'warp'\|\"warp\"" "$P" --include=*.py | grep -v test | head -40
M=$(python -c "import mujoco.mjx,os;print(os.path.dirname(mujoco.mjx.__file__))")
grep -n "def make_data" -A 30 "$M/_src/io.py" | grep -n "nconmax\|njmax\|naconmax\|impl\|def " | head -20
