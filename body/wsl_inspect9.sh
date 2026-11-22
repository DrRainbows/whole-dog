#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
P=$(python -c "import mujoco_playground,os;print(os.path.dirname(mujoco_playground.__file__))" 2>/dev/null)
sed -n 485,530p "$P/tasks/whole_body/tracking/task.py"
echo "=================== sim / impl plumbing"
grep -rn "put_model\|make_data\|mjx.step\|naconmax\|nconmax" "$P/_src" --include=*.py | head -30
