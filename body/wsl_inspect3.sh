#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
P=$(python -c "import mujoco_playground, os; print(os.path.dirname(mujoco_playground.__file__))")
echo "=== rl_training.rst"; sed -n '1,140p' "$P/docs/rl_training.rst"
echo "=== tracking task files"; find "$P/tasks/whole_body/tracking" -name "*.py" | head -20
