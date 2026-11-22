#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
P=$(python -c "import mujoco_playground, os; print(os.path.dirname(mujoco_playground.__file__))")
T="$P/tasks/whole_body/tracking"
wc -l $T/*.py "$P/_src/rl/env_registry.py" | tail -12
echo "=== task.py (head)"; sed -n '1,200p' $T/task.py | grep -v "^#" | grep -v '^\s*$' | head -150
