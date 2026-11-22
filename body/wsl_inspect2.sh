#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
P=$(python -c "import mujoco_playground, os; print(os.path.dirname(mujoco_playground.__file__))")
echo "=== wrapper.py"; sed -n '1,60p' "$P/wrapper.py" | grep -v "^#"
echo "=== _src/sim"; ls "$P/_src/sim"; echo "=== _src/rl"; ls "$P/_src/rl"; echo "=== rl"; ls "$P/rl"; echo "=== tasks"; ls "$P/tasks"; ls "$P/tasks/"* | head -40
echo "=== docs"; ls "$P/docs" | head -20
