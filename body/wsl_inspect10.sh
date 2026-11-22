#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
SP=$(python -c "import site;print(site.getsitepackages()[0])")
grep -rln "naconmax" "$SP" --include=*.py 2>/dev/null | grep -v "/mujoco/mjx/_src/\|/mujoco_warp/" | head -10
echo ---
F=$(grep -rln "naconmax" "$SP" --include=*.py 2>/dev/null | grep -v "/mujoco/mjx/_src/\|/mujoco_warp/\|test" | head -1)
echo "FILE $F"
grep -n "naconmax\|put_model\|make_data\|impl\|vmap\|mjx.step\|forward(" "$F" | head -40
echo "--- simulation timestep in tracking"
P=$(python -c "import mujoco_playground,os;print(os.path.dirname(mujoco_playground.__file__))")
grep -n "_CONTROL_TIMESTEP =\|_SIMULATION_TIMESTEP =" "$P/tasks/whole_body/tracking/task.py"
