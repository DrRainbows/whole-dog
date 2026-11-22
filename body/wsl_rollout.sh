#!/usr/bin/env bash
# Usage: wsl_rollout.sh RUN_NAME REF_OR_GLOB STARTS STEPS
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
python rollout.py "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/$1" "$2" --starts "$3" --steps "${4:-400}"
