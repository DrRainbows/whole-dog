#!/usr/bin/env bash
# Usage: wsl_train2.sh REF_OR_GLOB STEPS ENVS OUT_NAME [extra train_track.py args...]
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
REF="$1"; STEPS="$2"; ENVS="$3"; NAME="$4"; shift 4
python train_track.py "$REF" --steps "$STEPS" --envs "$ENVS" --out "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/$NAME" "$@"
