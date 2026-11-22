#!/usr/bin/env bash
# Usage: wsl_train.sh REF STEPS ENVS [OUT_NAME]
#   REF is a clip name in refs/ (e.g. D1_009_KAN01_001) or a path/glob relative to body/train (e.g. "../refs/all/*.npz")
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
case "$1" in
  */*|*\**) REF="$1"; NAME="${4:-multi}" ;;
  *) REF="../refs/$1.npz"; NAME="${4:-$1}" ;;
esac
python train_track.py "$REF" --steps "${2:-30000000}" --envs "${3:-4096}" --out "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/$NAME"
