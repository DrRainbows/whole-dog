#!/usr/bin/env bash
# Short GPU run on the Warp backend: checks the Brax wrappers accept warp data, and measures end-to-end speed.
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
python train_track.py ../refs/D1_009_KAN01_001.npz --steps 2000000 --envs 512 --batch 256 --minibatches 8 \
  --evals 3 --eval-envs 64 --out "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/smoke_gpu_warp"
