#!/usr/bin/env bash
# Shoves (RESULTS.md, 21 Nov evening, S1-S2, fixed beforehand): anat_tired continued for 100M steps with random shoves
# (mean every 3 s, up to 0.2 body weights for 0.15 s), capacity 0.55-1.1, same references and settings.
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
R=${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs
mkdir -p "$R/anat_push"
cd "$(git rev-parse --show-toplevel)/body/train"
python train_track.py "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/flat/*.npz" "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/mm/*.npz" \
  "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/gram/*.npz" --xml ../dog_anat.xml --muscles ../muscles_dog.json \
  --restore "$R/anat_tired" --steps 100000000 --envs 4096 --evals 5 --gamma 0.98 --lookahead2 15 \
  --paws-obs --w-rootvel 0.1 --transition-boost 4 --obs-drive --capacity 0.55,1.1 --push-every 3 --push-force 0.2 \
  --keep-checkpoints --out "$R/anat_push" > "$R/anat_push/train.log" 2>&1
