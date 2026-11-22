#!/usr/bin/env bash
# The ligament body (RESULTS.md, 21 Nov, fixed beforehand): anat_dog continued for 150M steps on dog_anat_lig.xml
# (out-of-plane hinges of elbow, carpus, stifle and hock held by ligaments) with muscles_dog_lig.json; same
# references and settings.
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
R=${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs
mkdir -p "$R/anat_lig"
cd "$(git rev-parse --show-toplevel)/body/train"
python train_track.py "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/flat/*.npz" "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/mm/*.npz" \
  "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/gram/*.npz" --xml ../dog_anat_lig.xml --muscles ../muscles_dog_lig.json \
  --restore "$R/anat_dog" --steps 150000000 --envs 4096 --evals 6 --gamma 0.98 --lookahead2 15 \
  --paws-obs --w-rootvel 0.1 --transition-boost 4 --obs-drive --keep-checkpoints \
  --out "$R/anat_lig" > "$R/anat_lig/train.log" 2>&1
