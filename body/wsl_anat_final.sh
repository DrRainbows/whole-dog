#!/usr/bin/env bash
# The consolidated dog (RESULTS.md, 21 Nov evening): anat_lig (the ligament body) continued for 150M steps with
# practice tired (capacity 0.55-1.1), same references and settings; paired with the reflex scramble_up_lig.
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
R=${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs
mkdir -p "$R/anat_final"
cd "$(git rev-parse --show-toplevel)/body/train"
python train_track.py "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/flat/*.npz" "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/mm/*.npz" \
  "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/gram/*.npz" --xml ../dog_anat_lig.xml --muscles ../muscles_dog_lig.json \
  --restore "$R/anat_lig" --steps 150000000 --envs 4096 --evals 6 --gamma 0.98 --lookahead2 15 \
  --paws-obs --w-rootvel 0.1 --transition-boost 4 --obs-drive --capacity 0.55,1.1 \
  --keep-checkpoints --out "$R/anat_final" > "$R/anat_final/train.log" 2>&1
