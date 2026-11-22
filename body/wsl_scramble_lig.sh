#!/usr/bin/env bash
# The scramble_up reflex on the ligament body (dog_anat_lig.xml, muscles_dog_lig.json), continued from the dog-strength
# reflex (scramble_up_dog3, 72M) for 40M steps on the proper standard, with the brain's capacity drawn from 0.55-1.1
# so it also practises getting up tired.
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
R=${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs
mkdir -p "$R/scramble_up_lig"
cd "$(git rev-parse --show-toplevel)/body/train"
python train_track.py "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/flat/*.npz" --task right --xml ../dog_anat_lig.xml \
  --muscles ../muscles_dog_lig.json --restore "$R/scramble_up_dog3_72M" --steps 40000000 --envs 1024 --evals 4 \
  --gamma 0.97 --net 256,128 --capacity 0.55,1.1 --keep-checkpoints --out "$R/scramble_up_lig" \
  > "$R/scramble_up_lig/train.log" 2>&1
