#!/usr/bin/env bash
# The deciding run (RESULTS.md, 21 Nov afternoon, fixed beforehand): anat_strong's 314.6M-step policy continued for
# 300M steps on the same references, same settings, with a dog's muscles (muscles_dog.json).
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
R=${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs
mkdir -p "$R/anat_strong_314M" "$R/anat_dog"
cp "$R/anat_strong/params_314572800.pkl" "$R/anat_strong_314M/params.pkl"
cp "$R/anat_strong/net.json" "$R/anat_strong/run.json" "$R/anat_strong_314M/"
cd "$(git rev-parse --show-toplevel)/body/train"
python train_track.py "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/flat/*.npz" "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/mm/*.npz" \
  "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/gram/*.npz" --xml ../dog_anat.xml --muscles ../muscles_dog.json \
  --restore "$R/anat_strong_314M" --steps 300000000 --envs 4096 --evals 12 --gamma 0.98 --lookahead2 15 \
  --paws-obs --w-rootvel 0.1 --transition-boost 4 --obs-drive --keep-checkpoints --out "$R/anat_dog" \
  > "$R/anat_dog/train.log" 2>&1
