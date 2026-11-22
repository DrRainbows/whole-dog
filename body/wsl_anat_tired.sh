#!/usr/bin/env bash
# Fatigue as a determiner (RESULTS.md, 21 Nov evening, fixed beforehand): anat_dog continued for 150M steps on the
# same references and settings, with each episode's muscle capacity drawn from 0.55-1.1 (the range
# brain.muscle_capacity() produces: fatigue, being winded, arousal). The policy is not told its capacity; it can only
# feel it through its drive sense (how hard each muscle pulls against its strength).
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
R=${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs
mkdir -p "$R/anat_tired"
cd "$(git rev-parse --show-toplevel)/body/train"
python train_track.py "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/flat/*.npz" "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/mm/*.npz" \
  "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/gram/*.npz" --xml ../dog_anat.xml --muscles ../muscles_dog.json \
  --restore "$R/anat_dog" --steps 150000000 --envs 4096 --evals 6 --gamma 0.98 --lookahead2 15 \
  --paws-obs --w-rootvel 0.1 --transition-boost 4 --obs-drive --capacity 0.55,1.1 --keep-checkpoints \
  --out "$R/anat_tired" > "$R/anat_tired/train.log" 2>&1
