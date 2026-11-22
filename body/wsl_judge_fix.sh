#!/usr/bin/env bash
# Usage: wsl_judge_fix.sh RUN   (S1-S2 / P1-P2 in RESULTS.md, 21 Nov evening)
# The capture (validity), the CPU export, then the same 100 social sessions as anat_tired's baseline (_base: 50 per
# dog, 60 s, a person present, seeds 1-50, reflex scramble_up_dog3 at 72M).
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
RUN="${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/$1"
python eval_track.py "$RUN" "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/flat/*.npz" --anchor 0.1 --term fall --tag flat_fall 2>&1 | grep -E "^EVAL|Error" || true
python export_policy.py "$RUN" 2>&1 | grep -E "^EXPORTED|Error" || true
bash "$(git rev-parse --show-toplevel)/body/wsl_online_batch.sh" "$1" scramble_up_dog3_72M _base 50 60 14 --present
