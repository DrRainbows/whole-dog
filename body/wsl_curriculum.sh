#!/usr/bin/env bash
# Usage: wsl_curriculum.sh REF_GLOB PREFIX STEPS_PER_STAGE "SCALE1 SCALE2 ..." [extra train_track.py args...]
# A trunk-assist curriculum (after Yu, Turk & Liu 2018): train with the twist assist at SCALE1, then carry on
# from that policy with SCALE2, and so on. Use 1e-9 for "none" (the policy keeps its six assist outputs).
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
REF="$1"; PREFIX="$2"; STEPS="$3"; SCALES="$4"; shift 4
prev="${INIT:-}"  # optional: start the first stage from this run directory
i=0
for s in $SCALES; do
  out="${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/${PREFIX}_s${i}"
  mkdir -p "$out"
  extra=()
  [ -n "$prev" ] && extra=(--restore "$prev")
  python train_track.py "$REF" --steps "$STEPS" --envs 4096 --out "$out" --rfc-torque "$s" "${extra[@]}" "$@" > "$out/train.log" 2>&1
  echo "stage $i (twist assist $s): $(grep -o 'EVAL.*' "$out/train.log" | tail -1 | cut -c1-200)"
  prev="$out"
  i=$((i+1))
done
