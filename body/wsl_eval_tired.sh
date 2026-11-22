#!/usr/bin/env bash
# Usage: wsl_eval_tired.sh RUN   (the fatigue test, RESULTS.md 21 Nov, T1-T4)
# Capture at muscle capacity 1.0 and 0.65; the link test (8 seeds) at 1.0 and 0.65, with the pre-registered fall
# (relative to the reference) and with the absolute topple (thorax rolled past 60 degrees).
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
RUN="${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/$1"
A=${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat
S=0,1,2,3,4,5,6,7
python eval_track.py "$RUN" "$A/flat/*.npz" --anchor 0.1 --term fall --capacity 1.0 --tag flat_fall_cap100 2>&1 | grep -E "^EVAL|Error" || true
python eval_track.py "$RUN" "$A/flat/*.npz" --anchor 0.1 --term fall --capacity 0.65 --tag flat_fall_cap065 2>&1 | grep -E "^EVAL|Error" || true
python eval_links.py "$RUN" "$A/links_anat" --anchor 0.1 --term fall --capacity 1.0 --out links_anat --seeds $S 2>&1 | grep -E "^LINKS|Error" || true
python eval_links.py "$RUN" "$A/links_anat" --anchor 0.1 --term fall --capacity 0.65 --out links_anat_cap065 --seeds $S 2>&1 | grep -E "^LINKS|Error" || true
python eval_links.py "$RUN" "$A/links_anat" --anchor 0.1 --term topple --capacity 1.0 --out links_anat_topple --seeds $S 2>&1 | grep -E "^LINKS|Error" || true
python eval_links.py "$RUN" "$A/links_anat" --anchor 0.1 --term topple --capacity 0.65 --out links_anat_topple_cap065 --seeds $S 2>&1 | grep -E "^LINKS|Error" || true
