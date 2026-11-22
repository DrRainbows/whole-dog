#!/usr/bin/env bash
# Usage: wsl_eval_all.sh RUN_NAME   (evaluates a run on the anatomical test sets; the run's own body and muscles)
#   capture (refs_anat/flat), held-out grammar sentences (refs_anat/gram_test), and the link test (refs_anat/links_anat,
#   blended in anatomical joints; 8 start seeds, since the GPU physics is not exactly reproducible), each with the reference anchored to the dog (0.1) and ending only on a real fall.
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
RUN="${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/$1"
A=${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat
python eval_track.py "$RUN" "$A/flat/*.npz" --anchor 0.1 --term fall --tag flat_fall 2>&1 | grep -E "^EVAL|Error" || true
python eval_track.py "$RUN" "$A/gram_test/*.npz" --anchor 0.1 --term fall --tag gram_test 2>&1 | grep -E "^EVAL|Error" || true
python eval_links.py "$RUN" "$A/links_anat" --anchor 0.1 --term fall --out links_anat --seeds 0,1,2,3,4,5,6,7 2>&1 | grep -E "^LINKS|Error" || true
