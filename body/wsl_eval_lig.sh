#!/usr/bin/env bash
# The ligament body's evaluation (RESULTS.md, 21 Nov, L1-L2): the standard evaluation (capture, held-out sentences,
# link test over 8 seeds), the link test with real topples, and which muscles work at their limit (L2 is judged on
# the 38 muscles that are left; the 16 ligament-held hinges have no strength by design).
set -e
bash "$(git rev-parse --show-toplevel)/body/wsl_eval_all.sh" anat_lig
bash "$(git rev-parse --show-toplevel)/body/wsl_links_topple.sh" anat_lig
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
python diag_muscles.py ${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/anat_lig "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/flat/*.npz" --every 10 2>&1 | grep -A 16 "most often" || true
