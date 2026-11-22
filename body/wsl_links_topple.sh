#!/usr/bin/env bash
# Usage: wsl_links_topple.sh RUN [RUN ...]
# The link test (refs_anat/links_anat, 8 seeds) scored with an absolute fall: the episode ends only when the trunk is
# rolled more than 60 degrees about the long axis, onto its side or back (term "topple"). The pre-registered criterion ("fall")
# is relative to the reference, so a dog lagging behind a fast rise counts as fallen; this separates the two.
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
for run in "$@"; do
  python eval_links.py "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/$run" ${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/refs_anat/links_anat --anchor 0.1 \
    --term topple --out links_anat_topple --seeds 0,1,2,3,4,5,6,7 2>&1 | grep -E "^LINKS|Error|Traceback" || true
done
