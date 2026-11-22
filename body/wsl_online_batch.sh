#!/usr/bin/env bash
# Usage: wsl_online_batch.sh TRACKER REFLEX TAG N SECONDS [PARALLEL] [EXTRA] [FIRST_SEED]
# N sessions per dog (Lab and Malinois) on the CPU (online_dog.py --cpu), PARALLEL at a time. With no EXTRA each dog
# lives on its own (endemic mode: the brain decides everything); EXTRA is passed to online_dog.py (e.g. "--present":
# a person is there, so the dog's own social sentences open). Writes TRACKER/online_batch{TAG}_{dog}_{seed}.json, for
# dog/online_batch_summary.py.
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
export BATCH_RUN="${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/$1" BATCH_REFLEX="${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/$2" BATCH_TAG="$3"
export BATCH_SECONDS="$5" BATCH_EXTRA="${7:-}"
P=${6:-12}
one() {
  # $1 dog, $2 seed
  python online_dog.py "$BATCH_RUN" --seconds "$BATCH_SECONDS" --brain "$1" --seed "$2" --cpu $BATCH_EXTRA \
    --reflex "$BATCH_REFLEX" --tag "_batch${BATCH_TAG}_$1_$2" > /dev/null 2>&1 || echo "failed $1 $2"
}
export -f one
F=${8:-1}
for dog in lab mal; do for s in $(seq "$F" $((F + $4 - 1))); do echo "$dog $s"; done; done | xargs -P "$P" -n 2 bash -c 'one "$0" "$1"'
echo "batch done"
