#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
nice -n 10 taskset -c 0-3 python train_track.py ../refs/walk/D1_007_KAN01_001_p06.npz --steps 6000 --envs 16 \
  --batch 16 --minibatches 4 --evals 2 --eval-envs 4 --impl jax --gamma 0.98 --lookahead2 15 --paws-obs --w-rootvel 0.1 \
  --out "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/smoke_flags"
