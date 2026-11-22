#!/usr/bin/env bash
# Tiny CPU-only training run to exercise the full loop (train, save, vectorized eval) on three clips. Pinned to 4 cores.
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
nice -n 10 taskset -c 0-3 python train_track.py ../refs/all/D1_003_KAN01_001.npz ../refs/all/D1_005_KAN01_001.npz \
  ../refs/flat/D1_009_KAN01_001_p09.npz --steps 6000 --envs 16 \
  --batch 16 --minibatches 4 --evals 2 --eval-envs 4 --impl jax --out "${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs/smoke_cpu"
