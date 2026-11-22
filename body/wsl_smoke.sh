#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
python smoke.py ../refs/D1_009_KAN01_001.npz "${1:-256}"
