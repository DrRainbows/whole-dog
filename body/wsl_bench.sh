#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
python bench_backends.py "${1:-1024}" "${2:-100}"
