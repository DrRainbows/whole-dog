#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
python test_naconmax.py "$1" "$2"
