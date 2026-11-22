#!/usr/bin/env bash
# Usage: wsl_py.sh SCRIPT.py [args...]   (runs a body/train script in the WSL venv)
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
python "$@"
