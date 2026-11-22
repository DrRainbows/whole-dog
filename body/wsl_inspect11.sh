#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
F=$(python -c "import mujoco.mjx.warp.testspeed as t;print(t.__file__)")
grep -n "put_model\|make_data\|vmap\|jit\|naconmax\|njmax\|nworld\|batch\|def \|step(" "$F" | head -60
