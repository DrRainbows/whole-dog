#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
M=$(python -c "import mujoco.mjx,os;print(os.path.dirname(mujoco.mjx.__file__))")
sed -n '/^def _make_data_warp(/,/^def /p' "$M/_src/io.py" | head -80
echo "------ ffi marshal (how batched worlds map to warp nworld / contact buffers)"
grep -n "naconmax\|nworld\|def marshal_custom_vmap\|def _" "$M/warp/ffi.py" | head -40
