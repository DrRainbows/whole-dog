#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
python -c "import jax, brax; print('jax', jax.__version__, 'brax', brax.__version__); print('pmap' , hasattr(jax,'pmap'))"
uv pip index versions brax 2>/dev/null || pip index versions brax 2>/dev/null | head -3 || true
B=$(python -c "import brax,os;print(os.path.dirname(brax.__file__))")
grep -rn "device_put_replicated\|jax.pmap\|pmap(" $B/training/agents/ppo/train.py $B/training/*.py | head -30
