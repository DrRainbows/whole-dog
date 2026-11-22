#!/usr/bin/env bash
cd "$HOME/dogsim" && . .venv/bin/activate
python - <<'PY'
import inspect
from brax.training.agents.ppo import networks as n
print(inspect.signature(n.make_ppo_networks))
from brax.training import distribution as d
print([c for c in dir(d) if "Distribution" in c])
print(inspect.signature(d.NormalTanhDistribution.__init__))
PY
