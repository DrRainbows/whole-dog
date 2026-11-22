"""Check that the env's clock stays matched to the body through Brax's auto-resets (run in WSL).

  python test_restart.py [--off]

Wraps the env exactly as Brax PPO does (episode limit + auto-reset), drives it with random actions so that
episodes fail and hit the time limit, and at every step recomputes the observation from the body and the
env's clock. They must agree: a mismatch means the policy is shown frames the body isn't at.
--off runs the old behaviour (clock not put back) for comparison.
"""
import glob
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np
from brax.envs.wrappers import training as wrappers

from dog_track import DogTrack, default_config

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(os.path.dirname(HERE), "dog_mocap_mjx.xml")


def main():
    refs = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "refs", "walk", "*.npz")))
    cfg = default_config()
    cfg.lookahead2, cfg.obs_paws, cfg.w_rootvel = 15, True, 0.1
    cfg.episode_length = 60  # short, so the time limit is hit often too
    cfg.restart_info = "--off" not in sys.argv
    n = 64
    env = DogTrack(XML, refs, config=cfg, max_dogs=n)
    wenv = wrappers.wrap(env, episode_length=cfg.episode_length, action_repeat=1)
    reset = jax.jit(wenv.reset)
    step = jax.jit(wenv.step)
    check = jax.jit(jax.vmap(lambda d, k, e, th, tr: env._obs(d, jp.minimum(k, e), e, th, tr)))
    st = reset(jax.random.split(jax.random.PRNGKey(0), n))
    worst, ends, bad_steps = 0.0, 0, 0
    for i in range(400):
        act = 0.5 * jax.random.normal(jax.random.PRNGKey(i + 1), (n, env.action_size))
        st = step(st, act)
        ends += int(st.done.sum())
        again = check(st.pipeline_state, st.info["k"], st.info["end"], st.info["th"], st.info["tr"])
        err = np.abs(np.nan_to_num(np.array(again)) - np.array(st.obs)).max(1)
        worst = max(worst, float(err.max()))
        bad_steps += int((err > 1e-3).sum())
    print(f"restart_info={cfg.restart_info}: {ends} episode ends in 400 steps x {n} envs; "
          f"observations that disagree with the body: {bad_steps} of {400 * n} (worst {worst:.3f})")


if __name__ == "__main__":
    main()
