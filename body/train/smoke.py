"""Smoke test: build the tracking env, run random actions in many parallel dogs, report shapes and speed."""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp

jax.config.update("jax_default_matmul_precision", "tensorfloat32")
from dog_track import DogTrack  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
env = DogTrack(os.path.join(os.path.dirname(HERE), "dog_mocap_mjx.xml"), sys.argv[1])
print("obs size", env.observation_size, "| actions", env.action_size, "| clip steps", env._T)
reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))
st = reset(jax.random.split(jax.random.PRNGKey(0), n))
zero = jp.zeros((n, env.action_size))
t = time.time()
st = step(st, zero)
jax.block_until_ready(st.obs)
print(f"compile + first step: {time.time() - t:.1f}s")
t = time.time()
steps = 50
alive = jp.ones(n)
for i in range(steps):
    st = step(st, zero)  # zero corrections: the capture's own joint targets, open loop
    alive = alive * (1 - st.done)
jax.block_until_ready(st.obs)
dt = time.time() - t
print(f"{steps} control steps x {n} dogs in {dt:.2f}s -> {steps * n / dt:,.0f} env-steps/s "
      f"({steps * n * env._n_sub / dt:,.0f} physics steps/s)")
print("open-loop tracking (no policy): share of dogs still on the capture after 1 s:", float(alive.mean()))
finite = jp.all(jp.isfinite(st.pipeline_state.qpos), axis=1)
print("dogs with finite state:", float(finite.mean()), "| mean reward (finite dogs):",
      float(jp.where(finite, st.reward, 0).sum() / jp.maximum(finite.sum(), 1)))
