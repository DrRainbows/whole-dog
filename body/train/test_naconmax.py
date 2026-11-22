"""Is warp's naconmax per world or global under jax.vmap? Stand 1024 dogs with a small per-call budget."""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jp, mujoco, numpy as np
from mujoco import mjx
HERE = os.path.dirname(os.path.abspath(__file__))
mj = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(HERE), "dog_mocap_mjx.xml"))
mj.opt.iterations, mj.opt.ls_iterations = 10, 10
ref = np.load(os.path.join(os.path.dirname(HERE), "refs", "D1_009_KAN01_001.npz"))
q0 = jp.array(ref["qpos"][100], dtype=jp.float32)
act_q = jp.array([mj.jnt_qposadr[mj.actuator_trnid[i, 0]] for i in range(mj.nu)])
mx = mjx.put_model(mj, impl="warp")
nenv, per = int(sys.argv[1]), int(sys.argv[2])
def init(key):
    d = mjx.make_data(mj, impl="warp", naconmax=per, njmax=400)
    d = d.replace(qpos=q0 + 0.03 * jax.random.normal(key, q0.shape).at[0:7].set(0.0), ctrl=q0[act_q])
    return mjx.forward(mx, d)
d = jax.jit(jax.vmap(init))(jax.random.split(jax.random.PRNGKey(0), nenv))
print("contact buffer leaf shapes:", {k: v.shape for k, v in vars(d._impl).items() if hasattr(v, "shape") and v.ndim and v.shape[0] == nenv and v.ndim > 1 and v.shape[1] == per})
step = jax.vmap(mjx.step, in_axes=(None, 0))
unroll = jax.jit(lambda d: jax.lax.scan(lambda d, _: (step(mx, d), None), d, None, length=250)[0])
t = time.time(); d = unroll(d); jax.block_until_ready(d.qpos); dt = time.time() - t
q = np.array(d.qpos)
print(f"nenv {nenv} naconmax per call {per}: {nenv*250/dt:.0f} physics steps/s; hip height after 1 s median {np.median(q[:,2]):.2f} min {q[:,2].min():.2f}; "
      f"lowest pad z min {np.array(d.geom_xpos)[:, [14,19,23,27], 2].min():.3f}; ncon per world {np.array(d._impl.nacon if hasattr(d._impl,'nacon') else -1).ravel()[:4]}")
import subprocess; print(subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
