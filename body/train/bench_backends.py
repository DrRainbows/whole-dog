"""Raw physics throughput of the dog model under different backends and contact sets (run in WSL).

  python bench_backends.py [nenv] [nstep]

Holds a standing pose from the trot capture with the joint motors, from slightly perturbed starts,
and reports physics steps per second, compile time, and whether the dogs stay numerically sound.
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(os.path.dirname(HERE), "dog_mocap_mjx.xml")
REF = os.path.join(os.path.dirname(HERE), "refs", "D1_009_KAN01_001.npz")
# what a dog lands on: pelvis haunches, belly, chest, head, forearms, hind hock-to-paw, the four pads
LITE = {"Hips": (0, 1), "Spine": (0,), "Spine1": (0,), "Head": (0,), "LeftForeArm": (0,), "RightForeArm": (0,),
        "LeftHand": (1,), "RightHand": (1,), "LeftFoot": (0, 1), "RightFoot": (0, 1)}


def lite_contacts(mj):
    seen = {}
    for g in range(mj.ngeom):
        b = mj.body(mj.geom_bodyid[g]).name
        if b == "world":
            continue
        k = seen.get(b, 0)
        seen[b] = k + 1
        if k not in LITE.get(b, ()):
            mj.geom_contype[g] = 0
            mj.geom_conaffinity[g] = 0
    return mj


def bench(impl, lite, dt, nenv, nstep):
    mj = mujoco.MjModel.from_xml_path(XML)
    mj.opt.timestep = dt
    if lite:
        lite_contacts(mj)
    ncol = sum(1 for g in range(mj.ngeom) if mj.geom_contype[g] and mj.body(mj.geom_bodyid[g]).name != "world")
    ref = np.load(REF)
    q0 = jp.array(ref["qpos"][100], dtype=jp.float32)
    act_q = jp.array([mj.jnt_qposadr[mj.actuator_trnid[i, 0]] for i in range(mj.nu)])
    mx = mjx.put_model(mj, impl=impl)

    def init(key):
        if impl == "warp":
            d = mjx.make_data(mj, impl="warp", naconmax=nenv * 64, njmax=400)
        else:
            d = mjx.make_data(mx)
        q = q0 + 0.05 * jax.random.normal(key, q0.shape).at[0:7].set(0.0)
        d = d.replace(qpos=q, ctrl=q0[act_q])
        return mjx.forward(mx, d)

    keys = jax.random.split(jax.random.PRNGKey(0), nenv)
    d = jax.jit(jax.vmap(init))(keys)
    step = jax.vmap(mjx.step, in_axes=(None, 0))

    @jax.jit
    def unroll(d):
        return jax.lax.scan(lambda d, _: (step(mx, d), None), d, None, length=nstep)[0]

    t = time.time()
    d1 = unroll(d)
    jax.block_until_ready(d1.qpos)
    tc = time.time() - t
    t = time.time()
    d2 = unroll(d1)
    jax.block_until_ready(d2.qpos)
    tr = time.time() - t
    q = np.array(d2.qpos)
    ok = np.isfinite(q).all(1)
    rate = nenv * nstep / tr
    print(f"{impl:4s} {'lite' if lite else 'full':4s} contacts={ncol:2d} dt={dt}: compile {tc:5.1f}s  "
          f"{rate:9.0f} physics steps/s = {rate * dt / 0.02:8.0f} control steps/s; finite {ok.mean():.3f}; "
          f"median hip height after {2 * nstep * dt:.1f}s {np.median(q[ok, 2]):.2f} m", flush=True)


if __name__ == "__main__":
    nenv = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    nstep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    print(jax.devices(), flush=True)
    for impl, lite, dt in (("jax", False, 0.004), ("jax", True, 0.004), ("jax", True, 0.005),
                           ("warp", False, 0.004), ("warp", True, 0.005)):
        try:
            bench(impl, lite, dt, nenv, nstep)
        except Exception as e:  # noqa: BLE001
            print(f"{impl} lite={lite} dt={dt}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
