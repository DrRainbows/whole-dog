"""Check that the trunk assist reaches the physics on the Warp backend (run in WSL).

  python test_rfc.py

Holds the captured pose with the motors and pushes the trunk up with 2x body weight through the assist
actions; the trunk must rise. Then a pure twist about the trunk's own x axis must roll it.
"""
import glob
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp

from dog_track import DogTrack, default_config, quat_to_mat

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(os.path.dirname(HERE), "dog_mocap_mjx.xml")
REF = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "refs", "walk", "*.npz")))[0]


def main():
    cfg = default_config()
    cfg.rfc_force, cfg.rfc_torque = 1.0, 1.0
    env = DogTrack(XML, [REF], config=cfg, max_dogs=8)
    print("body weight N", round(env._mg, 1), "actions", env.action_size)
    st0 = jax.jit(env.reset_at)(jax.random.PRNGKey(0), jp.int32(0))
    step = jax.jit(env.step)
    R = quat_to_mat(st0.pipeline_state.qpos[3:7])
    up_body = R.T @ jp.array([0.0, 0.0, 1.0])  # world up, in the trunk frame
    for name, wrench in (("none", jp.zeros(6)), ("up 1x weight per unit, 2 units", jp.concatenate([2 * up_body / jp.linalg.norm(up_body), jp.zeros(3)]).clip(-1, 1)),
                         ("roll twist", jp.array([0, 0, 0, 1.0, 0, 0]))):
        st = st0
        act = jp.concatenate([jp.zeros(env._mj.nu), wrench])
        for _ in range(10):
            st = step(st, act)
        q = st.pipeline_state.qpos
        R1 = quat_to_mat(q[3:7])
        roll = jp.arccos(jp.clip(jp.dot(R[:, 1], R1[:, 1]), -1, 1))
        print(f"{name:32s} trunk height {float(q[2]):.3f} (start {float(st0.pipeline_state.qpos[2]):.3f}), "
              f"trunk y-axis turned {float(roll):.3f} rad, assist_f {float(st.metrics['assist_f']):.2f}")


if __name__ == "__main__":
    main()
