"""Compile a dog model, report its build, test whether it can stand, and render pictures.

  .venv\\Scripts\\python check_model.py MODEL.xml [--key stand] [--seconds 3]
"""
import argparse
import os

import mujoco
import numpy as np
from PIL import Image


def render(model, data, path, cams=("side", "front")):
    r = mujoco.Renderer(model, 480, 640)
    imgs = []
    for cam in cams:
        opt = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, opt)
        opt.lookat[:] = data.qpos[:3] if model.nq >= 3 else [0, 0, 0.3]
        opt.lookat[2] = 0.3
        opt.distance = 1.8
        opt.elevation = -12
        opt.azimuth = 90 if cam == "side" else 0
        r.update_scene(data, opt)
        imgs.append(r.render())
    Image.fromarray(np.hstack(imgs)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--key", default="stand")
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(a.xml)
    d = mujoco.MjData(m)
    print(f"nq {m.nq}  nv {m.nv}  nu {m.nu}  bodies {m.nbody}  geoms {m.ngeom}  timestep {m.opt.timestep}")
    masses = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): round(float(m.body_mass[i]), 2) for i in range(1, m.nbody)}
    print(f"total mass {sum(masses.values()):.1f} kg:", masses)
    keys = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)]
    print("keyframes:", keys)
    k = keys.index(a.key) if a.key in keys else None
    if k is not None:
        mujoco.mj_resetDataKeyframe(m, d, k)
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    base = os.path.splitext(a.xml)[0]
    render(m, d, base + f"_{a.key}_t0.png")
    # hold the pose with the position actuators
    ctrl = d.ctrl.copy() if k is None else m.key_ctrl[k].copy()
    if not np.any(ctrl):
        # actuators target their joint's keyframe position
        for i in range(m.nu):
            j = m.actuator_trnid[i, 0]
            ctrl[i] = d.qpos[m.jnt_qposadr[j]]
    heights = []
    steps = int(a.seconds / m.opt.timestep)
    for s in range(steps):
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
        if s % int(0.25 / m.opt.timestep) == 0:
            heights.append(round(float(d.qpos[2]), 3))
    print(f"root height every 0.25 s while holding '{a.key}':", heights)
    print("upright (root z axis up):", round(float(d.xmat[1][8]), 3))
    render(m, d, base + f"_{a.key}_t{a.seconds:g}.png")
    print("rendered", base + f"_{a.key}_t0.png", "and", base + f"_{a.key}_t{a.seconds:g}.png")


if __name__ == "__main__":
    main()
