"""Build a MuJoCo dog from the motion-capture skeleton, and replay captured motion on it.

Every BVH joint becomes a body with three hinges. Their order is chosen per joint (joint_orders.json) to
keep away from gimbal lock; each captured frame is re-expressed in those orders, so it maps onto the
model exactly. Bodies are written in the
capture's coordinates (y up, centimetres converted to metres); one fixed rotation at the
root turns them into MuJoCo's world (z up, the dog walking along +x).

  .venv\\Scripts\\python mocap_model.py build          write dog_mocap.xml
  .venv\\Scripts\\python mocap_model.py replay FILE    render frames of a capture on the model
"""
import glob
import json
import os
import sys

import numpy as np

import bvh

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sorted(glob.glob(os.path.join(HERE, "data", "mann_mocap", "bvh", "**", "*.bvh"), recursive=True))
CM = 0.01
# capture axes (x, y up, z forward) -> MuJoCo axes (x forward, y left, z up)
M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
AXES = {"Xrotation": "1 0 0", "Yrotation": "0 1 0", "Zrotation": "0 0 1"}
RADIUS = {"Hips": 0.075, "Spine": 0.085, "Spine1": 0.08, "Neck": 0.045, "Head": 0.05, "Tail": 0.018, "Tail1": 0.012}
STRONG = {"Hips", "Spine", "Spine1", "LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm", "LeftShoulder", "RightShoulder"}
ORDER_FILE = os.path.join(HERE, "joint_orders.json")
# Soft parts trail passively (a spring toward neutral, and damping; no motor): the first notes' "soft body as a
# passive trailing parameter". The tail also sits near gimbal lock in any hinge order, so tracking it was noise.
PASSIVE = {"Tail", "Tail1"}


def joint_orders():
    """Per joint, the hinge order (e.g. "ZYX"), chosen so the middle hinge stays far from +-90 deg (gimbal lock)
    over the whole capture. In the capture's own order (ZXY) 11 of 20 joints came within 65-89 deg of it, so
    the hinge angles spun at up to 75 rad/s while the limbs barely moved. Without the file: the capture's order."""
    return json.load(open(ORDER_FILE)) if os.path.exists(ORDER_FILE) else {}


def hinge_channels(j, orders):
    o = orders.get(j.name)
    return [f"{c}rotation" for c in o] if o else list(j.channels)


def mat2quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "Xrotation":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Yrotation":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def radius(name):
    if name in RADIUS:
        return RADIUS[name]
    if "Shoulder" in name or "UpLeg" in name:
        return 0.035
    if "Arm" in name and "Fore" not in name:
        return 0.03
    return 0.022


def build(path=os.path.join(HERE, "dog_mocap.xml")):
    joints, _, ft = bvh.read(FILES[0])
    orders = joint_orders()
    lines = []

    def body(j, depth):
        pad = "  " * depth
        off = j.offset * CM if j.parent else np.zeros(3)
        lines.append(f'{pad}<body name="{j.name}" pos="{off[0]:.4f} {off[1]:.4f} {off[2]:.4f}">')
        if j.parent is None:
            lines.append(f'{pad}  <freejoint name="root"/>')
        else:
            for ch in hinge_channels(j, orders):
                lim = "-180 180"
                cls = "passive" if j.name in PASSIVE else "strong" if j.name in STRONG else "limb"
                lines.append(f'{pad}  <joint name="{j.name}_{ch[0].lower()}" axis="{AXES[ch]}" range="{lim}" class="{cls}"/>')
        # bone geometry toward each child (or the end site)
        tips = [c.offset * CM for c in j.children] or ([j.end * CM] if j.end is not None else [])
        r = radius(j.name)
        for t in tips:
            if np.linalg.norm(t) > 1e-3:
                lines.append(f'{pad}  <geom type="capsule" fromto="0 0 0 {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}" size="{r:.3f}"/>')
        if j.end is not None and not j.children and ("Hand" in j.name or "Foot" in j.name):
            e = j.end * CM
            lines.append(f'{pad}  <site name="{j.name}_paw" pos="{e[0]:.4f} {e[1]:.4f} {e[2]:.4f}"/>')
            lines.append(f'{pad}  <geom name="{j.name}_pad" type="sphere" pos="{e[0]:.4f} {e[1]:.4f} {e[2]:.4f}" size="0.022" class="paw"/>')
        for c in j.children:
            body(c, depth + 1)
        lines.append(f"{pad}</body>")

    root = joints[0]
    root_quat = mat2quat(M)
    body_lines_start = len(lines)
    body(root, 3)
    # the root body carries the capture-to-world rotation inside its free joint's pose
    acts = []
    for j in joints[1:]:
        if j.name in PASSIVE:
            continue
        for ch in hinge_channels(j, orders):
            n = f"{j.name}_{ch[0].lower()}"
            acts.append(f'    <position name="{n}" joint="{n}" class="{"strong" if j.name in STRONG else "limb"}"/>')
    xml = f"""<mujoco model="dog_mocap">
  <compiler angle="degree" autolimits="true" settotalmass="28"/>
  <option timestep="0.004" solver="Newton" integrator="implicitfast" iterations="4" ls_iterations="6"/>
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom density="1000" rgba="0.82 0.72 0.55 1" condim="3" friction="0.9 0.02 0.01" contype="2" conaffinity="0"/>
    <default class="paw"><geom contype="2" conaffinity="0" friction="1.0 0.02 0.01" rgba="0.2 0.2 0.2 1"/></default>
    <default class="strong"><joint damping="2" armature="0.02"/><position kp="300" kv="8" forcerange="-200 200"/></default>
    <default class="limb"><joint damping="0.8" armature="0.01"/><position kp="120" kv="3" forcerange="-80 80"/></default>
    <default class="passive"><joint stiffness="1.5" damping="0.08" armature="0.002"/></default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".25 .26 .28" rgb2=".32 .33 .35" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid" contype="1" conaffinity="2"/>
{chr(10).join(lines)}
  </worldbody>
  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
"""
    open(path, "w", encoding="utf-8").write(xml)
    print(f"wrote {path}: {len(acts)} hinges, root rotation quat {np.round(root_quat, 4)}")
    return path


def clip_qpos(model, joints, D):
    """All capture frames -> MuJoCo qpos. Each joint's rotation is rebuilt from the capture's Euler angles and
    re-expressed in the model's hinge order for that joint (angles in (-pi, pi]; unwrap over time afterwards)."""
    import mujoco
    from scipy.spatial.transform import Rotation as Rot

    orders = joint_orders()
    sl = bvh.channel_slices(joints)
    D = np.atleast_2d(D)
    q = np.zeros((len(D), model.nq))
    r = D[:, sl["Hips"]]
    q[:, :3] = (r[:, :3] * CM) @ M.T
    for t in range(len(D)):
        R_b = np.eye(3)
        for ch, v in zip(joints[0].channels[3:], r[t, 3:]):
            R_b = R_b @ rot(ch, v)
        q[t, 3:7] = mat2quat(M @ R_b)
    for j in joints[1:]:
        R = Rot.from_euler("".join(c[0] for c in j.channels), D[:, sl[j.name]], degrees=True)  # upper case: intrinsic
        chans = hinge_channels(j, orders)
        E = R.as_euler("".join(c[0] for c in chans))
        for k, ch in enumerate(chans):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{j.name}_{ch[0].lower()}")
            q[:, model.jnt_qposadr[jid]] = E[:, k]
    return q


def frame_qpos(model, joints, row):
    """One capture frame -> MuJoCo qpos for dog_mocap.xml."""
    return clip_qpos(model, joints, np.asarray(row)[None])[0]


def replay(name, frames=(0, 30, 60, 90)):
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "dog_mocap.xml"))
    data = mujoco.MjData(model)
    f = next(p for p in FILES if os.path.basename(p).startswith(name))
    joints, D, ft = bvh.read(f)
    r = mujoco.Renderer(model, 360, 480)
    imgs = []
    for k in frames:
        data.qpos[:] = frame_qpos(model, joints, D[min(k, len(D) - 1)])
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = data.qpos[:3]
        cam.lookat[2] = 0.25
        cam.distance, cam.elevation, cam.azimuth = 1.6, -10, 90
        r.update_scene(data, cam)
        imgs.append(r.render())
    out = os.path.join(HERE, f"replay_{name}.png")
    Image.fromarray(np.hstack(imgs)).save(out)
    print("wrote", out, "| paw heights:",
          [round(float(data.site_xpos[i][2]), 3) for i in range(model.nsite)])


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        replay(sys.argv[2])
