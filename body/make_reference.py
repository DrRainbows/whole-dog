"""Turn a capture into a tracking reference for physics training.

  .venv\\Scripts\\python make_reference.py CLIP [START_S END_S]   -> refs/CLIP.npz

The reference holds, at the 50 Hz control rate: qpos, qvel, paw positions, centre of mass.
It also writes dog_mocap_mjx.xml, a copy of the model with unlimited joints (cheaper in MJX).
The capture is lifted or lowered so the paws touch the floor.
"""
import os
import sys

import mujoco
import numpy as np

import bvh
import mocap_model as mm

HERE = os.path.dirname(os.path.abspath(__file__))
CTRL_DT = 0.02


def mjx_model():
    src = open(os.path.join(HERE, "dog_mocap.xml"), encoding="utf-8").read()
    out = src.replace(' range="-180 180"', "")
    path = os.path.join(HERE, "dog_mocap_mjx.xml")
    open(path, "w", encoding="utf-8").write(out)
    return path


def slerp(q0, q1, t):
    d = np.dot(q0, q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def repair_glitches(m, q, paw_limit=12.0, root_limit=7.0):
    """Capture glitches (a paw or the pelvis teleporting for 1-4 frames, e.g. marker swaps) are rebuilt by
    interpolating from the good frames on either side. Returns the repaired qpos and the repaired frames."""
    d = mujoco.MjData(m)
    pad = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in
           ("LeftHand_pad", "RightHand_pad", "LeftFoot_pad", "RightFoot_pad")]
    P = []
    for qi in q:
        d.qpos[:] = qi
        mujoco.mj_kinematics(m, d)
        P.append([d.geom_xpos[g][:2].copy() for g in pad])
    P = np.array(P)
    sp = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=2).max(1) / CTRL_DT]
    root_sp = np.r_[0, np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1) / CTRL_DT]
    bad = (sp > paw_limit) | (root_sp > root_limit)
    idx = np.where(bad)[0]
    for a, b in zip(idx[:-1], idx[1:]):  # frames between two nearby bad frames are bad too
        if b - a <= 15:
            bad[a:b + 1] = True
    bad[0] = bad[-1] = False
    q = q.copy()
    t = 1
    while t < len(q) - 1:
        if not bad[t]:
            t += 1
            continue
        a = t - 1
        b = t
        while b < len(q) - 1 and bad[b]:
            b += 1
        # each joint along the shortest angular path between the good frames (unwrapping the whole track
        # would misread the glitch's own jumps as wraps and spin the joint a full turn)
        delta = (q[b, 7:] - q[a, 7:] + np.pi) % (2 * np.pi) - np.pi
        for i in range(a + 1, b):
            w = (i - a) / (b - a)
            q[i, 0:3] = (1 - w) * q[a, 0:3] + w * q[b, 0:3]
            q[i, 3:7] = slerp(q[a, 3:7], q[b, 3:7], w)
            q[i, 7:] = q[a, 7:] + w * delta
        t = b
    return q, int(bad.sum())


def build(m, clip, start=0.0, end=None, repair=True):
    """Capture -> reference at the control rate: qpos, qvel, paw-pad positions, centre of mass."""
    d = mujoco.MjData(m)
    f = next(p for p in mm.FILES if os.path.basename(p).startswith(clip))
    joints, D, ft = bvh.read(f)
    q60 = mm.clip_qpos(m, joints, D)
    q60[:, 7:] = np.unwrap(q60[:, 7:], axis=0)  # hinge angles continuous over time: no 360 deg jumps
    for i in range(1, len(q60)):  # root quaternion continuous in sign too (q and -q are one rotation)
        if np.dot(q60[i, 3:7], q60[i - 1, 3:7]) < 0:
            q60[i, 3:7] *= -1
    t60 = np.arange(len(q60)) * ft
    end = end or t60[-1]
    t50 = np.arange(start, end, CTRL_DT)
    q = np.zeros((len(t50), m.nq))
    for i, t in enumerate(t50):
        k = min(int(t / ft), len(q60) - 2)
        a = (t - k * ft) / ft
        q[i] = (1 - a) * q60[k] + a * q60[k + 1]
        q[i, 3:7] = slerp(q60[k, 3:7], q60[k + 1, 3:7], a)
    n_fixed = 0
    if repair:
        for _ in range(4):  # repeat: a repaired stretch can expose a neighbouring bad frame
            q, n = repair_glitches(m, q)
            n_fixed += n
            if n == 0:
                break
        q[:, 7:] = np.unwrap(q[:, 7:], axis=0)
    # each hinge's typical angle into the principal branch, so different clips describe a pose alike
    q[:, 7:] -= 2 * np.pi * np.round(np.median(q[:, 7:], axis=0) / (2 * np.pi))
    for i in range(1, len(q)):
        if np.dot(q[i, 3:7], q[i - 1, 3:7]) < 0:
            q[i, 3:7] *= -1
    # floor: the lowest paw-pad bottom sits on the floor (5th percentile, so a stray frame can't set it)
    pad = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in
           ("LeftHand_pad", "RightHand_pad", "LeftFoot_pad", "RightFoot_pad")]
    lows = []
    for qi in q:
        d.qpos[:] = qi
        mujoco.mj_kinematics(m, d)
        lows.append(min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in pad))
    q[:, 2] -= np.percentile(lows, 5 if repair else 2)
    qv = np.zeros((len(q), m.nv))
    for i in range(len(q) - 1):
        mujoco.mj_differentiatePos(m, qv[i], CTRL_DT, q[i], q[i + 1])
    qv[-1] = qv[-2]
    paws, com = [], []
    for qi in q:
        d.qpos[:] = qi
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        paws.append(np.array([d.geom_xpos[g].copy() for g in pad]))
        com.append(d.subtree_com[1].copy())
    return {"qpos": q, "qvel": qv, "paws": np.array(paws), "com": np.array(com), "dt": CTRL_DT, "repaired": n_fixed}


def finish(m, q):
    """A qpos sequence at the control rate -> a full reference: joint velocities, paw-pad positions, centre of mass.
    Used for composed motion (motion matching), which arrives as poses only."""
    d = mujoco.MjData(m)
    pad = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in
           ("LeftHand_pad", "RightHand_pad", "LeftFoot_pad", "RightFoot_pad")]
    q = np.asarray(q, float)
    qv = np.zeros((len(q), m.nv))
    for i in range(len(q) - 1):
        mujoco.mj_differentiatePos(m, qv[i], CTRL_DT, q[i], q[i + 1])
    qv[-1] = qv[-2]
    paws, com = [], []
    for qi in q:
        d.qpos[:] = qi
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        paws.append(np.array([d.geom_xpos[g].copy() for g in pad]))
        com.append(d.subtree_com[1].copy())
    return {"qpos": q, "qvel": qv, "paws": np.array(paws), "com": np.array(com), "dt": CTRL_DT}


def main():
    clip = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    end = float(sys.argv[3]) if len(sys.argv) > 3 else None
    path = mjx_model()
    m = mujoco.MjModel.from_xml_path(path)
    ref = build(m, clip, start, end)
    q, qv = ref["qpos"], ref["qvel"]
    os.makedirs(os.path.join(HERE, "refs"), exist_ok=True)
    out = os.path.join(HERE, "refs", f"{clip}.npz")
    np.savez(out, **ref)
    speed = np.linalg.norm(qv[:, :2], axis=1).mean()
    print(f"wrote {out}: {len(q)} steps ({len(q) * CTRL_DT:.1f} s), mean speed {speed:.2f} m/s, "
          f"hips height {q[:, 2].mean():.2f} m; model {path}: nq {m.nq} nv {m.nv} nu {m.nu}, "
          f"mass {m.body_mass.sum():.1f} kg")


if __name__ == "__main__":
    main()
