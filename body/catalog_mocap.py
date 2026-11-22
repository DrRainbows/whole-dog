"""Name every moment of the dog motion capture in the footage's regime vocabulary, from kinematics alone.

  .venv\\Scripts\\python catalog_mocap.py

For each of the 51 MANN captures: build the 50 Hz reference, label every frame from hip height, trunk
pitch and roll, speed, paw contacts and footfall timing, then list the regime segments and the
transitions between them. Writes refs/all/CLIP.npz (reference + per-frame regime) and MOCAP_CATALOG.md.

This is the bridge between the router (learned from the footage, in regime names) and the physics
body (which learns from capture): the router asks for "down", the body knows which captured moments
are lying down, and which are the way down from standing.
"""
import os
from collections import Counter, defaultdict

import mujoco
import numpy as np

import make_reference as mr
import mocap_model as mm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "refs", "all")
SUPPORT = ["stand", "walk", "trot", "run", "jump", "sit", "down", "side", "back", "crouch", "rear_up", "swim", "wade"]
PADS = ("LeftHand_pad", "RightHand_pad", "LeftFoot_pad", "RightFoot_pad")  # front left, front right, hind left, hind right
BODIES = ("Hips", "Spine1", "Neck", "Head", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg")
DT = mr.CTRL_DT


def smooth(x, n):
    k = np.ones(n) / n
    return np.convolve(np.pad(x.astype(float), (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def kinematics(m, q):
    d = mujoco.MjData(m)
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in BODIES}
    gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in PADS]
    rad = [m.geom_size[g][0] for g in gid]
    pos, padz = defaultdict(list), []
    for qi in q:
        d.qpos[:] = qi
        mujoco.mj_kinematics(m, d)
        for n, i in bid.items():
            pos[n].append(d.xpos[i].copy())
        padz.append([d.geom_xpos[g][2] - r for g, r in zip(gid, rad)])
    return {n: np.array(v) for n, v in pos.items()}, np.array(padz)


def features(m, ref):
    B, padz = kinematics(m, ref["qpos"])
    fwd = B["Neck"] - B["Hips"]
    pitch = np.degrees(np.arctan2(fwd[:, 2], np.linalg.norm(fwd[:, :2], axis=1)))

    def roll(left, right):
        v = B[left] - B[right]
        return np.degrees(np.arctan2(v[:, 2], np.linalg.norm(v[:, :2], axis=1)))

    # a paw is planted when it is near its own lowest height in this clip and not sliding
    pv = np.vstack([np.zeros((1, 4)), np.linalg.norm(np.diff(ref["paws"][:, :, :2], axis=0), axis=2) / DT])
    body_v = np.linalg.norm(ref["qvel"][:, :2], axis=1)[:, None]
    contact = (padz < np.percentile(padz, 5, axis=0)[None] + 0.03) & (pv < np.maximum(0.3, 0.3 * body_v))
    air = ~contact.any(1)
    air_run = np.zeros(len(air), int)  # length of the airborne stretch each frame belongs to
    s = None
    for t in range(len(air) + 1):
        if t < len(air) and air[t]:
            s = t if s is None else s
        elif s is not None:
            air_run[s:t] = t - s
            s = None
    return {
        "hips_z": B["Hips"][:, 2], "neck_z": B["Neck"][:, 2], "pitch": smooth(pitch, 5),
        "roll": smooth((roll("LeftArm", "RightArm") + roll("LeftUpLeg", "RightUpLeg")) / 2, 5),
        "speed": smooth(np.linalg.norm(ref["qvel"][:, :2], axis=1), 25),
        "duty": smooth(contact.mean(1), 50),
        "diag": smooth((contact[:, 0] == contact[:, 3]) & (contact[:, 1] == contact[:, 2]), 50),
        "front_off": smooth(~contact[:, 0] & ~contact[:, 1], 10) > 0.8,
        "air_run": air_run,
    }


def label_frames(f):
    T = len(f["hips_z"])
    lab = np.empty(T, object)
    for t in range(T):
        hz, pitch, sp = f["hips_z"][t], f["pitch"][t], f["speed"][t]
        if abs(f["roll"][t]) > 120 and hz < 0.3:
            lab[t] = "back"
        elif abs(f["roll"][t]) > 45 and hz < 0.3:
            lab[t] = "side"
        elif hz < 0.3 and pitch >= 30 and not f["front_off"][t] and sp < 0.4:
            lab[t] = "sit"  # haunches down, chest up (D1_003)
        elif hz < 0.24 and pitch < 30:
            lab[t] = "down"  # hips low and the neck about level with them (D1_005)
        elif pitch > 40 and f["front_off"][t] and hz >= 0.3:
            lab[t] = "rear_up"
        elif f["air_run"][t] >= 8:
            lab[t] = "jump"
        elif pitch < -15 or (hz < 0.36 and sp < 0.4):  # play bow, or a low stance on the way up or down
            lab[t] = "crouch"
        elif sp > 0.3:
            # trot: diagonal pairs land together; canter and gallop: they don't, and paws are down briefly
            if sp > 2.6 or (f["diag"][t] < 0.5 and f["duty"][t] < 0.4):
                lab[t] = "run"
            elif f["diag"][t] >= 0.55 and f["duty"][t] < 0.55:
                lab[t] = "trot"
            else:
                lab[t] = "walk"
        else:
            lab[t] = "stand"
    return lab


def mode_filter(lab, n=21):
    out = lab.copy()
    h = n // 2
    for t in range(len(lab)):
        out[t] = Counter(lab[max(0, t - h):t + h + 1]).most_common(1)[0][0]
    return out


def segments(lab, min_len=15):
    segs, s = [], 0
    for t in range(1, len(lab) + 1):
        if t == len(lab) or lab[t] != lab[s]:
            segs.append([lab[s], s, t])
            s = t
    changed = True
    while changed and len(segs) > 1:  # fold segments under 0.3 s into the longer neighbour
        changed = False
        for i, (r, a, b) in enumerate(segs):
            if b - a < min_len:
                j = i - 1 if i == len(segs) - 1 or (i > 0 and segs[i - 1][2] - segs[i - 1][1] >= segs[i + 1][2] - segs[i + 1][1]) else i + 1
                lo, hi = min(segs[i][1], segs[j][1]), max(segs[i][2], segs[j][2])
                segs[j] = [segs[j][0], lo, hi]
                del segs[i]
                changed = True
                break
    merged = []
    for s in segs:
        if merged and merged[-1][0] == s[0]:
            merged[-1][2] = s[2]
        else:
            merged.append(s)
    return merged


def main():
    m = mujoco.MjModel.from_xml_path(mr.mjx_model())
    os.makedirs(OUT, exist_ok=True)
    total = Counter()
    trans = defaultdict(list)
    lines = []
    for path in mm.FILES:
        clip = os.path.splitext(os.path.basename(path))[0]
        ref = mr.build(m, clip)
        f = features(m, ref)
        segs = segments(mode_filter(label_frames(f)))
        regime = np.zeros(len(ref["qpos"]), np.int8)
        for r, a, b in segs:
            regime[a:b] = SUPPORT.index(r)
            total[r] += (b - a) * DT
        for (r0, _, b0), (r1, a1, _) in zip(segs[:-1], segs[1:]):
            trans[(r0, r1)].append(f"{clip} {a1 * DT:.1f}s")
        np.savez(os.path.join(OUT, clip + ".npz"), regime=regime, **ref)
        tl = " → ".join(f"{r} {(b - a) * DT:.1f}" for r, a, b in segs)
        lines.append(f"| {clip} | {len(regime) * DT:.1f} | {np.mean(f['speed']):.2f} | {tl} |")
        print(f"{clip}: {tl}", flush=True)
    with open(os.path.join(HERE, "MOCAP_CATALOG.md"), "w", encoding="utf-8") as fh:
        fh.write("# Motion capture catalog (MANN dog, 51 clips)\n\n")
        fh.write("Every 50 Hz frame named in the footage's regime vocabulary from kinematics alone (hip height, trunk "
                 "pitch and roll, speed, paw contacts, footfall timing). Written by `catalog_mocap.py`. "
                 "Capture licence CC BY-NC 4.0: prototyping only.\n\n## Time per regime\n\n| regime | seconds |\n|---|---|\n")
        for r, s in total.most_common():
            fh.write(f"| {r} | {s:.0f} |\n")
        fh.write("\n## Transitions found\n\n| from → to | count | examples |\n|---|---|---|\n")
        for (a, b), ex in sorted(trans.items(), key=lambda kv: -len(kv[1])):
            fh.write(f"| {a} → {b} | {len(ex)} | {', '.join(ex[:3])} |\n")
        fh.write("\n## Clips\n\n| clip | seconds | mean speed m/s | regimes (seconds each) |\n|---|---|---|---|\n")
        fh.write("\n".join(lines) + "\n")
    print("\n" + ", ".join(f"{r} {s:.0f}s" for r, s in total.most_common()))
    print(f"{sum(len(v) for v in trans.values())} transitions of {len(trans)} kinds")


if __name__ == "__main__":
    main()
