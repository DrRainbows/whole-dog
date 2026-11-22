"""Keep only the flat-floor parts of the capture, for physics training on flat ground.

  .venv\\Scripts\\python split_flat.py

Some MANN captures have the dog on platforms or obstacles that are not in our scene: no paw near the
floor for seconds at a time. A physics dog on a flat floor cannot follow those stretches, so they are
cut out. Kept: stretches of at least 2 s where no airborne spell lasts over 0.4 s (a real jump) and
nothing moves impossibly fast. Each part is re-floored on its own. Writes refs/flat/CLIP_pN.npz.
"""
import glob
import os
from collections import Counter

import numpy as np
import mujoco

import catalog_mocap as cm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "refs", "flat")
DT = 0.02
NEAR, MAX_AIR, PAD, MIN_LEN = 0.04, 20, 10, 100  # metres, frames, frames, frames


def runs(mask):
    out, s = [], None
    for t, v in enumerate(list(mask) + [False]):
        if v and s is None:
            s = t
        elif not v and s is not None:
            out.append((s, t))
            s = None
    return out


def main():
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "dog_mocap_mjx.xml"))
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.npz")):
        os.remove(f)
    kept, dropped = Counter(), 0.0
    n_parts = 0
    for f in sorted(glob.glob(os.path.join(HERE, "refs", "all", "*.npz"))):
        z = dict(np.load(f))
        clip = os.path.basename(f)[:-4]
        T = len(z["qpos"])
        _, padz = cm.kinematics(m, z["qpos"])
        low = padz.min(1)
        bad = np.zeros(T, bool)
        for a, b in runs(low >= NEAR):  # airborne spells longer than a jump
            if b - a > MAX_AIR:
                bad[a:b] = True
        root_v = np.linalg.norm(z["qvel"][:, :2], axis=1)
        pad_v = np.r_[0, np.linalg.norm(np.diff(z["paws"][:, :, :2], axis=0), axis=2).max(1) / DT]
        bad |= (root_v > 7.0) | (pad_v > 15.0)  # glitches
        grow = np.convolve(bad.astype(int), np.ones(2 * PAD + 1, int), mode="same") > 0
        for a, b in runs(~grow):
            if b - a < MIN_LEN:
                continue
            part = {k: (v[a:b].copy() if isinstance(v, np.ndarray) and v.ndim and len(v) == T else v) for k, v in z.items()}
            part["qpos"][:, 7:] -= 2 * np.pi * np.round(np.median(part["qpos"][:, 7:], axis=0) / (2 * np.pi))
            shift = float(np.percentile(low[a:b], 2))
            if abs(shift) < 0.05:  # re-floor this part on its own
                part["qpos"][:, 2] -= shift
                part["paws"][:, :, 2] -= shift
                part["com"][:, 2] -= shift
            np.savez(os.path.join(OUT, f"{clip}_p{n_parts % 100:02d}.npz"), **part)
            n_parts += 1
            for r, c in zip(*np.unique(part["regime"], return_counts=True)):
                kept[cm.SUPPORT[int(r)]] += c * DT
        dropped += grow.sum() * DT + sum(b - a for a, b in runs(~grow) if b - a < MIN_LEN) * DT
    total = sum(kept.values())
    print(f"kept {total / 60:.1f} min in {n_parts} parts; dropped {dropped / 60:.1f} min")
    print("kept per regime (s): " + ", ".join(f"{r} {s:.0f}" for r, s in kept.most_common()))


if __name__ == "__main__":
    main()
