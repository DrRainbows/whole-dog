"""Clean tracks and strip what isn't movement.

Size, facing direction and steady travel are removed, so a judge can't tell two dogs apart
by how big they are or which way they walked. The optional canonical skeleton also removes
build: every dog is redrawn with the same bone lengths, keeping only its joint angles and timing.
"""
import numpy as np
from scipy.signal import savgol_filter

from .tracks import Tracks

# SuperAnimal-Quadruped names used here. Missing ones are skipped.
FRONT, REAR = "neck_base", "tail_base"
PAWS = ["front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"]

# Canonical skeleton: (parent, child, length in body lengths), in build order from the root.
# The root is the midpoint of neck_base and tail_base.
BONES = [
    ("root", "neck_base", 0.5), ("root", "tail_base", 0.5), ("root", "back_middle", 0.05),
    ("neck_base", "left_eye", 0.50), ("left_eye", "nose", 0.16),
    ("left_eye", "left_earbase", 0.10), ("left_earbase", "left_earend", 0.12),
    ("tail_base", "tail_end", 0.45),
    ("neck_base", "front_left_thai", 0.25), ("front_left_thai", "front_left_knee", 0.30),
    ("front_left_knee", "front_left_paw", 0.32),
    ("neck_base", "front_right_thai", 0.25), ("front_right_thai", "front_right_knee", 0.30),
    ("front_right_knee", "front_right_paw", 0.32),
    ("tail_base", "back_left_thai", 0.15), ("back_left_thai", "back_left_knee", 0.32),
    ("back_left_knee", "back_left_paw", 0.38),
    ("tail_base", "back_right_thai", 0.15), ("back_right_thai", "back_right_knee", 0.32),
    ("back_right_knee", "back_right_paw", 0.38),
]


def fill_gaps(v, max_gap):
    """Linearly fill NaN runs no longer than max_gap that have valid samples on both sides."""
    v = v.copy()
    bad = np.isnan(v)
    if bad.all() or not bad.any():
        return v
    i, n = 0, len(v)
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                v[i:j] = np.interp(np.arange(i, j), [i - 1, j], [v[i - 1], v[j]])
            i = j
        else:
            i += 1
    return v


def _smooth_runs(v, win):
    """Savitzky-Golay smoothing applied to each run of valid samples long enough for the window."""
    out = v.copy()
    ok = ~np.isnan(v)
    i, n = 0, len(v)
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j]:
                j += 1
            if j - i >= win:
                out[i:j] = savgol_filter(v[i:j], win, 2)
            i = j
        else:
            i += 1
    return out


def clean(tr, min_conf=0.5, max_gap_s=0.25, smooth_s=0.12):
    xy = tr.xy.copy()
    xy[tr.conf < min_conf] = np.nan
    max_gap = max(1, int(round(max_gap_s * tr.fps)))
    win = max(5, int(round(smooth_s * tr.fps)))
    win += 1 - win % 2  # odd
    for k in range(xy.shape[1]):
        for c in range(2):
            xy[:, k, c] = _smooth_runs(fill_gaps(xy[:, k, c], max_gap), win)
    return Tracks(tr.names, xy, tr.conf, tr.fps, tr.size, tr.source, dict(tr.meta, cleaned=True))


def _nan_moving_average(v, win):
    out = np.full_like(v, np.nan)
    h = max(1, win // 2)
    for i in range(len(v)):
        seg = v[max(0, i - h): i + h + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg)
    return out


def normalize(tr, follow_s=1.0):
    """Return world and display coordinates in body lengths, y up, dog facing +x.

    world:   ground at y=0, x still includes travel (for footfalls and speed)
    display: steady travel removed with a moving average, so the dog trots in place
             but speeding up and stopping stay visible
    """
    xy = tr.xy.copy()
    xy[..., 1] = -xy[..., 1]
    f, r = tr.idx(FRONT), tr.idx(REAR)
    facing = np.nanmedian(xy[:, f, 0] - xy[:, r, 0])
    if facing < 0:
        xy[..., 0] = -xy[..., 0]
    body = np.nanmedian(np.linalg.norm(xy[:, f] - xy[:, r], axis=1))
    xy = xy / body
    paws = [tr.idx(p) for p in PAWS if tr.has(p)]
    ground = np.nanpercentile(xy[:, paws, 1], 5) if paws else np.nanmin(xy[..., 1])
    xy[..., 1] -= ground
    world = xy.copy()
    center = (xy[:, f, 0] + xy[:, r, 0]) / 2
    travel = _nan_moving_average(center, max(3, int(round(follow_s * tr.fps))))
    display = xy.copy()
    display[..., 0] -= travel[:, None]
    meta = {"body_px": float(body), "facing": "right" if facing >= 0 else "left (mirrored)"}
    return world, display, meta


def canonical(display, names):
    """Rebuild every frame with fixed bone lengths, keeping only bone directions."""
    idx = {n: i for i, n in enumerate(names)}
    T = display.shape[0]
    out = np.full((T, len(names), 2), np.nan)
    root = (display[:, idx[FRONT]] + display[:, idx[REAR]]) / 2
    pos = {"root": root}
    src = {"root": root}
    for n in names:
        src[n] = display[:, idx[n]]
    for parent, child, length in BONES:
        if child not in idx or parent not in src or parent not in pos:
            continue
        d = src[child] - src[parent]
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        unit = np.where(norm > 1e-9, d / np.maximum(norm, 1e-9), np.nan)
        pos[child] = pos[parent] + unit * length
        out[:, idx[child]] = pos[child]
    # canonical ground: lowest paw sits at 0
    paws = [idx[p] for p in PAWS if p in idx]
    if paws:
        out[..., 1] -= np.nanpercentile(out[:, paws, 1], 5)
    return out
