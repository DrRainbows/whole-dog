"""How fast the head responds to a sound.

Head direction is the angle of the line from the neck to the nose. In a side view a turn
toward or away from the camera also changes this angle, through foreshortening, so both
pitch and yaw turns register.
"""
import numpy as np


HEAD_BASES = (("left_eye", "right_eye"), ("neck_end",), ("neck_base",))


def head_base(coords, names):
    """The most complete head reference: the eyes' midpoint, else the end of the neck, else its base."""
    best, best_n = None, -1
    for cand in HEAD_BASES:
        if all(c in names for c in cand):
            pts = np.nanmean(np.stack([coords[:, names.index(c)] for c in cand]), axis=0)
            n = int(np.isfinite(pts).all(axis=1).sum())
            if n > best_n:
                best, best_n = pts, n
    return best


def head_angle(coords, names, tip="nose"):
    base = head_base(coords, names)
    j = names.index(tip)
    d = coords[:, j] - base
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    ok = np.isfinite(ang)
    if ok.sum() > 1:
        ang[ok] = np.degrees(np.unwrap(np.radians(ang[ok])))
    return ang


def turn_latency(angle, fps, stim_t, base_s=0.5, max_wait_s=2.0, k_sd=4.0, min_rate=60.0, hold=2):
    """Seconds from the stimulus to the start of the first head movement, or None.

    A movement starts when angular speed exceeds both min_rate (deg/s) and the pre-stimulus
    noise level (mean + k_sd standard deviations) for `hold` frames in a row. The onset is then
    moved back to where the speed first rose above a third of that threshold.
    """
    w = np.abs(np.gradient(angle) * fps)
    i0 = int(round(stim_t * fps))
    b0 = max(0, i0 - int(round(base_s * fps)))
    base = w[b0:i0]
    base = base[np.isfinite(base)]
    thr = min_rate if base.size < 3 else max(min_rate, base.mean() + k_sd * base.std())
    end = min(len(w), i0 + int(round(max_wait_s * fps)))
    for i in range(i0, end - hold + 1):
        seg = w[i:i + hold]
        if np.all(np.isfinite(seg)) and np.all(seg > thr):
            j = i
            while j > i0 and np.isfinite(w[j - 1]) and w[j - 1] > thr / 3:
                j -= 1
            return (j - i0) / fps
    return None
