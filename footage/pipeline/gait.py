"""Footfalls and stride timing from paw tracks in world coordinates (body lengths, travel kept).

A paw is in stance when it is nearly still in the world. Touchdown is the first still frame,
liftoff the first moving one.
"""
import numpy as np


def stance_mask(x, fps, low=0.05, high=0.15, floor=0.02, min_frames=2):
    """True where the paw is still in the world.

    Thresholds adapt to the paw's own speed, so slow walks and fast trots both work: stance
    starts when speed drops below `low` x the 95th-percentile speed and ends when it rises
    above `high` x that speed (hysteresis). Speeds are in body lengths per second.
    Known bias: a swing that slows smoothly to zero counts its last few percent as stance.
    """
    v = np.abs(np.gradient(x) * fps)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.zeros(len(x), bool)
    top = np.percentile(v[ok], 95)
    lo_thr, hi_thr = max(floor, low * top), max(floor, high * top)
    m = np.zeros(len(x), bool)
    state = bool(ok[0] and v[0] < lo_thr)
    for i in range(len(x)):
        if not ok[i]:
            m[i] = False
            continue
        if state and v[i] > hi_thr:
            state = False
        elif not state and v[i] < lo_thr:
            state = True
        m[i] = state
    # remove stance or swing runs shorter than min_frames
    out = m.copy()
    i, n = 0, len(m)
    while i < n:
        j = i
        while j < n and m[j] == m[i]:
            j += 1
        if j - i < min_frames and 0 < i and j < n:
            out[i:j] = not m[i]
        i = j
    return out


def events(mask):
    """Touchdown and liftoff frame indices from a stance mask."""
    d = np.diff(mask.astype(int))
    td = list(np.where(d == 1)[0] + 1)
    lo = list(np.where(d == -1)[0] + 1)
    return td, lo


def stride_stats(mask, fps):
    td, lo = events(mask)
    if len(td) < 2:
        return None
    periods = np.diff(td) / fps
    duties = []
    for a, b in zip(td[:-1], td[1:]):
        stance = mask[a:b].sum()
        duties.append(stance / (b - a))
    return {
        "touchdowns": [t / fps for t in td],
        "period_s": float(np.mean(periods)),
        "period_cv": float(np.std(periods) / np.mean(periods)) if len(periods) > 1 else 0.0,
        "frequency_hz": float(1.0 / np.mean(periods)),
        "duty": float(np.mean(duties)),
    }


def dissociation(td_hind, td_fore, period):
    """Mean time from each hind touchdown to the next fore touchdown, as a fraction of the stride.
    For a trot, pass a hind paw and its diagonal fore paw. Positive: the fore lands later."""
    out = []
    for h in td_hind:
        later = [f for f in td_fore if f >= h - 0.5 * period]
        if later:
            d = later[0] - h
            if abs(d) < 0.5 * period:
                out.append(d / period)
    return float(np.mean(out)) if out else None
