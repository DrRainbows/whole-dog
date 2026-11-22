"""Turn tracked keypoints into movement features that don't depend on camera distance or facing.

Per frame and dog: keypoints centred on the torso, scaled by body size, mirrored so the dog faces
right, y pointing up; their frame-to-frame velocities; tracking confidence; and the torso's
own motion relative to its size. A sample is a window of these frames around one moment.
"""
import numpy as np

TORSO = ["back_middle", "back_base", "back_end", "neck_base", "tail_base", "body_middle_left", "body_middle_right"]
HEAD = ["nose", "upper_jaw", "left_eye", "right_eye"]
MIN_CONF = 0.4


UP = {"top": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, 1.0)}


def upright(xy, up):
    """Rotate image coordinates so the world's up is +y: returns coordinates with x right, y up.
    up: (T,) array of "top"/"left"/"right"/"bottom" (where the sky or ceiling is in the image)."""
    out = np.empty_like(xy)
    for t in range(len(xy)):
        ux, uy = UP.get(up[t], UP["top"])
        rx, ry = -uy, ux
        out[t, ..., 0] = xy[t, ..., 0] * rx + xy[t, ..., 1] * ry
        out[t, ..., 1] = xy[t, ..., 0] * ux + xy[t, ..., 1] * uy
    return out


def per_frame(names, xy, conf, up=None):
    """xy (T, K, 2) image pixels for one dog, conf (T, K); up (T,) image direction of the sky.
    Returns (T, F) features and (T,) validity. Output coordinates are x right, y up."""
    T, K = conf.shape
    if up is not None:
        xy = upright(xy, up)
    else:
        xy = xy * np.array([1.0, -1.0])
    ok = conf >= MIN_CONF
    ti = [names.index(n) for n in TORSO if n in names]
    hi = [names.index(n) for n in HEAD if n in names]
    tail = names.index("tail_base")
    center = np.full((T, 2), np.nan)
    scale = np.full(T, np.nan)
    facing = np.ones(T)
    for t in range(T):
        pts = xy[t][ok[t]]
        if len(pts) < 6:
            continue
        tor = xy[t, ti][ok[t, ti]]
        center[t] = tor.mean(0) if len(tor) >= 2 else pts.mean(0)
        lo, hi_ = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
        scale[t] = max(np.hypot(*(hi_ - lo)), 1.0)
        head = xy[t, hi][ok[t, hi]]
        if len(head) and ok[t, tail]:
            facing[t] = 1.0 if head[:, 0].mean() >= xy[t, tail, 0] else -1.0
    valid = np.isfinite(scale)
    # smooth scale and facing over half a second to avoid flicker
    if valid.any():
        idx = np.where(valid)[0]
        scale = np.interp(np.arange(T), idx, scale[idx])
        center = np.stack([np.interp(np.arange(T), idx, center[idx, c]) for c in range(2)], 1)
        k = 15
        f_s = np.convolve(facing, np.ones(k) / k, mode="same")
        facing = np.where(f_s >= 0, 1.0, -1.0)
    rel = (xy - center[:, None, :]) / scale[:, None, None]
    rel[..., 0] *= facing[:, None]
    rel = np.where(ok[..., None], rel, 0.0)
    vel = np.zeros_like(rel)
    vel[1:] = rel[1:] - rel[:-1]
    both = ok.copy()
    both[1:] &= ok[:-1]
    vel = np.where(both[..., None], vel, 0.0)
    cvel = np.zeros((T, 2))
    cvel[1:] = (center[1:] - center[:-1]) / scale[1:, None]
    cvel[:, 0] *= facing
    srate = np.zeros(T)
    srate[1:] = np.log(scale[1:] / scale[:-1])
    feats = np.concatenate([rel.reshape(T, -1), vel.reshape(T, -1) * 10, ok.astype(float), cvel * 10, srate[:, None] * 10], 1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats, valid & (ok.sum(1) >= 8)


def window(feats, t, half=15, step=2):
    """Frames t-half..t+half every `step` frames, padded at the clip edges. Returns (L, F)."""
    T = len(feats)
    idx = np.clip(np.arange(t - half, t + half + 1, step), 0, T - 1)
    return feats[idx]


def bbox(xy_t, conf_t):
    pts = xy_t[conf_t >= MIN_CONF]
    if len(pts) < 4:
        return None
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    return x0, y0, x1, y1


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
