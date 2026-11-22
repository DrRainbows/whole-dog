"""Join teacher labels to tracked keypoints.

Each teacher label names a dog ("lab" or "mal") and draws its box. Each track was named by coat
brightness. A label is kept only if its box overlaps the track of the same name; when it
overlaps the other track instead, that is a naming error, and it is counted.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

from .features import bbox, iou, per_frame

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(HERE, "library", "work")
LABELS = os.path.join(HERE, "library", "labels", "gemini.jsonl")
SUPPORT = ["stand", "walk", "trot", "run", "jump", "sit", "down", "side", "back", "crouch", "rear_up", "swim", "wade"]
ACTION = ["none", "sniff", "look", "carry", "tug", "mouth_play", "chew", "eat_drink", "lick", "pant"]
CONTEXT = ["rest", "explore", "play_dog", "play_person", "retrieve", "eat_drink", "watch", "interact_person", "travel",
           "other"]
DOGS = ["lab", "mal"]


def load_tracks(clip):
    p = os.path.join(WORK, "tracks", clip + ".dogs.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=False)
    return list(d["names"]), d["xy"], d["conf"], float(d["fps"])


def frame_size(clip):
    import cv2

    cap = cv2.VideoCapture(os.path.join(WORK, clip + ".mp4"))
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return w, h


def teacher_rows(include_repeats=False):
    rows = []
    if not os.path.exists(LABELS):
        return rows
    for line in open(LABELS, encoding="utf-8"):
        r = json.loads(line)
        if r.get("repeat") and not include_repeats:
            continue
        for dog in DOGS:  # the teacher now and then writes a box coordinate as text ("585"): read it as a number
            d = r.get(dog)
            if isinstance(d, dict) and isinstance(d.get("box"), list):
                try:
                    d["box"] = [float(x) for x in d["box"]]
                except (TypeError, ValueError):
                    d["box"] = None
        rows.append(r)
    return rows


def up_track(rows, T, fps):
    """Per-frame 'up' from the teacher's once-a-second labels (nearest label wins)."""
    pts = sorted((int(round(r["t"] * fps)), r.get("up") or "top") for r in rows)
    if not pts:
        return np.array(["top"] * T)
    idx = np.array([p[0] for p in pts])
    vals = [p[1] for p in pts]
    near = np.abs(np.arange(T)[:, None] - idx[None, :]).argmin(1)
    return np.array([vals[k] for k in near])


def build(min_iou=0.2):
    """Returns samples [(clip, dog, frame, support, action, context)], per-clip features, and audit counts."""
    rows = teacher_rows()
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(r)
    samples, feats, audit = [], {}, Counter()
    for clip, rs in by_clip.items():
        tr = load_tracks(clip)
        if tr is None:
            audit["no_tracks_yet"] += len(rs)
            continue
        names, xy, conf, fps = tr
        w, h = frame_size(clip)
        up = up_track(rs, xy.shape[0], fps)
        per = {}
        for di, dog in enumerate(DOGS):
            per[dog] = per_frame(names, xy[:, di], conf[:, di], up)
        feats[clip] = per
        for r in rs:
            f = int(round(r["t"] * fps))
            if f >= xy.shape[0]:
                continue
            if (r.get("other_dogs") or 0) > 0:
                audit["other_dogs_present"] += 1
                continue
            for di, dog in enumerate(DOGS):
                lab = r.get(dog) or {}
                if not lab.get("visible"):
                    continue
                audit["teacher_visible"] += 1
                box = lab.get("box")
                if not (isinstance(box, list) and len(box) == 4):
                    audit["no_box"] += 1
                    continue
                tb = (box[1] / 1000 * w, box[0] / 1000 * h, box[3] / 1000 * w, box[2] / 1000 * h)
                mine = bbox(xy[f, di], conf[f, di])
                other = bbox(xy[f, 1 - di], conf[f, 1 - di])
                o_mine = iou(tb, mine) if mine else 0.0
                o_other = iou(tb, other) if other else 0.0
                if o_mine >= min_iou and o_mine >= o_other:
                    audit["matched"] += 1
                elif o_other >= min_iou:
                    audit["name_swapped"] += 1
                    continue
                else:
                    audit["not_tracked"] += 1
                    continue
                if not per[dog][1][f]:
                    audit["too_few_keypoints"] += 1
                    continue
                sup = lab.get("support")
                act = lab.get("action")
                ctx = lab.get("context")
                samples.append((clip, dog, f,
                                SUPPORT.index(sup) if sup in SUPPORT else -1,
                                ACTION.index(act) if act in ACTION else -1,
                                CONTEXT.index(ctx) if ctx in CONTEXT else -1))
    return samples, feats, audit
