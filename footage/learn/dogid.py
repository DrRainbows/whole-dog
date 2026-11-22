"""Which dog is this? A small image classifier (Labrador vs Malinois) trained on the teacher's boxes.

  .venv\\Scripts\\python -m learn.dogid train     crops from teacher boxes -> models/dogid.pt, accuracy on held-out clips
  .venv\\Scripts\\python -m learn.dogid apply     name every tracked detection, rewrite tracks/*.dogs.npz

The tracker finds dogs but not which dog. This classifier looks at each detection's crop.
Detections it isn't sure about (other dogs, blur, a sliver at the frame edge) stay unnamed.
"""
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn

from .dataset import DOGS, teacher_rows
from .features import bbox

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(HERE, "library", "work")
MODELS = os.path.join(HERE, "models")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop(frame, x0, y0, x1, y1, pad=0.12):
    h, w = frame.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = int(max(0, x0 - pad * bw)), int(min(w, x1 + pad * bw))
    y0, y1 = int(max(0, y0 - pad * bh)), int(min(h, y1 + pad * bh))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    c = frame[y0:y1, x0:x1]
    s = SIZE / max(c.shape[:2])
    c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((SIZE, SIZE, 3), np.uint8)
    oy, ox = (SIZE - c.shape[0]) // 2, (SIZE - c.shape[1]) // 2
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out


def to_tensor(crops):
    x = np.stack(crops)[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    x = (x - MEAN) / STD
    return torch.tensor(x.transpose(0, 3, 1, 2).copy())


def net():
    import torchvision

    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def teacher_crops():
    rows = defaultdict(list)
    for r in teacher_rows():
        if (r.get("other_dogs") or 0) == 0:
            rows[r["clip"]].append(r)
    X, y, groups = [], [], []
    for clip, rs in rows.items():
        cap = cv2.VideoCapture(os.path.join(WORK, clip + ".mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for r in sorted(rs, key=lambda r: r["t"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(r["t"] * fps)))
            ok, fr = cap.read()
            if not ok:
                continue
            h, w = fr.shape[:2]
            for di, dog in enumerate(DOGS):
                lab = r.get(dog) or {}
                b = lab.get("box")
                if lab.get("visible") and isinstance(b, list) and len(b) == 4:
                    c = crop(fr, b[1] / 1000 * w, b[0] / 1000 * h, b[3] / 1000 * w, b[2] / 1000 * h, pad=0.05)
                    if c is not None:
                        X.append(c)
                        y.append(di)
                        groups.append(clip)
        cap.release()
    return X, np.array(y), np.array(groups)


def augment(batch, rng):
    out = []
    for c in batch:
        c = c.copy()
        if rng.random() < 0.5:
            c = c[:, ::-1]
        g = rng.uniform(0.6, 1.4)
        c = np.clip(c.astype(np.float32) * g + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            k = int(rng.integers(0, 4))
            c = np.ascontiguousarray(np.rot90(c, k))
        out.append(c)
    return out


def fit(X, y, epochs=6, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = net().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    w = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2) + 1), dtype=torch.float32).to(DEV)
    lossf = nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        m.train()
        perm = rng.permutation(len(X))
        for b in range(0, len(X), 64):
            i = perm[b:b + 64]
            xb = to_tensor(augment([X[k] for k in i], rng)).to(DEV)
            loss = lossf(m(xb), torch.tensor(y[i]).to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def probs(m, crops):
    m.eval()
    out = []
    for b in range(0, len(crops), 256):
        out.append(m(to_tensor(crops[b:b + 256]).to(DEV)).softmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2))


def train():
    X, y, groups = teacher_crops()
    clips = sorted(set(groups))
    print(f"{len(X)} crops from {len(clips)} clips: lab {int((y == 0).sum())}, mal {int((y == 1).sum())}")
    rng = np.random.default_rng(0)
    fold_of = {c: i % 5 for i, c in enumerate(rng.permutation(clips))}
    pred = np.zeros(len(y), int)
    conf = np.zeros(len(y))
    for k in range(5):
        tr = np.array([fold_of[g] != k for g in groups])
        if tr.all() or not tr.any():
            continue
        m = fit([X[i] for i in np.where(tr)[0]], y[tr])
        p = probs(m, [X[i] for i in np.where(~tr)[0]])
        pred[~tr] = p.argmax(1)
        conf[~tr] = p.max(1)
        print(f"fold {k + 1}: held-out accuracy {np.mean(pred[~tr] == y[~tr]):.3f}", flush=True)
    acc = float(np.mean(pred == y))
    sure = conf >= 0.8
    print(f"held-out accuracy {acc:.3f}; when confident (p>=0.8, {sure.mean():.0%} of crops): "
          f"{np.mean(pred[sure] == y[sure]):.3f}")
    m = fit(X, y, epochs=8)
    os.makedirs(MODELS, exist_ok=True)
    torch.save(m.state_dict(), os.path.join(MODELS, "dogid.pt"))
    json.dump({"crops": len(X), "clips": len(clips), "heldout_acc": acc,
               "confident_share": float(sure.mean()), "confident_acc": float(np.mean(pred[sure] == y[sure]))},
              open(os.path.join(MODELS, "dogid.json"), "w"), indent=1)


def apply(min_p=0.8):
    import sys as _s
    _s.path.insert(0, HERE)
    from pipeline.tracks import dlc_h5_all

    m = net().to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODELS, "dogid.pt"), map_location=DEV))
    rows = defaultdict(list)
    for r in teacher_rows():
        rows[r["clip"]].append(r)
    dlc = os.path.join(WORK, "dlc")
    summary = {}
    for f in sorted(os.listdir(os.path.join(WORK, "tracks"))):
        if not f.endswith(".dogs.npz"):
            continue
        clip = f[:-9]
        h5 = [x for x in os.listdir(dlc) if x.startswith(clip + "_superanimal") and x.endswith(".h5")]
        names, xy, cf = dlc_h5_all(os.path.join(dlc, h5[0]))
        T, I, K = cf.shape
        crowded = np.zeros(T, bool)
        for r in rows.get(clip, []):
            if (r.get("other_dogs") or 0) > 0:
                fr = int(round(r["t"] * 30))
                crowded[max(0, fr - 15):fr + 16] = True
        out_xy = np.full((T, 2, K, 2), np.nan)
        out_cf = np.zeros((T, 2, K))
        cap = cv2.VideoCapture(os.path.join(WORK, clip + ".mp4"))
        t = 0
        while t < T:
            ok, fr = cap.read()
            if not ok:
                break
            if not crowded[t]:
                cand, crops = [], []
                for i in range(I):
                    bb = bbox(xy[t, i], cf[t, i])
                    if bb is None or (cf[t, i] >= 0.4).sum() < 8:
                        continue
                    c = crop(fr, *bb)
                    if c is not None:
                        cand.append(i)
                        crops.append(c)
                if crops:
                    p = probs(m, crops)
                    taken = {}
                    for i, pr in zip(cand, p):
                        d = int(pr.argmax())
                        if pr[d] >= min_p and (d not in taken or pr[d] > taken[d][0]):
                            taken[d] = (pr[d], i)
                    for d, (_, i) in taken.items():
                        out_xy[t, d], out_cf[t, d] = xy[t, i], cf[t, i]
            t += 1
        cap.release()
        old = np.load(os.path.join(WORK, "tracks", f), allow_pickle=False)
        np.savez_compressed(os.path.join(WORK, "tracks", f), names=np.array(names), xy=out_xy, conf=out_cf,
                            fps=30.0, brightness=old["brightness"], split=0.0, source=str(old["source"]),
                            identity="dogid")
        seen = (out_cf.mean(axis=2) > 0.3).mean(axis=0)
        summary[clip] = {"lab_seen": round(float(seen[0]), 3), "mal_seen": round(float(seen[1]), 3)}
        print(f"named {clip}: lab seen {seen[0]:.0%}, mal seen {seen[1]:.0%}", flush=True)
    json.dump(summary, open(os.path.join(MODELS, "dogid_apply.json"), "w"), indent=1)


if __name__ == "__main__":
    {"train": train, "apply": apply}[sys.argv[1]]()
