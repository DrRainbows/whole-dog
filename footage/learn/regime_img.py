"""Regime classifier from the image of the dog (instead of from 2D keypoints).

  .venv\\Scripts\\python -m learn.regime_img cv [frames]    cross-validate: across clips and within clips
  .venv\\Scripts\\python -m learn.regime_img final [frames] train on everything -> models/regime_img.pt

frames = 1 uses the crop at the labelled moment; frames = 3 stacks crops 0.2 s before, at, and after it
(9 input channels), so the network can see movement as well as shape.

Keypoints alone failed on handheld video: seen from above, a standing dog's keypoints look like a lying
dog's. A network looking at the dog itself can learn cues that survive the camera angle.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn

from .dataset import ACTION, CONTEXT, DOGS, SUPPORT, teacher_rows
from .dogid import DEV, MEAN, STD, crop

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(HERE, "library", "work")
MODELS = os.path.join(HERE, "models")
HEADS = [("support", SUPPORT), ("action", ACTION), ("context", CONTEXT)]
OFFSETS = {1: (0.0,), 3: (-0.2, 0.0, 0.2)}


def samples(frames=1, with_dog=False):
    """Crops of each labelled dog: X images, Y head labels, G clip; with_dog adds D (index into DOGS)."""
    rows = defaultdict(list)
    for r in teacher_rows():
        if (r.get("other_dogs") or 0) == 0:
            rows[r["clip"]].append(r)
    X, Y, G, D = [], [], [], []
    for clip, rs in rows.items():
        cap = cv2.VideoCapture(os.path.join(WORK, clip + ".mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for r in sorted(rs, key=lambda r: r["t"]):
            imgs = []
            for off in OFFSETS[frames]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round((r["t"] + off) * fps))))
                ok, fr = cap.read()
                imgs.append(fr if ok else None)
            if any(i is None for i in imgs):
                continue
            h, w = imgs[0].shape[:2]
            for dog in DOGS:
                lab = r.get(dog) or {}
                b = lab.get("box")
                if not (lab.get("visible") and isinstance(b, list) and len(b) == 4):
                    continue
                box = (b[1] / 1000 * w, b[0] / 1000 * h, b[3] / 1000 * w, b[2] / 1000 * h)
                crops = [crop(im, *box, pad=0.08) for im in imgs]
                if any(c is None for c in crops):
                    continue
                y = tuple(cls.index(lab.get(k)) if lab.get(k) in cls else -1 for k, cls in HEADS)
                X.append(np.concatenate(crops, axis=2))
                Y.append(y)
                G.append(clip)
                D.append(DOGS.index(dog))
        cap.release()
    if with_dog:
        return X, np.array(Y), np.array(G), np.array(D)
    return X, np.array(Y), np.array(G)


def to_tensor(batch):
    x = np.stack(batch).astype(np.float32) / 255.0
    n = x.shape[-1] // 3
    x = x.reshape(*x.shape[:-1], n, 3)[..., ::-1]  # BGR -> RGB per frame
    x = ((x - MEAN) / STD).reshape(*x.shape[:-2], n * 3)
    return torch.tensor(x.transpose(0, 3, 1, 2).copy())


class Net(nn.Module):
    def __init__(self, frames=1):
        super().__init__()
        import torchvision

        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        if frames > 1:
            w = m.conv1.weight.data
            m.conv1 = nn.Conv2d(3 * frames, 64, 7, 2, 3, bias=False)
            m.conv1.weight.data = w.repeat(1, frames, 1, 1) / frames
        m.fc = nn.Identity()
        self.body = m
        self.heads = nn.ModuleList([nn.Linear(512, len(c)) for _, c in HEADS])

    def forward(self, x):
        z = self.body(x)
        return [h(z) for h in self.heads]


def augment(batch, rng):
    out = []
    for c in batch:
        c = c.copy()
        if rng.random() < 0.5:
            c = c[:, ::-1]
        g = rng.uniform(0.6, 1.4)
        c = np.clip(c.astype(np.float32) * g + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
        out.append(np.ascontiguousarray(c))
    return out


def fit(X, Y, frames, epochs=8, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Net(frames).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    losses = []
    for i, (_, cls) in enumerate(HEADS):
        cnt = np.bincount(Y[:, i][Y[:, i] >= 0], minlength=len(cls)).astype(float)
        w = 1.0 / np.sqrt(np.maximum(cnt, 1))
        losses.append(nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32).to(DEV), ignore_index=-1))
    for _ in range(epochs):
        m.train()
        perm = rng.permutation(len(X))
        for b in range(0, len(X), 64):
            i = perm[b:b + 64]
            out = m(to_tensor(augment([X[k] for k in i], rng)).to(DEV))
            yb = torch.tensor(Y[i]).to(DEV)
            loss = sum(l(o, yb[:, k]) for k, (l, o) in enumerate(zip(losses, out)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def predict(m, X):
    m.eval()
    outs = [[], [], []]
    for b in range(0, len(X), 128):
        for k, o in enumerate(m(to_tensor(X[b:b + 128]).to(DEV))):
            outs[k].append(o.softmax(1).cpu().numpy())
    return [np.concatenate(o) for o in outs]


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        if tp + fn:
            f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f)) if f else float("nan")


def evaluate(X, Y, G, frames, scheme):
    clips = sorted(set(G))
    if scheme == "across clips":
        rng = np.random.default_rng(0)
        fold_of = {c: i % 5 for i, c in enumerate(rng.permutation(clips))}
        fold = np.array([fold_of[g] for g in G])
    else:  # within clips: every 5th labelled second held out
        fold = np.zeros(len(G), int)
        for c in clips:
            idx = np.where(G == c)[0]
            fold[idx] = np.arange(len(idx)) % 5
    preds = [np.full(len(Y), -1) for _ in HEADS]
    for k in range(5):
        tr = fold != k
        m = fit([X[i] for i in np.where(tr)[0]], Y[tr], frames)
        P = predict(m, [X[i] for i in np.where(~tr)[0]])
        for h in range(3):
            preds[h][~tr] = P[h].argmax(1)
    res = {}
    for h, (name, cls) in enumerate(HEADS):
        y = Y[:, h]
        ok = y >= 0
        maj = Counter(y[ok]).most_common(1)[0][1] / ok.sum()
        res[name] = {"n": int(ok.sum()), "majority": round(float(maj), 3),
                     "acc": round(float(np.mean(preds[h][ok] == y[ok])), 3),
                     "macro_f1": round(macro_f1(y[ok], preds[h][ok], len(cls)), 3)}
    return res


def cv(frames=1):
    X, Y, G = samples(frames)
    print(f"{len(X)} crops ({frames} frame{'s' if frames > 1 else ''} each) from {len(set(G))} clips", flush=True)
    out = {}
    for scheme in ("within clips", "across clips"):
        out[scheme] = evaluate(X, Y, G, frames, scheme)
        print(scheme + ":", flush=True)
        for name, r in out[scheme].items():
            print(f"  {name:8s} n={r['n']:4d}  majority {r['majority']:.2f}  accuracy {r['acc']:.2f}  macro-F1 {r['macro_f1']:.2f}",
                  flush=True)
    os.makedirs(MODELS, exist_ok=True)
    json.dump(out, open(os.path.join(MODELS, f"regime_img_cv_{frames}f.json"), "w"), indent=1)


def final(frames=1):
    X, Y, G = samples(frames)
    m = fit(X, Y, frames, epochs=10)
    torch.save({"state": m.state_dict(), "frames": frames}, os.path.join(MODELS, "regime_img.pt"))
    print(f"saved models/regime_img.pt ({len(X)} crops)")


if __name__ == "__main__":
    mode = sys.argv[1]
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {"cv": cv, "final": final}[mode](frames)
