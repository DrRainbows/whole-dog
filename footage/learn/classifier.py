"""Regime classifier: learn the teacher's labels from keypoint movement alone.

  .venv\\Scripts\\python -m learn.classifier cv       grouped cross-validation (whole clips held out), with baselines
  .venv\\Scripts\\python -m learn.classifier final    train on everything, save models/classifier.pt

Three heads share one small temporal convolution network: support (body), action (head and
mouth), context (what the dog is doing). Folds hold out whole clips, so the score says how
well it recognises regimes in footage it has never seen.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from .dataset import ACTION, CONTEXT, SUPPORT, build
from .features import window

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")
HEADS = [("support", SUPPORT), ("action", ACTION), ("context", CONTEXT)]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, f_in, hidden=128, drop=0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(f_in, hidden, 5, padding=2), nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden * 2, len(c)) for _, c in HEADS])

    def forward(self, x):  # x (B, L, F)
        h = self.body(x.transpose(1, 2))
        z = torch.cat([h.mean(2), h.amax(2)], 1)
        return [head(z) for head in self.heads]


def tensors(samples, feats, jitter=0, rng=None):
    X, Y = [], []
    for clip, dog, f, s, a, c in samples:
        ff = feats[clip][dog][0]
        t = f + (rng.integers(-jitter, jitter + 1) if jitter else 0)
        X.append(window(ff, int(np.clip(t, 0, len(ff) - 1))))
        Y.append((s, a, c))
    return torch.tensor(np.stack(X)), torch.tensor(np.array(Y), dtype=torch.long)


def class_weights(y, n):
    cnt = np.bincount(y[y >= 0], minlength=n).astype(float)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train(samples, feats, epochs=40, seed=0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f_in = next(iter(next(iter(feats.values())).values()))[0].shape[1]
    net = Net(f_in).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    Y_all = np.array([(s, a, c) for _, _, _, s, a, c in samples])
    losses = [nn.CrossEntropyLoss(weight=class_weights(Y_all[:, i], len(c)).to(DEV), ignore_index=-1)
              for i, (_, c) in enumerate(HEADS)]
    for ep in range(epochs):
        X, Y = tensors(samples, feats, jitter=3, rng=rng)
        perm = torch.randperm(len(X))
        net.train()
        for b in range(0, len(X), 64):
            i = perm[b:b + 64]
            out = net(X[i].to(DEV))
            loss = sum(l(o, Y[i, k].to(DEV)) for k, (l, o) in enumerate(zip(losses, out)))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if verbose and ep % 10 == 0:
            print(f"epoch {ep} loss {loss.item():.3f}", flush=True)
    return net


@torch.no_grad()
def predict(net, samples, feats):
    net.eval()
    X, _ = tensors(samples, feats)
    outs = [[], [], []]
    for b in range(0, len(X), 256):
        for k, o in enumerate(net(X[b:b + 256].to(DEV))):
            outs[k].append(o.softmax(1).cpu().numpy())
    return [np.concatenate(o) for o in outs]


def macro_f1(y, p, n):
    f1s = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        if tp + fn == 0:
            continue
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def baseline_features(samples, feats):
    X = []
    for clip, dog, f, *_ in samples:
        w = window(feats[clip][dog][0], f)
        X.append(np.concatenate([w.mean(0), w.std(0)]))
    return np.stack(X)


def cv(folds=5):
    from sklearn.ensemble import HistGradientBoostingClassifier

    samples, feats, audit = build()
    print("join audit:", dict(audit))
    clips = sorted({s[0] for s in samples})
    print(f"{len(samples)} labelled samples from {len(clips)} clips")
    Y = np.array([(s, a, c) for _, _, _, s, a, c in samples])
    for i, (h, cls) in enumerate(HEADS):
        cnt = Counter(Y[:, i][Y[:, i] >= 0])
        print(f"  {h}: " + ", ".join(f"{cls[k]} {v}" for k, v in cnt.most_common()))
    rng = np.random.default_rng(0)
    order = rng.permutation(clips)
    fold_of = {c: k % folds for k, c in enumerate(order)}
    preds = {h: np.full(len(samples), -1) for h, _ in HEADS}
    base = {h: np.full(len(samples), -1) for h, _ in HEADS}
    major = {h: np.full(len(samples), -1) for h, _ in HEADS}
    for k in range(folds):
        tr = [s for s in samples if fold_of[s[0]] != k]
        te_idx = [j for j, s in enumerate(samples) if fold_of[s[0]] == k]
        te = [samples[j] for j in te_idx]
        if not te or not tr:
            continue
        net = train(tr, feats)
        P = predict(net, te, feats)
        Xtr, Xte = baseline_features(tr, feats), baseline_features(te, feats)
        Ytr = np.array([(s, a, c) for _, _, _, s, a, c in tr])
        for i, (h, cls) in enumerate(HEADS):
            preds[h][te_idx] = P[i].argmax(1)
            m = Ytr[:, i] >= 0
            if len(set(Ytr[m, i])) > 1:
                gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xtr[m], Ytr[m, i])
                base[h][te_idx] = gb.predict(Xte)
            vals = Ytr[m, i]
            major[h][te_idx] = Counter(vals).most_common(1)[0][0] if len(vals) else 0
        print(f"fold {k + 1}/{folds} done", flush=True)
    report = {}
    print("\nhead      n     majority acc  boosted-trees acc/F1   network acc/F1")
    for i, (h, cls) in enumerate(HEADS):
        y = Y[:, i]
        m = y >= 0
        r = {
            "n": int(m.sum()),
            "majority_acc": float(np.mean(major[h][m] == y[m])),
            "trees_acc": float(np.mean(base[h][m] == y[m])), "trees_f1": macro_f1(y[m], base[h][m], len(cls)),
            "net_acc": float(np.mean(preds[h][m] == y[m])), "net_f1": macro_f1(y[m], preds[h][m], len(cls)),
        }
        report[h] = r
        print(f"{h:8s} {r['n']:5d}   {r['majority_acc']:.2f}          {r['trees_acc']:.2f} / {r['trees_f1']:.2f}"
              f"            {r['net_acc']:.2f} / {r['net_f1']:.2f}")
        conf = np.zeros((len(cls), len(cls)), int)
        for a, b in zip(y[m], preds[h][m]):
            conf[a, b] += 1
        report[h]["confusion"] = conf.tolist()
        report[h]["classes"] = cls
    os.makedirs(MODELS, exist_ok=True)
    with open(os.path.join(MODELS, "classifier_cv.json"), "w") as f:
        json.dump({"audit": dict(audit), "samples": len(samples), "clips": len(clips), "report": report}, f, indent=1)
    return report


def final():
    samples, feats, audit = build()
    net = train(samples, feats, epochs=50, verbose=True)
    os.makedirs(MODELS, exist_ok=True)
    f_in = next(iter(next(iter(feats.values())).values()))[0].shape[1]
    torch.save({"state": net.state_dict(), "f_in": f_in, "heads": HEADS}, os.path.join(MODELS, "classifier.pt"))
    print(f"saved models/classifier.pt from {len(samples)} samples")


if __name__ == "__main__":
    {"cv": cv, "final": final}[sys.argv[1] if len(sys.argv) > 1 else "cv"]()
