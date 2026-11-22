"""Regime classifier on DINOv2 features (a strong general-purpose image model), with a linear head.

  .venv\\Scripts\\python -m learn.regime_dino cv [small|base]

Same crops and teacher labels as regime_img; the backbone is frozen and only a logistic
regression is trained per head. Frozen general features often transfer to new footage better
than a network fine-tuned on a few dozen clips.
"""
import json
import os
import sys
from collections import Counter

os.environ.setdefault("USE_TF", "0")  # this venv stubs TensorFlow out; keep transformers from importing it
import numpy as np
import torch

from .regime_img import HEADS, MODELS, macro_f1, samples

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed(X, size="base"):
    from transformers import AutoImageProcessor, AutoModel

    name = f"facebook/dinov2-{size}"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(DEV).eval()
    out = []
    for b in range(0, len(X), 64):
        imgs = [x[..., ::-1] for x in X[b:b + 64]]  # BGR -> RGB
        inp = proc(images=imgs, return_tensors="pt").to(DEV)
        h = model(**inp).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())  # CLS + mean patch
    return np.concatenate(out)


def cv(size="base"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, Y, G, D = samples(1, with_dog=True)
    print(f"{len(X)} crops from {len(set(G))} clips; embedding with DINOv2-{size}", flush=True)
    E = embed(X, size)
    clips = sorted(set(G))
    rng = np.random.default_rng(0)
    across = {c: i % 5 for i, c in enumerate(rng.permutation(clips))}
    folds = {"across clips": np.array([across[g] for g in G])}
    within = np.zeros(len(G), int)
    for c in clips:
        idx = np.where(G == c)[0]
        within[idx] = np.arange(len(idx)) % 5
    folds["within clips"] = within
    out = {}
    for scheme in ("within clips", "across clips"):
        fold = folds[scheme]
        res = {}
        for h, (name, cls) in enumerate(HEADS):
            y = Y[:, h]
            pred = np.full(len(y), -1)
            for k in range(5):
                tr = (fold != k) & (y >= 0)
                te = (fold == k) & (y >= 0)
                if len(set(y[tr])) < 2 or not te.any():
                    continue
                sc = StandardScaler().fit(E[tr])
                clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(sc.transform(E[tr]), y[tr])
                pred[te] = clf.predict(sc.transform(E[te]))
            ok = y >= 0
            maj = Counter(y[ok]).most_common(1)[0][1] / ok.sum()
            res[name] = {"n": int(ok.sum()), "majority": round(float(maj), 3),
                         "acc": round(float(np.mean(pred[ok] == y[ok])), 3),
                         "macro_f1": round(macro_f1(y[ok], pred[ok], len(cls)), 3)}
        out[scheme] = res
        print(scheme + ":", flush=True)
        for name, r in res.items():
            print(f"  {name:8s} n={r['n']:4d}  majority {r['majority']:.2f}  accuracy {r['acc']:.2f}  macro-F1 {r['macro_f1']:.2f}",
                  flush=True)
    json.dump(out, open(os.path.join(MODELS, f"regime_dino_cv_{size}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(MODELS, f"dino_{size}_embeddings.npz"), E=E, Y=Y, G=G, D=D)


def cv_temporal(size="base"):
    """Same test, with motion: embeddings 0.2 s before, at and after each labelled moment, plus the change
    across them. Walking, trotting and running differ mostly in movement, which one frame can't show."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, Y, G = samples(3)
    print(f"{len(X)} labelled moments x 3 frames from {len(set(G))} clips; embedding with DINOv2-{size}", flush=True)
    E3 = [embed([x[..., 3 * i:3 * i + 3] for x in X], size) for i in range(3)]
    feats = {"one frame": E3[1], "three frames + change": np.concatenate([E3[0], E3[1], E3[2], E3[2] - E3[0]], 1)}
    clips = sorted(set(G))
    rng = np.random.default_rng(0)
    across = {c: i % 5 for i, c in enumerate(rng.permutation(clips))}
    fold = np.array([across[g] for g in G])
    out = {}
    for fname, E in feats.items():
        res = {}
        for h, (name, cls) in enumerate(HEADS):
            y = Y[:, h]
            pred = np.full(len(y), -1)
            for k in range(5):
                tr, te = (fold != k) & (y >= 0), (fold == k) & (y >= 0)
                if len(set(y[tr])) < 2 or not te.any():
                    continue
                sc = StandardScaler().fit(E[tr])
                clf = LogisticRegression(max_iter=3000, C=0.2, class_weight="balanced").fit(sc.transform(E[tr]), y[tr])
                pred[te] = clf.predict(sc.transform(E[te]))
            ok = y >= 0
            res[name] = {"acc": round(float(np.mean(pred[ok] == y[ok])), 3),
                         "macro_f1": round(macro_f1(y[ok], pred[ok], len(cls)), 3)}
        out[fname] = res
        print(f"clips never seen, {fname}: " + ", ".join(f"{n} {r['acc']:.2f} (F1 {r['macro_f1']:.2f})" for n, r in res.items()),
              flush=True)
    json.dump(out, open(os.path.join(MODELS, f"regime_dino_temporal_{size}.json"), "w"), indent=1)


def identity(size="base", sure=0.8):
    """Which dog is in the crop, on clips the classifier never saw (same embeddings and folds as cv). Also: accuracy
    when the classifier is sure (probability >= sure), and how many crops that covers; the worst clips."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    path = os.path.join(MODELS, f"dino_{size}_embeddings.npz")
    z = np.load(path) if os.path.exists(path) else None
    if z is None or "D" not in z.files:
        cv(size)
        z = np.load(path)
    E, G, D = z["E"], z["G"], z["D"]
    clips = sorted(set(G))
    rng = np.random.default_rng(0)
    across = {c: i % 5 for i, c in enumerate(rng.permutation(clips))}
    fold = np.array([across[g] for g in G])
    p = np.zeros(len(D))
    for k in range(5):
        tr, te = fold != k, fold == k
        sc = StandardScaler().fit(E[tr])
        clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(E[tr]), D[tr])
        p[te] = clf.predict_proba(sc.transform(E[te]))[:, 1]
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p) >= sure
    per = {c: float(np.mean(pred[G == c] == D[G == c])) for c in clips}
    worst = sorted(per.items(), key=lambda kv: kv[1])[:5]
    out = {"crops": int(len(D)), "clips": len(clips), "acc": round(float(np.mean(pred == D)), 3),
           "acc_when_sure": round(float(np.mean(pred[conf] == D[conf])), 3), "sure_share": round(float(conf.mean()), 3),
           "worst_clips": {c: round(a, 3) for c, a in worst}}
    print(f"which dog, clips never seen: {out['acc']:.1%} of {out['crops']} crops from {out['clips']} clips; "
          f"{out['acc_when_sure']:.1%} when sure ({out['sure_share']:.0%} of crops); worst: "
          + ", ".join(f"{c} {a:.0%}" for c, a in worst), flush=True)
    json.dump(out, open(os.path.join(MODELS, f"identity_dino_{size}.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "temporal":
        cv_temporal(sys.argv[2] if len(sys.argv) > 2 else "base")
    elif sys.argv[1] == "identity":
        identity(sys.argv[2] if len(sys.argv) > 2 else "base")
    else:
        cv(sys.argv[2] if len(sys.argv) > 2 else "base")
