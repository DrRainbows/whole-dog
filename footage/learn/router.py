"""Routers: learn from the dogs' own footage which regime comes next, and when.

  .venv\\Scripts\\python -m learn.router infer     run the trained classifier over every clip (10 per second)
  .venv\\Scripts\\python -m learn.router fit       smooth into segments, fit and compare routers on held-out clips

A router here is the part of the dog that decides what the body does next. Candidates, from
simplest to richest:
  R0 base rates           next regime by overall frequency
  R1 Markov               next regime given the current one (one table per dog)
  R2 Markov + context     ... also given the activity mode and what the other dog is doing
  R3 neural router        small network over the current regime, its duration, context, the other dog,
                          panting, and a learned vector per dog (the dog's "individuality")
Scored on clips the router never saw: average log-likelihood of the true next regime (higher is
better) and top-1 accuracy. Also: can the routers alone tell the Lab from the Malinois?
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

from .classifier import HEADS, Net
from .dataset import CONTEXT, DOGS, SUPPORT, load_tracks, teacher_rows, up_track
from .features import per_frame, window

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")
SEQ = os.path.join(MODELS, "sequences")
STRIDE = 3  # frames: 10 predictions per second at 30 fps
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NS, NC = len(SUPPORT), len(CONTEXT)
PANT = HEADS[1][1].index("pant")


# ---------- 1. run the classifier everywhere ----------
@torch.no_grad()
def infer():
    ck = torch.load(os.path.join(MODELS, "classifier.pt"), map_location=DEV)
    net = Net(ck["f_in"]).to(DEV)
    net.load_state_dict(ck["state"])
    net.eval()
    rows = defaultdict(list)
    for r in teacher_rows():
        rows[r["clip"]].append(r)
    os.makedirs(SEQ, exist_ok=True)
    clips = sorted(f[:-9] for f in os.listdir(os.path.join(HERE, "library", "work", "tracks")) if f.endswith(".dogs.npz"))
    for clip in clips:
        names, xy, conf, fps = load_tracks(clip)
        up = up_track(rows.get(clip, []), xy.shape[0], fps)
        out = {}
        for di, dog in enumerate(DOGS):
            feats, valid = per_frame(names, xy[:, di], conf[:, di], up)
            idx = np.arange(0, len(feats), STRIDE)
            X = torch.tensor(np.stack([window(feats, t) for t in idx]))
            probs = [[], [], []]
            for b in range(0, len(X), 512):
                for k, o in enumerate(net(X[b:b + 512].to(DEV))):
                    probs[k].append(o.softmax(1).cpu().numpy())
            out[dog] = {"t": idx / fps, "valid": valid[idx], "support": np.concatenate(probs[0]),
                        "action": np.concatenate(probs[1]), "context": np.concatenate(probs[2]),
                        "center": np.nanmean(np.where(conf[idx, di, :, None] >= 0.4, xy[idx, di], np.nan), axis=1)}
        np.save(os.path.join(SEQ, clip + ".npy"), out, allow_pickle=True)
        print(f"inferred {clip}", flush=True)


# ---------- 2. smooth probabilities into segments ----------
def viterbi(probs, stay=0.95):
    """Sticky HMM decoding of per-step class probabilities."""
    T, K = probs.shape
    logp = np.log(np.clip(probs, 1e-6, 1))
    trans = np.full((K, K), np.log((1 - stay) / (K - 1)))
    np.fill_diagonal(trans, np.log(stay))
    score = logp[0].copy()
    back = np.zeros((T, K), int)
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.zeros(T, int)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs(path):
    out, s = [], 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[s]:
            out.append((int(path[s]), s, t))
            s = t
    return out


def segments():
    """Per dog: list of sequences; each sequence is a list of segment dicts."""
    seqs = {d: [] for d in DOGS}
    for f in sorted(os.listdir(SEQ)):
        clip = f[:-4]
        out = np.load(os.path.join(SEQ, f), allow_pickle=True).item()
        for di, dog in enumerate(DOGS):
            o = out[dog]
            other = out[DOGS[1 - di]]
            valid = o["valid"]
            # split into stretches where the dog is tracked, bridging gaps under one second
            stretches, s, gap = [], None, 0
            for t, v in enumerate(valid):
                if v:
                    if s is None:
                        s = t
                    gap = 0
                elif s is not None:
                    gap += 1
                    if gap > 10:
                        stretches.append((s, t - gap + 1))
                        s, gap = None, 0
            if s is not None:
                stretches.append((s, len(valid)))
            for a, b in stretches:
                if b - a < 30:
                    continue
                sup = viterbi(o["support"][a:b])
                ctx = viterbi(o["context"][a:b], stay=0.98)
                seq = []
                for k, s0, s1 in runs(sup):
                    seg = slice(a + s0, a + s1)
                    ov = other["valid"][seg]
                    dist = np.nan
                    if ov.any():
                        d = np.linalg.norm(o["center"][seg] - other["center"][seg], axis=1)
                        dist = float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan
                    seq.append({
                        "clip": clip, "regime": k, "dur": (s1 - s0) * STRIDE / 30.0,
                        "context": int(Counter(ctx[s0:s1]).most_common(1)[0][0]),
                        "other_seen": bool(ov.mean() > 0.5),
                        "other_regime": int(Counter(other["support"][seg][ov].argmax(1)).most_common(1)[0][0]) if ov.any() else -1,
                        "dist_px": dist,
                        "pant": float(o["action"][seg, PANT].mean()),
                    })
                if len(seq) >= 2:
                    seqs[dog].append(seq)
    return seqs


def teacher_segments():
    """Per dog: sequences of segments built directly from the teacher's once-a-second labels."""
    by_clip = defaultdict(list)
    for r in teacher_rows():
        by_clip[r["clip"]].append(r)
    seqs = {d: [] for d in DOGS}
    for clip, rs in by_clip.items():
        rs = sorted(rs, key=lambda r: r["t"])
        for di, dog in enumerate(DOGS):
            other = DOGS[1 - di]
            run, prev_t = [], None
            stretches = []
            for r in rs:
                lab = r.get(dog) or {}
                ok = lab.get("visible") and lab.get("support") in SUPPORT and (r.get("other_dogs") or 0) == 0
                if not ok or (prev_t is not None and r["t"] - prev_t > 1.5):
                    if len(run) >= 2:
                        stretches.append(run)
                    run = []
                if ok:
                    run.append(r)
                    prev_t = r["t"]
            if len(run) >= 2:
                stretches.append(run)
            for st in stretches:
                seq = []
                for r in st:
                    lab, oth = r[dog], (r.get(other) or {})
                    reg = SUPPORT.index(lab["support"])
                    ctx = CONTEXT.index(lab["context"]) if lab.get("context") in CONTEXT else len(CONTEXT) - 1
                    o_seen = bool(oth.get("visible"))
                    o_reg = SUPPORT.index(oth["support"]) if o_seen and oth.get("support") in SUPPORT else -1
                    dist = np.nan
                    if o_seen and isinstance(lab.get("box"), list) and isinstance(oth.get("box"), list):
                        a, b = np.array(lab["box"], float), np.array(oth["box"], float)
                        dist = float(np.hypot((a[1] + a[3] - b[1] - b[3]) / 2, (a[0] + a[2] - b[0] - b[2]) / 2))
                    pant = 1.0 if lab.get("action") == "pant" else 0.0
                    if seq and seq[-1]["regime"] == reg:
                        seq[-1]["dur"] += 1.0
                        continue
                    seq.append({"clip": clip, "regime": reg, "dur": 1.0, "context": ctx, "other_seen": o_seen,
                                "other_regime": o_reg, "dist_px": dist, "pant": pant})
                if len(seq) >= 2:
                    seqs[dog].append(seq)
    return seqs


# ---------- 3. routers ----------
def markov_table(seqs, n=NS, alpha=0.5, key=None):
    counts = defaultdict(lambda: np.full(n, alpha))
    for seq in seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[key(a) if key else a["regime"]][b["regime"]] += 1
    return {k: v / v.sum() for k, v in counts.items()}


def ll_markov(table, seqs, key=None, fallback=None):
    lls = []
    for seq in seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            k = key(a) if key else a["regime"]
            p = table.get(k, fallback)
            lls.append(np.log(p[b["regime"]]) if p is not None else np.log(1.0 / NS))
    return lls


def feat(seg, dog_id):
    x = np.zeros(NS + NC + NS + 1 + 4, np.float32)
    x[seg["regime"]] = 1
    x[NS + seg["context"]] = 1
    if seg["other_regime"] >= 0:
        x[NS + NC + seg["other_regime"]] = 1
    x[NS + NC + NS] = float(seg["other_seen"])
    x[-4] = np.log1p(seg["dur"])
    x[-3] = seg["pant"]
    x[-2] = 0.0 if not np.isfinite(seg["dist_px"]) else min(seg["dist_px"] / 500.0, 3.0)
    x[-1] = dog_id
    return x


class Router(nn.Module):
    def __init__(self, f_in, n_dogs=2, emb=4, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(n_dogs, emb)
        self.net = nn.Sequential(nn.Linear(f_in - 1 + emb, hidden), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, NS))

    def forward(self, x):
        dog = x[:, -1].long()
        return self.net(torch.cat([x[:, :-1], self.emb(dog)], 1))


def pairs(seqs_by_dog):
    X, y = [], []
    for di, dog in enumerate(DOGS):
        for seq in seqs_by_dog[dog]:
            for a, b in zip(seq[:-1], seq[1:]):
                X.append(feat(a, di))
                y.append(b["regime"])
    return np.array(X), np.array(y)


def train_router(seqs_by_dog, epochs=200, seed=0):
    torch.manual_seed(seed)
    X, y = pairs(seqs_by_dog)
    net = Router(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    Xt, yt = torch.tensor(X).to(DEV), torch.tensor(y).to(DEV)
    for _ in range(epochs):
        net.train()
        loss = nn.functional.cross_entropy(net(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def fit(folds=5, source="classifier"):
    seqs = teacher_segments() if source == "teacher" else segments()
    for dog in DOGS:
        n_seg = sum(len(s) for s in seqs[dog])
        occ = Counter(seg["regime"] for s in seqs[dog] for seg in s)
        dur = defaultdict(list)
        for s in seqs[dog]:
            for seg in s:
                dur[seg["regime"]].append(seg["dur"])
        print(f"{dog}: {len(seqs[dog])} sequences, {n_seg} segments")
        print("   " + ", ".join(f"{SUPPORT[k]} {v} (median {np.median(dur[k]):.1f}s)" for k, v in occ.most_common()))
    clips = sorted({seg["clip"] for d in DOGS for s in seqs[d] for seg in s})
    rng = np.random.default_rng(0)
    fold_of = {c: i % folds for i, c in enumerate(rng.permutation(clips))}
    res = defaultdict(list)
    ident = []
    for k in range(folds):
        tr = {d: [s for s in seqs[d] if fold_of[s[0]["clip"]] != k] for d in DOGS}
        te = {d: [s for s in seqs[d] if fold_of[s[0]["clip"]] == k] for d in DOGS}
        if not any(te.values()):
            continue
        base = {}
        for d in DOGS:
            c = Counter(seg["regime"] for s in tr[d] for seg in s[1:])
            p = np.array([c.get(i, 0) + 0.5 for i in range(NS)])
            base[d] = p / p.sum()
        r1 = {d: markov_table(tr[d]) for d in DOGS}
        ctxkey = lambda a: (a["regime"], a["context"], a["other_regime"])  # noqa: E731
        r2 = {d: markov_table(tr[d], key=ctxkey) for d in DOGS}
        net = train_router(tr)
        for di, d in enumerate(DOGS):
            for seq in te[d]:
                for a, b in zip(seq[:-1], seq[1:]):
                    res["R0 base rates"].append(np.log(base[d][b["regime"]]))
            res["R1 Markov"] += ll_markov(r1[d], te[d], fallback=base[d])
            res["R2 Markov + context"] += ll_markov(r2[d], te[d], key=ctxkey, fallback=None) if False else [
                np.log((r2[d].get(ctxkey(a)) if ctxkey(a) in r2[d] else r1[d].get(a["regime"], base[d]))[b["regime"]])
                for seq in te[d] for a, b in zip(seq[:-1], seq[1:])]
            X, y = pairs({dd: (te[d] if dd == d else []) for dd in DOGS})
            if len(X):
                with torch.no_grad():
                    lp = torch.log_softmax(net(torch.tensor(X).to(DEV)), 1).cpu().numpy()
                res["R3 neural router"] += list(lp[np.arange(len(y)), y])
                res["_acc_R3"] += list(lp.argmax(1) == y)
                res["_acc_R1"] += [r1[d].get(a["regime"], base[d]).argmax() == b["regime"]
                                   for seq in te[d] for a, b in zip(seq[:-1], seq[1:])]
            # identity: does the Lab's router or the Malinois's router explain this sequence better?
            for seq in te[d]:
                if len(seq) < 4:
                    continue
                l_lab = sum(ll_markov(r1["lab"], [seq], fallback=base["lab"]))
                l_mal = sum(ll_markov(r1["mal"], [seq], fallback=base["mal"]))
                ident.append(("lab" if l_lab > l_mal else "mal") == d)
    print("\nrouter                  next-regime log-likelihood per transition (higher is better)")
    summary = {}
    for name in ["R0 base rates", "R1 Markov", "R2 Markov + context", "R3 neural router"]:
        if res[name]:
            summary[name] = float(np.mean(res[name]))
            print(f"  {name:22s} {np.mean(res[name]):7.3f}  (n={len(res[name])})")
    if res["_acc_R1"]:
        print(f"  top-1 accuracy: Markov {np.mean(res['_acc_R1']):.2f}, neural {np.mean(res['_acc_R3']):.2f}")
    if ident:
        print(f"\nidentity from the router alone: {np.mean(ident):.0%} of held-out sequences assigned to the right dog "
              f"(n={len(ident)}, chance 50%)")
    with open(os.path.join(MODELS, f"router_cv_{source}.json"), "w") as f:
        json.dump({"loglik": summary, "acc_markov": float(np.mean(res["_acc_R1"])) if res["_acc_R1"] else None,
                   "acc_neural": float(np.mean(res["_acc_R3"])) if res["_acc_R3"] else None,
                   "identity_acc": float(np.mean(ident)) if ident else None, "identity_n": len(ident)}, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "infer":
        infer()
    else:
        fit(source=sys.argv[2] if len(sys.argv) > 2 else "classifier")
