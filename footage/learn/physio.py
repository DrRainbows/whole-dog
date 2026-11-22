"""Does physiology help the router? Breath and exertion as determiners, tested on the teacher's labels.

  .venv\\Scripts\\python -m learn.physio

The router here works per second and has two parts:
  WHEN  does the current regime end this second?      (a hazard: switch or stay)
  WHAT  if it ends, which regime comes next?          (a transition)
Both are linear (logistic) models, cross-validated on clips they never saw.

Physiology enters as two leaky integrators of exertion, computed from the dog's own recent regimes:
a fast one (breath and heart, tens of seconds) and a slow one (fatigue, minutes). They see no pixels.

Tests:
  1. Grounding: does the exertion state predict panting beyond the current regime? Does it throttle
     sniffing, as the first notes say ("panting throttles sniffing")?
  2. Routing: does it improve the held-out per-second likelihood of what the dog does next?
  3. Generation: with a hazard for WHEN, does generated rest finally look like rest?
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .dataset import CONTEXT, DOGS, SUPPORT, teacher_rows

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")
NS, NC = len(SUPPORT), len(CONTEXT)

# Rough exertion per regime as a fraction of a flat-out gallop. Orders of magnitude from canine exercise
# physiology, not fitted. Lying or standing in play (wrestling) is work; lying at rest is not.
INTENSITY = {"stand": 0.08, "walk": 0.2, "trot": 0.45, "run": 1.0, "jump": 0.9, "sit": 0.04, "down": 0.02,
             "side": 0.01, "back": 0.02, "crouch": 0.15, "rear_up": 0.8, "swim": 0.85, "wade": 0.35}
IN_PLAY = {"stand": 0.3, "sit": 0.2, "down": 0.35, "side": 0.4, "back": 0.45, "crouch": 0.4}
PLAY = {CONTEXT.index("play_dog"), CONTEXT.index("play_person")}
TAU_FAST, TAU_SLOW = 30.0, 300.0
MOUTH_FREE = ("none", "look", "sniff", "pant")


def intensity(s):
    name = SUPPORT[s["reg"]]
    x = INTENSITY[name]
    if s["ctx"] in PLAY:
        x = max(x, IN_PLAY.get(name, 0.0))
    return x


# ---------- per-second records ----------
def record(clip, r, lab, oth):
    o_seen = bool(oth.get("visible"))
    dist = np.nan
    if o_seen and isinstance(lab.get("box"), list) and isinstance(oth.get("box"), list):
        a, b = np.array(lab["box"], float), np.array(oth["box"], float)
        dist = float(np.hypot((a[1] + a[3] - b[1] - b[3]) / 2, (a[0] + a[2] - b[0] - b[2]) / 2)) / 1000.0
    return {"clip": clip, "t": r["t"], "reg": SUPPORT.index(lab["support"]), "act": lab.get("action"),
            "ctx": CONTEXT.index(lab["context"]) if lab.get("context") in CONTEXT else NC - 1,
            "o_seen": o_seen, "o_reg": SUPPORT.index(oth["support"]) if o_seen and oth.get("support") in SUPPORT else -1,
            "dist": dist}


def annotate(run, tau_fast=TAU_FAST, tau_slow=TAU_SLOW):
    """Adds elapsed time in the current regime and the two exertion states, computed causally."""
    kf, ks = 1 - np.exp(-1.0 / tau_fast), 1 - np.exp(-1.0 / tau_slow)
    ef = es = intensity(run[0])  # assume the dog had been doing its first regime for a while
    start, known = 0, False
    for i, s in enumerate(run):
        if i:
            x = intensity(s)
            ef += kf * (x - ef)
            es += ks * (x - es)
            if s["reg"] != run[i - 1]["reg"]:
                start, known = i, True
        s["ef"], s["es"] = ef, es
        s["elapsed"], s["known"] = i - start + 1, known
    return run


def stretches(tau_fast=TAU_FAST, tau_slow=TAU_SLOW):
    """Per dog: continuous runs of once-a-second labels (no gap over 1.5 s, no stranger dogs)."""
    by_clip = defaultdict(list)
    for r in teacher_rows():
        by_clip[r["clip"]].append(r)
    out = {d: [] for d in DOGS}
    for clip, rs in by_clip.items():
        rs = sorted(rs, key=lambda r: r["t"])
        for di, dog in enumerate(DOGS):
            other = DOGS[1 - di]
            run, prev = [], None
            for r in rs:
                lab = r.get(dog) or {}
                ok = bool(lab.get("visible")) and lab.get("support") in SUPPORT and (r.get("other_dogs") or 0) == 0
                if run and (not ok or r["t"] - prev > 1.5):
                    if len(run) >= 3:
                        out[dog].append(annotate(run, tau_fast, tau_slow))
                    run = []
                if ok:
                    run.append(record(clip, r, lab, r.get(other) or {}))
                    prev = r["t"]
            if len(run) >= 3:
                out[dog].append(annotate(run, tau_fast, tau_slow))
    return out


def steps(runs):
    return [(di, a, b) for di, d in enumerate(DOGS) for run in runs[d] for a, b in zip(run[:-1], run[1:])]


# ---------- features and models ----------
def onehot(i, n):
    v = np.zeros(n, np.float32)
    if i >= 0:
        v[i] = 1
    return v


def fvec(s, dog_id, use):
    parts = [onehot(s["reg"], NS), onehot(s["ctx"], NC), [float(dog_id)]]
    if "elapsed" in use:
        parts.append([np.log1p(s["elapsed"]), float(not s["known"])])
    if "other" in use:
        parts += [onehot(s["o_reg"], NS), [float(s["o_seen"]), 0.0 if not np.isfinite(s["dist"]) else min(s["dist"], 1.5)]]
    if "pant" in use:
        parts.append([float(s["act"] == "pant")])
    if "physio" in use:
        parts.append([s["ef"], s["es"], s["ef"] * intensity(s)])
    return np.concatenate([np.asarray(p, np.float32) for p in parts])


class Markov:
    """Per dog, per regime: a constant chance of switching each second, and a table of what comes next."""

    def fit(self, st):
        self.sw = np.full((2, NS, 2), 0.5)
        self.nx = np.full((2, NS, NS), 0.5)
        for di, a, b in st:
            s = b["reg"] != a["reg"]
            self.sw[di, a["reg"], int(s)] += 1
            if s:
                self.nx[di, a["reg"], b["reg"]] += 1
        for di in range(2):
            np.fill_diagonal(self.nx[di], 0)
        return self

    def when(self, st):
        return np.array([self.sw[di, a["reg"], 1] / self.sw[di, a["reg"]].sum() for di, a, _ in st])

    def what(self, st):
        return np.stack([self.nx[di, a["reg"]] / self.nx[di, a["reg"]].sum() for di, a, _ in st])


class Durations(Markov):
    """WHEN from each dog's own regime durations (what generate.py does now): a smoothed discrete hazard."""
    EMAX = 30

    def fit(self, st):
        super().fit(st)
        self.ev = np.zeros((2, NS, self.EMAX + 1))
        self.risk = np.zeros((2, NS, self.EMAX + 1))
        for di, a, b in st:
            e = min(a["elapsed"], self.EMAX)
            self.risk[di, a["reg"], e] += 1
            self.ev[di, a["reg"], e] += b["reg"] != a["reg"]
        return self

    def when(self, st):
        h0 = super().when(st)
        out = []
        for (di, a, _), h in zip(st, h0):
            e = min(a["elapsed"], self.EMAX)
            out.append((self.ev[di, a["reg"], e] + 2 * h) / (self.risk[di, a["reg"], e] + 2))
        return np.array(out)


class Linear:
    def __init__(self, use_when, use_what, C=0.3):
        self.uw, self.ut, self.C = use_when, use_what, C

    def fit(self, st):
        sw = np.array([b["reg"] != a["reg"] for _, a, b in st])
        Xw = np.stack([fvec(a, di, self.uw) for di, a, _ in st])
        self.h = LogisticRegression(max_iter=4000, C=self.C).fit(Xw, sw)
        Xt = np.stack([fvec(a, di, self.ut) for (di, a, _), s in zip(st, sw) if s])
        yt = np.array([b["reg"] for (_, _, b), s in zip(st, sw) if s])
        self.t = LogisticRegression(max_iter=4000, C=self.C).fit(Xt, yt)
        return self

    def when(self, st):
        return self.h.predict_proba(np.stack([fvec(a, di, self.uw) for di, a, _ in st]))[:, 1]

    def what(self, st):
        X = np.stack([fvec(a, di, self.ut) for di, a, _ in st])
        P = np.full((len(st), NS), 1e-4)
        P[:, self.t.classes_] += self.t.predict_proba(X)
        P[np.arange(len(st)), [a["reg"] for _, a, _ in st]] = 0
        return P / P.sum(1, keepdims=True)


def folds_by_clip(runs, k=5):
    clips = sorted({run[0]["clip"] for d in DOGS for run in runs[d]})
    rng = np.random.default_rng(0)
    return {c: i % k for i, c in enumerate(rng.permutation(clips))}, k


def score(make, runs):
    """Held-out per-second log-likelihood: WHEN over all seconds, WHAT over switches only."""
    fold_of, k = folds_by_clip(runs)
    lw, lt = [], []
    for f in range(k):
        tr = steps({d: [r for r in runs[d] if fold_of[r[0]["clip"]] != f] for d in DOGS})
        te = steps({d: [r for r in runs[d] if fold_of[r[0]["clip"]] == f] for d in DOGS})
        if not te:
            continue
        m = make().fit(tr)
        h = np.clip(m.when(te), 1e-4, 1 - 1e-4)
        sw = np.array([b["reg"] != a["reg"] for _, a, b in te])
        lw += list(np.where(sw, np.log(h), np.log(1 - h)))
        if sw.any():
            sub = [x for x, s in zip(te, sw) if s]
            P = m.what(sub)
            lt += list(np.log(np.clip(P[np.arange(len(sub)), [b["reg"] for _, _, b in sub]], 1e-6, 1)))
    return float(np.mean(lw)), float(np.mean(lt)), len(lw), len(lt)


# ---------- 1. grounding ----------
def grounding(runs):
    fold_of, k = folds_by_clip(runs)
    rows = [(di, s) for di, d in enumerate(DOGS) for run in runs[d] for s in run if s["act"] in MOUTH_FREE]
    fold = np.array([fold_of[s["clip"]] for _, s in rows])
    out = {}
    for target in ("pant", "sniff"):
        y = np.array([s["act"] == target for _, s in rows])
        res = {}
        for name, use in (("regime + activity + dog", set()), ("+ exertion state", {"physio"})):
            X = np.stack([fvec(s, di, use) for di, s in rows])
            p = np.full(len(y), np.nan)
            for f in range(k):
                tr, te = fold != f, fold == f
                if te.any() and len(set(y[tr])) == 2:
                    p[te] = LogisticRegression(max_iter=4000, C=0.3).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            ok = np.isfinite(p)
            pp = np.clip(p[ok], 1e-4, 1 - 1e-4)
            res[name] = {"logloss": float(-np.mean(np.where(y[ok], np.log(pp), np.log(1 - pp)))),
                         "auc": float(roc_auc_score(y[ok], p[ok])) if len(set(y[ok])) == 2 else None}
        Xf = np.stack([fvec(s, di, {"physio"}) for di, s in rows])
        full = LogisticRegression(max_iter=4000, C=0.3).fit(Xf, y)
        res["coef fast exertion"] = float(full.coef_[0][-3])
        res["coef slow exertion"] = float(full.coef_[0][-2])
        # raw rates by fast-exertion tercile, for a plain-language read
        ef = np.array([s["ef"] for _, s in rows])
        q = np.quantile(ef, [1 / 3, 2 / 3])
        res["rate by exertion (low, mid, high)"] = [float(y[ef <= q[0]].mean()), float(y[(ef > q[0]) & (ef <= q[1])].mean()),
                                                    float(y[ef > q[1]].mean())]
        res["n"], res["positives"] = int(len(y)), int(y.sum())
        out[target] = res
    return out


# ---------- 3. generation ----------
def generate(m, dog_id, ctx, o_reg, start, seconds, rng, tau_fast=TAU_FAST, tau_slow=TAU_SLOW):
    kf, ks = 1 - np.exp(-1.0 / tau_fast), 1 - np.exp(-1.0 / tau_slow)
    s = {"reg": start, "ctx": ctx, "o_reg": o_reg, "o_seen": o_reg >= 0, "dist": 0.3, "act": "none",
         "elapsed": 1, "known": True}
    s["ef"] = s["es"] = intensity(s)
    regs = []
    for _ in range(seconds):
        regs.append(s["reg"])
        st = [(dog_id, s, None)]
        if rng.random() < m.when(st)[0]:
            p = m.what(st)[0]
            s = dict(s, reg=int(rng.choice(NS, p=p / p.sum())), elapsed=1)
        else:
            s = dict(s, elapsed=s["elapsed"] + 1)
        x = intensity(s)
        s["ef"] += kf * (x - s["ef"])
        s["es"] += ks * (x - s["es"])
    return regs


def occupancy(regs):
    c = np.bincount(regs, minlength=NS).astype(float)
    return c / c.sum()


def generation(runs, models, contexts=("rest", "play_dog", "explore", "retrieve"), seconds=300, reps=20):
    from . import generate as G

    old_lin, old_durs = G.train()
    out = {}
    for ci in [CONTEXT.index(c) for c in contexts]:
        for di, dog in enumerate(DOGS):
            secs = [s for run in runs[dog] for s in run if s["ctx"] == ci]
            if len(secs) < 60:
                continue
            real = occupancy([s["reg"] for s in secs])
            start = int(np.argmax(real))
            o = Counter(s["o_reg"] for s in secs).most_common(1)[0][0]
            row = {"real seconds": len(secs), "real": {SUPPORT[i]: round(float(v), 2) for i, v in enumerate(real) if v >= 0.03}}
            rng = np.random.default_rng(1)
            # the current generator: linear router over segments + each dog's empirical durations
            occ = np.zeros(NS)
            for _ in range(reps):
                tl = G.run(old_lin, old_durs, dog, CONTEXT[ci], SUPPORT[o] if o >= 0 else None, seconds, rng)
                for _, r, d in tl:
                    occ[SUPPORT.index(r)] += d
            occ /= occ.sum()
            row["segment router + durations (current)"] = round(float(0.5 * np.abs(occ - real).sum()), 3)
            for name, m in models.items():
                occ = np.mean([occupancy(generate(m, di, ci, o, start, seconds, rng)) for _ in range(reps)], 0)
                row[name] = round(float(0.5 * np.abs(occ - real).sum()), 3)
                row[name + " occupancy"] = {SUPPORT[i]: round(float(v), 2) for i, v in enumerate(occ) if v >= 0.03}
            out[f"{CONTEXT[ci]} / {dog}"] = row
    return out


def main():
    runs = stretches()
    n_sec = sum(len(r) for d in DOGS for r in runs[d])
    n_sw = sum(b["reg"] != a["reg"] for _, a, b in steps(runs))
    print(f"{n_sec} labelled seconds in {sum(len(runs[d]) for d in DOGS)} runs; {n_sw} regime switches", flush=True)
    report = {"seconds": n_sec, "switches": int(n_sw)}

    print("\n1. grounding: does the exertion state explain breath (panting) and sniffing?")
    g = grounding(runs)
    for target, res in g.items():
        print(f"  {target}: n={res['n']} ({res['positives']} positive)")
        for name in ("regime + activity + dog", "+ exertion state"):
            print(f"    {name:26s} held-out log-loss {res[name]['logloss']:.3f}  AUC {res[name]['auc']:.3f}")
        print(f"    coefficient: fast exertion {res['coef fast exertion']:+.2f}, slow {res['coef slow exertion']:+.2f}; "
              f"rate low/mid/high exertion {[round(v, 3) for v in res['rate by exertion (low, mid, high)']]}")
    report["grounding"] = g

    print("\n2. routing: held-out log-likelihood (higher is better)")
    base = {"elapsed"}
    variants = {
        "Markov per dog (constant hazard)": lambda: Markov(),
        "durations per dog (current WHEN)": lambda: Durations(),
        "linear: regime, activity, dog": lambda: Linear(set(), set()),
        "  + time in regime": lambda: Linear(base, base),
        "  + other dog": lambda: Linear(base | {"other"}, base | {"other"}),
        "  + panting seen": lambda: Linear(base | {"other", "pant"}, base | {"other", "pant"}),
        "  + exertion state (no panting)": lambda: Linear(base | {"other", "physio"}, base | {"other", "physio"}),
        "  + both": lambda: Linear(base | {"other", "pant", "physio"}, base | {"other", "pant", "physio"}),
    }
    rows = {}
    print(f"  {'model':36s} {'WHEN /s':>8s} {'WHAT /switch':>13s} {'total /s':>9s}")
    for name, make in variants.items():
        lw, lt, nw, nt = score(make, runs)
        tot = lw + lt * nt / nw
        rows[name] = {"when": lw, "what": lt, "total_per_second": tot}
        print(f"  {name:36s} {lw:8.3f} {lt:13.3f} {tot:9.3f}", flush=True)
    report["routing"] = rows
    print("  time constants for the exertion state (fast / slow, seconds):")
    taus = {}
    for tf, tsl in ((10, 300), (30, 300), (90, 300), (30, 60), (30, 1200)):
        r2 = stretches(tf, tsl)
        lw, lt, nw, nt = score(variants["  + exertion state (no panting)"], r2)
        taus[f"{tf}/{tsl}"] = lw + lt * nt / nw
        print(f"    {tf:4d} / {tsl:4d}: total {taus[f'{tf}/{tsl}']:.3f}", flush=True)
    report["taus"] = taus

    print("\n3. generation: occupancy vs the real dogs (total variation distance, lower is better)")
    st = steps(runs)
    models = {"per-second hazard router": Linear(base | {"other"}, base | {"other"}).fit(st),
              "per-second hazard router + exertion": Linear(base | {"other", "physio"}, base | {"other", "physio"}).fit(st)}
    gen = generation(runs, models)
    for key, row in gen.items():
        print(f"  {key} ({row['real seconds']} s real): real {row['real']}")
        for name in ["segment router + durations (current)"] + list(models):
            occ = row.get(name + " occupancy", "")
            print(f"    {name:40s} TVD {row[name]:.3f}  {occ}")
    report["generation"] = gen
    json.dump(report, open(os.path.join(MODELS, "physio_router.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
