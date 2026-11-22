"""Let the learned router run: generate a behaviour timeline for each dog, procedurally.

  .venv\\Scripts\\python -m learn.generate [seconds]

The linear router (trained on the teacher's per-second sequences of both dogs) picks the next
body regime from the current one, how long it has lasted, the activity, what the other dog is
doing, and which dog this is. How long each regime lasts is drawn from that dog's own
durations in the footage. No footage is replayed: every timeline is new.
"""
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import router as R
from .dataset import CONTEXT, DOGS, SUPPORT


def train():
    seqs = R.teacher_segments()
    X, y = R.pairs(seqs)
    lin = LogisticRegression(max_iter=3000, C=0.3).fit(X, y)
    durs = {d: defaultdict(list) for d in DOGS}
    for d in DOGS:
        for s in seqs[d]:
            for seg in s:
                durs[d][seg["regime"]].append(seg["dur"])
    return lin, durs


def run(lin, durs, dog, context, other_regime, seconds, rng):
    di = DOGS.index(dog)
    reg = SUPPORT.index("stand")
    t, out = 0.0, []
    while t < seconds:
        pool = durs[dog].get(reg) or [1.0]
        dur = float(rng.choice(pool))
        out.append((t, SUPPORT[reg], dur))
        t += dur
        seg = {"regime": reg, "dur": dur, "context": CONTEXT.index(context), "other_seen": other_regime is not None,
               "other_regime": SUPPORT.index(other_regime) if other_regime else -1, "dist_px": 200.0, "pant": 0.0}
        p = np.zeros(len(SUPPORT))
        p[lin.classes_] = lin.predict_proba(R.feat(seg, di)[None])[0]
        p[reg] = 0.0  # a segment ends when the regime changes
        reg = int(rng.choice(len(p), p=p / p.sum()))
    return out


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 90
    lin, durs = train()
    rng = np.random.default_rng(7)
    for context, other in (("play_dog", "stand"), ("rest", "down")):
        print(f"\n=== context: {context}, other dog: {other} ===")
        for dog in DOGS:
            tl = run(lin, durs, dog, context, other, seconds, rng)
            occ = defaultdict(float)
            for _, r, d in tl:
                occ[r] += d
            line = " -> ".join(f"{r}({d:.0f}s)" for _, r, d in tl[:18])
            print(f"{dog}: {line}{' ...' if len(tl) > 18 else ''}")
            tot = sum(occ.values())
            print("     time share: " + ", ".join(f"{r} {100 * v / tot:.0f}%" for r, v in sorted(occ.items(), key=lambda kv: -kv[1])[:6]))


if __name__ == "__main__":
    main()
