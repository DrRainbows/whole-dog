"""Export the learned segment router as plain numbers, for the dog's brain to run without sklearn.

  .venv\\Scripts\\python -m learn.export_router     -> models/router_linear.json

Holds: the linear router's weights (next regime given the current one, how long it lasted, the activity,
the other dog, distance, panting, which dog), how many transitions each dog has in each activity (the
brain trusts the learned router in proportion), and each dog's regime durations per activity.
"""
import json
import os
from collections import Counter, defaultdict

from . import router as R
from .dataset import CONTEXT, DOGS, SUPPORT
from .generate import train

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    lin, _ = train()
    seqs = R.teacher_segments()
    n_trans = {d: dict(Counter(CONTEXT[seg["context"]] for s in seqs[d] for seg in s[:-1])) for d in DOGS}
    durs = {d: defaultdict(lambda: defaultdict(list)) for d in DOGS}
    for d in DOGS:
        for s in seqs[d]:
            for i, seg in enumerate(s):
                # first and last segments of a sequence are cut by the clip: keep only whole bouts
                if 0 < i < len(s) - 1:
                    durs[d][CONTEXT[seg["context"]]][SUPPORT[seg["regime"]]].append(seg["dur"])
    out = {"support": SUPPORT, "context": CONTEXT, "dogs": DOGS, "coef": lin.coef_.tolist(),
           "intercept": lin.intercept_.tolist(), "classes": lin.classes_.tolist(), "n_transitions": n_trans,
           "durations": {d: {c: dict(v) for c, v in durs[d].items()} for d in DOGS},
           "feature": "onehot regime(13), onehot context(10), onehot other regime(13), other seen, "
                      "log1p(duration s), pant, min(dist_px/500, 3), dog id (lab 0, mal 1)"}
    path = os.path.join(HERE, "models", "router_linear.json")
    json.dump(out, open(path, "w"), indent=0)
    print(f"wrote {path}: {len(lin.classes_)} classes, {lin.coef_.shape[1]} features; transitions {n_trans}")


if __name__ == "__main__":
    main()
