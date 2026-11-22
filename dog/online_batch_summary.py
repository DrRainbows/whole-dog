"""Summarise a batch of online sessions (body/wsl_online_batch.sh): how often each dog really falls, doing what,
how fast it gets back up, and how often it only lags.

  body\.venv\Scripts\python dog\online_batch_summary.py RUN_DIR TAG
  body\.venv\Scripts\python dog\online_batch_summary.py RUN_A TAG_A RUN_B TAG_B   (fewer falls in A than in B?)
"""
import collections
import glob
import json
import os
import sys

import numpy as np


def main():
    run, tag = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(run, f"online_batch{tag}_*.json")))
    by_dog = collections.defaultdict(list)
    for f in files:
        d = json.load(open(f))
        by_dog[d["brain"]].append(d)
    out = {}
    for dog, ds in sorted(by_dog.items()):
        secs = sum(d["seconds"] for d in ds)
        falls = [f for d in ds for f in d["fall_log"] if f.get("what") == "fall"]
        during = collections.Counter(f["during"] for f in falls)
        rec = [s["until"] - s["fell"] for d in ds for s in d.get("scrambles", []) if s.get("to") == "stand"]
        lags = collections.Counter(lg["during"] for d in ds for lg in d.get("lags", []))
        out[dog] = {"sessions": len(ds), "minutes": round(secs / 60, 1),
                    "falls per minute": round(len(falls) / (secs / 60), 2),
                    "falls, doing": dict(during.most_common()),
                    "back on its feet": f"{sum(d['rises_completed'] for d in ds)} of {len(falls)}",
                    "seconds to its feet (median, max)": [round(float(np.median(rec)), 2), round(float(max(rec)), 2)] if rec else None,
                    "lags per minute": round(sum(lags.values()) / (secs / 60), 2), "lags, doing": dict(lags.most_common()),
                    "ms per 20 ms step (median)": round(float(np.median([d["ms_per_step"] for d in ds])), 1),
                    "walked m per minute": round(sum(d["walked_m"] for d in ds) / (secs / 60), 1)}
    print(json.dumps(out, indent=1))


def compare(run_a, tag_a, run_b, tag_b):
    """Fewer real falls in batch A than in batch B, over equal minutes? One-sided conditional binomial test:
    given n = a + b falls, the chance of a or fewer in A if both had the same rate (sessions must match)."""
    from scipy.stats import binomtest
    counts, minutes = [], []
    for run, tag in ((run_a, tag_a), (run_b, tag_b)):
        fs = sorted(glob.glob(os.path.join(run, f"online_batch{tag}_*.json")))
        ds = [json.load(open(f)) for f in fs]
        counts.append(sum(1 for d in ds for f in d["fall_log"] if f.get("what") == "fall"))
        minutes.append(sum(d["seconds"] for d in ds) / 60)
    a, b = counts
    share = minutes[0] / sum(minutes)
    p = binomtest(a, a + b, share, alternative="less").pvalue if a + b else float("nan")
    return {"A": f"{os.path.basename(run_a)}{tag_a}: {a} falls in {minutes[0]:.0f} min",
            "B": f"{os.path.basename(run_b)}{tag_b}: {b} falls in {minutes[1]:.0f} min",
            "A fewer than B, one-sided p": round(float(p), 4)}


if __name__ == "__main__":
    if len(sys.argv) == 5:  # RUN_A TAG_A RUN_B TAG_B
        print(json.dumps(compare(*sys.argv[1:5]), indent=1))
    else:
        main()
