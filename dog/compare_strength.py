"""Forced vs routed changes at different motor strengths (train/eval_links.py results, same references, same policy).

  body\\.venv\\Scripts\\python dog\\compare_strength.py LINKS_A.json [LINKS_B.json ...]

If the physics dog only gets through changes a real dog couldn't because its motors are too strong, then as the
motors weaken, forced changes should fall faster than routed ones, and the words' "no direct link" should start to
predict the falls.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grammar import Grammar  # noqa: E402

GOOD = {"stand", "walk", "trot", "sit", "down", "crouch"}  # what the policy can do at full strength


def summary(path, G):
    d = json.load(open(path))
    rows = [r for r in d["rows"] if r["steps"] >= r["link"]]
    out = {"file": os.path.basename(path), "reached the change": len(rows), "of": len(d["rows"])}
    for name, sel in (("all", rows), ("competent subset", [r for r in rows if r["a"] in GOOD and r["b"] in GOOD])):
        f = [r for r in sel if r["mode"] == "forced"]
        ro = [r for r in sel if r["mode"] == "routed"]
        ff, rf = sum(r["fell"] for r in f), sum(r["fell"] for r in ro)
        _, p = fisher_exact([[ff, len(f) - ff], [rf, len(ro) - rf]], alternative="greater")
        wn = [r for r in f if not G.coarse_text.get((r["a"], r["b"]))]
        wy = [r for r in f if G.coarse_text.get((r["a"], r["b"]))]
        fn, fy = sum(r["fell"] for r in wn), sum(r["fell"] for r in wy)
        _, pw = fisher_exact([[fn, len(wn) - fn], [fy, len(wy) - fy]], alternative="greater")
        rn = [r for r in ro if not G.coarse_text.get((r["a"], r["b"]))]  # P2's other half: routed shows no gap
        ry = [r for r in ro if G.coarse_text.get((r["a"], r["b"]))]
        gn, gy = sum(r["fell"] for r in rn), sum(r["fell"] for r in ry)
        _, pr = fisher_exact([[gn, len(rn) - gn], [gy, len(ry) - gy]], alternative="greater")
        lig = {}
        if sel and "lig_after" in sel[0]:  # ligament strain after the ask (radians past the ranges, per step)
            for mode, grp in (("forced", f), ("routed", ro)):
                xn = [r["lig_after"] for r in grp if not G.coarse_text.get((r["a"], r["b"]))]
                xy = [r["lig_after"] for r in grp if G.coarse_text.get((r["a"], r["b"]))]
                pv = mannwhitneyu(xn, xy, alternative="greater").pvalue if xn and xy else float("nan")
                lig[f"{mode}: ligament strain, words forbid vs allow (median, p)"] = (
                    f"{np.median(xn):.4f} vs {np.median(xy):.4f}, p={pv:.4f}")
        if sel and "pose_last" in sel[0]:  # exploratory (added before anat_dog's result): did it follow the ask?
            for mode, grp in (("forced", f), ("routed", ro)):
                up = [r for r in grp if not r["fell"]]
                xn = [r["pose_last"] for r in up if not G.coarse_text.get((r["a"], r["b"]))]
                xy = [r["pose_last"] for r in up if G.coarse_text.get((r["a"], r["b"]))]
                pv = mannwhitneyu(xn, xy, alternative="less").pvalue if xn and xy else float("nan")
                lig[f"{mode}, still up: pose match over its last second, words forbid vs allow (median, p less)"] = (
                    f"{np.median(xn):.3f} vs {np.median(xy):.3f}, p={pv:.4f}")
        out[name] = {**lig, "forced fell": f"{ff}/{len(f)} ({ff / max(len(f), 1):.0%})",
                     "routed fell": f"{rf}/{len(ro)} ({rf / max(len(ro), 1):.0%})", "forced > routed, p": round(p, 4),
                     "forced, words forbid": f"{fn}/{len(wn)} ({fn / max(len(wn), 1):.0%})",
                     "forced, words allow": f"{fy}/{len(wy)} ({fy / max(len(wy), 1):.0%})", "forbid > allow, p": round(pw, 4),
                     "forced gap (points)": round(100 * (fn / max(len(wn), 1) - fy / max(len(wy), 1)), 1),
                     "routed, words forbid": f"{gn}/{len(rn)} ({gn / max(len(rn), 1):.0%})",
                     "routed, words allow": f"{gy}/{len(ry)} ({gy / max(len(ry), 1):.0%})",
                     "routed forbid > allow, p": round(pr, 4),
                     "routed gap (points)": round(100 * (gn / max(len(rn), 1) - gy / max(len(ry), 1)), 1)}
    return out


def gap_difference(path_a, path_b, G, n_boot=10000, seed=0):
    """P3, supplementary: forced gap (words forbid minus allow, fall rate, competent subset) of run B minus run A,
    with a bootstrap over the link references (same references in both runs; each run keeps only the episodes that
    reached the change, as in summary())."""
    runs = []
    for path in (path_a, path_b):
        rows = {r["file"]: r for r in json.load(open(path))["rows"]
                if r["steps"] >= r["link"] and r["mode"] == "forced" and r["a"] in GOOD and r["b"] in GOOD}
        runs.append(rows)
    refs = sorted(set(runs[0]) | set(runs[1]))
    forbid = np.array([not G.coarse_text.get(tuple(f.split("__")[1:3])) for f in refs])
    fell = np.array([[float(run[f]["fell"]) if f in run else np.nan for f in refs] for run in runs])

    def gaps(idx):
        out = []
        for k in range(2):
            x, w = fell[k, idx], forbid[idx]
            ok = ~np.isnan(x)
            out.append(np.mean(x[ok & w]) - np.mean(x[ok & ~w]) if (ok & w).any() and (ok & ~w).any() else np.nan)
        return out

    ga, gb = gaps(np.arange(len(refs)))
    rng = np.random.default_rng(seed)
    iw, ia = np.flatnonzero(forbid), np.flatnonzero(~forbid)
    boots = []
    for _ in range(n_boot):  # stratified: keep the number of forbid and allow references
        idx = np.concatenate([rng.choice(iw, len(iw)), rng.choice(ia, len(ia))])
        a, b = gaps(idx)
        boots.append(b - a)
    boots = np.array(boots)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"P3 (supplementary)": f"forced gap {os.path.basename(os.path.dirname(path_b))} {100 * gb:.1f} points vs "
                                  f"{os.path.basename(os.path.dirname(path_a))} {100 * ga:.1f}; difference "
                                  f"{100 * (gb - ga):.1f} points, 95% bootstrap CI {100 * lo:.1f} to {100 * hi:.1f}; "
                                  f"share of resamples with B's gap <= A's: {np.nanmean(boots <= 0):.4f}"}


def main():
    G = Grammar()
    for p in sys.argv[1:]:
        print(json.dumps(summary(p, G), indent=1))
    if len(sys.argv) == 3:
        print(json.dumps(gap_difference(sys.argv[1], sys.argv[2], G), indent=1))


if __name__ == "__main__":
    main()
