"""The link test repeated (train/eval_links.py --seeds 0,...,7), judged per link rather than per episode.

  body\\.venv\\Scripts\\python dog\\link_test_repeated.py RUN_A[@NAME] RUN_B[@NAME] [--name links_anat] [--seeds 0-7]

RUN@NAME reads that run's links_NAME*.json (e.g. the same run tested at another muscle capacity).

Why: the physics on the GPU is not exactly reproducible, so one pass gives one sample of each reference's outcome
(anat_strong, the same policy twice: 102 of 576 references changed outcome). And the 4 references of a link (the
same two postures, the same mode) are not independent trials. So:
- each link's fall rate pools its references over all seeds (episodes that reached the change);
- forbid vs allow is tested by permuting the words' verdict across links (one-sided, the pre-registered direction);
- P3 is paired by link: d = rate(B) - rate(A); the gap in d between forbid and allow links, permuted the same way.
Fixed on 21 Nov, after the non-reproducibility was found and before anat_dog's link results existed.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grammar import Grammar  # noqa: E402

GOOD = {"stand", "walk", "trot", "sit", "down", "crouch"}  # the pre-registered competent subset
N_PERM = 200000


def seeds_arg(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def load(run, name, seeds):
    """link (mode, a, b) -> lists over episodes that reached the change: fell, lig_after, pose_last (if still up)."""
    links = {}
    for sd in seeds:
        path = os.path.join(run, f"links_{name}.json" if sd == 0 else f"links_{name}_s{sd}.json")
        for r in json.load(open(path))["rows"]:
            if r["a"] not in GOOD or r["b"] not in GOOD or r["steps"] < r["link"]:
                continue
            L = links.setdefault((r["mode"], r["a"], r["b"]), {"fell": [], "lig": [], "pose": []})
            L["fell"].append(float(r["fell"]))
            L["lig"].append(r["lig_after"])
            if not r["fell"] and "pose_last" in r:
                L["pose"].append(r["pose_last"])
    return links


def perm_gap(x, forbid, rng, n=N_PERM):
    """Observed mean(x | forbid) - mean(x | allow), one-sided permutation p (shuffling forbid across links)."""
    obs = x[forbid].mean() - x[~forbid].mean()
    k = int(forbid.sum())
    perms = np.argsort(rng.random((n, len(x))), axis=1)[:, :k]
    sel = np.zeros((n, len(x)), bool)
    np.put_along_axis(sel, perms, True, axis=1)
    tot = x.sum()
    g = (sel @ x) / k - (tot - sel @ x) / (len(x) - k)
    return obs, (1 + np.sum(g >= obs - 1e-12)) / (n + 1)


def boot_ci(x, forbid, rng, n=20000):
    iw, ia = np.flatnonzero(forbid), np.flatnonzero(~forbid)
    b = [x[rng.choice(iw, len(iw))].mean() - x[rng.choice(ia, len(ia))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def sign_flip(d, rng, n=N_PERM):
    """Mean of paired differences, one-sided sign-flip permutation p (mean > 0)."""
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return obs, (1 + np.sum((flips * d).mean(1) >= obs - 1e-12)) / (n + 1)


def routing(runs, names, G, rng, m1="forced", m2="routed"):
    """Does routing protect? Per link: fall rate forced minus routed (same link, same run), on the links the words
    forbid and on those they allow."""
    out = {}
    for name, r in zip(names, runs):
        keys = sorted({k[1:] for k in r if k[0] == m1} & {k[1:] for k in r if k[0] == m2})
        forbid = np.array([not G.coarse_text.get(k) for k in keys])
        d = np.array([np.mean(r[(m1,) + k]["fell"]) - np.mean(r[(m2,) + k]["fell"]) for k in keys])
        res = {}
        for lab, sel in (("words forbid", forbid), ("words allow", ~forbid)):
            obs, p = sign_flip(d[sel], rng)
            res[lab] = f"{m1} minus {m2} {100 * obs:.1f} points over {int(sel.sum())} links, sign-flip p={p:.4f}"
        out[name] = res
    return out


def judge(runs, names, G, mode, rng):
    keys = sorted(set.intersection(*[{k for k in r if k[0] == mode} for r in runs]))
    forbid = np.array([not G.coarse_text.get((a, b)) for _, a, b in keys])
    out = {"links": f"{len(keys)} ({int(forbid.sum())} the words forbid, {int((~forbid).sum())} they allow)"}
    rates = []
    for name, r in zip(names, runs):
        x = np.array([np.mean(r[k]["fell"]) for k in keys])
        rates.append(x)
        obs, p = perm_gap(x, forbid, rng)
        lo, hi = boot_ci(x, forbid, rng)
        eps = sum(len(r[k]["fell"]) for k in keys)
        out[name] = {"episodes": eps,
                     "fall rate, words forbid (mean of links)": round(float(x[forbid].mean()), 3),
                     "fall rate, words allow (mean of links)": round(float(x[~forbid].mean()), 3),
                     "gap (points)": round(100 * obs, 1), "gap 95% CI (bootstrap over links)": [round(100 * lo, 1), round(100 * hi, 1)],
                     "forbid > allow, permutation p": round(float(p), 4)}
        lig = np.array([np.mean(r[k]["lig"]) for k in keys])
        obs, p = perm_gap(lig, forbid, rng)
        out[name]["ligament strain after the ask, forbid - allow (per step), permutation p"] = (
            f"{obs:.5f}, p={p:.4f} (means {lig[forbid].mean():.5f} vs {lig[~forbid].mean():.5f})")
        ok = np.array([len(r[k]["pose"]) > 0 for k in keys])
        pose = np.array([np.mean(r[k]["pose"]) if r[k]["pose"] else np.nan for k in keys])
        if ok.all():
            obs, p = perm_gap(-pose, forbid, rng)
            out[name]["still up: pose match in its last second, forbid vs allow, p (forbid lower)"] = (
                f"{pose[forbid].mean():.3f} vs {pose[~forbid].mean():.3f}, p={p:.4f}")
    if len(runs) == 2:
        d = rates[1] - rates[0]
        obs, p = perm_gap(d, forbid, rng)
        lo, hi = boot_ci(d, forbid, rng)
        out[f"P3 paired: gap of {names[1]} minus gap of {names[0]} (points)"] = (
            f"{100 * obs:.1f}, 95% CI {100 * lo:.1f} to {100 * hi:.1f}, one-sided permutation p={p:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", default="links_anat")
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--modes", default="forced,routed", help="the two modes to compare (rush test: forced,eased)")
    a = ap.parse_args()
    G = Grammar()
    seeds = seeds_arg(a.seeds)
    specs = [(r.split("@")[0], r.split("@")[1] if "@" in r else a.name) for r in a.runs]
    runs = [load(r, n, seeds) for r, n in specs]
    names = [os.path.basename(r.rstrip("/\\")) + ("" if n == a.name else f"@{n}") for r, n in specs]
    rng = np.random.default_rng(0)
    m1, m2 = a.modes.split(",")
    res = {"seeds": seeds, m1: judge(runs, names, G, m1, rng), m2: judge(runs, names, G, m2, rng),
           f"{m2} protects (paired by link)": routing(runs, names, G, rng, m1, m2)}
    print(json.dumps(res, indent=1))
    out = os.path.join(specs[-1][0], f"link_test_repeated_{specs[-1][1]}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
