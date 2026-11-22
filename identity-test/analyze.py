"""Score the identity test against the pre-registered rules in decisions/001-first-build.md.

  python analyze.py results/*.csv

M4  a carrier holds identity if group accuracy >= 40% and above chance:
    exact one-sided binomial, alpha 0.05, Holm-corrected across carriers
M5  reaction against steady trot, paired across judges (Wilcoxon signed-rank; sign test if few judges)
"""
import csv
import glob
import sys
from collections import defaultdict

from scipy.stats import binomtest, wilcoxon


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main(paths):
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    if not rows:
        raise SystemExit("no results")
    by_carrier = defaultdict(list)
    by_judge = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ok = int(r["correct"])
        by_carrier[r["carrier"]].append((ok, int(r["n_options"])))
        by_judge[r["judge"]][r["carrier"]].append(ok)
    judges = sorted(by_judge)
    print(f"{len(rows)} trials from {len(judges)} judges: {', '.join(judges)}")
    if len(judges) < 8:
        print("Rule M3 asks for at least 8 judges; treat the result as a pilot.")
    names = sorted(by_carrier)
    stats = []
    for c in names:
        xs = by_carrier[c]
        k, n, chance = sum(x for x, _ in xs), len(xs), 1 / xs[0][1]
        stats.append((c, k, n, chance, binomtest(k, n, chance, alternative="greater").pvalue))
    adj = holm([s[4] for s in stats])
    print("\ncarrier                 correct     accuracy  chance  p (Holm)  M4")
    for (c, k, n, chance, _), pa in zip(stats, adj):
        acc = k / n
        verdict = "holds identity" if acc >= 0.40 and pa < 0.05 else "not shown"
        print(f"{c:22s} {k:4d}/{n:<5d}  {acc:7.1%}  {chance:6.0%}  {pa:8.4f}  {verdict}")
    pairs = [("react-dots", "gait-dots"), ("react-canonical", "gait-canonical")]
    for a, b in pairs:
        diffs = []
        for j in judges:
            if by_judge[j][a] and by_judge[j][b]:
                diffs.append(sum(by_judge[j][a]) / len(by_judge[j][a]) - sum(by_judge[j][b]) / len(by_judge[j][b]))
        if len(diffs) < 3:
            continue
        nonzero = [d for d in diffs if d != 0]
        mean = sum(diffs) / len(diffs)
        if len(nonzero) >= 6:
            p = wilcoxon(nonzero).pvalue
            test = "Wilcoxon"
        else:
            pos = sum(d > 0 for d in nonzero)
            p = binomtest(pos, len(nonzero), 0.5).pvalue if nonzero else 1.0
            test = "sign test"
        winner = a if mean > 0 else b
        print(f"\nM5 {a} vs {b}: mean difference {mean:+.1%} across {len(diffs)} judges, {test} p = {p:.4f}"
              f" -> {'build around ' + winner if p < 0.05 else 'no difference shown'}")


if __name__ == "__main__":
    main([p for arg in sys.argv[1:] for p in glob.glob(arg)])
