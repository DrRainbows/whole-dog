"""Does the grammar's price of a change predict whether the physics dog can make it?

  body\\.venv\\Scripts\\python dog\\test_links.py runs/body-runs/uni_mm2/links_test.json

Reads train/eval_links.py's result (one row per reference: forced or routed, regime a then b, how long the physics
dog lasted) and sets it against dog/grammar.py: is there a direct link a -> b in the words, in the capture, and
what does the cheapest route cost in bits. Predictions written before the run (RESULTS.md):
  1. forced changes fall more often than routed ones;
  2. among forced changes, those the words have no direct link for fall more than those the words allow;
  3. the forced fall rate rises with the route's price in bits;
  4. routed changes fall rarely, whatever the price.
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grammar import REP, Grammar  # noqa: E402


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def main():
    path = sys.argv[1]
    rows = json.load(open(path))["rows"]
    G = Grammar()
    out = {"source": path}
    before = [r for r in rows if r["steps"] < r["link"]]
    rows = [r for r in rows if r["steps"] >= r["link"]]  # reached the change on its feet (or in its posture)
    out["fell_before_the_change"] = len(before)
    pair = defaultdict(lambda: {"forced": [], "routed": []})
    for r in rows:
        pair[(r["a"], r["b"])][r["mode"]].append(1.0 if r["fell"] else 0.0)
    table = []
    for (a, b), d in sorted(pair.items()):
        bits, route = G.route(REP[a], REP[b])
        words = bool(G.coarse_text.get((a, b)))
        cap = G.cap.get((a, b), 0)
        direct = REP[b] in G.adj.get(REP[a], {})
        f = sum(d["forced"]) / len(d["forced"]) if d["forced"] else None
        r_ = sum(d["routed"]) / len(d["routed"]) if d["routed"] else None
        table.append({"a": a, "b": b, "words": words, "capture": cap, "direct": direct,
                      "route_bits": round(bits, 2) if math.isfinite(bits) else None, "hops": len(route) - 1,
                      "forced_fall": f, "routed_fall": r_, "n_forced": len(d["forced"]), "n_routed": len(d["routed"])})
    out["pairs"] = table
    allf = [v for (a, b), d in pair.items() for v in d["forced"]]
    allr = [v for (a, b), d in pair.items() for v in d["routed"]]
    out["p1_forced_vs_routed"] = {"forced_fall": round(sum(allf) / len(allf), 3), "routed_fall": round(sum(allr) / len(allr), 3),
                                  "n": [len(allf), len(allr)]}
    wy = [v for t in table if t["words"] for v in pair[(t["a"], t["b"])]["forced"]]
    wn = [v for t in table if not t["words"] for v in pair[(t["a"], t["b"])]["forced"]]
    # permutation test on the pair labels (words allow / not), pairs kept whole
    obs = (sum(wn) / len(wn)) - (sum(wy) / len(wy))
    labels = [t["words"] for t in table]
    rnd, more = random.Random(0), 0
    for _ in range(5000):
        perm = labels[:]
        rnd.shuffle(perm)
        y = [v for t, l in zip(table, perm) if l for v in pair[(t["a"], t["b"])]["forced"]]
        n = [v for t, l in zip(table, perm) if not l for v in pair[(t["a"], t["b"])]["forced"]]
        if y and n and (sum(n) / len(n) - sum(y) / len(y)) >= obs:
            more += 1
    out["p2_forced_by_words"] = {"words_allow": round(sum(wy) / len(wy), 3), "words_forbid": round(sum(wn) / len(wn), 3),
                                 "n_pairs": [sum(labels), len(labels) - sum(labels)], "p": round((more + 1) / 5001, 4)}
    tb = [t for t in table if t["route_bits"] is not None and t["forced_fall"] is not None]
    rho = spearman([t["route_bits"] for t in tb], [t["forced_fall"] for t in tb])
    rnd, more = random.Random(1), 0
    ys = [t["forced_fall"] for t in tb]
    for _ in range(5000):
        sh = ys[:]
        rnd.shuffle(sh)
        if spearman([t["route_bits"] for t in tb], sh) >= rho:
            more += 1
    out["p3_forced_vs_price"] = {"spearman": round(rho, 3), "p": round((more + 1) / 5001, 4), "n_pairs": len(tb)}
    rho_r = spearman([t["route_bits"] for t in tb if t["routed_fall"] is not None],
                     [t["routed_fall"] for t in tb if t["routed_fall"] is not None])
    out["p4_routed_vs_price"] = {"spearman": round(rho_r, 3), "routed_fall": out["p1_forced_vs_routed"]["routed_fall"]}
    # baseline: how often the same policy falls from a start in each regime (no change asked): the starts' regime a
    by_a = defaultdict(list)
    for r in rows:
        if r["mode"] == "routed":
            by_a[r["a"]].append(1.0 if r["fell"] else 0.0)
    out["routed_fall_by_start"] = {a: round(sum(v) / len(v), 2) for a, v in by_a.items()}
    by_b = defaultdict(list)
    for r in rows:
        by_b[(r["mode"], r["b"])].append(1.0 if r["fell"] else 0.0)
    out["fall_by_target"] = {f"{m}>{b}": round(sum(v) / len(v), 2) for (m, b), v in sorted(by_b.items())}
    res = os.path.join(HERE, "test_links_result.json")
    json.dump(out, open(res, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    print("\npairs, sorted by the forced fall rate:")
    for t in sorted(table, key=lambda t: -(t["forced_fall"] or 0)):
        print(f"  {t['a']:6s}>{t['b']:6s} words {'yes' if t['words'] else ' no'} capture {t['capture']:3d} "
              f"route {t['route_bits'] if t['route_bits'] is not None else '-':>5} bits ({t['hops']} hops)  "
              f"forced falls {t['forced_fall']:.2f} (n {t['n_forced']})  routed {t['routed_fall'] if t['routed_fall'] is not None else '-'}")
    print(f"wrote {res}")


if __name__ == "__main__":
    main()
