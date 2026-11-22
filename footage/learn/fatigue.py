"""Does play slow down over minutes? The fatigue test, on the one long play session (IMG_2650, 6 minutes).

  .venv\\Scripts\\python -m learn.fatigue [clip]

The brain (dog/brain.py) has a designed slow exertion integrator (fatigue, time constant 5 min) that shortens
active bouts and lengthens rest. The short clips can't test it; this one can. Per dog and per minute, from the
teacher's once-a-second labels:
  vigour   share of seconds running, trotting, jumping, rearing up, tugging or mouth-playing
  rest     share of seconds lying, sitting or standing still with nothing in the mouth
  panting  share of seconds the teacher saw panting
  switches regime changes per minute (how busy the play is)
A trend is a least-squares slope per minute; its p-value comes from shuffling whole 30 s blocks (keeps the
autocorrelation of play bouts). Writes models/fatigue_<clip>.json.
"""
import json
import os
import sys

import numpy as np

from .dataset import DOGS, teacher_rows

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")
VIGOROUS = {"run", "trot", "jump", "rear_up"}
MOUTH_PLAY = {"tug", "mouth_play"}
RESTING = {"down", "side", "sit"}


def per_second(clip):
    rows = sorted((r for r in teacher_rows() if r["clip"] == clip), key=lambda r: r["t"])
    out = {}
    for dog in DOGS:
        t, vig, rest, pant, reg = [], [], [], [], []
        for r in rows:
            d = r.get(dog) or {}
            if not d.get("visible"):
                continue
            s, a = d.get("support"), d.get("action")
            t.append(r["t"])
            vig.append(s in VIGOROUS or a in MOUTH_PLAY)
            rest.append(s in RESTING or (s == "stand" and a in ("none", "look")))
            pant.append(a == "pant")
            reg.append(s)
        out[dog] = {"t": np.array(t), "vigour": np.array(vig, float), "rest": np.array(rest, float),
                    "panting": np.array(pant, float), "regime": reg}
    return out


def slope_and_p(t, y, block=30.0, n=4000, seed=0):
    """Slope of y against time (per minute) and a block-permutation p-value (two-sided)."""
    if len(t) < 20:
        return float("nan"), float("nan")
    x = t / 60.0
    b = np.polyfit(x, y, 1)[0]
    blocks = (t // block).astype(int)
    ids = np.unique(blocks)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        perm = dict(zip(ids, rng.permutation(ids)))
        xs = np.array([perm[k] for k in blocks]) * block / 60.0 + (t % block) / 60.0
        hits += abs(np.polyfit(xs, y, 1)[0]) >= abs(b)
    return float(b), (hits + 1) / (n + 1)


def designed_fatigue(seconds, play_intensity=0.4, tau=300.0, start=0.1):
    """The brain's slow integrator during sustained play: F(t) rises toward the play intensity."""
    t = np.arange(seconds)
    return play_intensity + (start - play_intensity) * np.exp(-t / tau)


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "IMG_2650"
    data = per_second(clip)
    report = {"clip": clip}
    for dog, d in data.items():
        t = d["t"]
        if len(t) == 0:
            continue
        minutes = int(t.max() // 60) + 1
        per_min = []
        for k in range(minutes):
            sel = (t >= 60 * k) & (t < 60 * (k + 1))
            if sel.sum() < 15:
                continue
            regs = [d["regime"][i] for i in np.where(sel)[0]]
            switches = sum(a != b for a, b in zip(regs, regs[1:])) * 60.0 / max(sel.sum(), 1)
            per_min.append({"minute": k + 1, "seconds": int(sel.sum()), "vigour": round(float(d["vigour"][sel].mean()), 2),
                            "rest": round(float(d["rest"][sel].mean()), 2), "panting": round(float(d["panting"][sel].mean()), 2),
                            "switches_per_min": round(switches, 1)})
        trends = {}
        for key in ("vigour", "rest", "panting"):
            b, p = slope_and_p(t, d[key])
            trends[key] = {"slope_per_min": round(b, 3), "p": round(p, 3)}
        report[dog] = {"labelled_seconds": int(len(t)), "per_minute": per_min, "trends": trends}
        print(f"\n{dog}: {len(t)} labelled seconds over {t.max() / 60:.1f} min")
        print("  minute  seconds  vigour  rest  panting  switches/min")
        for m in per_min:
            print(f"  {m['minute']:6d}  {m['seconds']:7d}  {m['vigour']:6.2f}  {m['rest']:4.2f}  {m['panting']:7.2f}  {m['switches_per_min']:12.1f}")
        print("  trends per minute: " + ", ".join(f"{k} {v['slope_per_min']:+.3f} (p {v['p']:.3f})" for k, v in trends.items()))
    F = designed_fatigue(360)
    report["designed_fatigue_at_minutes"] = [round(float(F[min(60 * k, 359)]), 3) for k in range(7)]
    print("\nthe brain's designed fatigue F during sustained play, at minutes 0-6:",
          " ".join(f"{x:.2f}" for x in report["designed_fatigue_at_minutes"]))
    json.dump(report, open(os.path.join(MODELS, f"fatigue_{clip}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
