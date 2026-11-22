"""A library of the motion matcher's own output, as the dogs' brains would ask for it, for physics training.

  .venv\\Scripts\\python make_mm_library.py [episodes] [seconds]

DReCon (Bergamin et al. 2019) trains its physics tracker on the motion matcher's output under random user
input, not on the raw capture: online, the tracker meets the matcher's jumps and blends (paws sliding while
on the floor; 27-38% of on-floor frames over 0.3 m/s in composed motion against 7-13% in the capture clips),
so it must practise on them. Here the "user" is each dog's brain (dog/brain.py) in a mix of activities,
called to random spots 3-8 m away now and then. Writes refs/mm/mm_NNN.npz with the regime played per frame.
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "dog"))
import make_reference as mr  # noqa: E402
import motion_match as mmod  # noqa: E402
from brain import LAB, MAL, Brain, Router  # noqa: E402

OUT = os.path.join(HERE, "refs", "mm")
DT = 0.02
CONTEXTS = (("play_dog", 0.35), ("explore", 0.2), ("travel", 0.15), ("rest", 0.15), ("interact_person", 0.15))


def episode(mm, router, seed, seconds):
    rng = np.random.default_rng(seed)
    dog = (LAB, MAL)[seed % 2]
    ctx = CONTEXTS[rng.choice(len(CONTEXTS), p=[p for _, p in CONTEXTS])][0]
    start = str(rng.choice(["stand", "walk", "sit", "down", "side"]))
    brain = Brain(dog, router, rng, regime=start, context=ctx)
    player = mmod.Player(mm, start)
    next_call = rng.uniform(2, 12)
    Q, R = [], []
    for f in range(int(seconds / DT)):
        t = f * DT
        if t >= next_call:
            ang, dist = rng.uniform(-np.pi, np.pi), rng.uniform(3, 8)
            target = player.pos + dist * np.array([np.cos(ang), np.sin(ang)])
            if brain.goal is None:
                brain.request(t, "come", target)
            next_call = t + rng.uniform(4, 10)
        reg, speed, want = brain.tick(t, DT, player.pos)
        q, played = player.step(reg, speed, want)
        Q.append(q)
        R.append(mmod.SUPPORT.index(played))
    return np.array(Q), np.array(R, np.int32), dog["name"], ctx


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "dog_mocap_mjx.xml"))
    mm = mmod.Matcher(m)
    router = Router()
    os.makedirs(OUT, exist_ok=True)
    from collections import Counter

    tally = Counter()
    for s in range(n):
        Q, R, who, ctx = episode(mm, router, 1000 + s, seconds)
        ref = mr.finish(m, Q)
        ref["regime"] = R
        np.savez(os.path.join(OUT, f"mm_{s:03d}.npz"), **ref)
        for r, c in zip(*np.unique(R, return_counts=True)):
            tally[mmod.SUPPORT[r]] += c * DT
        moved = float(np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1).sum())
        print(f"mm_{s:03d}: {who} {ctx:15s} moved {moved:5.1f} m", flush=True)
    total = sum(tally.values())
    print(f"{n} episodes, {total / 60:.1f} min: " + ", ".join(f"{k} {v:.0f} s" for k, v in tally.most_common()))


if __name__ == "__main__":
    main()
