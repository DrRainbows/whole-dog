"""New motion from new sentences: the dogs' grammar strings words, the matcher turns them into motion.

  .venv\\Scripts\\python make_grammar_library.py [episodes] [seconds] [first]

Each episode is one dog (the Labrador or the Malinois) in one mode:
  endemic  nobody there: the dog strings its own sentences from the words open to it alone (A, D) in an activity;
  social   a person stands nearby: the human-coupled words (H) open too, and every so often the person asks
           for something (sit, down, come, place, off, jump ...), which the dog weighs and may do.
Sentences are drawn from dog/grammar.py with probability 2^(-bits / tau): tau 0.7 keeps to the dog's usual
links, 1.5 reaches the rare ones the body still has a route for. Links the body has no route for are never
drawn here (forced links are for train/eval_links.py). Every sentence is planned into body steps by the
dog's brain and played by the motion matcher, so the tracker can practise sentences no capture clip contains.
Writes refs/gram/gram_NNN.npz (with the regime played per frame) and refs/gram/sentences.jsonl.
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "dog"))
import make_reference as mr  # noqa: E402
import motion_match as mmod  # noqa: E402
from brain import LAB, MAL, Brain, Router  # noqa: E402
from grammar import Grammar  # noqa: E402

OUT = os.path.join(HERE, "refs", "gram")
DT = 0.02
ASKS = ["sit", "down", "come", "place", "off", "stand", "jump", "settle", "heel", "up", "platz", "here"]
CTX = {"endemic": ["explore", "rest", "play_dog", "travel"], "social": ["interact_person", "play_person", "explore"]}


def episode(mm, G, router, seed, seconds):
    rng = np.random.default_rng(seed)
    dog = (LAB, MAL)[seed % 2]
    mode = ("endemic", "social")[(seed // 2) % 2]
    ctx = str(rng.choice(CTX[mode]))
    tau = float(rng.choice([0.7, 1.0, 1.5]))
    start = str(rng.choice(["stand", "walk", "sit", "down", "side"]))
    person = np.array([rng.uniform(3, 6), rng.uniform(-2, 2)]) if mode == "social" else None
    brain = Brain(dog, router, rng, regime=start, context=ctx, grammar=G, person=person)
    brain.next_social = None  # sentences come from the grammar here, not from the dog's social habit
    player = mmod.Player(mm, start)
    Q, R, said = [], [], []
    next_ask = rng.uniform(8, 20) if mode == "social" else 1e9
    for f in range(int(seconds / DT)):
        t = f * DT
        if not brain.plan and brain.goal is None:
            s = G.generate(1, length=int(rng.integers(4, 9)), mode=mode, ctx=ctx, dog=dog["name"], tau=tau,
                           seed=int(rng.integers(1 << 30)), start=brain.node, state=brain.state(),
                           body_only=True)[0]
            for lk in s["links"]:
                lk.setdefault("said", lk["to"])
                lk.setdefault("target", None)
            here = player.pos + rng.uniform(-4, 4, 2)
            brain.plan = brain._plan_from({"links": s["links"]}, here, 0.0, heard=False)
            brain.plan_from = t
            said.append({"t": round(t, 1), "who": "dog", "bits": s["bits"], "words": [lk["to"] for lk in s["links"]]})
        if t >= next_ask:
            word = str(rng.choice(ASKS))
            brain.plan = []
            brain.hear(t, word, where=player.pos + rng.uniform(-4, 4, 2))
            said.append({"t": round(t, 1), "who": "person", "words": [word]})
            next_ask = t + rng.uniform(10, 25)
        reg, speed, want = brain.tick(t, DT, player.pos)
        q, played = player.step(reg, speed, want)
        Q.append(q)
        R.append(mmod.SUPPORT.index(played))
    return np.array(Q), np.array(R, np.int32), dog["name"], mode, ctx, tau, said, brain


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out = sys.argv[4] if len(sys.argv) > 4 else OUT
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "dog_mocap_mjx.xml"))
    mm, G, router = mmod.Matcher(m), Grammar(), Router()
    os.makedirs(out, exist_ok=True)
    from collections import Counter

    tally, changes = Counter(), Counter()
    log = open(os.path.join(out, "sentences.jsonl"), "a")
    for s in range(first, first + n):
        Q, R, who, mode, ctx, tau, said, brain = episode(mm, G, router, 5000 + s, seconds)
        ref = mr.finish(m, Q)
        ref["regime"] = R
        np.savez(os.path.join(out, f"gram_{s:03d}.npz"), **ref)
        for r, c in zip(*np.unique(R, return_counts=True)):
            tally[mmod.SUPPORT[r]] += c * DT
        runs = [int(R[0])] + [int(b) for a, b in zip(R, R[1:]) if a != b]
        for a, b in zip(runs, runs[1:]):
            changes[(mmod.SUPPORT[a], mmod.SUPPORT[b])] += 1
        log.write(json.dumps({"file": f"gram_{s:03d}.npz", "dog": who, "mode": mode, "ctx": ctx, "tau": tau,
                              "said": said, "heard": brain.heard, "events": brain.events[-40:]}, default=str) + "\n")
        print(f"gram_{s:03d}: {who} {mode:8s} {ctx:15s} tau {tau}: {len(said)} sentences, "
              f"{len(runs) - 1} regime changes", flush=True)
    total = sum(tally.values())
    print(f"{n} episodes, {total / 60:.1f} min: " + ", ".join(f"{k} {v:.0f} s" for k, v in tally.most_common()))
    print("regime changes: " + ", ".join(f"{a}>{b} {v}" for (a, b), v in changes.most_common(30)))


if __name__ == "__main__":
    main()
