"""Every change between two regimes, made two ways for the physics dog to try.

  .venv\\Scripts\\python make_links.py OUTDIR [seeds] [first_seed] [--anat]

--anat: play from the capture in anatomical joints (anat.py, dog_anat.xml), so a blend moves each joint along its
own axes inside its range; "straight across" then means straight across for a body with ligaments.

forced: the ask goes straight across. The matcher may jump to any frame of the new regime and blends there in
        0.2 s (inertialization): what a body is asked to do when a word is said with no regard for where it is.
routed: the matcher's planner takes the capture's own path between the two (side -> down -> ... -> stand) and
        only changes regime through transitions the capture contains: what the grammar would do.
--rush: the six competent regimes, forced against eased (the same jump, blended over 1 s instead of 0.2 s, from
        the same start): separates asking for a change the body has no path for from asking for it too fast.
Each reference holds regime a for 2 s (0.6 s for a jump), then asks for b for 3 s. Written to
OUTDIR/{mode}__{a}__{b}__{seed}.npz with `link` = the frame the ask changes and the regime played per frame,
for train/eval_links.py.
"""
import os
import sys
import zlib

ANAT = "--anat" in sys.argv
if ANAT:
    sys.argv.remove("--anat")
RUSH = "--rush" in sys.argv  # the rush test: forced (0.2 s blend) vs eased (1 s blend) from the same start
if RUSH:
    sys.argv.remove("--rush")
    os.environ.setdefault("DOG_MM_FLAT", "runs/refs_anat/flat")
    os.environ.setdefault("DOG_MM_CACHE", "runs/refs_anat/mm_db.npz")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_reference as mr  # noqa: E402
import motion_match as mmod  # noqa: E402

DT = 0.02
REGS = ["stand", "walk", "trot", "run", "jump", "sit", "down", "side", "crouch"]  # rear_up: 3 s of capture, too few


def one(mm, a, b, mode, rng):
    before = 0.6 if a == "jump" else 2.0
    p = mmod.Player(mm, a, plan=(mode == "routed"), same_regime=(mode == "routed"))
    pool = mm.cand[mm.db["regime"][mm.cand] == mmod.SUPPORT.index(a)]
    # start somewhere inside a run of regime a with room to stay in it
    ok = [c for c in pool if c + 30 < mm.end[c] and mm.db["regime"][min(c + 30, mm.end[c])] == mmod.SUPPORT.index(a)]
    p.c = int(rng.choice(ok if ok else pool))
    Q, R = [], []
    n_before, n = int(round(before / DT)), int(round((before + 3.0) / DT))
    for f in range(n):
        if mode == "eased" and f == n_before:
            p.blend = 50  # from the ask on, jumps blend over 1 s; before it, identical to forced
        q, played = p.step(a if f < n_before else b)
        Q.append(q)
        R.append(mmod.SUPPORT.index(played))
    return np.array(Q), np.array(R, np.int32), n_before


def main():
    out = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(out, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "dog_anat.xml" if ANAT else "dog_mocap_mjx.xml"))
    mm = mmod.Matcher(m)
    k = 0
    regs = ["stand", "walk", "trot", "sit", "down", "crouch"] if RUSH else REGS  # rush: the competent regimes
    for a in regs:
        for b in regs:
            if a == b:
                continue
            for mode in (("forced", "eased") if RUSH else ("forced", "routed")):
                for s in range(first, first + seeds):
                    # the same start for every mode of a link (crc32: reproducible across processes, unlike hash())
                    rng = np.random.default_rng(zlib.crc32(f"{a}|{b}|{s}".encode()))
                    Q, R, link = one(mm, a, b, mode, rng)
                    ref = mr.finish(m, Q)
                    ref.update(regime=R, link=np.int32(link), a=np.array(a), b=np.array(b), mode=np.array(mode))
                    np.savez(os.path.join(out, f"{mode}__{a}__{b}__{s}.npz"), **ref)
                    k += 1
        print(f"{a}: done ({k} references)", flush=True)


if __name__ == "__main__":
    main()
