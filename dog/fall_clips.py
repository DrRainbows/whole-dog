"""Which captured clips does the dog fall out of? (online sessions that log the matcher's frame, mm_frame_0)

  body\\.venv\\Scripts\\python dog\\fall_clips.py RUN_DIR TAG [TAG ...]

For every real fall: the clips and regimes the matcher played in the second before it. For every clip: how long it was
played in all sessions (exposure) and how many falls followed it, so a clip is blamed by its rate, not its count.
"""
import collections
import glob
import json
import os
import sys

import numpy as np

FLAT = "runs/refs_anat/flat"
DB = "runs/refs_anat/mm_db.npz"
SUPPORT = ["stand", "walk", "trot", "run", "jump", "sit", "down", "side", "crouch", "rear_up"]


def main():
    run, tags = sys.argv[1], sys.argv[2:]
    db = np.load(DB)
    clip, regime = db["clip"], db["regime"]
    names = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(FLAT, "*.npz")))]
    exposure = collections.Counter()
    falls_by_clip = collections.Counter()
    fall_rows = []
    n_sessions = 0
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(run, f"online_batch{tag}_*.json"))):
            z = np.load(f.replace(".json", ".npz"), allow_pickle=True)
            if "mm_frame_0" not in z.files:
                continue
            n_sessions += 1
            fr = z["mm_frame_0"]
            for c, n in collections.Counter(clip[fr].tolist()).items():
                exposure[c] += n
            d = json.load(open(f))
            for fl in d["fall_log"]:
                if fl.get("what") != "fall":
                    continue
                i = int(round(fl["t"] / 0.02)) - 1
                win = fr[max(0, i - 50): i + 1]
                cs = collections.Counter(clip[win].tolist())
                rg = collections.Counter(SUPPORT[r] for r in regime[win])
                main_clip = cs.most_common(1)[0][0]
                falls_by_clip[main_clip] += 1
                fall_rows.append((d["brain"], fl["t"], fl["during"], names[main_clip], dict(rg.most_common(3)),
                                  int(fr[i]) if i < len(fr) else None))
    print(f"{n_sessions} sessions, {len(fall_rows)} falls")
    for r in fall_rows:
        print("  fall", r)
    tot = sum(exposure.values())
    print("clips by falls (falls, minutes played, falls per minute played):")
    for c, k in falls_by_clip.most_common(12):
        mins = exposure[c] * 0.02 / 60
        print(f"  {names[c]:28s} falls {k:2d}   played {mins:6.1f} min ({exposure[c] / tot:.1%})   rate {k / max(mins, 1e-6):.2f}/min")


if __name__ == "__main__":
    main()
