"""Name the tracked dogs per clip, calibrated by the teacher's labels.

Per clip, the teacher's boxes say where the Lab and the Malinois are once a second. The
tracked detections inside those boxes show how bright each dog's torso looks in that
clip's light. Every detection in the clip then goes to the dog it is closer to in brightness.
Detections that match neither, such as other dogs, are dropped. Clips with no usable teacher
samples fall back to a rule learned across all clips.

  .venv\\Scripts\\python -m learn.identity      rewrite library/work/tracks/*.dogs.npz and report agreement
"""
import json
import os
from collections import defaultdict

import numpy as np

from .dataset import DOGS, frame_size, teacher_rows
from .features import bbox, iou

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(HERE, "library", "work")


def raw(clip):
    import sys
    sys.path.insert(0, HERE)
    from pipeline.tracks import dlc_h5_all

    dlc = os.path.join(WORK, "dlc")
    h5 = [f for f in os.listdir(dlc) if f.startswith(clip + "_superanimal") and f.endswith(".h5")]
    names, xy, conf = dlc_h5_all(os.path.join(dlc, h5[0]))
    old = np.load(os.path.join(WORK, "tracks", clip + ".dogs.npz"), allow_pickle=False)
    return names, xy, conf, old["brightness"]


def teacher_samples(clip, rows, xy, conf, bright, w, h, fps=30.0):
    """(brightness, dog) for detections that sit inside a teacher box for that dog."""
    out = []
    for r in rows:
        if (r.get("other_dogs") or 0) > 0:
            continue
        f = int(round(r["t"] * fps))
        if f >= xy.shape[0]:
            continue
        for dog in DOGS:
            lab = r.get(dog) or {}
            box = lab.get("box")
            if not (lab.get("visible") and isinstance(box, list) and len(box) == 4):
                continue
            tb = (box[1] / 1000 * w, box[0] / 1000 * h, box[3] / 1000 * w, box[2] / 1000 * h)
            for i in range(xy.shape[1]):
                bb = bbox(xy[f, i], conf[f, i])
                if bb and iou(tb, bb) > 0.3 and np.isfinite(bright[f, i]):
                    out.append((f, i, float(bright[f, i]), dog))
    return out


def decide(b, ref):
    """ref: {'lab': median brightness or None, 'mal': ...}. Returns 'lab', 'mal' or None."""
    if not np.isfinite(b):
        return None
    lb = np.log(max(b, 1.0))
    have = {d: np.log(max(v, 1.0)) for d, v in ref.items() if v is not None}
    if len(have) == 2:
        d = min(have, key=lambda k: abs(have[k] - lb))
        gap = abs(have["lab"] - have["mal"])
        return d if abs(have[d] - lb) < max(0.35, 0.75 * gap) else None
    if len(have) == 1:
        d, v = next(iter(have.items()))
        return d if abs(v - lb) < 0.3 else None
    return None


def main():
    rows = defaultdict(list)
    for r in teacher_rows():
        rows[r["clip"]].append(r)
    clips = sorted(f[:-9] for f in os.listdir(os.path.join(WORK, "tracks")) if f.endswith(".dogs.npz"))
    all_samples, per_clip = [], {}
    for clip in clips:
        names, xy, conf, bright = raw(clip)
        w, h = frame_size(clip)
        s = teacher_samples(clip, rows.get(clip, []), xy, conf, bright, w, h)
        per_clip[clip] = (names, xy, conf, bright, s)
        all_samples += [(clip, *x) for x in s]
    # global fallback: brightness threshold between the two dogs across all clips
    lab_b = [b for _, _, _, b, d in all_samples if d == "lab"]
    mal_b = [b for _, _, _, b, d in all_samples if d == "mal"]
    glob_ref = {"lab": float(np.median(lab_b)) if lab_b else None, "mal": float(np.median(mal_b)) if mal_b else None}
    report = {"global": glob_ref, "clips": {}}
    agree, total = 0, 0
    for clip, (names, xy, conf, bright, s) in per_clip.items():
        T, I, K = conf.shape
        # leave-one-out agreement on the teacher samples, then the final per-clip reference
        for j, (f, i, b, d) in enumerate(s):
            rest = [x for k, x in enumerate(s) if k != j]
            ref = {dd: (float(np.median([x[2] for x in rest if x[3] == dd])) if any(x[3] == dd for x in rest) else None)
                   for dd in DOGS}
            if all(v is None for v in ref.values()):
                ref = glob_ref
            agree += decide(b, ref) == d
            total += 1
        clip_rows = rows.get(clip, [])
        teacher_saw = {dd: any((r.get(dd) or {}).get("visible") for r in clip_rows) for dd in DOGS}
        ref = {}
        source = {}
        for dd in DOGS:
            mine = [x[2] for x in s if x[3] == dd]
            if mine:
                ref[dd], source[dd] = float(np.median(mine)), "clip"
            elif teacher_saw[dd] and glob_ref[dd] is not None:
                ref[dd], source[dd] = glob_ref[dd], "global"
            else:
                ref[dd], source[dd] = None, "absent"
        crowded = np.zeros(T, bool)
        for r in clip_rows:
            if (r.get("other_dogs") or 0) > 0:
                f = int(round(r["t"] * 30))
                crowded[max(0, f - 15):f + 16] = True
        out_xy = np.full((T, 2, K, 2), np.nan)
        out_cf = np.zeros((T, 2, K))
        for t in range(T):
            if crowded[t]:
                continue
            taken = {}
            for i in range(I):
                if conf[t, i].mean() < 0.2:
                    continue
                d = decide(bright[t, i], ref)
                if d is None:
                    continue
                score = conf[t, i].mean()
                if d not in taken or score > taken[d][0]:
                    taken[d] = (score, i)
            for d, (_, i) in taken.items():
                slot = DOGS.index(d)
                out_xy[t, slot], out_cf[t, slot] = xy[t, i], conf[t, i]
        seen = (out_cf.mean(axis=2) > 0.3).mean(axis=0)
        np.savez_compressed(os.path.join(WORK, "tracks", clip + ".dogs.npz"), names=np.array(names), xy=out_xy,
                            conf=out_cf, fps=30.0, brightness=bright, split=0.0, source=os.path.join(WORK, clip + ".mp4"),
                            identity=json.dumps({"ref": ref, "source": source, "teacher_samples": len(s)}))
        report["clips"][clip] = {"ref": ref, "source": source, "samples": len(s),
                                 "lab_seen": round(float(seen[0]), 3), "mal_seen": round(float(seen[1]), 3)}
    report["leave_one_out_agreement"] = agree / total if total else None
    report["teacher_samples"] = total
    json.dump(report, open(os.path.join(HERE, "models", "identity.json"), "w"), indent=1)
    print(f"identity: {total} teacher-matched detections; leave-one-out agreement "
          f"{(agree / total if total else float('nan')):.1%}; global medians {glob_ref}")
    for c, r in report["clips"].items():
        print(f"  {c:40s} {r['source']:6s} samples {r['samples']:3d}  lab seen {r['lab_seen']:.0%}  mal seen {r['mal_seen']:.0%}")


if __name__ == "__main__":
    main()
