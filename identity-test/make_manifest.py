"""Build manifest.js from a folder of clips.

Name each clip   DOG__CARRIER__ANYTHING.mp4   (or .m4a / .wav for sound-only clips), for example
  clips/juno__gait-dots__2026-11-21_pass3.mp4
and describe the dogs in dogs.json:
  [{"id": "juno", "name": "Juno", "breed": "labrador", "target": true},
   {"id": "lab_b", "name": "Lab B", "breed": "labrador", "target": false}, ...]
"target": true marks the dogs judges are asked to find; the others are same-breed foils.

  python make_manifest.py [--trials 20] [--options 4]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

CARRIERS = [
    {"id": "gait-dots", "label": "Movement only: trotting", "audio": False},
    {"id": "react-dots", "label": "Movement only: reacting", "audio": False},
    {"id": "gait-canonical", "label": "Movement only, same skeleton: trotting", "audio": False},
    {"id": "react-canonical", "label": "Movement only, same skeleton: reacting", "audio": False},
    {"id": "pauses-dots", "label": "Movement only: in between", "audio": False},
    {"id": "sound", "label": "Sound only", "audio": True},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--options", type=int, default=4)
    ap.add_argument("--dir", default=".", help="folder holding dogs.json and clips/ (default: this folder)")
    a = ap.parse_args()
    root = Path(__file__).parent
    here = (root / a.dir).resolve()
    prefix = here.relative_to(root.resolve()).as_posix()
    prefix = "" if prefix == "." else prefix + "/"
    dogs = json.loads((here / "dogs.json").read_text(encoding="utf-8"))
    ids = {d["id"] for d in dogs}
    known_carriers = {c["id"] for c in CARRIERS}
    clips = []
    for f in sorted((here / "clips").glob("*")):
        if f.suffix.lower() not in (".mp4", ".webm", ".m4a", ".wav", ".mp3"):
            continue
        parts = f.stem.split("__")
        if len(parts) < 3 or parts[0] not in ids or parts[1] not in known_carriers:
            print(f"skipped (name doesn't match DOG__CARRIER__X or unknown dog/carrier): {f.name}")
            continue
        clips.append({"file": f"{prefix}clips/{f.name}", "dog": parts[0], "carrier": parts[1], "tag": parts[2]})
    used = [c for c in CARRIERS if any(k["carrier"] == c["id"] for k in clips)]
    manifest = {"dogs": dogs, "carriers": used, "clips": clips, "trialsPerCarrier": a.trials, "options": a.options}
    (here / "manifest.js").write_text("window.MANIFEST = " + json.dumps(manifest, indent=1) + ";\n", encoding="utf-8")
    count = Counter((c["dog"], c["carrier"]) for c in clips)
    print(f"manifest.js: {len(dogs)} dogs, {len(clips)} clips, carriers: {[c['id'] for c in used]}")
    for d in dogs:
        row = ", ".join(f"{c['id']} {count.get((d['id'], c['id']), 0)}" for c in used)
        print(f"  {d['id']:12s} ({d['breed']}, {'target' if d.get('target') else 'foil'}): {row}")
    short = [d["id"] for d in dogs if d.get("target") for c in used if count.get((d["id"], c["id"]), 0) < 10]
    if short:
        print("Rule M3 needs at least 10 independent clips per target dog per carrier. Short:", sorted(set(short)))


if __name__ == "__main__":
    main()
