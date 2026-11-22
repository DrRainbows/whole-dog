"""Requests in the footage: what was asked of each dog, and what each dog did about it.

  .venv\\Scripts\\python -m learn.requests transcribe     speech -> library/labels/speech.jsonl (local, CPU)
  .venv\\Scripts\\python -m learn.requests analyze        match requests to the dogs' regimes before and after

The first notes treat commands as requests: the dog has the final say. The videos' audio holds the
requests actually made of these two dogs. Speech is transcribed locally (faster-whisper, word
timestamps); nothing leaves the machine. Request words are matched to the teacher's per-second
labels, giving per dog: how often a request is followed by the asked-for regime, and how fast.
"""
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

from .dataset import DOGS, teacher_rows

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(HERE, "library")
AUDIO = os.path.join(LIB, "work", "audio")
SPEECH = os.path.join(LIB, "labels", "speech.jsonl")
SRC = os.environ.get("DOG_PHOTOS_DIR", "footage/raw_photos")

# request -> what compliance looks like (a regime the dog should enter, or a change it should make)
REQUESTS = {
    "sit": ("sit", r"\bsit\b"),
    "down": ("down", r"\b(lie down|lay down|down)\b"),
    "come": ("approach", r"\b(come|come here|here|let'?s go|this way)\b"),
    "stay": ("still", r"\b(stay|wait)\b"),
    "drop": ("release", r"\b(drop|drop it|leave it|give|out)\b"),
    "fetch": ("go", r"\b(get it|go get|fetch|find it|go on|go)\b"),
    "settle": ("calm", r"\b(easy|gentle|settle|calm|enough|off)\b"),
    "no": ("stop", r"\b(no|hey|ah)\b"),
}
PRAISE = r"\b(good (girl|boy|dog|job)|yes|nice|atta)\b"
STATIC = {"stand", "sit", "down", "side", "back", "crouch"}
MOVING = {"walk", "trot", "run", "jump", "swim", "wade"}


# ---------- 1. transcribe ----------
def extract(name, src):
    os.makedirs(AUDIO, exist_ok=True)
    out = os.path.join(AUDIO, name + ".wav")
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", out], check=True)
    return out


def transcribe(model_size="small.en"):
    from faster_whisper import WhisperModel

    inv = json.load(open(os.path.join(LIB, "inventory.json")))
    done = set()
    if os.path.exists(SPEECH):
        done = {json.loads(line)["clip"] for line in open(SPEECH, encoding="utf-8")}
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=6)
    with open(SPEECH, "a", encoding="utf-8") as fh:
        for r in inv:
            name = os.path.splitext(r["file"])[0]
            if name in done:
                continue
            wav = extract(name, os.path.join(SRC, r["file"]))
            segs, _ = model.transcribe(wav, language="en", word_timestamps=True, vad_filter=True,
                                       condition_on_previous_text=False)
            words = [{"w": w.word.strip(), "t0": round(w.start, 2), "t1": round(w.end, 2), "p": round(w.probability, 2)}
                     for s in segs for w in (s.words or [])]
            fh.write(json.dumps({"clip": name, "words": words, "model": model_size}) + "\n")
            fh.flush()
            print(f"{name}: {len(words)} words", flush=True)


# ---------- 2. analyze ----------
def phrases(words):
    """Yield (t, text) for every word with up to two following words, so multi-word requests match."""
    for i, w in enumerate(words):
        text = " ".join(x["w"] for x in words[i:i + 3]).lower()
        yield w["t0"], re.sub(r"[^a-z' ]", "", text)


def find_requests(words, names):
    events = []
    last = -10.0
    for t, text in phrases(words):
        head = text.split(" ")[0] if text else ""
        for req, (_, pat) in REQUESTS.items():
            m = re.match(pat, text)
            if m:
                if t - last > 1.0:  # repeated words inside one second count once
                    addressed = [n for n in names if re.search(rf"\b{n}\b", " ".join(x["w"].lower() for x in words
                                                                                     if t - 3 <= x["t0"] <= t + 1.5))]
                    events.append({"t": t, "req": req, "text": m.group(0), "addressed": addressed})
                    last = t
                break
        _ = head
    return events


def regime_at(rows, dog, t):
    """The teacher's label for this dog nearest to time t (within 0.6 s), or None."""
    best = None
    for r in rows:
        if abs(r["t"] - t) <= 0.6:
            lab = r.get(dog) or {}
            if lab.get("visible"):
                best = lab
    return best


def outcome(req, before, after_rows, dog):
    """Did the dog do what was asked within 6 s, and how fast? after_rows: [(dt, label)] sorted by dt."""
    want = REQUESTS[req][0]
    b_sup = (before or {}).get("support")
    b_act = (before or {}).get("action")
    b_area = None
    if before and isinstance(before.get("box"), list):
        y0, x0, y1, x1 = before["box"]
        b_area = (y1 - y0) * (x1 - x0)
    for dt, lab in after_rows:
        sup, act = lab.get("support"), lab.get("action")
        if want in ("sit", "down") and sup == want:
            return True, dt
        if want == "approach":
            if isinstance(lab.get("box"), list) and b_area:
                y0, x0, y1, x1 = lab["box"]
                if (y1 - y0) * (x1 - x0) > 1.5 * b_area and sup in MOVING | {"stand"}:
                    return True, dt
        if want == "release" and b_act in ("carry", "tug", "mouth_play", "chew") and act not in ("carry", "tug", "mouth_play", "chew"):
            return True, dt
        if want == "go" and sup in ("run", "trot", "swim", "jump"):
            return True, dt
        if want in ("calm", "stop") and b_sup in MOVING | {"rear_up", "jump"} and sup in STATIC:
            return True, dt
    if want == "still":
        moved = any(lab.get("support") in MOVING for _, lab in after_rows)
        return (not moved and len(after_rows) >= 3), None
    return False, None


def analyze():
    speech = [json.loads(line) for line in open(SPEECH, encoding="utf-8")]
    rows = defaultdict(list)
    for r in teacher_rows():
        rows[r["clip"]].append(r)
    # the dogs' names: frequent capitalised words that are not request words
    caps = Counter(w["w"].strip(".,!?").lower() for s in speech for w in s["words"]
                   if w["w"].strip(".,!?")[:1].isupper() and len(w["w"].strip(".,!?")) > 2)
    stop = {"come", "good", "the", "and", "you", "yes", "okay", "sit", "down", "drop", "stay", "get", "here", "let's",
            "that", "what", "she", "he", "it's", "this", "there", "come", "go", "no", "hey", "oh", "all", "right", "i'm"}
    names = [w for w, n in caps.most_common(20) if w not in stop and n >= 5][:4]
    n_words = sum(len(s["words"]) for s in speech)
    print(f"{len(speech)} clips transcribed, {n_words} words; frequent names (candidates): {names}")
    events = []
    for s in speech:
        for e in find_requests(s["words"], names):
            e["clip"] = s["clip"]
            e["labelled"] = s["clip"] in rows
            events.append(e)
    praise = sum(1 for s in speech for t, text in phrases(s["words"]) if re.match(PRAISE, text))
    print(f"request words found: {len(events)} in {len({e['clip'] for e in events})} clips "
          f"({sum(e['labelled'] for e in events)} in clips with teacher labels); praise words: {praise}")
    print("  by request: " + ", ".join(f"{k} {v}" for k, v in Counter(e["req"] for e in events).most_common()))
    res = defaultdict(lambda: {"asked": 0, "seen": 0, "complied": 0, "lat": []})
    examples = []
    for e in events:
        if not e["labelled"]:
            continue
        rs = sorted(rows[e["clip"]], key=lambda r: r["t"])
        for dog in DOGS:
            before = regime_at(rs, dog, e["t"] - 0.5)
            after = [(r["t"] - e["t"], r[dog]) for r in rs if 0 < r["t"] - e["t"] <= 6 and (r.get(dog) or {}).get("visible")]
            key = (dog, e["req"])
            res[key]["asked"] += 1
            if before is None or not after:
                continue
            res[key]["seen"] += 1
            ok, lat = outcome(e["req"], before, after, dog)
            res[key]["complied"] += int(ok)
            if lat is not None:
                res[key]["lat"].append(lat)
            if len(examples) < 40:
                examples.append(f"{e['clip']} {e['t']:6.1f}s {dog}: '{e['req']}' before {before.get('support')}/"
                                f"{before.get('action')} -> {'did it' if ok else 'did not'}{f' after {lat:.1f}s' if lat else ''}")
    print("\nper dog and request (only where the dog is visible before and after, in labelled clips):")
    out = {}
    for (dog, req), v in sorted(res.items()):
        if v["seen"]:
            lat = f", median latency {np.median(v['lat']):.1f}s" if v["lat"] else ""
            print(f"  {dog:3s} {req:6s} seen {v['seen']:3d}  followed by the asked-for change {v['complied']:3d} "
                  f"({v['complied'] / v['seen']:.0%}){lat}")
        out[f"{dog}/{req}"] = {k: (v[k] if k != "lat" else [round(x, 2) for x in v[k]]) for k in v}
    print("\nexamples:")
    for x in examples[:25]:
        print("  " + x)
    json.dump({"names": names, "events": events, "results": out}, open(os.path.join(HERE, "models", "requests.json"), "w"),
              indent=1)


if __name__ == "__main__":
    {"transcribe": transcribe, "analyze": analyze}[sys.argv[1]]()
