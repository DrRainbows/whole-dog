"""Teacher labels from Gemini Carbon: which regime each dog is in, once per second.

  python label_gemini.py pilot        about 36 requests across varied clips
  python label_gemini.py all          every clip, one frame per second
  python label_gemini.py repeat N     relabel N already-labeled groups to measure the teacher's own consistency

Needs the Hermes proxy:  hermes proxy start --provider nous   (serves http://127.0.0.1:8645/v1)
Frames come from library/work/*.mp4. Labels append to library/labels/gemini.jsonl.
"""
import base64
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "library", "work")
OUT_DIR = os.path.join(HERE, "library", "labels")
OUT = os.path.join(OUT_DIR, "gemini.jsonl")
URL = "http://127.0.0.1:8645/v1/chat/completions"
MODEL = "google/gemini-4-carbon"
GROUP, STEP_S = 4, 1.0

SUPPORT = ["stand", "walk", "trot", "run", "jump", "sit", "down", "side", "back", "crouch", "rear_up", "swim", "wade",
           "unclear"]
ACTION = ["none", "sniff", "look", "carry", "tug", "mouth_play", "chew", "eat_drink", "lick", "pant", "unclear"]
CONTEXT = ["rest", "explore", "play_dog", "play_person", "retrieve", "eat_drink", "watch", "interact_person", "travel",
           "other"]

PROMPT = f"""These are {GROUP} frames from one home video, one second apart, in order.
Two particular dogs may appear:
- "lab": a cream/yellow Labrador Retriever
- "mal": a dark sable Belgian Malinois with a black mask and upright ears
Other dogs may also appear; count them but do not label them.

For EACH frame and EACH of lab and mal, report:
- visible: true or false
- box: [ymin, xmin, ymax, xmax] around the dog, integers 0-1000 (omit if not visible)
- support: what the legs and body are doing, one of {SUPPORT}
    stand = all feet down, not travelling; walk/trot/run = travelling at that gait; jump = airborne or leaping;
    sit = rump on the ground, front legs straight; down = lying on chest or belly (sphinx or relaxed);
    side = lying on one side; back = on its back or rolling, belly up; crouch = front end lowered (play bow, stalking);
    rear_up = up on hind legs; swim = swimming; wade = standing or walking in water
- action: what the head and mouth are doing, one of {ACTION}
    look = head up, attending to something; carry = holding an object in the mouth;
    tug = pulling on an object someone else holds; mouth_play = mouthing or play-biting the other dog
- context: what the dog is doing overall, one of {CONTEXT}
- contact: true if touching the other dog
Also for EACH frame: "up": which edge of the image the sky or ceiling is toward, one of ["top", "left", "right", "bottom"] (the camera is sometimes turned sideways).
Use the neighbouring frames to judge movement. Use "unclear" rather than guessing.

Answer ONLY with JSON:
{{"frames": [{{"i": 0, "up": "top", "lab": {{...}} or null, "mal": {{...}} or null, "other_dogs": 0}}, ...]}}"""

lock = threading.Lock()
spent = {"usd": 0.0, "calls": 0, "fails": 0}


def frames_at(video, times):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        s = 768 / max(h, w)
        if s < 1:
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append(base64.b64encode(buf.tobytes()).decode())
    cap.release()
    return out


def ask(images, temperature=0.2):
    content = [{"type": "text", "text": PROMPT}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}} for b in images]
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "temperature": temperature,
            "max_tokens": 6000, "reasoning": {"effort": "low"}}
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            text = data["choices"][0]["message"].get("content") or ""
            m = re.search(r"\{.*\}", text, re.S)
            parsed = json.loads(m.group(0)) if m else None
            with lock:
                spent["usd"] += float(data.get("usage", {}).get("cost") or 0)
                spent["calls"] += 1
            if parsed and "frames" in parsed:
                return parsed
            last = "unparseable: " + text[:200]
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code == 400 and "reasoning" in body:
                body.pop("reasoning")
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(4 * (attempt + 1))
    with lock:
        spent["fails"] += 1
    raise RuntimeError(last)


def done_keys():
    keys = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
                if not r.get("repeat"):
                    keys.add((r["clip"], round(r["t"], 2)))
            except Exception:  # noqa: BLE001
                pass
    return keys


def label_group(clip, times, repeat=False, temperature=0.2):
    video = os.path.join(WORK, clip + ".mp4")
    imgs = frames_at(video, times)
    if len(imgs) < len(times):
        times = times[:len(imgs)]
    if not imgs:
        return 0
    res = ask(imgs, temperature)
    rows = []
    for f in res.get("frames", []):
        i = f.get("i")
        if isinstance(i, int) and 0 <= i < len(times):
            rows.append({"clip": clip, "t": round(times[i], 2), "up": f.get("up", "top"), "lab": f.get("lab"), "mal": f.get("mal"),
                         "other_dogs": f.get("other_dogs", 0), "repeat": repeat, "model": MODEL})
    with lock:
        with open(OUT, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return len(rows)


def clip_groups(clip):
    cap = cv2.VideoCapture(os.path.join(WORK, clip + ".mp4"))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    dur = n / fps if n else 0
    ts = [round(0.5 + k * STEP_S, 2) for k in range(int((dur - 0.5) / STEP_S))]
    return [ts[i:i + GROUP] for i in range(0, len(ts), GROUP)]


def run(groups, workers=4, repeat=False, temperature=0.2):
    total = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(label_group, c, ts, repeat, temperature) for c, ts in groups]
        for k, f in enumerate(futs):
            try:
                total += f.result()
            except Exception as e:  # noqa: BLE001
                print("failed:", e, flush=True)
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{len(futs)} groups, {total} frames, ${spent['usd']:.2f}", flush=True)
    print(f"done: {total} frames from {len(groups)} groups; calls {spent['calls']}, failures {spent['fails']}, "
          f"cost ${spent['usd']:.2f}", flush=True)


def ready_clips():
    return sorted(f[:-4] for f in os.listdir(WORK) if f.endswith(".mp4") and not f.endswith(".part.mp4"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    done = done_keys()
    clips = ready_clips()
    if mode == "pilot":
        rng = random.Random(1)
        groups = []
        for c in clips[:]:
            gs = [g for g in clip_groups(c) if (c, g[0]) not in done]
            groups += [(c, g) for g in rng.sample(gs, min(3, len(gs)))]
        groups = rng.sample(groups, min(36, len(groups)))
        run(groups)
    elif mode == "all":
        groups = [(c, g) for c in clips for g in clip_groups(c) if (c, g[0]) not in done]
        run(groups, workers=6)
    elif mode == "watch":
        # label clips as they finish transcoding, until every clip in the inventory is done
        inv = json.load(open(os.path.join(HERE, "library", "inventory.json")))
        wanted = {os.path.splitext(r["file"])[0] for r in inv}
        while True:
            done = done_keys()
            labelled = {c for c, _ in done}
            groups = [(c, g) for c in ready_clips() for g in clip_groups(c) if (c, g[0]) not in done]
            if groups:
                run(groups, workers=6)
            elif wanted <= labelled:
                print("every clip labelled", flush=True)
                break
            else:
                time.sleep(60)
    elif mode == "repeat":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        seen = {}
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("repeat"):
                seen.setdefault(r["clip"], set()).add(r["t"])
        groups = []
        for c, ts in seen.items():
            ts = sorted(ts)
            groups += [(c, ts[i:i + GROUP]) for i in range(0, len(ts), GROUP)]
        groups = random.Random(2).sample(groups, min(n, len(groups)))
        run(groups, repeat=True, temperature=0.7)


if __name__ == "__main__":
    main()
