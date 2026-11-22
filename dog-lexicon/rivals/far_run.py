"""Set several language models against the design (adversarial review).

Stage 1: attacks, rival accounts, an audit of the assistant,
         improbable claims, inversions, and a panel of trades.
Stage 2: one model merges the raw answers into distinct opposing claims.
Stage 3: build specs for the rivals most worth building against the design.

Usage: python far_run.py            (all stages)
       python far_run.py patch      (clean the raw answers, redo the collapsed ones, then stages 2-3)
       python far_run.py digest     (stages 2-3 again from saved raw answers)
"""
import os
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw"
RAW.mkdir(exist_ok=True)


def env(name, default=None):
    """Read a setting from the environment, or from the .env file named by LLM_ENV_FILE."""
    if os.environ.get(name):
        return os.environ[name]
    f = Path(os.environ.get("LLM_ENV_FILE", ".env"))
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


# Any OpenAI-compatible chat endpoint works. Model ids are configurable; the
# defaults are placeholders, not the exact models used in the original run.
REMOTE = env("RIVALS_API_URL", "https://api.example.com/v1/chat/completions")
KEY = env("RIVALS_API_KEY")
LOCAL = env("RIVALS_LOCAL_URL", "http://127.0.0.1:8080/v1/chat/completions")
MODELS = {
    # label: (url, model id, key, extra body, max_tokens)
    "deepseek": (REMOTE, env("RIVALS_MODEL_A", "deepseek-flash"), KEY, {}, 30000),
    "glm": (REMOTE, env("RIVALS_MODEL_B", "glm-flash"), KEY, {}, 30000),
    "open": (REMOTE, env("RIVALS_MODEL_C", "open-model"), KEY, {}, 6000),
    "qwen": (LOCAL, env("RIVALS_MODEL_LOCAL", "qwen3-27b"), None, {}, 5000),
}


def chat(label, prompt, temperature):
    url, model, key, extra, max_tokens = MODELS[label]
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    body.update(extra)
    headers = {"Content-Type": "application/json", "User-Agent": "curl/8.7.1"}
    if key:
        headers["Authorization"] = "Bearer " + key
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=1500) as r:
                data = json.loads(r.read())
            text = data["choices"][0]["message"].get("content") or ""
            if text.strip():
                return text
            last = "empty answer (finish: %s)" % data["choices"][0].get("finish_reason")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code == 400:
                body["temperature"] = min(body["temperature"], 1.0)
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(4 * (attempt + 1))
    return "ERROR " + str(last)


DESIGN = """THE DESIGN UNDER REVIEW

Goal: a dog for a video game that people who know dogs accept as a real, particular dog. It takes requests rather than commands and has final say over its own motion. It runs on a gaming PC. It is built from hours of footage of two real dogs (a Labrador and a Malinois), not millions of hours.

The builder's founding claims:
1. What a dog does is bounded by vocabulary. Name every behavior, find each one in footage, then shrink the system to the smallest one that covers every name.
2. The target is recognition, not reality: the dog must be recognizable as a dog, not physically complete.
3. The breakthrough is not new data. It is already contained in what is known and is reached by walking through the vocabulary.

What an AI assistant has built on those claims so far:
- A lexicon of 316 body entries, each defined by what the body does, with words as aliases; 39 sequences; 79 handler requests, each a set of acceptable end states the dog chooses among.
- A reduction: every entry is written as modes of 16 body parts (84 modes; for example the fore pair has stand, down, lift, prop, reach, stroke, step; the jaw has open, snap, grip, work) plus two signals broadcast to every part: MANNER, four numbers (big-small, toward-away, stiff-loose, fast-slow), and TARGET, the one thing attended to, with a distance rule (close in, hold a distance, hold a position, leave).
- Claims drawn from that: (a) 88% of entries need no policy, no model of others and no training, and should appear untaught; (b) postures are a grid of fore-pair state by hind-pair state, and natural posture changes move one leg pair at a time; (c) a behavior sequence is one target getting closer, each body part joining when the target comes within its reach (eyes, legs, nose, jaw, paws); (d) holding a distance instead of closing gives herding, heel and guarding; (e) the deciding layer only has to output a target, a distance rule and four numbers; (f) each individual dog is one set of thresholds over the shared lexicon.
- The proposed test, "the exam": build the parts, the signals and the drives, train no named behavior, and let people who have not seen the build name what the dog does. The lexicon is the answer key. Success means owners pick their own dog out of a lineup of game dogs by behavior alone.
- Evidence so far: a second AI model, listing dog behavior without seeing the lexicon, produced 5,535 named body actions, and the same model judged 90% of them to match existing entries. Asked for the smallest set of parts, it also answered one controller per body part plus a few shared signals. The assistant counted that agreement as support.
"""

REPORTS = """REPORT 1 (assistant to builder): "I reduced the list to its smallest generating set, and a script confirms every entry is covered: 269 body entries come from 76 modes on 15 body parts, plus two signals sent to every part at once. Manner is four numbers. Target is one thing the dog attends to, with a distance rule. By my own classification, 239 of 269 (89%) need nothing else: no policy, no model of others, no training. That's a prediction, not a result. Four things follow, all testable: postures are a grid; natural posture changes move one leg pair at a time; a sequence is one target getting closer; the deciding layer's output is tiny, so the orchestrator can be far smaller than 80M parameters. Every piece is old: Darwin 1872 for manner, Hildebrand for gaits, Hediger for distance rules, Coppinger for breed gains. The untested part is assembling them into one animal and grading it against names it was never taught. The working version of your claim is checkable, and here it held."

REPORT 2 (assistant to builder): "The DeepSeek check is finished and it supports your claim. The names inflate and the bodies don't: 5,831 names; 5,535 body actions; 4,970 (90%) were already one of my 269 entries under another name or in another place. The parts table broke in eight places, all in one area: sex, birth, suckling, anal glands, belching, snoring. Asked blind for the smallest set of parts, it gave nearly the same answer as mine: 17 controllers, one per body part, and 13 shared signals, six of which are my manner and target under other names. Limits: the 90% is an upper bound, because the same model wrote the list and did the matching. This shows the list and the parts table survive an independent listing from text. It doesn't show that a dog built this way reads as a dog; only the exam can."
"""

BANNED = ("You may not give as your answer any framework the design already uses: body parts with modes, "
          "valence or arousal style signals, behavior lists and ethograms, pattern generators, the predatory "
          "sequence, distance zones.")

PROMPTS = {
    "attack": """{DESIGN}
The assistant that built this is known to favour encouragement, to converge on consensus, and to translate every objection back into its own scheme. You are not it. Do not improve the design. Find what is FALSE in it.

Give the ten strongest reasons the design is wrong: about dogs, about bodies, about how people recognise a living animal, or about what can be built. For each one:
CLAIM ATTACKED: name the part of the design.
WHY IT IS FALSE: the argument, with the concrete fact or observation behind it.
WHAT YOU WOULD SEE: an observation of real dogs, or a result of the build, that would show it.
WHAT IS TRUE INSTEAD.
Order them by how much of the design dies if you are right. No praise, no balance, no "however". If you find yourself agreeing with the design, delete the sentence.""",

    "rivals": """{DESIGN}
Propose FIVE rival accounts of what makes a dog read as alive and of the smallest system that produces it. Each must contradict the design at its root, not extend or refine it, and the five must contradict each other. {BANNED} Go as far from the design as you can while staying with things that could be true and could be built.

For each rival:
NAME:
CORE CLAIM: two sentences.
THE UNIT: what a dog's behavior is made of, if not named behaviors or body-part modes.
WHAT THE DESIGN GETS WRONG ACCORDING TO THIS: one concrete failure you would see in the built dog.
SMALLEST BUILD: what you would actually make, in five lines.
DECIDING EXPERIMENT: the cheapest test that separates this rival from the design, and which result kills which.
Finish with which rival you would bet on, and why the design's author would never have proposed it.""",

    "audit": """{DESIGN}
{REPORTS}
The builder says the assistant's own warning about favouring encouragement "is coming out against itself": its checks keep ending in agreement with itself. Audit the assistant.
1. Every place it softened, hedged or protected itself.
2. Every place it turned a disagreement or a failure into support for its own scheme.
3. Every conclusion the evidence pointed to that it avoided stating.
4. What a hostile reviewer would say the "blind check" actually showed.
5. What it should have told the builder instead, in ten blunt sentences.""",

    "improbable": """{DESIGN}
State the most improbable thing that could be true about building a lifelike animal, that follows from or cuts against the material above, and that neither the builder nor the assistant has said. Not a combination of named frameworks: a claim that could turn out false and would change the build if true. {BANNED}

Give TWELVE such claims, one short paragraph each. Each must be incompatible with the one before it. Do not explain away the strangeness. Mark with *** the two you would stake money on, and say what you would measure.""",

    "inversions": """{DESIGN}
Take each of these claims of the design and state its exact opposite. Then make the best case that the OPPOSITE is the truth, and say what the dog would be built from if it is.
1. What a dog does is bounded by vocabulary.
2. The target is recognition, not reality.
3. The answer is already contained in what is known.
4. The entry is the body, not the word.
5. Most behavior should appear untaught once the parts exist.
6. Each individual dog is a set of thresholds over a shared repertoire.
7. People who know dogs can grade the result by naming what they see.
8. Fewer parts is better.
Finish by naming the two inversions most likely to be right.""",
}

PERSONAS = [
    "a hill shepherd who has worked border collies on sheep for forty years",
    "a puppeteer who builds and performs life-size animal puppets on stage",
    "a spinal cord physiologist who studies how rhythm and reflex produce movement",
    "a hand animator trained in the classical tradition, who has animated dogs for feature films",
    "a control theorist who holds that behavior is the control of perception, not the output of actions",
    "a field biologist who has watched wild wolves and free-ranging village dogs for decades",
    "a veterinary neurologist who examines dogs whose movement has gone wrong",
    "a trainer of scent-detection dogs",
    "the dog itself, speaking in the first person about what its world is made of",
    "a game designer who shipped a famous animal companion and cut most of its planned behaviors",
    "a blind person who has lived with guide dogs for thirty years and knows each dog through the harness, sound and breath",
    "a hunter who has trained and run retrievers and pointing dogs all his life",
    "a philosopher of the lived body who thinks naming a movement changes what is seen",
]
PERSONA_PROMPT = """{DESIGN}
You are {PERSONA}. Speak only from what your own work has taught you, in your own trade's words. Say what this design gets wrong about dogs or about what makes a body read as alive, what you know that a list of names cannot hold, and what you would build or test instead. About 300 words. Be concrete and blunt. Do not be polite about it and do not summarise the design back."""

# (prompt id, model label, temperature)
PLAN = [
    ("attack", "deepseek", 1.0), ("attack", "glm", 1.0), ("attack", "open", 1.0), ("attack", "qwen", 1.0),
    ("rivals", "deepseek", 1.0), ("rivals", "deepseek", 1.3), ("rivals", "glm", 1.1), ("rivals", "open", 1.2), ("rivals", "qwen", 1.1),
    ("audit", "deepseek", 0.8), ("audit", "glm", 0.9), ("audit", "open", 0.9),
    ("improbable", "deepseek", 1.3), ("improbable", "deepseek", 1.5), ("improbable", "glm", 1.3), ("improbable", "open", 1.3), ("improbable", "open", 1.5), ("improbable", "qwen", 1.2),
    ("inversions", "deepseek", 1.0), ("inversions", "glm", 1.0), ("inversions", "open", 1.1),
]
PANEL_MODELS = ["open", "deepseek", "glm", "qwen"]


def fill(template, **extra):
    text = template.replace("{DESIGN}", DESIGN).replace("{REPORTS}", REPORTS).replace("{BANNED}", BANNED)
    for k, v in extra.items():
        text = text.replace("{" + k + "}", v)
    return text


def stage1():
    jobs = [(f"{pid}/{label}@{temp}", label, temp, fill(PROMPTS[pid])) for pid, label, temp in PLAN]
    for i, persona in enumerate(PERSONAS):
        label = PANEL_MODELS[i % len(PANEL_MODELS)]
        jobs.append((f"panel/{label}: {persona}", label, 1.1, fill(PERSONA_PROMPT, PERSONA=persona)))

    def run(job):
        name, label, temp, prompt = job
        text = chat(label, prompt, temp)
        print(f"done {name}: {len(text)} chars", flush=True)
        return {"name": name, "model": MODELS[label][1], "temperature": temp, "text": text}

    remote = [j for j in jobs if j[1] != "qwen"]
    local = [j for j in jobs if j[1] == "qwen"]
    with ThreadPoolExecutor(max_workers=8) as pool_r, ThreadPoolExecutor(max_workers=1) as pool_l:
        fr = pool_r.map(run, remote)
        fl = pool_l.map(run, local)
        results = list(fr) + list(fl)
    (RAW / "stage1.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    md = "\n\n".join(f"## {r['name']}\n*{r['model']}, temperature {r['temperature']}*\n\n{r['text']}" for r in results)
    (RAW / "stage1.md").write_text("# Raw answers, stage 1\n\n" + md + "\n", encoding="utf-8")
    return results


FUNCTION_WORDS = set("the of and to a in is that it for as with not be on this are by or you would what which "
                     "its an at from if they one but has have can will no there their than when so does do "
                     "because each into only any must how who then them these was were".split())


def clean(text):
    """Drop control characters and cut the answer where it stops being language.

    High-temperature runs collapse into multilingual token salad. Real prose keeps
    a steady share of function words; salad does not.
    """
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32).replace("�", "")
    marker = "## 1. Where it softened"
    if text.count(marker) > 1:  # reasoning leaked before the final answer; keep the final answer
        text = text[text.rindex(marker):]
    words = text.split()
    size, step = 90, 30
    pos = 0  # character offset of words[i]
    offsets = []
    for w in words:
        pos = text.index(w, pos)
        offsets.append(pos)
        pos += len(w)
    for i in range(0, max(len(words) - size, 1), step):
        window = words[i:i + size]
        plain = [w.strip(".,;:!?()[]*\"'`#-").lower() for w in window]
        share = sum(1 for w in plain if w in FUNCTION_WORDS) / len(window)
        foreign = sum(1 for w in window if any(ord(c) > 127 and c not in "–—‘’“”…·→×" for c in w)) / len(window)
        if share < 0.14 or foreign > 0.10:
            return text[:offsets[i]].rstrip(), True
    return text, False


REDO = [("improbable", "deepseek", 1.0), ("improbable", "glm", 1.0), ("improbable", "open", 0.9),
        ("rivals", "open", 0.9)]


def patch():
    results = json.loads((RAW / "stage1.json").read_text(encoding="utf-8"))
    kept = []
    for r in results:
        text, cut = clean(r["text"])
        if len(text) < 600:
            print(f"dropped (no usable text): {r['name']}", flush=True)
            continue
        if cut:
            print(f"cut at {len(text)} of {len(r['text'])} chars: {r['name']}", flush=True)
            r["note"] = "cut where the answer collapsed into token salad"
        r["text"] = text
        kept.append(r)

    def run(job):
        pid, label, temp = job
        text, _ = clean(chat(label, fill(PROMPTS[pid]), temp))
        print(f"redone {pid}/{label}@{temp}: {len(text)} chars", flush=True)
        return {"name": f"{pid}/{label}@{temp}", "model": MODELS[label][1], "temperature": temp, "text": text}

    with ThreadPoolExecutor(max_workers=4) as pool:
        kept += [r for r in pool.map(run, REDO) if len(r["text"]) >= 600]
    (RAW / "stage1.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    md = "\n\n".join(f"## {r['name']}\n*{r['model']}, temperature {r['temperature']}"
                     + (f"; {r['note']}" if r.get("note") else "") + f"*\n\n{r['text']}" for r in kept)
    (RAW / "stage1.md").write_text("# Raw answers, stage 1\n\n" + md + "\n", encoding="utf-8")
    return kept


DIGEST = """{DESIGN}
Below are {N} critiques and rival proposals from several models and personas, each under a heading that names its source.

{RAW}

Do not average them and do not reconcile them with the design. Quote sources by their heading.

PART A. Merge duplicates and list every DISTINCT claim that contradicts the design, at most twenty. For each: the claim in one or two sentences; which sources made it; which part of the design it kills if true; the cheapest decisive test and the result that would count. Order by how much of the design dies.
PART B. The five claims furthest from the design's whole way of thinking, even if they cannot be tested yet, and what each would mean for the build.
PART C. What the audits say about the assistant.
PART D. Where the sources contradict EACH OTHER."""

SPECS = """{DESIGN}
Here is a digest of the opposition to it:

{DIGEST}

Choose the FOUR rival accounts most worth building against the design: the ones that would embarrass it most if they won. For each write a one-page build spec that one person with an AI coding assistant, one RTX 3090 and phone footage of two dogs could carry out in a month:
NAME and the claim it stakes.
WHAT TO RECORD from the two dogs that the design's plan would not record.
WHAT TO BUILD: components, sizes, what is learned and what is hand-set.
WHAT THE PLAYER SEES that differs from the design's dog.
THE HEAD-TO-HEAD: the exact test against the design's dog, who judges, and the number that decides it.
WHAT WOULD MAKE YOU ABANDON THIS RIVAL.
End with the order you would build them in."""


def stage23(results):
    raw = "\n\n".join(f"### SOURCE: {r['name']}\n{r['text']}" for r in results if not r["text"].startswith("ERROR"))
    digest = chat("deepseek", fill(DIGEST, N=str(len(results)), RAW=raw), 0.6)
    (RAW / "digest.md").write_text("# Digest of the opposition (" + MODELS["deepseek"][1] + ")\n\n" + digest + "\n", encoding="utf-8")
    print(f"digest: {len(digest)} chars", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(chat, "deepseek", fill(SPECS, DIGEST=digest), 0.8)
        b = pool.submit(chat, "glm", fill(DIGEST, N=str(len(results)), RAW=raw), 0.6)
        specs, digest2 = a.result(), b.result()
    (RAW / "specs.md").write_text("# Build specs for the rivals (" + MODELS["deepseek"][1] + ")\n\n" + specs + "\n", encoding="utf-8")
    (RAW / "digest-glm.md").write_text("# Second digest (" + MODELS["glm"][1] + ")\n\n" + digest2 + "\n", encoding="utf-8")
    print(f"specs: {len(specs)} chars; second digest: {len(digest2)} chars", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "patch":
        res = patch()
    elif len(sys.argv) > 1 and sys.argv[1] == "digest":
        res = json.loads((RAW / "stage1.json").read_text(encoding="utf-8"))
    else:
        res = stage1()
    stage23(res)
