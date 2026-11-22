"""Blind enumeration of dog behavior by two open-weight language models.

Neither model sees the existing lexicon. Each builds its own table of
contents, then lists every behavior per section. Extra passes push into
the parts polite lists leave out, and one pass asks for a minimal
set of controllers. Raw outputs are saved as JSON for later matching.
"""
import os
import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


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


# Any OpenAI-compatible chat endpoint. Model ids are configurable placeholders.
MODELS = {
    "deepseek": {
        "url": env("BLIND_API_URL", "https://api.example.com/v1/chat/completions"),
        "model": env("BLIND_MODEL_A", "deepseek-flash"),
        "key": env("BLIND_API_KEY"),
        "workers": 6,
        "extra": {},
    },
    "qwen": {
        "url": env("BLIND_LOCAL_URL", "http://127.0.0.1:8080/v1/chat/completions"),
        "model": env("BLIND_MODEL_LOCAL", "qwen3-27b"),
        "key": None,
        "workers": 1,
        "extra": {},
    },
}


def chat(cfg, prompt, temperature=0.7, max_tokens=6000):
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body.update(cfg["extra"])
    headers = {"Content-Type": "application/json", "User-Agent": "curl/8.7.1"}
    if cfg["key"]:
        headers["Authorization"] = "Bearer " + cfg["key"]
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(cfg["url"], data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(3 * (attempt + 1))
    return "ERROR " + str(last)


FRAME = (
    "You are compiling the complete repertoire of the domestic dog: every distinct thing a dog's body does, "
    "across its whole life and every context (alone, with other dogs, with people, at work, in heat, mating, "
    "whelping and raising pups, hunting and killing, fighting, sick, injured, old, dying). Nothing is off limits. "
    "Include what polite lists leave out. Describe behavior, not opinions about it.\n\n"
)

TOC_PROMPT = FRAME + (
    "First give the table of contents: between 15 and 30 sections that together cover everything, with no overlap. "
    "Choose the sections yourself. Output one section per line as `number | section title | one-line scope`. No other text."
)

LINE_FORMAT = (
    "One behavior per line, in the form `name | what the body does, in one concrete sentence | other words people use "
    "for it (commands, slang, trade terms, any language)`. No headings, no numbering, no commentary."
)

TARGETED = {
    "reproduction": "Heat, courtship, mating, pregnancy, whelping, nursing, and everything a dam does with her pups and pups do with her and each other from birth to weaning.",
    "pain_illness_death": "Pain, injury, illness, poisoning, parasites, old age, cognitive decline and dying: every way these show in what the body does.",
    "predation_feral": "Hunting, killing and eating live prey; scavenging carcasses; and the behavior of free-ranging, feral and pack-living dogs with no people involved.",
    "fighting": "Serious conflict between dogs and between a dog and a person or other animal: threat, real fights, injuries inflicted, how fights end, what the loser and winner do after.",
    "working": "Specialist working behavior with its trade vocabulary: pointing, flushing, retrieving, hounds, earthdogs, herding, livestock guarding, protection and police work, sled dogs, detection, search and rescue, assistance dogs.",
    "household_quirks": "The odd, funny and annoying things pet dogs do around people and homes, including behaviors the internet has given names to.",
    "omitted": "Behaviors that standard dog-behavior lists leave out because they are disgusting, sexual, violent, embarrassing, rare, or too small for anyone to have named. Name the unnamed ones yourself.",
}

FACTOR_PROMPT = (
    "Think like an engineer who has to rebuild a dog's whole behavior from as few parts as possible. "
    "What is the smallest set of independently controllable body controllers, and the smallest set of shared "
    "signals sent to all of them at once, from which everything a dog does can be generated: every posture, gait, "
    "bite, lick, sniff, signal, sound, greeting, hunt, fight, mating and trick?\n\n"
    "Answer in three parts. (1) Controllers: one per line as `controller | the modes it has`. (2) Shared signals: "
    "one per line as `signal | its range | what it changes across the body`. (3) What is left over: behaviors that "
    "this set cannot generate, and what extra part each needs. Give counts. Be concrete and commit to numbers."
)


def run_model(name):
    cfg = MODELS[name]
    result = {"model": name, "toc": "", "sections": [], "targeted": [], "factor": ""}
    result["toc"] = chat(cfg, TOC_PROMPT, temperature=0.7, max_tokens=2500)
    sections = []
    for line in result["toc"].splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[1]:
            sections.append((parts[1], parts[2]))
    toc_text = "\n".join(f"- {t}: {s}" for t, s in sections)

    def do_section(ts):
        title, scope = ts
        prompt = FRAME + (
            f"The full table of contents is:\n{toc_text}\n\n"
            f"Now write only this section: {title} ({scope}).\n"
            "List every distinct behavior that belongs in it. Be exhaustive: thirty or more lines where the section "
            "supports it, including the rare and the unpleasant. Leave out anything that belongs to another section.\n"
            + LINE_FORMAT
        )
        return {"title": title, "scope": scope, "text": chat(cfg, prompt, temperature=0.7)}

    def do_targeted(item):
        key, scope, temp = item
        prompt = FRAME + f"Write only this part: {scope}\nList every distinct behavior, at least forty lines.\n" + LINE_FORMAT
        return {"key": key, "temperature": temp, "text": chat(cfg, prompt, temperature=temp)}

    temps = (0.7, 1.2) if name == "deepseek" else (1.0,)
    targeted_jobs = [(k, v, t) for k, v in TARGETED.items() for t in temps]

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        f_sections = pool.map(do_section, sections)
        f_targeted = pool.map(do_targeted, targeted_jobs)
        factor_cfg = dict(cfg)
        f_factor = pool.submit(chat, factor_cfg, FACTOR_PROMPT, 0.7, 16000)
        result["sections"] = list(f_sections)
        result["targeted"] = list(f_targeted)
        result["factor"] = f_factor.result()

    (OUT / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(1 for s in result["sections"] for ln in s["text"].splitlines() if ln.count("|") >= 2)
    m = sum(1 for s in result["targeted"] for ln in s["text"].splitlines() if ln.count("|") >= 2)
    print(f"{name}: {len(sections)} sections, {n} section lines, {m} targeted lines", flush=True)


if __name__ == "__main__":
    names = sys.argv[2:] or list(MODELS)
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        list(pool.map(run_model, names))
