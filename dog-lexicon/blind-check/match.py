"""Match the blind lists against the lexicon, then test the leftovers against the modes.

Stage A: every blind name -> a lexicon id, NEW, or OUT (body-first rule).
Stage B: every NEW name -> a recipe in the 76 modes, or MISSING (needs a mode
         the table lacks), or OUT.
Results are cached, so a model that finishes later is added incrementally.

Usage: python match.py
"""
import importlib.util
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
LEX = HERE.parent
BLIND = HERE / "blind"
OUT = HERE / "match"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("blind_run", HERE / "blind_run.py")
_argv = sys.argv
sys.argv = ["blind_run.py", str(BLIND)]
blind_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blind_run)
sys.argv = _argv
CFG = blind_run.MODELS["deepseek"]
chat = blind_run.chat

BODY_FILES = ["01-postures.md", "03-locomotion.md", "04-head-mouth-nose.md",
              "05-ears-eyes-tail-coat.md", "06-limbs-whole-body.md", "07-voice.md", "09-body-upkeep.md",
              "13-added-from-blind-lists.md"]


def table_rows(path):
    for line in (LEX / path).read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if re.fullmatch(r"[a-z_]+", cells[0]) and cells[0] not in {"id", "request", "from", "part", "signal"}:
                yield cells


def catalogue():
    body, lines = set(), []
    for f in BODY_FILES:
        for c in table_rows(f):
            body.add(c[0])
            lines.append(f"{c[0]} | {c[1]}")
    for c in table_rows("08-sequences.md"):
        lines.append(f"seq_{c[0]} | sequence: {c[1]}")
    for c in table_rows("10-requests.md"):
        lines.append(f"ask_{c[0]} | trained request ({c[1]}): {c[2]}")
    return body, lines


def modes_table():
    modes, lines = set(), []
    for c in table_rows("11-modules.md"):
        if len(c) == 3 and re.fullmatch(r"[a-z]+", c[1]):
            modes.add(f"{c[0]}.{c[1]}")
            lines.append(f"{c[0]}.{c[1]} | {c[2]}")
    return modes, lines


def parse_items():
    items = []
    for model in ("deepseek", "qwen"):
        path = BLIND / f"{model}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        blocks = [("general", s["title"], s["text"]) for s in data["sections"]]
        blocks += [("targeted", t["key"], t["text"]) for t in data["targeted"]]
        seen = set()
        for kind, where, text in blocks:
            for line in text.splitlines():
                parts = [p.strip(" *`-") for p in line.split("|")]
                if len(parts) < 2 or len(parts[1]) < 12:
                    continue
                name = re.sub(r"^\d+[.)]\s*", "", parts[0]).strip()
                key = (model, name.lower())
                if not name or key in seen:
                    continue
                seen.add(key)
                items.append({"model": model, "kind": kind, "where": where, "name": name, "body": parts[1]})
    return items


def key(it):
    return f"{it['model']}|{it['name']}|{it['body']}"


def run_cached(cache_path, todo_items, make_prompt, parse_line, batch_size, max_tokens):
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for _ in range(2):  # second pass picks up lines the model skipped
        todo = [it for it in todo_items if key(it) not in cache]
        if not todo:
            break

        def work(start):
            batch = todo[start:start + batch_size]
            listing = "\n".join(f"{start + i} | {it['name']} | {it['body']}" for i, it in enumerate(batch))
            text = chat(CFG, make_prompt(listing), temperature=0.0, max_tokens=max_tokens)
            found = {}
            for line in text.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if parts and parts[0].isdigit() and start <= int(parts[0]) < start + len(batch) and len(parts) >= 2:
                    found[int(parts[0])] = parse_line(parts[1:])
            return found

        with ThreadPoolExecutor(max_workers=6) as pool:
            for found in pool.map(work, range(0, len(todo), batch_size)):
                for idx, value in found.items():
                    if value is not None:
                        cache[key(todo[idx])] = value
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def main():
    body_ids, cat_lines = catalogue()
    valid_ids = {ln.split(" | ")[0] for ln in cat_lines}
    modes, mode_lines = modes_table()
    items = parse_items()
    print("names:", {m: sum(1 for i in items if i["model"] == m) for m in ("deepseek", "qwen")}, flush=True)

    # ---- Stage A
    cat_text = "\n".join(cat_lines)

    def prompt_a(listing):
        return (
            "Below is a catalogue of dog behaviors, one per line as `id | definition`. After it is a numbered list of "
            "behaviors from another source.\n\nFor each numbered item decide:\n"
            "- the catalogue id, if the catalogue already has what the body is doing. The catalogue defines entries "
            "by the body, so the same action in another place, on another object, toward another target, at another "
            "age or for another reason is the SAME entry (sleeping on a roof is sleep; barking at the mail carrier "
            "is bark_alert; licking a wound is lick_self). Pick the entry that carries the main action.\n"
            "- NEW, only if the body does something that no catalogue definition describes.\n"
            "- OUT, if the item is not a visible action of the dog's body (a fact about breeds, an opinion, a human's "
            "action, a symptom with no action, the absence of a behavior).\n\n"
            "Answer with one line per item, `number | id-or-NEW-or-OUT`, and nothing else.\n\n"
            f"CATALOGUE\n{cat_text}\n\nITEMS\n{listing}"
        )

    def parse_a(parts):
        label = parts[0].split()[0] if parts[0] else ""
        return label if label in valid_ids or label in {"NEW", "OUT"} else "NEW"

    labels = run_cached(OUT / "labels.json", items, prompt_a, parse_a, 50, 4000)
    for it in items:
        it["label"] = labels.get(key(it), "UNMAPPED")

    # ---- Stage B
    mode_text = "\n".join(mode_lines)

    def prompt_b(listing):
        return (
            "Below is a parts list for a dog: every body part and the modes it has, one per line as `part.mode | what "
            "it is`. Every part also receives four manner signals (big or small, toward or away, stiff or loose, fast "
            "or slow) and a target (none, self, object, place, person, dog, prey, scent, sound, two).\n\n"
            "After the parts list is a numbered list of dog behaviors. Write each one using only modes from the list.\n"
            "- If the listed modes can express what the body does: `number | OK | modes separated by spaces | target`.\n"
            "- If the body does something no listed mode can express: `number | MISSING | part.mode you would add | "
            "what it is, in a few words`.\n"
            "- If the item is not a visible action of the dog's body (a symptom with no action, a trait, a human's "
            "action, the absence of a behavior): `number | OUT`.\n"
            "Do not stretch a mode to cover something it does not describe. One line per item, nothing else.\n\n"
            f"PARTS LIST\n{mode_text}\n\nITEMS\n{listing}"
        )

    def parse_b(parts):
        verdict = parts[0].upper()
        if verdict == "OUT":
            return {"v": "OUT"}
        if verdict == "OK" and len(parts) >= 2:
            used = [m for m in parts[1].split() if "." in m]
            unknown = [m for m in used if m not in modes]
            if unknown:
                return {"v": "MISSING", "mode": unknown[0], "what": "used an undefined mode"}
            return {"v": "OK", "modes": sorted(used), "target": parts[2] if len(parts) > 2 else "-"}
        if verdict == "MISSING" and len(parts) >= 2:
            return {"v": "MISSING", "mode": parts[1], "what": parts[2] if len(parts) > 2 else ""}
        return None

    new_items = [it for it in items if it["label"] == "NEW"]
    recipes = run_cached(OUT / "recipes.json", new_items, prompt_b, parse_b, 40, 5000)
    for it in new_items:
        it["recipe"] = recipes.get(key(it), {"v": "UNMAPPED"})

    (OUT / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- Report
    for model in ("deepseek", "qwen"):
        mine = [i for i in items if i["model"] == model]
        if not mine:
            continue
        matched = [i for i in mine if i["label"] in valid_ids]
        body_hit = {i["label"] for i in matched if i["label"] in body_ids}
        new = [i for i in mine if i["label"] == "NEW"]
        verdicts = Counter(i["recipe"]["v"] for i in new)
        ok = [i for i in new if i["recipe"]["v"] == "OK"]
        shapes = {tuple(i["recipe"]["modes"]) for i in ok}
        shapes_t = {(tuple(i["recipe"]["modes"]), i["recipe"]["target"]) for i in ok}
        out = sum(1 for i in mine if i["label"] == "OUT") + verdicts.get("OUT", 0)
        actions = len(mine) - out
        missing = verdicts.get("MISSING", 0)
        print(f"\n== {model} ==")
        print(f"names {len(mine)}; not a body action {out}; body actions {actions}")
        print(f"already in the lexicon: {len(matched)} names on {len(body_hit)} of {len(body_ids)} body entries "
              f"(+ sequences and requests)")
        print(f"not in the lexicon: {len(new)} -> expressible in the existing modes {len(ok)}, needs a new mode {missing}, "
              f"not an action {verdicts.get('OUT', 0)}, unjudged {verdicts.get('UNMAPPED', 0)}")
        print(f"expressible leftovers use {len(shapes)} distinct mode combinations ({len(shapes_t)} with target)")
        print(f"share of body actions the modes can express: {100 * (len(matched) + len(ok)) / max(actions, 1):.1f}%")
        proposed = Counter(i["recipe"]["mode"].lower() for i in new if i["recipe"]["v"] == "MISSING")
        print("most proposed new modes:", proposed.most_common(40))
        print("my body entries it never listed:", sorted(body_ids - body_hit))


if __name__ == "__main__":
    main()
