"""Check 12-recipes.md against the lexicon and 11-modules.md, and count the result.

Run from anywhere:  python check_recipes.py
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
BODY_FILES = [
    "01-postures.md", "03-locomotion.md", "04-head-mouth-nose.md",
    "05-ears-eyes-tail-coat.md", "06-limbs-whole-body.md", "07-voice.md", "09-body-upkeep.md",
    "13-added-from-blind-lists.md",
]
MANNER = {"big", "small", "toward", "away", "stiff", "loose", "fast", "slow"}
TARGETS = {"-", "none", "self", "object", "place", "person", "dog", "prey", "scent", "sound", "two", "any"}
CLASSES = {
    "P": "modes, manner and target only",
    "X": "physics",
    "R": "reflex trigger",
    "D": "policy over time",
    "O": "estimate of another's attention or intent",
    "L": "trained to a cue",
}


def rows(path):
    """Yield the cells of every table row that is not a header or a separator."""
    for line in (HERE / path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] not in {"id", "part", "signal", "from", "request", "file", "column", ""} \
                and not cells[0].startswith("**"):
            yield cells


def main():
    lexicon = {}
    for f in BODY_FILES:
        for cells in rows(f):
            if re.fullmatch(r"[a-z_]+", cells[0]):
                lexicon[cells[0]] = f

    modes = {f"{c[0]}.{c[1]}" for c in rows("11-modules.md")
             if len(c) == 3 and re.fullmatch(r"[a-z]+", c[0]) and re.fullmatch(r"[a-z]+", c[1])}
    parts = {m.split(".")[0] for m in modes}

    recipes = {}
    problems = []
    for cells in rows("12-recipes.md"):
        if len(cells) != 5 or not re.fullmatch(r"[a-z_]+", cells[0]):
            continue
        rid, mode_text, manner_text, target, cls = cells
        if rid in recipes:
            problems.append(f"duplicate recipe: {rid}")
        used = [] if mode_text == "-" else mode_text.split()
        manner = [] if manner_text == "-" else manner_text.split()
        for m in used:
            if m not in modes:
                problems.append(f"{rid}: unknown mode {m}")
        for w in manner:
            if w not in MANNER:
                problems.append(f"{rid}: unknown manner {w}")
        if target not in TARGETS:
            problems.append(f"{rid}: unknown target {target}")
        if cls not in CLASSES:
            problems.append(f"{rid}: unknown class {cls}")
        recipes[rid] = (tuple(sorted(used)), tuple(manner), target, cls)

    for rid in lexicon.keys() - recipes.keys():
        problems.append(f"no recipe: {rid} ({lexicon[rid]})")
    for rid in recipes.keys() - lexicon.keys():
        problems.append(f"recipe for unknown entry: {rid}")

    print(f"body entries: {len(lexicon)}   recipes: {len(recipes)}")
    print(f"parts: {len(parts)}   modes defined: {len(modes)}")
    used_modes = Counter(m for r in recipes.values() for m in r[0])
    print(f"modes used: {len(used_modes)}   unused: {sorted(modes - used_modes.keys())}")

    by_class = Counter(r[3] for r in recipes.values())
    print("\nclass counts:")
    for cls, label in CLASSES.items():
        n = by_class.get(cls, 0)
        print(f"  {cls}  {n:3d}  {100 * n / len(recipes):4.1f}%  {label}")
    free = sum(by_class.get(c, 0) for c in "PXR")
    print(f"  P+X+R = {free} of {len(recipes)} ({100 * free / len(recipes):.0f}%) need no policy, no model of others, no training")

    shapes = defaultdict(list)
    for rid, r in recipes.items():
        if r[0]:
            shapes[r[0]].append(rid)
    shared = {k: v for k, v in shapes.items() if len(v) > 1}
    print(f"\ndistinct mode combinations: {len(shapes)}")
    print(f"combinations shared by several entries: {len(shared)}, covering {sum(len(v) for v in shared.values())} entries")
    for k, v in sorted(shared.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"  {' '.join(k)}: {', '.join(v)}")

    print(f"\nproblems: {len(problems)}")
    for p in problems:
        print("  " + p)


if __name__ == "__main__":
    main()
