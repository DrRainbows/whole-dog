"""The dog's grammar: which words name which bodies, which bodies may follow which, and what every link costs.

  body\\.venv\\Scripts\\python dog\\grammar.py build                     compile dog-lexicon/*.md into dog/grammar.json
  body\\.venv\\Scripts\\python dog\\grammar.py audit                     the posture graph written from words, checked against
                                                                    the capture and the two dogs
  body\\.venv\\Scripts\\python dog\\grammar.py price "platz jump" [--dog mal] [--from stand] [--hot] [--tired]
  body\\.venv\\Scripts\\python dog\\grammar.py generate 8 [--mode endemic|social] [--dog lab] [--tau 1.0]

Words are aliases; an entry is a body (dog-lexicon/README.md, rule 1). Every entry is written as modes of body
parts with a manner and a target (11-modules.md, 12-recipes.md), and classed as
  reflexive: R (one trigger from the body) or X (physics acting on the parts): never chosen, only triggered;
  planned:   D (a policy over time), O (an estimate of someone else), L (a trained cue), and every sequence
             and request;
  state:     P (modes, manner and target are enough): held or expressed, chosen by drives.
A link from one word to the next is priced in bits (-log2 of a probability), from four sources kept apart:
  body     can the body get there, and by which route: the posture graph written from words (02), with gaits
           as stepping from a stand (03, 11), weighted by how often the captured dog takes each step. A link the
           graph lacks is routed through the cheapest intermediate postures; a link with no route can only
           be forced, and a forced body falls (02: any -> fall -> scramble up).
  habit    this dog's own surprise, relative to any dog: its footage, smoothed toward the body. Negative when
           the dog does it more than a dog in general.
  meaning  does the next word make sense after this one: next in a sequence (08), later in one, a shared drive.
  skill    taught or not: a trained cue (L), or a body used for a purpose it doesn't serve by nature (a tug
           that opens a door) costs what the dog's training says.
Three modes choose which words are open to the dog: endemic (alone: A words, and D with another dog), social
(a person present, not asking: H words join, aimed at the person), guided (a request: the accept-set, and the
dog picks its cheapest way, or declines).
"""
import argparse
import heapq
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEX = os.path.join(ROOT, "dog-lexicon")
OUT = os.path.join(HERE, "grammar.json")
CATALOG = os.path.join(ROOT, "body", "MOCAP_CATALOG.md")
ENTRY_FILES = ["01-postures.md", "03-locomotion.md", "04-head-mouth-nose.md", "05-ears-eyes-tail-coat.md",
               "06-limbs-whole-body.md", "07-voice.md", "09-body-upkeep.md", "13-added-from-blind-lists.md"]
SUPPORT = ["stand", "walk", "trot", "run", "jump", "sit", "down", "side", "back", "crouch", "rear_up", "swim", "wade"]
REFLEX, PLANNED = {"R", "X"}, {"D", "O", "L"}

# Each held shape (01) on the node of the posture graph (02) it belongs to; the posture grid of 11-modules
# decides the rest (fore pair state x hind pair state).
POSTURE = {
    **{k: "stand" for k in ["stand", "stand_stacked", "stand_forward", "stand_back", "stand_braced", "stand_freeze",
                            "stand_point", "stand_paw_lift", "stand_cower", "stand_over", "stand_hunched"]},
    "stand_hind": "stand_hind", "forepaws_up": "forepaws_up", "sit_pretty": "sit_pretty",
    "sit": "sit", "sit_hip": "sit", "sit_slump": "sit", "sit_prop": "sit",
    "platz": "platz", "cower_flat": "platz", "down_sphinx": "down_sphinx", "loaf": "down_sphinx",
    "down_hip": "down_hip", "sploot_half": "sploot", "sploot": "sploot",
    "crouch_flat": "crouch", "crouch_spring": "crouch",
    "lateral": "lateral", "curl": "curl", "dorsal": "dorsal", "dorsal_appease": "dorsal",
    "play_bow": "play_bow", "stretch_bow": "play_bow", "prayer": "play_bow",
}
# Travel (03): the gait falls out of speed and manner (11-modules, move.go).
GAIT = {
    **{k: "walk" for k in ["walk", "amble", "stalk", "meander", "slink", "back_up", "side_step", "pivot_rear",
                           "pivot_front", "climb", "descend", "stairs_up", "stairs_down", "plow", "pull", "pull_load",
                           "limp_fore", "limp_hind", "stagger", "knuckle", "circle_one_way", "patrol", "pace_restless",
                           "seclude", "turn_away", "body_rub", "rump_present", "drive_in", "hoard", "eat_carry",
                           "scavenge", "site_circle", "sniff_ground", "carry", "circle_opponent", "crab"]},
    **{k: "trot" for k in ["trot", "pace", "trot_flying", "prance", "quarter"]},
    **{k: "run" for k in ["canter", "gallop_transverse", "gallop_rotary", "bound", "zoomies", "chase", "flee",
                          "harry", "fence_run", "u_turn", "skid_stop"]},
    **{k: "jump" for k in ["jump_long", "jump_vertical", "jump_over", "jump_onto", "jump_down", "leap_catch", "pounce",
                           "water_entry", "pronk", "startle"]},
    "swim": "swim", "wade": "wade", "crawl": "crouch",
    **{k: "fall" for k in ["slip", "stumble", "fall", "collide", "bad_landing", "collapse"]},
    "scramble_up": "stand",
}
# Bodies that need a particular support though they are not themselves postures.
SUPPORT_OF = {"spin": "stand", "tail_chase": "stand", "pivot_front": "stand", "scratch_self": "sit", "scratch_reflex": "lateral", "scoot": "sit", "rem": "lateral",
              "seizure": "lateral", "nurse": "lateral", "sunbathe": "lateral", "seek_warm": "curl", "seek_cool": "sploot",
              "back_rub": "dorsal", "scent_roll": "dorsal", "solicit_rub": "dorsal", "shake_off": "stand",
              "shake_off_dry": "stand", "urinate_leg": "stand", "urinate_squat": "stand", "urinate_tripod": "stand",
              "urinate_handstand": "stand", "defecate": "stand", "ground_scratch": "stand", "dig": "stand",
              "dig_bed": "stand", "tug": "stand", "shake_kill": "stand", "box": "stand_hind", "hug": "stand_hind",
              "mount": "stand", "tie": "stand", "flag_tail": "stand", "high_five": "sit", "give_paw": "sit",
              "wave": "sit", "sleep": "lateral", "drowsy": "down_sphinx", "contact_rest": "down_hip"}
# Posture-graph nodes and gaits as the capture and the footage name them.
NODE_REGIME = {"stand": "stand", "sit": "sit", "sit_pretty": "rear_up", "stand_hind": "rear_up",
               "forepaws_up": "rear_up", "platz": "down", "down_sphinx": "down", "down_hip": "down", "sploot": "down",
               "lateral": "side", "curl": "side", "dorsal": "back", "play_bow": "crouch", "crouch": "crouch",
               "walk": "walk", "trot": "trot", "run": "run", "jump": "jump", "swim": "swim", "wade": "wade",
               "fall": None}
REP = {"stand": "stand", "sit": "sit", "down": "platz", "side": "lateral", "back": "dorsal", "crouch": "crouch",
       "rear_up": "stand_hind", "walk": "walk", "trot": "trot", "run": "run", "jump": "jump", "swim": "swim",
       "wade": "wade"}
# Travel as 03 and 11 define it: stepping from a stand at any speed, leaping from a stand or any gait, landing
# into a stand or a gait; water from the shallows. Written from the definitions, not from data.
GAIT_EDGES = ([("stand", g) for g in ("walk", "trot", "run")] + [(g, "stand") for g in ("walk", "trot", "run")]
              + [(a, b) for a in ("walk", "trot", "run") for b in ("walk", "trot", "run") if a != b]
              + [(a, "jump") for a in ("stand", "walk", "trot", "run")]
              + [("jump", b) for b in ("stand", "walk", "trot", "run", "swim")]
              + [("walk", "wade"), ("wade", "walk"), ("wade", "swim"), ("swim", "wade")])
# Requests whose accept-set 10-requests.md writes as a description: the bodies that satisfy it, as alternatives
# the dog chooses among (each alternative in order).
REQUEST_BODIES = {
    "come": [["walk", "stand"], ["trot", "stand"], ["trot", "sit"], ["gallop_transverse", "sit"]],
    "front": [["sit"]], "finish": [["walk", "sit"]], "middle": [["walk", "stand"]], "weave": [["walk"]],
    "around": [["trot", "sit"]], "out": [["release"]], "leave_it": [["head_turn_away", "eye_contact"]],
    "stay": [[]], "wait": [[]], "release": [[]], "quiet": [[]],
    "place": [["walk", "platz"], ["walk", "down_hip"]], "settle": [["down_hip"], ["lateral"], ["curl"]],
    "kennel": [["walk", "platz"]], "off": [["stand"]], "potty": [["urinate_squat"], ["urinate_leg"]],
    "steady": [["stand_freeze"]], "stop": [["sit"]], "watch": [["orient", "stare_hard"]],
    "mark": [["watch_moving"]], "back": [["gallop_transverse"]], "over": [["canter"]],
    "send_away": [["gallop_transverse", "platz"]], "touch": [["nudge"]], "load": [["jump_onto"]],
    "directional": [["walk"], ["trot"]], "walk_up": [["stalk"]], "stop_herding": [["platz"], ["stand_freeze"]],
    "hike": [["pull_load"]], "gee": [["trot"]], "haw": [["trot"]], "whoa": [["stand"]], "on_by": [["trot"]],
    "line_out": [["stand_braced"]], "honor": [["stand_freeze"]], "flush": [["chase"]],
    "drive": [["trot"]], "shed": [["block"]], "pen": [["trot", "stand_freeze"]], "brace": [["stand_braced"]],
    "alert_sound": [["nudge", "walk"]], "refuse_unsafe": [["stand_braced"]], "show_me": [["walk", "stand"]],
    "come_bye": [["trot"]], "away": [["trot"]], "play_dead": [["lateral", "go_limp"]], "heel": [["walk"], ["trot"], ["sit"]], "heel_right": [["walk"], ["trot"]],
}
# Sequence steps 08-sequences.md writes as phrases: the body each phrase means (None: not a body, a cue).
STEP_PHRASES = {
    "track the fall": "watch_moving", "watch the throw": "watch_moving", "send cue": None, "gallop out": "gallop_transverse",
    "carry back": "carry", "sit in front": "sit", "release on": "release", "drop near": "release", "drop at": "release",
    "stare at the item": "stare_hard", "cast out": "trot", "narrowing zigzags": "quarter", "source": "sniff_detail",
    "follow step by step": "sniff_ground", "end at the target": "indicate", "wide arc around": "gallop_transverse",
    "stop behind it": "stand_freeze", "bring it toward": "walk", "circle to turn": "trot",
    "move between the handler": "block", "go to the hidden person": "trot", "face them and bark": "bark_threat",
    "curved approach": "walk", "one dog runs": "chase", "eat faster": "eat_gulp", "circle 1": "site_circle",
    "go to the door": "walk", "follow to the door": "walk", "wag": "wag_loose", "following a person": "walk",
    "lying pressed against": "down_hip", "gently gripping another": "mouthing", "eat the afterbirth": "eat_gulp",
    "the adult regurgitates": "regurgitate", "pups eat": "eat_gulp", "lie among the stock": "down_hip",
    "search": "quarter", "find the person": "orient", "lead the handler there": "walk",
    "run back to the handler": "trot", "trot to the item": "trot", "grab whatever": "grab",
}
# What sets off a reflex (R) that a word may also name: without it, the reflex is not a likely reading.
TRIGGER = {"shake_off": "wet", "shake_off_dry": "shaken", "pant": "hot", "pant_stress": "stressed", "yawn": "tired",
           "scramble_up": "fallen", "sneeze": "irritant", "scratch_self": "itch", "scratch_reflex": "itch",
           "head_shake": "irritant", "startle": "sudden", "tremble": "cold", "hackles": "threat"}
CHANNEL = {"fore": "legs", "hind": "legs", "trunk": "legs", "move": "legs", "head": "head", "eyes": "eyes",
           "ears": "ears", "tail": "tail", "jaw": "mouth", "tongue": "mouth", "lips": "lips", "air": "airway",
           "voice": "voice", "gut": "gut", "skin": "skin", "sex": "sex"}
# What a person says that is not in the lexicon's words but names a request or a target.
EXTRA_WORDS = {"fetch": [("request", "retrieve")], "bring": [("request", "retrieve")],
               "go to": [("travel", "walk")], "run to": [("travel", "run")], "walk to": [("travel", "walk")],
               "trot to": [("travel", "trot")], "open": [("world", "open")], "close": [("world", "close")],
               "lie down": [("request", "down")], "sit down": [("request", "sit")], "get up": [("request", "stand")],
               "roll over": [("request", "roll_over")], "jump up": [("request", "up")]}
STOP = {"the", "a", "an", "it", "to", "on", "of", "by", "and", "then", "me", "my", "your", "please", "now", "with",
        "at", "in", "into", "from", "for", "that", "this", "her", "him", "good", "girl", "boy", "dog", "go"}
PERSON = {"me", "my", "mine", "us", "here"}
# Requests that mean "to the one who said it" whatever follows them: recall and handler-position cues (10-requests.md)
TO_PERSON = {"come", "here", "front", "finish", "middle", "heel", "heel_right", "around", "touch"}
DRIVES = ["REST", "CMF", "ING", "ELM", "EXP", "PRY", "PLY", "SOC", "FEA", "DEF", "WRK", "SEX", "RFX", "EDGE"]
CONTEXT_DRIVES = {"rest": {"REST", "CMF"}, "explore": {"EXP", "ELM", "ING"}, "play_dog": {"PLY"},
                  "play_person": {"PLY", "SOC"}, "retrieve": {"WRK", "PRY", "PLY"}, "interact_person": {"SOC"},
                  "travel": {"EXP"}, "watch": {"EXP", "DEF"}, "eat_drink": {"ING"}}


# ---------- reading the lexicon ----------
def table(path, first_only=False):
    """Rows of every markdown table in the file (each read with its own header), or of the first table only."""
    rows, head = [], None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s.startswith("|"):
            if head is not None and first_only and rows:
                break
            head = None
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if head is None:
            head = cells
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(dict(zip(head, (cells + [""] * len(head))[:len(head)])))
    return rows


def aliases(cell):
    out = []
    for w in cell.split(","):
        w = re.sub(r"\(.*?\)", "", w).strip().strip('"').strip("'").lower()
        if w and w not in ("-", "(trained, no word)"):
            out.append(w)
    return out


def build():
    entries = {}
    for f in ENTRY_FILES:
        for r in table(os.path.join(LEX, f)):
            eid = r["id"].strip()
            entries[eid] = {"id": eid, "body": r.get("body", ""), "words": aliases(r.get("words", "")),
                            "with": r.get("with", "").replace(" ", ""), "why": r.get("why", "").split(), "file": f}
    missing = []
    for r in table(os.path.join(LEX, "12-recipes.md")):
        e = entries.get(r["id"])
        if e is None:
            missing.append(r["id"])
            continue
        e["modes"] = [m for m in r["modes"].split() if m != "-"]
        e["manner"] = [m for m in r["manner"].split() if m != "-"]
        e["target"] = None if r["target"] in ("-", "") else r["target"]
        e["cls"] = r["class"]
    for e in entries.values():
        e.setdefault("modes", []), e.setdefault("manner", []), e.setdefault("target", None), e.setdefault("cls", "P")
        e["channels"] = sorted({CHANNEL[m.split(".")[0]] for m in e["modes"] if m.split(".")[0] in CHANNEL})
        e["kind"] = "reflexive" if e["cls"] in REFLEX else "planned" if e["cls"] in PLANNED else "state"
        node = POSTURE.get(e["id"]) or GAIT.get(e["id"]) or SUPPORT_OF.get(e["id"])
        if node is None:
            ms = set(e["modes"])
            if "move.leap" in ms:
                node = "jump"
            elif "move.go" in ms:
                node = "run" if "fast" in e["manner"] else "walk"
            elif "trunk.side" in ms:
                node = "lateral"
            elif "trunk.back" in ms:
                node = "dorsal"
            elif "trunk.curl" in ms:
                node = "curl"
        e["node"] = node  # None: the body runs on whatever support the dog is on (an overlay)
        e["regime"] = NODE_REGIME.get(node) if node else None
    sequences = {}
    for r in table(os.path.join(LEX, "08-sequences.md")):
        steps = []
        for part in r["steps"].split("→"):
            key = re.sub(r"\s+", " ", part.strip().lower())
            hit = next((v for k, v in STEP_PHRASES.items() if key.startswith(k)), "")
            found = [hit] if hit else [] if hit is None else resolve_phrase(part, entries)
            steps.append({"text": part.strip(), "ids": found})
        sequences[r["id"]] = {"id": r["id"], "steps": steps, "trigger": r.get("trigger", ""),
                              "why": r.get("why", "").split(), "breed": r.get("breed loading", "")}
    requests = {}
    for r in table(os.path.join(LEX, "10-requests.md")):
        rid, text = r["request"], r["accepts"]
        seqs = [s for s in sequences if re.search(r"(?<![a-z_])" + re.escape(s) + r"(?![a-z_])", text.lower())]
        if rid in REQUEST_BODIES:  # accepts written as a description ("arrive at the handler"): read by hand
            alts = REQUEST_BODIES[rid]
        elif "→" in text:  # an ordered accept (roll over): one sentence, not alternatives
            alts = [[x for part in text.split("→") for x in resolve_phrase(part, entries)[:1]]]
        elif seqs:  # accepts named as sequences (get: retrieve_point)
            alts = [[st["ids"][0] for st in sequences[s]["steps"] if st["ids"]] for s in seqs]
        else:
            alts = [[x] for x in resolve_phrase(text, entries, every=True)]
        alts = [[x for x in alt if x in entries] for alt in alts]
        requests[rid] = {"id": rid, "words": aliases(r["words"]), "accepts_text": text, "alts": alts,
                         "accepts": sorted({x for alt in alts for x in alt}), "sequences": seqs,
                         "strict": r.get("strict", ""), "notes": r.get("notes", "")}
    edges = [{"from": r["from"], "to": r["to"], "how": r["how"], "notes": r.get("notes", "")}
             for r in table(os.path.join(LEX, "02-posture-graph.md"))]
    g = {"entries": entries, "sequences": sequences, "requests": requests, "edges": edges,
         "gait_edges": [list(x) for x in GAIT_EDGES], "recipes_without_entry": missing,
         "capture": capture_counts(), "footage": footage_counts()}
    json.dump(g, open(OUT, "w"), indent=1)
    kinds = Counter(e["kind"] for e in entries.values())
    unresolved = sum(1 for s in sequences.values() for st in s["steps"] if not st["ids"])
    print(f"{len(entries)} entries ({dict(kinds)}), {len(sequences)} sequences ({unresolved} steps unresolved), "
          f"{len(requests)} requests, {len(edges)} posture edges; recipes without an entry: {missing}")
    print(f"wrote {OUT}")
    return g


def resolve_phrase(text, entries, every=False):
    """Entry ids named in a phrase, in order: ids written out ('sniff_detail'), or their words ('sit in front')."""
    t = text.lower()
    hits = []
    for eid in entries:
        for m in re.finditer(r"(?<![a-z_])" + re.escape(eid) + r"(?![a-z_])", t):
            hits.append((m.start(), eid))
    if not hits or every:
        for eid, e in entries.items():
            for w in [eid.replace("_", " ")] + e["words"]:
                if len(w) < 3:
                    continue
                for m in re.finditer(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t):
                    if not any(abs(m.start() - s) < 2 for s, _ in hits):
                        hits.append((m.start(), eid))
    seen, out = set(), []
    for _, eid in sorted(hits):
        if eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out if every else out[:2]


def capture_counts():
    """Transitions between regimes in the 41-minute capture, as catalog_mocap.py counted them (runs of 0.2 s+)."""
    out = {}
    if not os.path.exists(CATALOG):
        return out
    for line in open(CATALOG, encoding="utf-8"):
        m = re.match(r"\|\s*(\w+)\s*→\s*(\w+)\s*\|\s*(\d+)\s*\|", line)
        if m:
            out[f"{m.group(1)}>{m.group(2)}"] = int(m.group(3))
    return out


def footage_counts():
    """Per dog: transitions between the support regimes the teacher named once a second, within unbroken stretches."""
    sys.path.insert(0, os.path.join(ROOT, "footage"))
    try:
        from learn.dataset import SUPPORT as FS, DOGS, teacher_rows
    except Exception as ex:  # noqa: BLE001
        print("footage labels unavailable:", ex)
        return {}
    by_clip = defaultdict(list)
    for r in teacher_rows():
        by_clip[r["clip"]].append(r)
    out = {d: Counter() for d in DOGS}
    ctx_out = {d: Counter() for d in DOGS}
    for clip, rs in by_clip.items():
        rs = sorted(rs, key=lambda r: r["t"])
        for dog in DOGS:
            prev, prev_t = None, None
            for r in rs:
                lab = r.get(dog) or {}
                ok = lab.get("visible") and lab.get("support") in FS and (r.get("other_dogs") or 0) == 0
                if not ok or (prev_t is not None and r["t"] - prev_t > 1.5):
                    prev = None
                if not ok:
                    continue
                cur = lab["support"]
                if prev is not None and cur != prev:
                    out[dog][f"{prev}>{cur}"] += 1
                    ctx_out[dog][f"{lab.get('context', 'other')}|{prev}>{cur}"] += 1
                prev, prev_t = cur, r["t"]
    return {"by_dog": {d: dict(c) for d, c in out.items()}, "by_dog_context": {d: dict(c) for d, c in ctx_out.items()}}


# ---------- the grammar at run time ----------
class Grammar:
    A_TEXT = 2.0  # pseudo-counts a link written in the posture graph carries, observed or not
    K_HABIT = 5.0  # how many of a dog's own transitions it takes to move half way from any dog to this one
    FALL_BITS = 9.0  # a forced link the body has no route for: priced as a fall (to be measured in physics)
    SENSE_WEIGHT = 4.0  # how much more a reading's likelihood counts than the body's cost when settling a word

    def __init__(self, path=OUT):
        if not os.path.exists(path):
            build()
        g = json.load(open(path))
        self.g, self.E, self.S, self.Q = g, g["entries"], g["sequences"], g["requests"]
        self._graphs()
        self._seq_index()
        self._alias_index()

    # body graph: nodes are posture-graph nodes and gaits; edges from words (02 + travel), weighted by the capture
    def _graphs(self):
        text = set()
        for e in self.g["edges"]:
            a, b = e["from"], e["to"]
            if a == "any" or b == "any":
                continue
            text.add((a, b))
        text |= {tuple(x) for x in self.g["gait_edges"]}
        self.text_edges = set(text)
        # within one regime a written change implies its reverse: a dog that unloads from platz into a sphinx
        # down can gather itself back; a hip rolled out can roll back under (manner and one leg pair, 11-modules)
        self.reverse_edges = {(b, a) for a, b in text if NODE_REGIME.get(a) and NODE_REGIME.get(a) == NODE_REGIME.get(b)
                              and (b, a) not in text}
        text |= self.reverse_edges
        coarse_text = Counter()
        for a, b in text:
            ra, rb = NODE_REGIME.get(a), NODE_REGIME.get(b)
            if ra and rb and ra != rb:
                coarse_text[(ra, rb)] = 1
        self.coarse_text = coarse_text
        cap = Counter({tuple(k.split(">")): v for k, v in self.g["capture"].items()})
        self.cap = cap
        out = defaultdict(float)
        for (a, b), v in cap.items():
            out[a] += v
        for (a, b) in coarse_text:
            out[a] += self.A_TEXT
        self.p_body = {}
        for a in SUPPORT:
            for b in SUPPORT:
                if a != b and out[a] > 0:
                    n = cap[(a, b)] + self.A_TEXT * coarse_text[(a, b)]
                    if n > 0:
                        self.p_body[(a, b)] = n / out[a]
        # fine graph
        adj = defaultdict(dict)
        for a, b in text:
            adj[a][b] = ("reverse" if (a, b) in self.reverse_edges else "words",
                         self._edge_bits(a, b) + (0.5 if (a, b) in self.reverse_edges else 0.0))
        for (ra, rb), v in cap.items():  # the capture's own transitions the words did not write
            if v > 0 and (ra, rb) not in coarse_text and ra in REP and rb in REP:
                a, b = REP[ra], REP[rb]
                if b not in adj[a]:
                    adj[a][b] = ("capture", self._edge_bits(a, b))
        self.adj = adj
        fd = self.g.get("footage", {}).get("by_dog", {})
        self.habit = {d: Counter({tuple(k.split(">")): v for k, v in c.items()}) for d, c in fd.items()}

    def _edge_bits(self, a, b):
        ra, rb = NODE_REGIME.get(a), NODE_REGIME.get(b)
        if not ra or not rb or ra == rb:
            return 1.0  # within one regime (platz -> sphinx), or into a fall: priced elsewhere
        return -math.log2(self.p_body.get((ra, rb), 1e-3))

    def p_dog(self, dog, ra, rb):
        c = self.habit.get(dog, Counter())
        tot = sum(v for (a, _), v in c.items() if a == ra)
        return (c[(ra, rb)] + self.K_HABIT * self.p_body.get((ra, rb), 1e-3)) / (tot + self.K_HABIT)

    def route(self, a, b):
        """Cheapest body route between two nodes: (bits, [nodes]); bits inf when only a fall gets there."""
        if a is None or b is None or a == b:
            return 0.0, [a] if a else []
        dist, prev, todo = {a: 0.0}, {}, [(0.0, a)]
        while todo:
            d, u = heapq.heappop(todo)
            if u == b:
                break
            if d > dist.get(u, math.inf):
                continue
            for v, (_, w) in self.adj.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(todo, (nd, v))
        if b not in dist:
            return math.inf, [a, b]
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        return dist[b], path[::-1]

    def habit_bits(self, dog, path):
        """This dog's surprise at the route, minus any dog's: from its footage, per regime change on the route."""
        if not dog or dog not in self.habit:
            return 0.0
        h = 0.0
        regs = [NODE_REGIME.get(n) for n in path]
        for ra, rb in zip(regs, regs[1:]):
            if ra and rb and ra != rb:
                pb = self.p_body.get((ra, rb), 1e-3)
                h += -math.log2(self.p_dog(dog, ra, rb)) + math.log2(pb)
        return h

    # meaning: what the lexicon's sequences and drives say about one word following another
    def _seq_index(self):
        self.adjacent, self.later = Counter(), Counter()
        for s in self.S.values():
            ids = [st["ids"] for st in s["steps"]]
            for i, a_ids in enumerate(ids):
                for a in a_ids:
                    for b in (ids[i + 1] if i + 1 < len(ids) else []):
                        self.adjacent[(a, b)] += 1
                    for later in ids[i + 2:]:
                        for b in later:
                            self.later[(a, b)] += 1

    def meaning_bits(self, a, b, goal=None):
        if a is None:
            return 0.0
        if goal and b in goal:
            return 1.0
        if self.adjacent[(a, b)]:
            return 1.0
        if self.later[(a, b)]:
            return 2.0
        ea, eb = self.E.get(a), self.E.get(b)
        if ea and eb and eb["kind"] == "reflexive":
            return 3.0  # a reflex is triggered, not chosen: its price is the trigger's
        if ea and eb and set(ea["why"]) & set(eb["why"]):
            return 3.0
        return 5.0

    def skill_bits(self, b, purpose=None, dog=None, taught=()):
        e = self.E.get(b)
        if e is None:
            return 0.0
        bits = 0.0
        if e["cls"] == "L" and b not in taught:
            bits += 2.0  # a cue it must have been taught; placeholder reliability 1/4 until set per dog
        if purpose == "WRK" and "WRK" not in e["why"] and e["kind"] != "reflexive" and b not in taught:
            bits += 4.0  # a body put to a use it doesn't serve by nature: a tug that opens a door
        return bits

    def conflicts(self, a, b):
        """Two bodies at once that share a part they can't share (README rule 5)."""
        ma, mb = set(self.E[a]["modes"]), set(self.E[b]["modes"])
        out = []
        if ("air.sniff" in ma and "air.pant" in mb) or ("air.pant" in ma and "air.sniff" in mb):
            out.append("the airway sniffs or pants, not both")
        grip = {"jaw.grip"}
        if (ma & grip and any(m.startswith(("voice.", "air.pant", "tongue.")) for m in mb)) or \
                (mb & grip and any(m.startswith(("voice.", "air.pant", "tongue.")) for m in ma)):
            out.append("a mouth that holds can't bark, pant or lick: it drops what it holds")
        work = {"fore.stroke", "fore.reach"}
        if (ma & work and "move.go" in mb) or (mb & work and "move.go" in ma):
            out.append("a forepaw that digs or reaches isn't walking")
        return out

    # ---------- words ----------
    def _alias_index(self):
        """Every word or phrase, with every body it can name and how readily it names it (bits): a word is most
        often its own entry or request; a sport's cue ('stand' to a herding dog) is a rare reading of a plain word."""
        idx, prior = defaultdict(list), {}
        special = ("herding", "sled", "gundog", "pointer", "spaniel", "agility", "protection", "assistance",
                   "search and rescue", "hearing", "guide")

        def add(w, sense, bits):
            idx[w].append(sense)
            prior[(w, sense)] = min(bits, prior.get((w, sense), 99))

        for eid, e in self.E.items():
            own = {eid.replace("_", " "), eid}
            for w in own | set(e["words"]):
                add(w, ("entry", eid), 0.0 if w in own or (e["words"] and w == e["words"][0]) else 1.0)
        for rid, r in self.Q.items():
            own = {rid.replace("_", " "), rid}
            rare = 3.0 if any(s in r["notes"].lower() for s in special) else 0.0
            for w in own | set(r["words"]):
                add(w, ("request", rid), (0.0 if w in own or (r["words"] and w == r["words"][0]) else 1.0) + rare)
        for sid in self.S:
            add(sid.replace("_", " "), ("sequence", sid), 0.0)
        for w, senses in EXTRA_WORDS.items():
            for s in senses:
                idx[w].insert(0, s)
                prior[(w, s)] = 0.0
        self.alias = {k: list(dict.fromkeys(v)) for k, v in idx.items() if k}
        self.sense_prior = prior
        self.maxlen = max(len(k.split()) for k in self.alias)

    @staticmethod
    def _stems(tok):
        out = [tok]
        for suf in ("ing", "ed"):
            if tok.endswith(suf) and len(tok) > len(suf) + 2:
                base = tok[: -len(suf)]
                out += [base, base + "e"] + ([base[:-1]] if len(base) > 2 and base[-1] == base[-2] else [])
        if tok.endswith("s") and len(tok) > 3:
            out.append(tok[:-1])
        return out

    def tokens(self, text):
        """Words found in the text, longest first, each with every body it can name; the rest become targets."""
        toks = re.findall(r"[a-zäöüß']+", text.lower())
        out, i, loose, world = [], 0, [], None
        while i < len(toks):
            hit = None
            for n in range(min(self.maxlen, len(toks) - i), 0, -1):
                for cand in self._phrase_forms(toks[i:i + n]):
                    if cand in self.alias and not (n == 1 and cand in STOP):
                        hit = (n, cand)
                        break
                if hit:
                    break
            if hit:
                if loose and out:
                    out[-1]["target"] = " ".join(loose)
                loose = []
                senses = self.alias[hit[1]]
                tok = {"said": " ".join(toks[i:i + hit[0]]), "senses": senses, "target": None,
                       "prior": {s: self.sense_prior.get((hit[1], s), 0.0) for s in senses}}
                if any(ident in TO_PERSON for _, ident in senses):
                    tok["to_person"] = True  # a recall cue: to the speaker, though no "me" follows it
                if any(k == "world" for k, _ in senses):
                    world = tok  # "open it ... by tugging": the next body is the means to this end
                elif world is not None:
                    tok["purpose"], tok["effect"] = "WRK", f"{world['said']} {world.get('target') or ''}".strip()
                    world = None
                out.append(tok)
                i += hit[0]
            else:
                if toks[i] in PERSON and out:
                    out[-1]["to_person"] = True
                elif toks[i] not in STOP:
                    loose.append(toks[i])
                i += 1
        if loose and out:
            out[-1]["target"] = " ".join(loose)
        return out

    def _phrase_forms(self, toks):
        forms = [" ".join(toks)]
        if len(toks) == 1:
            forms += self._stems(toks[0])
        else:
            forms += [" ".join(toks[:-1] + [s]) for s in self._stems(toks[-1])[1:]]
            forms += [" ".join([s] + toks[1:]) for s in self._stems(toks[0])[1:]]
        return forms

    # ---------- reading free speech with Gemini Carbon (closed vocabulary: the lexicon's own ids) ----------
    def carbon_tokens(self, text, cache=os.path.join(HERE, "carbon_parses.json")):
        """Free speech -> the same tokens tokens() makes, read by Gemini Carbon through the Hermes proxy. Carbon
        only maps words to the lexicon's ids, targets and means; every price is still the grammar's."""
        memo = json.load(open(cache)) if os.path.exists(cache) else {}
        if text not in memo:
            sys.path.insert(0, os.path.join(ROOT, "tools"))
            from ask import ask
            ents = "\n".join(f"entry:{k}: {e['body'][:70]}" for k, e in self.E.items())
            reqs = "\n".join(f"request:{k}: {r['accepts_text'][:60]}" for k, r in self.Q.items())
            seqs = "\n".join(f"sequence:{k}: " + " > ".join(st['text'][:20] for st in s['steps'])
                             for k, s in self.S.items())
            prompt = (
                "A person says something to their dog. Map it onto the dog's lexicon below, in order.\n"
                "Answer with JSON only: a list of steps, each {\"said\": the words, \"sense\": one id from the lists "
                "(or \"world:open\" / \"world:close\" for a change to the world the dog must bring about), "
                "\"target\": a short noun phrase or null, \"to_person\": true if it ends with the speaker, "
                "\"means_of\": the index of the world step this step brings about, or null}.\n"
                "Use requests for handler cues, sequences for errands, entries for particular bodies. "
                "Leave out words that ask for nothing.\n\n"
                f"SAID: {text}\n\nREQUESTS\n{reqs}\n\nSEQUENCES\n{seqs}\n\nENTRIES\n{ents}\n")
            answer, usage = ask("gemini", prompt, max_tokens=4000, temperature=0.0)
            m = re.search(r"\[.*\]", answer, re.S)
            memo[text] = json.loads(m.group(0)) if m else []
            json.dump(memo, open(cache, "w"), indent=1)
        steps, toks = memo[text], []
        for i, st in enumerate(steps):
            kind, _, ident = str(st.get("sense", "")).partition(":")
            ok = (kind == "entry" and ident in self.E) or (kind == "request" and ident in self.Q) or \
                 (kind == "sequence" and ident in self.S) or kind == "world"
            if not ok:
                continue
            tok = {"said": st.get("said", ""), "senses": [(kind, ident)], "target": st.get("target"),
                   "to_person": bool(st.get("to_person")), "prior": {(kind, ident): 0.0}}
            j = st.get("means_of")
            if isinstance(j, int) and 0 <= j < len(steps):
                tok["purpose"], tok["effect"] = "WRK", f"{steps[j].get('sense', '')} {steps[j].get('target') or ''}".strip()
            toks.append(tok)
        return toks

    # ---------- pricing a string of words ----------
    def expand(self, kind, ident, state):
        """A sense as the bodies it can be: an entry is itself; a request is its accept-set (the dog picks);
        a sequence is its steps; travel is a gait."""
        if kind == "entry":
            return [[ident]]
        if kind == "request":
            return [alt for alt in self.Q[ident]["alts"]] or [[]]
        if kind == "sequence":
            steps = [st["ids"][0] for st in self.S[ident]["steps"] if st["ids"]]
            return [steps]
        if kind == "travel":
            return [[ident]]
        return [[]]

    def state_bits(self, eid, state):
        """Blood and breath as determiners: the body's state makes some ways of complying cheaper (10-requests:
        a hot dog sploots, a keyed-up one goes to platz, a tired one lies on its hip)."""
        e = self.E.get(eid, {})
        n, b = e.get("node"), 0.0
        if state.get("hot"):  # belly and thighs to the cool floor
            b += {"sploot": -4.0, "lateral": -1.0, "platz": 1.5, "down_sphinx": 0.5, "curl": 2.0}.get(n, 0.0)
        if state.get("aroused"):  # loaded, ready to go again
            b += {"platz": -2.0, "crouch": -1.0, "lateral": 2.0, "curl": 2.0, "down_hip": 1.0}.get(n, 0.0)
        if state.get("tired"):  # hips rolled out, flat on the side
            b += {"down_hip": -2.0, "lateral": -2.0, "curl": -1.5, "platz": 0.5}.get(n, 0.0)
            b += 2.5 if NODE_REGIME.get(n) in ("run", "jump") else 0.0
        return b

    def link(self, a, b, dog=None, goal=None, purpose=None, state=None, node=None):
        """Price the move from body a (an entry id, or None at the start) on node `node` to body b."""
        state = state or {}
        eb = self.E.get(b) or {"node": b if b in NODE_REGIME else None, "kind": "state", "why": [], "cls": "P",
                                "modes": []}
        here = node or (self.E[a]["node"] if a in self.E else None)
        there = eb["node"] or here
        if here == "fall":
            here = "stand"  # a fall ends in a scramble up (02)
        body, path = self.route(here, there) if here and there else (0.0, [here] if here else [])
        forced = not math.isfinite(body)
        if forced:
            body = self.FALL_BITS
        habit = self.habit_bits(dog, path) if not forced else 0.0
        meaning = self.meaning_bits(a, b, goal)
        skill = self.skill_bits(b, purpose, dog)
        st = self.state_bits(b, state)
        trig = TRIGGER.get(b)
        if eb.get("kind") == "reflexive" and trig is not None:
            wet = state.get("wet") or here in ("swim", "wade")
            if not (wet if trig == "wet" else state.get(trig)):
                st += self.FALL_BITS  # a reflex is triggered, not chosen: asked for without its trigger, as unlikely as a fall
        # the direct change, as each source has it: does the body have this link, or only a route to it?
        direct = here is None or there is None or here == there or there in self.adj.get(here, {})
        ra, rb = NODE_REGIME.get(here), NODE_REGIME.get(there)
        ev = None
        if ra and rb and ra != rb:
            ev = {"words": bool(self.coarse_text.get((ra, rb))), "capture": self.cap.get((ra, rb), 0),
                  **{d: c.get((ra, rb), 0) for d, c in self.habit.items()}}
        return {"from": a, "to": b, "route": path, "forced": forced, "direct": direct, "evidence": ev,
                "overlay": self.E.get(b, {}).get("node", 1) is None,
                "body": round(body, 2), "habit": round(habit, 2), "meaning": meaning, "skill": skill, "state": st,
                "bits": round(body + habit + meaning + skill + st, 2),
                "kind": eb.get("kind", "state"), "node": there}

    def price(self, text, dog=None, start="stand", state=None, plan=None):
        """Words -> the cheapest sentence of bodies that says them -> every link priced. Ambiguous words
        ('shake', 'down', 'off') are settled by price: the reading the body and the dog find cheapest wins."""
        state = state or {}
        toks = plan if plan is not None else self.tokens(text)
        # beam over readings: each token offers one or more body strings; keep the cheapest few
        beams = [(0.0, [], start, None)]  # bits, links, node, last body
        for tok in toks:
            options = []
            for kind, ident in tok["senses"]:
                if kind == "world":
                    continue
                goal = set(self.Q[ident]["accepts"]) if kind == "request" else None
                sense_bits = tok.get("prior", {}).get((kind, ident), 0.0)
                for bodies in self.expand(kind, ident, state):
                    options.append((kind, ident, bodies, goal, sense_bits))
            if not options:
                continue
            new = []
            for bits, links, node, last in beams:
                for kind, ident, bodies, goal, sense_bits in options:
                    # what was meant comes first, how hard it is second: a sport's reading of a plain word must not
                    # win just because it asks less of the body
                    b2, l2, n2, last2 = bits + self.SENSE_WEIGHT * sense_bits, list(links), node, last
                    for k, bid in enumerate(bodies):
                        lk = self.link(last2, bid, dog=dog, goal=goal,
                                       purpose=tok.get("purpose"), state=state, node=n2)
                        lk.update({"said": tok["said"] if k == 0 else "", "sense": f"{kind}:{ident}",
                                   "sense_bits": sense_bits if k == 0 else 0.0, "effect": tok.get("effect"),
                                   "target": tok.get("target"), "to_person": tok.get("to_person", False)})
                        b2 += lk["bits"]
                        l2.append(lk)
                        n2 = lk["node"] or n2
                        last2 = bid
                    new.append((b2, l2, n2, last2))
            beams = sorted(new, key=lambda x: x[0])[:12]
        bits, links, node, _ = beams[0]
        bits = sum(lk["bits"] + lk.get("sense_bits", 0.0) for lk in links)
        return {"text": text, "dog": dog, "start": start, "bits": round(bits, 2), "links": links,
                "alternatives": [round(b[0], 2) for b in beams[1:4]]}

    # ---------- stringing new sentences ----------
    def open_words(self, mode, ctx, other_dog=False, body_only=False):
        """Which bodies the dog can reach for on its own in this mode and activity (requests aside); body_only:
        just the words that move the support (postures, gaits, leaps), for a body without jaw or ears."""
        drives = CONTEXT_DRIVES.get(ctx, {"EXP"})
        out = []
        for eid, e in self.E.items():
            w = e["with"]
            if "EDGE" in e["why"] or "SEX" in e["why"] or e["file"].startswith("13"):
                continue
            ok = "A" in w or (other_dog and "D" in w) or (mode == "social" and "H" in w)
            if not ok or e["kind"] == "reflexive" or e["node"] == "fall" or (body_only and e["node"] is None):
                continue
            if set(e["why"]) & drives:
                out.append(eid)
        return out

    def generate(self, n=8, length=10, mode="endemic", ctx="explore", dog=None, tau=1.0, seed=0, start="stand",
                 state=None, other_dog=False, body_only=False):
        import random
        rnd = random.Random(seed)
        words = self.open_words(mode, ctx, other_dog, body_only)
        out = []
        for _ in range(n):
            node, last, links, bits = start, None, [], 0.0
            for _k in range(length):
                cand = []
                for b in words:
                    if b == last:
                        continue
                    lk = self.link(last, b, dog=dog, state=state or {}, node=node)
                    if lk["forced"]:
                        continue
                    cand.append(lk)
                w = [2 ** (-lk["bits"] / tau) for lk in cand]
                lk = rnd.choices(cand, weights=w)[0]
                links.append(lk)
                bits += lk["bits"]
                node, last = lk["node"] or node, lk["to"]
            out.append({"mode": mode, "ctx": ctx, "dog": dog, "tau": tau, "bits": round(bits, 2), "links": links})
        return out


# ---------- the audit: words against the capture and the two dogs ----------
def audit():
    G = Grammar()
    regs = [r for r in SUPPORT if r not in ("swim", "wade")]
    pairs = [(a, b) for a in regs for b in regs if a != b]
    T = {p: G.coarse_text[p] for p in pairs}
    C = {p: G.cap[p] for p in pairs}
    F = Counter()
    for d, c in G.habit.items():
        for p in pairs:
            F[p] += c[p]
    n_text = sum(T.values())
    print(f"The posture graph written from words allows {n_text} of {len(pairs)} direct changes between "
          f"{len(regs)} regimes ({n_text / len(pairs):.0%}).")
    rows = []
    import random
    for name, D in (("capture (41 min, 50 Hz)", C), ("your two dogs (2,895 labelled s, 1 Hz)", F)):
        tot = sum(D.values())
        covered = sum(v for p, v in D.items() if T[p])
        # chance: the same number of allowed changes, placed at random, 2000 times
        rnd = random.Random(0)
        sims = []
        for _ in range(2000):
            pick = set(rnd.sample(pairs, n_text))
            sims.append(sum(v for p, v in D.items() if p in pick) / max(tot, 1))
        sims.sort()
        p_val = (sum(s >= covered / max(tot, 1) for s in sims) + 1) / (len(sims) + 1)
        print(f"\n{name}: {tot} changes; {covered / max(tot, 1):.0%} are ones the words allow "
              f"(random graphs of the same size: {sims[len(sims) // 2]:.0%} median, {sims[int(.95 * len(sims))]:.0%} at 95th; p={p_val:.4f})")
        miss = sorted(((v, p) for p, v in D.items() if v and not T[p]), reverse=True)
        print("  seen but not written: " + ", ".join(f"{a}>{b} {v}" for v, (a, b) in miss[:14]))
        rows.append((name, tot, covered, sims, p_val, miss))
    never = [f"{a}>{b}" for (a, b), v in T.items() if v and not C[(a, b)] and not F[(a, b)]]
    print(f"\nwritten but seen in neither: {', '.join(never) or 'none'}")
    forbidden = [p for p in pairs if not T[p]]
    both_zero = [p for p in forbidden if not C[p] and not F[p]]
    print(f"of the {len(forbidden)} changes the words forbid, {len(both_zero)} are never seen in either source")
    return rows


def show(res):
    print(f"\n\"{res['text']}\"  dog={res['dog'] or 'any'}  from={res['start']}  total {res['bits']} bits"
          + (f"  (next readings: {res['alternatives']})" if res["alternatives"] else ""))
    for lk in res["links"]:
        via = " > ".join(n for n in lk["route"] if n) if len(lk["route"]) > 2 else ""
        tag = "FORCED: falls" if lk["forced"] else lk["kind"]
        said = f"'{lk['said']}'" if lk["said"] else ""
        tgt = f" -> {lk['target']}" if lk.get("target") else ""
        tgt += " (to you)" if lk.get("to_person") else ""
        tgt += f" [means: {lk['effect']}]" if lk.get("effect") else ""
        print(f"  {said:12s} {lk['sense']:20s} {str(lk['to']):15s} {tag:9s} body {lk['body']:4.1f} "
              f"habit {lk['habit']:+4.1f} meaning {lk['meaning']:1.0f} skill {lk['skill']:1.0f}"
              f" state {lk['state']:+4.1f}" + (f" sense {lk['sense_bits']:.0f}" if lk.get("sense_bits") else "")
              + f" = {lk['bits'] + lk.get('sense_bits', 0):4.1f}{tgt}")
        if not lk["direct"]:
            ev = lk["evidence"] or {}
            seen = ", ".join(f"{k} {v}" for k, v in ev.items() if k != "words")
            print(f"  {'':12s}   no direct link {lk['route'][0]} -> {lk['route'][-1]} (words: {'yes' if ev.get('words') else 'no'};"
                  f" {seen}): routed via {via}; forced instead, it falls ({Grammar.FALL_BITS:.0f} bits, a fall and a scramble)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "audit", "price", "generate"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--dog", default=None)
    ap.add_argument("--from", dest="start", default="stand")
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--tired", action="store_true")
    ap.add_argument("--aroused", action="store_true")
    ap.add_argument("--mode", default="endemic", choices=["endemic", "social"])
    ap.add_argument("--ctx", default="explore")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--length", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--carbon", action="store_true", help="price: read free speech with Gemini Carbon (Hermes proxy)")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "audit":
        audit()
    elif a.cmd == "price":
        G = Grammar()
        show(G.price(a.arg, dog=a.dog, start=a.start, state={"hot": a.hot, "tired": a.tired, "aroused": a.aroused},
                     plan=G.carbon_tokens(a.arg) if a.carbon else None))
    elif a.cmd == "generate":
        G = Grammar()
        for s in G.generate(int(a.arg or 5), a.length, a.mode, a.ctx, a.dog, a.tau, a.seed):
            words = " ".join(lk["to"] for lk in s["links"])
            print(f"[{s['bits']:6.1f} bits] {words}")


if __name__ == "__main__":
    main()
