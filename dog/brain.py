"""The dog's decisions: which regime, where to go, how it breathes, and whether it answers a request.

Learned where the footage has data, designed where it doesn't:
  * Routing (what comes next, and when): the linear router learned from the teacher's labels of the two
    dogs (footage/models/router_linear.json), trusted in proportion to how many transitions that dog has
    in the current activity, and shrunk toward designed priors otherwise. Rest has 3-6 labelled
    transitions per dog, play about 140, so rest leans on the priors and play on the dogs themselves.
  * Physiology (designed, after the footage showed it can't be learned from short clips): a fast
    exertion integrator (breath and heart, 30 s) and a slow one (fatigue, 20 min) driven by how hard
    each regime works. Fatigue lengthens rest bouts and shortens active ones; exertion sets the
    breathing rate and switches to panting.
  * Requests (designed, final say): a request is weighed against arousal, fatigue and whether the dog is
    lying down; the dog may decline. Asking again raises arousal. Per-dog numbers are placeholders
    (the footage's audio holds almost no requests): to be set by the person who knows the dogs.
  * Words (dog/grammar.py): what a person says is parsed into bodies and priced in bits from where the dog
    is (body, its own habit, meaning, skill, its state). The price enters the final say: an unlikely ask is
    declined more often. A change the body has no direct link for is routed through the postures between;
    asked again with urgency, an aroused dog may take it straight across, and the physics decides whether it
    falls. Three modes: endemic (nobody there: the dog's own router), social (a person present, not asking:
    the dog strings its own sentences aimed at the person, from its individual words), guided (a person
    speaks: the heard plan runs, then the dog goes back to its own devices).
No learning happens while the dog runs.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTER = os.path.join(os.path.dirname(HERE), "footage", "models", "router_linear.json")

BODY = ["stand", "walk", "trot", "run", "sit", "down", "side", "crouch"]  # what the captured body can play
STATIC = {"stand", "sit", "down", "side", "crouch", "back", "rear_up"}
LYING = {"down", "side", "back"}
SPEED = {"stand": 0.0, "walk": 1.0, "trot": 1.9, "run": 4.0, "sit": 0.0, "down": 0.0, "side": 0.0, "crouch": 0.0,
         "jump": 3.0, "rear_up": 0.0, "back": 0.0}
DT_BRAIN = 0.02
try:  # the grammar's posture nodes, as regimes the captured body can play
    from grammar import NODE_REGIME as _NR
except ImportError:  # pragma: no cover
    _NR = {}


def NODE_REGIME_OF(n):  # noqa: N802
    r = _NR.get(n, n if n in SPEED else None)
    return r if r in SPEED else None
INTENSITY = {"stand": 0.08, "walk": 0.2, "trot": 0.45, "run": 1.0, "jump": 0.9, "sit": 0.04, "down": 0.02,
             "side": 0.01, "back": 0.02, "crouch": 0.15, "rear_up": 0.8, "swim": 0.85, "wade": 0.35}
IN_PLAY = {"stand": 0.3, "sit": 0.2, "down": 0.35, "side": 0.4, "back": 0.45, "crouch": 0.4}

# Designed priors: median bout length (s) and what follows what, when nothing calls the dog.
PRIOR_DUR = {
    "rest": {"down": 120, "side": 240, "sit": 20, "stand": 6, "crouch": 2, "walk": 4, "trot": 3, "run": 2},
    "default": {"down": 30, "side": 40, "sit": 10, "stand": 5, "crouch": 2, "walk": 5, "trot": 4, "run": 3},
}
PRIOR_NEXT = {
    "rest": {"down": {"side": .45, "sit": .1, "stand": .3, "crouch": .15}, "side": {"down": .8, "stand": .2},
             "sit": {"down": .55, "stand": .35, "walk": .1}, "stand": {"down": .45, "sit": .3, "walk": .25},
             "walk": {"stand": .6, "down": .2, "sit": .2}, "crouch": {"down": .5, "stand": .5},
             "trot": {"walk": .7, "stand": .3}, "run": {"trot": .5, "stand": .5}},
    "default": {"down": {"stand": .5, "sit": .3, "side": .2}, "side": {"down": .7, "stand": .3},
                "sit": {"stand": .6, "down": .3, "walk": .1}, "stand": {"walk": .5, "sit": .2, "down": .2, "trot": .1},
                "walk": {"stand": .5, "trot": .3, "sit": .1, "down": .1}, "crouch": {"stand": .6, "run": .4},
                "trot": {"walk": .5, "run": .2, "stand": .3}, "run": {"trot": .6, "stand": .4}},
}
TRUST_K = 50.0  # learned transitions needed for the learned router to get half the say

# Individuality. Request numbers are PLACEHOLDERS until set by the person who knows the dogs.
LAB = {"name": "lab", "dog_id": 0, "breath_rest": 18, "breath_max": 45, "pant": 220,
       "request": {"bias": -0.3, "arousal": 2.0, "fatigue": 1.5, "lying": 1.0, "repeat": 0.8, "latency": 1.5},
       "rest_scale": 1.3, "arrive": "sit",
       # with a person there and nothing asked: she picks something up to say hi (from the person who knows her)
       "social": [("bring gift", 1.0)], "social_every": 25.0, "impulse": 0.6,
       # with the other dog near: how often she invites play (s) and how readily she joins. PLACEHOLDERS
       "play_every": 45.0, "play_bias": -0.2}
MAL = {"name": "mal", "dog_id": 1, "breath_rest": 22, "breath_max": 50, "pant": 240,
       "request": {"bias": 0.8, "arousal": 2.0, "fatigue": 1.0, "lying": 0.6, "repeat": 0.8, "latency": 0.6},
       "rest_scale": 0.9, "arrive": "sit",
       # with a person there: she walks between the legs and to one side (handler-position control)
       "social": [("middle", 0.6), ("weave", 0.4)], "social_every": 18.0, "impulse": 1.0,
       "play_every": 30.0, "play_bias": 0.4}
BITS0, BITS_SCALE = 4.0, 4.0  # an ask priced 4 bits above this lowers the odds of a yes as much as one unit of bias
REP_NODE = {"stand": "stand", "sit": "sit", "down": "platz", "side": "lateral", "crouch": "crouch", "walk": "walk",
            "trot": "trot", "run": "run", "back": "dorsal", "rear_up": "stand_hind", "jump": "jump"}
HOLD = {"jump": 0.8, "walk": 3.0, "trot": 3.0, "run": 2.5}  # seconds a heard body is held when it has no place to go


def intensity(reg, ctx):
    x = INTENSITY.get(reg, 0.1)
    return max(x, IN_PLAY.get(reg, 0.0)) if ctx in ("play_dog", "play_person") else x


class Router:
    """The learned linear router, in numpy."""

    def __init__(self, path=ROUTER):
        z = json.load(open(path))
        self.S, self.C = z["support"], z["context"]
        self.W, self.b = np.array(z["coef"]), np.array(z["intercept"])
        self.classes = z["classes"]
        self.n = z["n_transitions"]
        self.durs = z["durations"]

    def next_probs(self, dog, reg, ctx, dur, pant):
        ns, nc = len(self.S), len(self.C)
        x = np.zeros(ns + nc + ns + 1 + 4)
        x[self.S.index(reg)] = 1
        x[ns + (self.C.index(ctx) if ctx in self.C else nc - 1)] = 1
        x[-4], x[-3], x[-1] = math.log1p(dur), float(pant), float(dog["dog_id"])
        logits = self.W @ x + self.b
        p = np.exp(logits - logits.max())
        p /= p.sum()
        full = np.zeros(ns)
        full[self.classes] = p
        return {r: full[self.S.index(r)] for r in BODY}

    def trust(self, dog, ctx):
        n = self.n.get(dog["name"], {}).get(ctx, 0)
        return n / (n + TRUST_K)


class Brain:
    def __init__(self, dog, router, rng, regime="down", context="rest", grammar=None, person=None):
        self.d, self.R, self.rng = dog, router, rng
        self.reg, self.ctx = regime, context
        self.E = self.F = intensity(regime, context)
        self.A = 0.1
        self.elapsed, self.dur = 0.0, self._duration(regime)
        self.breath_phase, self.n_asked = 0.0, 0
        self.goal, self.goal_from, self.pending = None, 0.0, []
        self.events = []
        self.phase = None  # within a goal: "rise", "go", "arrive"
        # words
        self.G, self.person = grammar, (None if person is None else np.asarray(person, float))
        self.node = REP_NODE.get(regime, regime)
        self.plan, self.plan_from, self.heard = [], 0.0, []
        self.forced = False  # the current change was taken straight across (over-commanded)
        self.next_social = None if person is None else float(self.d.get("social_every", 30.0))
        # dog with dog: where the other dog is (a shared array, set by whoever runs both), and its brain
        self.other, self.partner = None, None
        self.next_play = float(self.d.get("play_every", 40.0))
        self.play_until, self.play_role = 0.0, "chase"
        self.last_pos = np.zeros(2)

    def meet(self, other_pos, partner):
        """Another dog is here: its position (an array kept up to date by the world) and its brain."""
        self.other, self.partner = other_pos, partner

    def _free(self):
        return not self.plan and self.goal is None and self.ctx != "play_dog"

    def _invite(self, t):
        """Near the other dog and nothing to do: a play bow at it, then a dash at it."""
        self.ctx, self.play_role = "play_dog", "chase"
        self.play_until = t + 20.0 + 40.0 * self.rng.random()
        self.plan = [{"reg": "crouch", "node": "play_bow", "said": "", "heard": False, "secs": 1.2, "event": "play_bow",
                      "target": None},
                     {"reg": "run" if float(np.hypot(*(self.other - self.last_pos))) > 4.0 else "trot", "node": "run",
                      "said": "", "heard": False, "secs": 2.5, "event": "chase", "target": self.other, "play": True}]
        self.plan_from = t
        self.A = min(1.5, self.A + 0.4)
        self.events.append((t, "(her own idea) play bow at the other dog"))
        if self.partner is not None:
            self.partner.invited(t, self)

    def invited(self, t, by):
        """The other dog bows at her. She joins when she is free, and the more roused she is, the likelier."""
        if not self._free():
            return
        z = self.d.get("play_bias", 0.0) + 2.0 * self.A - 1.5 * self.F - 0.8 * (self.reg in LYING)
        p = 1 / (1 + math.exp(-z))
        if self.rng.random() < p:
            self.ctx, self.play_role = "play_dog", "flee"
            self.play_until = by.play_until
            self.A = min(1.5, self.A + 0.4)
            self._enter("crouch", t, f"answers the play bow (chance {p:.0%})")
        else:
            self.events.append((t, f"ignores the play bow (chance {p:.0%})"))

    @property
    def mode(self):
        if self.plan:
            return "guided" if self.plan[0].get("heard") else "social"
        return "endemic" if self.person is None else "social"

    # ---------- words ----------
    def state(self):
        return {"hot": self.E > 0.35, "tired": self.F > 0.3, "aroused": self.A > 0.6}

    def hear(self, t, text, where=None, urgency=0.0):
        """A person says something. It is parsed and priced from where the dog is now, then weighed (final say)."""
        self.A = min(1.5, self.A + 0.3 + 0.3 * urgency)
        self.pending.append((t + 0.3, "words", (text, None if where is None else np.asarray(where, float), urgency)))

    def _plan_from(self, res, where, urgency, heard):
        """Priced links -> body steps for the matcher. Routes become their postures in turn; a change the body has
        no direct link for may be taken straight across when the ask is urgent and the dog impulsive."""
        steps, cur = [], self.reg
        for lk in res["links"]:
            tgt = self.person if lk.get("to_person") and self.person is not None else where
            if lk["target"] and where is not None and not lk.get("to_person"):
                tgt = where
            if lk["node"] is None or lk.get("overlay"):  # mouth, eyes, ears, voice: no body change; an event
                steps.append({"reg": None, "secs": 0.8, "event": lk["to"], "said": lk["said"], "heard": heard})
                continue
            route = [n for n in lk["route"] if n] or [lk["node"]]
            regs = [NODE_REGIME_OF(n) for n in route]
            impulsive = (not lk["direct"]) and urgency > 0 and \
                self.rng.random() < 1 / (1 + math.exp(-(2.0 * urgency + self.d.get("impulse", 0.5) * self.A - 2.0)))
            if impulsive:
                regs, route = [regs[-1]], [route[-1]]
            for k, (r, n) in enumerate(zip(regs, route)):
                if r is None or (r == cur and k < len(regs) - 1):
                    continue
                last = k == len(regs) - 1
                moving = r in ("walk", "trot", "run")
                steps.append({"reg": r, "node": n, "said": lk["said"] if last else "", "heard": heard,
                              "secs": (HOLD.get(r, 3.0) if last else 1.0), "event": lk["to"] if last else None,
                              "target": tgt if (moving and last) else None, "forced": impulsive and k == 0})
                cur = r
        return steps

    def _take_words(self, t, text, where, urgency):
        if self.G is None:
            return
        res = self.G.price(text, dog=self.d["name"], start=self.node, state=self.state())
        r = self.d["request"]
        # what the dog weighs: its hardest link, plus a little effort for every further step (a long but easy
        # errand is not an unlikely one; one link the body or the dog's training lacks is)
        links = [lk["bits"] + lk.get("sense_bits", 0.0) for lk in res["links"]] or [0.0]
        ask = max(links) + 0.5 * (len(links) - 1)
        z = r["bias"] + r["arousal"] * self.A - r["fatigue"] * self.F - r["lying"] * (self.reg in LYING) \
            + r["repeat"] * self.n_asked - (ask - BITS0) / BITS_SCALE
        p = 1 / (1 + math.exp(-z))
        self.n_asked += 1
        self.heard.append({"t": round(t, 1), "said": text, "bits": res["bits"], "hardest": round(max(links), 1),
                           "weighed": round(ask, 1), "p_yes": round(p, 2)})
        if self.rng.random() < p:
            self.plan = self._plan_from(res, where, urgency, heard=True)
            self.plan_from = t + r["latency"] * math.exp(0.4 * self.rng.standard_normal()) * (0.5 if urgency else 1)
            self.goal = None
            forced = [s for s in self.plan if s.get("forced")]
            self.events.append((t, f"'{text}': yes (hardest link {max(links):.0f} bits, chance {p:.0%})"
                                   + (" - straight across!" if forced else "")))
        else:
            self.events.append((t, f"'{text}': not now (hardest link {max(links):.0f} bits, chance {p:.0%})"))

    def _social(self, t):
        """A person is there and asks nothing: the dog says something of its own, aimed at them."""
        words = self.d.get("social", [])
        if not words or self.G is None or self.person is None:
            return
        text = words[int(self.rng.choice(len(words), p=[w for _, w in words]))][0]
        res = self.G.price(text, dog=self.d["name"], start=self.node, state=self.state())
        self.plan = self._plan_from(res, self.person, 0.0, heard=False)
        for s in self.plan:  # her own idea: toward the person
            if s["reg"] in ("walk", "trot", "run"):
                s["target"] = self.person
        self.plan_from = t
        self.ctx = "interact_person"
        self.events.append((t, f"(her own idea) {text}"))

    def _run_plan(self, t, dt, pos):
        """One step of the current plan: returns the heading wanted, or None."""
        s = self.plan[0]
        want = None
        if s["reg"] is None:  # an overlay: the body holds while the mouth or eyes do it
            if s.get("event") and not s.get("_logged"):
                self.events.append((t, s["event"].replace("_", " ")))
                s["_logged"] = True
            s["secs"] -= dt
            if s["secs"] <= 0:
                self.plan.pop(0)
            return None
        if not s.get("_begun"):
            s["_begun"] = True
            if self.reg == s["reg"]:
                self.elapsed = 0.0
            if s.get("event") and s["event"] != s["reg"]:
                self.events.append((t, s["event"].replace("_", " ")))
        if self.reg != s["reg"]:
            self.forced = bool(s.get("forced"))
            self._enter(s["reg"], t, ("! straight across: " if s.get("forced") else "") + (s["said"] or s["reg"]))
            self.node = s.get("node") or REP_NODE.get(s["reg"], s["reg"])
        if s.get("target") is not None:
            gap = s["target"] - pos
            dist = float(np.hypot(*gap))
            want = math.atan2(gap[1], gap[0])
            # a dash at a playmate ends close to it (a dog can't stop on a point at a run) or after a few seconds
            done = (dist < 1.5 or self.elapsed > 5.0) if s.get("play") else (dist < 1.0 or self.elapsed > 20.0)
        else:
            done = self.elapsed >= s["secs"]
        if done:
            self.plan.pop(0)
            self.forced = False
            if not self.plan and self.ctx != "play_dog":  # a bout of play outlasts the dash that started it
                self.ctx = "interact_person" if self.person is not None else self.ctx
                self.dur = self._duration(self.reg)
                self.elapsed = 0.0
        return want

    # ---------- routing ----------
    def _duration(self, reg):
        prior = PRIOR_DUR.get(self.ctx, PRIOR_DUR["default"]).get(reg, 3.0)
        pool = self.R.durs.get(self.d["name"], {}).get(self.ctx, {}).get(reg, [])
        if pool and self.rng.random() < self.R.trust(self.d, self.ctx):
            base = float(self.rng.choice(pool))
        else:
            base = float(prior * math.exp(0.6 * self.rng.standard_normal()))
        if reg in STATIC:  # tired dogs stay put longer, roused dogs less
            base *= (1 + 1.5 * self.F) * (1 - 0.5 * min(self.A, 1)) * (self.d["rest_scale"] if reg in LYING else 1)
        else:
            base *= max(0.3, 1 - 0.5 * self.F)
        return max(1.0, base)

    def _next(self):
        if self.reg not in BODY:  # after a heard jump, rear-up or roll: back on its feet first
            return "stand"
        w = self.R.trust(self.d, self.ctx)
        learned = self.R.next_probs(self.d, self.reg, self.ctx, self.elapsed, self.E > 0.35)
        prior = PRIOR_NEXT.get(self.ctx, PRIOR_NEXT["default"]).get(self.reg, {})
        p = np.array([(w * learned[r] + (1 - w) * prior.get(r, 0.0)) if r != self.reg else 0.0 for r in BODY])
        if self.F > 0.3:  # tired: lying down gains, running loses
            p *= np.array([1.0 + (2.0 * self.F if r in LYING else -0.8 * self.F if r in ("run", "trot") else 0) for r in BODY])
        p = np.clip(p, 0, None)
        return BODY[int(self.rng.choice(len(BODY), p=p / p.sum()))]

    def _enter(self, reg, t, why=""):
        if reg != self.reg:
            self.reg, self.elapsed = reg, 0.0
            self.dur = self._duration(reg)
            if not self.plan:
                self.node = REP_NODE.get(reg, reg)
            if why:
                self.events.append((t, why))

    # ---------- requests ----------
    def request(self, t, kind, target):
        self.A = min(1.5, self.A + 0.3)
        self.pending.append((t + 0.3, kind, np.asarray(target, float)))

    def _decide(self, t, kind, target):
        r = self.d["request"]
        z = r["bias"] + r["arousal"] * self.A - r["fatigue"] * self.F - r["lying"] * (self.reg in LYING) \
            + r["repeat"] * self.n_asked
        p = 1 / (1 + math.exp(-z))
        self.n_asked += 1
        if self.goal is None and self.rng.random() < p:
            lat = r["latency"] * math.exp(0.4 * self.rng.standard_normal())
            self.goal, self.goal_from, self.phase = (kind, target), t + lat, "rise"
            self.events.append((t, f"'{kind}': yes (chance {p:.0%}), moves in {lat:.1f} s"))
        elif self.goal is None:
            self.events.append((t, f"'{kind}': not now (chance {p:.0%})"))

    # ---------- the body reports back (physics) ----------
    def fell(self, t):
        """The body fell (the physics said so, not the plan): a startle, and whatever was going on is interrupted."""
        self.A = min(1.5, self.A + 0.5)
        self.events.append((t, "falls"))

    def got_up(self, t):
        """Back on its feet after a fall: standing again; a goal in progress carries on."""
        self._enter("stand", t, "gets back up")
        if self.goal and self.phase == "rise":
            self.phase = "go"

    # ---------- the tick ----------
    def tick(self, t, dt, pos):
        """Returns (regime, speed, wanted heading or None)."""
        self.last_pos = np.asarray(pos, float)
        for item in [x for x in self.pending if x[0] <= t]:
            self.pending.remove(item)
            if item[1] == "words":
                self._take_words(t, *item[2])
            else:
                self._decide(t, item[1], item[2])
        if self.person is not None and not self.plan and self.goal is None and self.next_social is not None \
                and t >= self.next_social and self.ctx != "play_dog":
            self._social(t)
            self.next_social = t + self.d.get("social_every", 30.0) * math.exp(0.4 * self.rng.standard_normal())
        if self.other is not None and self._free() and t >= self.next_play:
            near = float(np.hypot(*(self.other - pos))) < 5.0
            self.next_play = t + self.d.get("play_every", 40.0) * math.exp(0.5 * self.rng.standard_normal())
            if near and self.reg not in LYING and self.E < 0.5 and self.rng.random() < 0.5 + 0.5 * min(self.A, 1.0):
                self._invite(t)
        if self.ctx == "play_dog" and (t >= self.play_until or self.E > 0.6):  # the bout ends, or she is winded
            self.ctx = "rest"
            self.events.append((t, "stops playing" + (" (winded)" if self.E > 0.6 else "")))
        want = None
        if self.plan and t >= self.plan_from:
            want = self._run_plan(t, dt, pos)
        elif self.plan:
            pass  # heard, about to move (latency)
        elif self.goal and t >= self.goal_from:
            kind, target = self.goal
            gap = target - pos
            dist = float(np.hypot(*gap))
            if self.phase == "rise":
                self._enter("stand", t, "gets up")
                if self.elapsed > 1.2:
                    self.phase = "go"
            elif self.phase == "go":
                if dist > 1.0:
                    self._enter("trot" if dist > 6 and self.A > 0.6 else "walk", t, "comes")
                    want = math.atan2(gap[1], gap[0])
                else:
                    self.phase = "arrive"
                    self._enter("stand", t, "arrives")
            elif self.phase == "arrive" and self.elapsed > 1.0:
                self._enter(self.d["arrive"], t, f"{self.d['arrive']}s by you")
                self.goal, self.phase, self.ctx, self.n_asked = None, None, "interact_person", 0
        elif self.elapsed >= self.dur:
            nxt = self._next()
            if self.ctx == "play_dog" and nxt in ("walk", "trot", "run"):  # chase and be chased, in turn
                self.play_role = "flee" if self.play_role == "chase" else "chase"
            self._enter(nxt, t, f"shifts to {nxt}" + (f" ({self.play_role})" if self.ctx == "play_dog" else ""))
        if want is None and self.ctx == "play_dog" and self.other is not None and not self.plan \
                and self.reg in ("walk", "trot", "run"):
            gap = self.other - pos
            dist = float(np.hypot(*gap))
            toward = math.atan2(gap[1], gap[0])
            if self.play_role == "chase":  # at it; on top of it, around it
                want = toward + (math.pi / 2 if dist < 1.2 else 0.0)
            else:  # away while it is close; with room, back toward it (play stays together)
                want = toward + (math.pi if dist < 3.0 else 0.0)
        if self.ctx == "interact_person" and self.reg in STATIC and self.elapsed > 20:
            self.ctx = "rest"
        # physiology
        x = intensity(self.reg, self.ctx)
        self.E += (1 - math.exp(-dt / 30.0)) * (x - self.E)
        # fatigue: 20 min, not 5. In the one long session (IMG_2650, 6 min of wrestling) neither dog slows
        # (footage/learn/fatigue.py); a 5 min integrator would already be shortening their bouts by then
        self.F += (1 - math.exp(-dt / 1200.0)) * (x - self.F)
        self.A += -self.A * dt / 20.0
        self.elapsed += dt
        self.breath_phase += 2 * math.pi * self.breath_rate() / 60.0 * dt
        return self.reg, SPEED[self.reg], want

    def panting(self):
        return self.E > 0.35

    def muscle_capacity(self):
        """Blood and breath as a determiner of the body: the share of its muscles' strength the dog has now.
        Fatigue (20 min) takes up to 35% (exhausting exercise lowers a muscle's peak force by a fifth to two
        fifths); being winded (exertion past panting) up to 15% more; arousal adds up to 8% (adrenaline).
        The physics (train/dog_track.py, info 'cap') scales every muscle's strength by it, so a tired dog asked
        to take a hard link straight across is likelier to fall, as the grammar's 'tired' bits say."""
        winded = max(0.0, self.E - 0.35) / 0.65
        c = (1 - 0.35 * min(self.F, 1.0)) * (1 - 0.15 * min(winded, 1.0)) * (1 + 0.08 * min(self.A, 1.0))
        return float(min(1.1, max(0.5, c)))

    def breath_rate(self):
        if self.panting():
            return self.d["pant"]
        k = min(1.0, self.E / 0.35) ** 1.2
        return self.d["breath_rest"] + (self.d["breath_max"] - self.d["breath_rest"]) * k + 15 * min(self.A, 1.0)

    def breath_offset(self):
        """Chest motion for the body: deep and slow at rest, shallow and fast when panting (radians)."""
        amp = 0.012 if self.panting() else 0.03
        return amp * math.sin(self.breath_phase)

    def tail_wag(self, t):
        """Arousal shows in the tail: side-to-side, wider when roused (radians)."""
        a = max(0.0, min(self.A, 1.0) - 0.15)
        return 0.5 * a * math.sin(2 * math.pi * 2.5 * t)
