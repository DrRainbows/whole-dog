"""The dog game, first playable: a physics dog with its own brain, in Panda3D, that you can talk to and shove.

  body\\.venv\\Scripts\\python game\\dog_game.py [--dog lab|mal] [--start stand] [--seed 3]
  body\\.venv\\Scripts\\python game\\dog_game.py --shots 3,12,14 --out shots/   (offscreen: save frames, no window)
  body\\.venv\\Scripts\\python game\\dog_game.py --video 20 --say "2:platz|6:jump:1.5" --out demo.mp4

Type to the dog and press Enter (it hears words, in its own grammar: "platz", "come", "go to the fridge door ...").
Quick keys: F1 sit, F2 down, F3 come, F4 stand, F5 jump, F6 shake.   F8: shove it.   Tab: switch the camera.
Arrow keys: you are the person; walk (relative to the camera). The dog's brain knows where you are: "come" means to you.
Things: a red toy lies near the dog, and a bottle stands where the dog is sent for named things ("get the beer"). When
its brain says "grab" near one, the dog carries it in its mouth until "release" or "present".
The simulation runs at 50 Hz in fixed steps (game/dog_runtime.py); the picture is MuJoCo's own geometry, drawn each
frame from the physics.
"""
import argparse
import math
import os
import subprocess
import sys
import time

import mujoco
import numpy as np
from panda3d.core import (AmbientLight, DirectionalLight, Geom, GeomNode, GeomTriangles, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, LQuaternionf, TextNode, loadPrcFileData, CardMaker,
                          Texture, PNMImage, AntialiasAttrib, Filename)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = "runs/body-runs"
QUICK = {"f1": "sit", "f2": "down", "f3": "come", "f4": "stand", "f5": "jump", "f6": "shake"}


def capsule_geom(radius, half, rgba, seg=16, rings=6):
    """A capsule along +z (MuJoCo's convention): cylinder of half-length `half` with hemispherical caps."""
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("cap", fmt, Geom.UHStatic)
    vw, nw, cw = (GeomVertexWriter(vdata, n) for n in ("vertex", "normal", "color"))
    rows = []
    # bottom cap (rings), cylinder (two rows), top cap (rings)
    for k in range(rings + 1):
        a = -math.pi / 2 + (math.pi / 2) * k / rings
        rows.append((math.cos(a), math.sin(a), -half))
    for k in range(rings + 1):
        a = (math.pi / 2) * k / rings
        rows.append((math.cos(a), math.sin(a), half))
    for (cr, sz, zc) in rows:
        for j in range(seg):
            phi = 2 * math.pi * j / seg
            nx, ny, nz = cr * math.cos(phi), cr * math.sin(phi), sz
            vw.addData3(radius * nx, radius * ny, zc + radius * nz)
            nw.addData3(nx, ny, nz)
            cw.addData4(*rgba)
    tris = GeomTriangles(Geom.UHStatic)
    for r in range(len(rows) - 1):
        for j in range(seg):
            a, b = r * seg + j, r * seg + (j + 1) % seg
            c, d = a + seg, b + seg
            tris.addVertices(a, b, d)
            tris.addVertices(a, d, c)
    g = Geom(vdata)
    g.addPrimitive(tris)
    node = GeomNode("capsule")
    node.addGeom(g)
    return node


def cone_geom(rx, ry, h, rgba, seg=18):
    """A cone along +z with an elliptical base (rx by ry) at z = 0: an upright ear, broad and thin."""
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("cone", fmt, Geom.UHStatic)
    vw, nw, cw = (GeomVertexWriter(vdata, n) for n in ("vertex", "normal", "color"))
    tris = GeomTriangles(Geom.UHStatic)
    for j in range(seg):  # sides: base point, next base point, apex (own normals per face, flat shaded)
        a0, a1 = 2 * math.pi * j / seg, 2 * math.pi * (j + 1) / seg
        p0 = np.array([rx * math.cos(a0), ry * math.sin(a0), 0.0])
        p1 = np.array([rx * math.cos(a1), ry * math.sin(a1), 0.0])
        top = np.array([0.0, 0.0, h])
        n = np.cross(p1 - p0, top - p0)
        n /= np.linalg.norm(n) + 1e-12
        for v in (p0, p1, top):
            vw.addData3(*v)
            nw.addData3(*n)
            cw.addData4(*rgba)
        tris.addVertices(3 * j, 3 * j + 1, 3 * j + 2)
    o = 3 * seg  # base
    vw.addData3(0, 0, 0)
    nw.addData3(0, 0, -1)
    cw.addData4(*rgba)
    for j in range(seg):
        a0 = 2 * math.pi * j / seg
        vw.addData3(rx * math.cos(a0), ry * math.sin(a0), 0.0)
        nw.addData3(0, 0, -1)
        cw.addData4(*rgba)
    for j in range(seg):
        tris.addVertices(o, o + 1 + (j + 1) % seg, o + 1 + j)
    g = Geom(vdata)
    g.addPrimitive(tris)
    node = GeomNode("cone")
    node.addGeom(g)
    return node


def box_geom(sx, sy, sz, rgba):
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("box", fmt, Geom.UHStatic)
    vw, nw, cw = (GeomVertexWriter(vdata, n) for n in ("vertex", "normal", "color"))
    tris = GeomTriangles(Geom.UHStatic)
    faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
             ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0))]
    s = np.array([sx, sy, sz])
    for fi, (n, u, v) in enumerate(faces):
        n, u, v = np.array(n), np.array(u), np.array(v)
        for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = (n + a * u + b * v) * s
            vw.addData3(*p)
            nw.addData3(*n)
            cw.addData4(*rgba)
        o = fi * 4
        tris.addVertices(o, o + 1, o + 2)
        tris.addVertices(o, o + 2, o + 3)
    g = Geom(vdata)
    g.addPrimitive(tris)
    node = GeomNode("box")
    node.addGeom(g)
    return node


BREED = {  # body, then mask / ears / muzzle, then nose and eyes
    "mal": ((0.76, 0.58, 0.36, 1), (0.13, 0.11, 0.10, 1), (0.05, 0.05, 0.05, 1), "erect"),
    "lab": ((0.92, 0.80, 0.56, 1), (0.86, 0.72, 0.48, 1), (0.18, 0.12, 0.10, 1), "drop"),
}


def quat_z_to(v):
    """The rotation taking +z onto direction v (Panda3D quaternion)."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, v)
    s, c = np.linalg.norm(axis), float(z @ v)
    if s < 1e-9:
        return LQuaternionf(1, 0, 0, 0) if c > 0 else LQuaternionf(0, 1, 0, 0)
    axis /= s
    ang = math.atan2(s, c)
    return LQuaternionf(math.cos(ang / 2), *(axis * math.sin(ang / 2)))


def mat_to_quat(xmat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, float).reshape(9))
    return LQuaternionf(q[0], q[1], q[2], q[3])


def parse_says(s):
    out = []
    for item in s.split("|"):
        if item.strip():
            p = item.split(":")
            out.append((float(p[0]), p[1], float(p[2]) if len(p) > 2 else 0.0))
    return sorted(out)


class DogDraw:
    """One dog on screen: its capsules (drawn from the physics every frame), its breed's face, and what it carries."""

    def __init__(self, render, rt):
        self.rt, self.render = rt, render
        self.breed = BREED[rt.dog_name]
        m = rt.model
        self.geoms = []
        for g in range(m.ngeom):
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            rgba = tuple(float(c) for c in m.geom_rgba[g])
            if rgba[3] == 0:
                continue
            paw = rgba[0] < 0.3 and rgba[1] < 0.3 and rgba[2] < 0.3
            rgba = rgba if paw else self.breed[0]  # the breed's colour; paw pads stay dark
            s = m.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                node = capsule_geom(float(s[0]), float(s[1]), rgba)
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                node = capsule_geom(float(s[0]), 0.0, rgba)
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                node = box_geom(float(s[0]), float(s[1]), float(s[2]), rgba)
            else:  # ellipsoid / cylinder / mesh: a capsule of its largest radius is enough for now
                node = capsule_geom(float(max(s[0], s[1])), float(s[2]) if t == mujoco.mjtGeom.mjGEOM_CYLINDER else 0.0,
                                    rgba)
            self.geoms.append((g, render.attachNewNode(node)))
        self._head_dressing()
        self.ev_seen = 0
        self.place()

    def _head_dressing(self):
        """Ears, muzzle, eyes and nose, fixed to the head body (drawn only; the physics is the capsules). In the
        head's own frame the skull runs along +x; its up is how the capture's dogs hold their heads standing."""
        rt, m = self.rt, self.rt.model
        self.head_id = m.body("Head").id
        dk = mujoco.MjData(m)
        ups = []
        for j in rt.standing[:: max(1, len(rt.standing) // 200)]:
            dk.qpos[:] = rt.mm.db["qpos"][j]
            mujoco.mj_kinematics(m, dk)
            ups.append(dk.xmat[self.head_id].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0]))
        x = np.array([1.0, 0.0, 0.0])
        u = np.median(ups, 0)
        u = u - (u @ x) * x
        u /= np.linalg.norm(u)
        sd = np.cross(x, u)
        self.mouth_local = 0.27 * x - 0.045 * u  # where a carried thing sits, in the head's frame
        self.mouth_across = sd  # a bottle is held across the mouth
        _, mask, dark, ears = self.breed
        self.head = self.render.attachNewNode("head")

        def part(node, pos, direction, flat=None):
            n = self.head.attachNewNode(node)
            n.setPos(*[float(v) for v in pos])
            if flat is None:
                n.setQuat(quat_z_to(direction))
            else:  # orient +z along direction and the node's x across `flat` (the thin way)
                z = np.asarray(direction, float) / np.linalg.norm(direction)
                xa = np.asarray(flat, float) - (np.asarray(flat, float) @ z) * z
                xa /= np.linalg.norm(xa)
                ya = np.cross(z, xa)
                R = np.stack([xa, ya, z], 1)
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, R.reshape(9))
                n.setQuat(LQuaternionf(*q))
            return n

        part(capsule_geom(0.034, 0.04, mask), 0.235 * x - 0.016 * u, x)  # muzzle, out past the skull
        part(capsule_geom(0.02, 0.0, dark), 0.30 * x + 0.016 * u, x)  # nose, on the top front of the muzzle
        for sgn in (1, -1):
            part(capsule_geom(0.009, 0.0, dark), 0.135 * x + 0.03 * u + sgn * 0.033 * sd, x)  # eyes
            if ears == "erect":  # the Malinois: broad upright triangles, a little out, thin front to back
                part(cone_geom(0.008, 0.032, 0.075, mask), 0.04 * x + 0.04 * u + sgn * 0.035 * sd, u + 0.3 * sgn * sd,
                     flat=x)
            else:  # the Lab: soft ears that hang against the cheeks, thin side to side
                n = part(capsule_geom(0.026, 0.025, mask), 0.055 * x - 0.02 * u + sgn * 0.055 * sd,
                         -u + 0.2 * sgn * sd + 0.3 * x, flat=sd)
                n.setScale(0.35, 1.0, 1.0)  # flatten across the face

    def place(self):
        d = self.rt.data
        gx, gm = d.geom_xpos, d.geom_xmat
        for g, np_ in self.geoms:
            np_.setPos(float(gx[g][0]), float(gx[g][1]), float(gx[g][2]))
            np_.setQuat(mat_to_quat(gm[g]))
        hp, hq = d.xpos[self.head_id], d.xquat[self.head_id]
        self.head.setPos(float(hp[0]), float(hp[1]), float(hp[2]))
        self.head.setQuat(LQuaternionf(float(hq[0]), float(hq[1]), float(hq[2]), float(hq[3])))

    def mouth(self):
        d = self.rt.data
        R = d.xmat[self.head_id].reshape(3, 3)
        return d.xpos[self.head_id] + R @ self.mouth_local, R @ self.mouth_across

    def carry(self, things):
        """Grab, carry, release: this dog's own plan events move the things."""
        ev = self.rt.brain.events
        new, self.ev_seen = ev[self.ev_seen:], len(ev)
        m, across = self.mouth()
        for _, msg in new:
            if msg == "grab" and not any(t["held"] is self for t in things):
                free = [t for t in things if t["held"] is None]
                if free:
                    near = min(free, key=lambda t: np.linalg.norm(t["pos"] - m))
                    if np.linalg.norm(near["pos"][:2] - m[:2]) < 1.5:  # within reach of a step and a dip of the head
                        near["held"] = self
            elif msg in ("release", "present", "drop"):
                for t in things:
                    if t["held"] is self:
                        t["held"] = None
                        t["pos"] = np.array([m[0], m[1], t["r"] if t["name"] == "toy" else 0.1])
                        t["node"].setQuat(LQuaternionf(1, 0, 0, 0))
                        t["node"].setPos(*[float(v) for v in t["pos"]])
        for t in things:
            if t["held"] is self:
                t["pos"] = m.copy()
                t["node"].setPos(*[float(v) for v in m])
                t["node"].setQuat(quat_z_to(across))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", default="lab", choices=["lab", "mal"])
    ap.add_argument("--dogs", default="", help="several dogs in one world, e.g. lab,mal (overrides --dog)")
    ap.add_argument("--start", default="", help="the regime the dogs start in (default: their brains decide)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tracker", default=f"{RUNS}/anat_tired")
    ap.add_argument("--reflex", default=f"{RUNS}/scramble_up_dog3_72M")
    ap.add_argument("--alone", action="store_true", help="no person: the dogs live on their own")
    ap.add_argument("--shots", default="", help="offscreen: save frames at these simulated seconds, then quit")
    ap.add_argument("--video", type=float, default=0.0, help="offscreen: record this many seconds to --out (mp4)")
    ap.add_argument("--skip", type=float, default=0.0, help="with --video: simulate this many seconds before recording")
    ap.add_argument("--say", default="", help="scripted words, 'T:words[:urgency]|...' (for shots and videos)")
    ap.add_argument("--out", default=os.path.join(HERE, "shots"))
    ap.add_argument("--cam", type=int, default=0, help="camera: 0 behind the dog, 1 your eyes, 2 in front of the dog, "
                    "3 all the dogs")
    ap.add_argument("--selftest", action="store_true", help="offscreen: drive the interactive controls, save a frame")
    a = ap.parse_args()
    dogs = [d.strip() for d in a.dogs.split(",") if d.strip()] or [a.dog]
    offscreen = bool(a.shots or a.video or a.selftest)
    interactive = not offscreen or a.selftest
    loadPrcFileData("", "window-title Dog\nwin-size 1280 720\nframebuffer-multisample 1\nmultisamples 4\n"
                    "sync-video 1\n" + ("window-type offscreen\naudio-library-name null\n" if offscreen else ""))
    from direct.gui.DirectGui import DirectEntry
    from direct.gui.OnscreenText import OnscreenText
    from direct.showbase.ShowBase import ShowBase
    from dog_runtime import DT, Pack

    class Game(ShowBase):
        def __init__(self):
            super().__init__()
            self.disableMouse()
            self.camLens.setNearFar(0.05, 200.0)  # Panda3D's default near plane (1 m) cut the dog's face off up close
            self.setBackgroundColor(0.62, 0.72, 0.82, 1)
            self.render.setAntialias(AntialiasAttrib.MMultisample)
            self.render.setShaderAuto()  # per-pixel lighting
            t0 = time.time()
            self.pack = Pack(dogs, a.tracker, a.reflex, present=not a.alone, seed=a.seed, start=a.start or None)
            print(f"{len(dogs)} dog(s) ready in {time.time() - t0:.1f} s", flush=True)
            self.person = self.pack.person
            self._scene()
            self.draws = [DogDraw(self.render, rt) for rt in self.pack.dogs]
            self.follow = 0
            self._things()
            self._hud()
            self.says = parse_says(a.say)
            self.acc, self.last = 0.0, None
            self.cam_mode = a.cam
            self.cam_pos = None
            self.shove_left = 0
            self.taskMgr.add(self.tick, "tick")

        @property
        def rt(self):  # the dog the camera follows
            return self.pack.dogs[self.follow]

        # -- the world ----------------------------------------------------------------------------------------
        def _scene(self):
            img = PNMImage(256, 256)
            for x in range(256):
                for y in range(256):
                    c = 0.55 if ((x // 32) + (y // 32)) % 2 else 0.47
                    img.setXel(x, y, c, c * 0.98, c * 0.9)
            tex = Texture()
            tex.load(img)
            cm = CardMaker("ground")
            cm.setFrame(-40, 40, -40, 40)
            cm.setUvRange((0, 0), (20, 20))
            g = self.render.attachNewNode(cm.generate())
            g.setP(-90)
            g.setTexture(tex)
            sun = DirectionalLight("sun")
            sun.setColor((0.9, 0.88, 0.82, 1))
            sn = self.render.attachNewNode(sun)
            sn.setHpr(30, -55, 0)
            self.render.setLight(sn)
            amb = AmbientLight("amb")
            amb.setColor((0.42, 0.44, 0.5, 1))
            self.render.setLight(self.render.attachNewNode(amb))
            self.person_node = None
            if not a.alone:  # the person (you): every brain reads this very array, so moving it re-aims the dogs
                self.person_node = self.render.attachNewNode(capsule_geom(0.17, 0.62, (0.35, 0.45, 0.75, 1)))
                self.person_node.setPos(float(self.person[0]), float(self.person[1]), 0.8)

        def _things(self):
            """Objects a dog can pick up: a toy near the dogs, and a bottle where named things are (`where`)."""
            self.things = []
            toy = self.render.attachNewNode(capsule_geom(0.045, 0.0, (0.85, 0.15, 0.12, 1)))
            beer = self.render.attachNewNode(capsule_geom(0.032, 0.07, (0.35, 0.2, 0.08, 1)))
            for name, node, pos, r in (("toy", toy, (1.2, -0.8), 0.045), ("beer", beer, tuple(self.rt.where), 0.032)):
                p3 = np.array([pos[0], pos[1], r if name == "toy" else 0.1])
                node.setPos(*[float(v) for v in p3])
                self.things.append({"name": name, "node": node, "pos": p3, "r": r, "held": None})

        # -- what you see and say -----------------------------------------------------------------------------
        def _hud(self):
            self.labels = [OnscreenText(text="", pos=(-1.75, 0.92 - 0.055 * k), scale=0.042, align=TextNode.ALeft,
                                        fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.8), mayChange=True)
                           for k in range(len(self.pack.dogs))]
            self.events = OnscreenText(text="", pos=(-1.75, 0.90 - 0.055 * len(self.pack.dogs)), scale=0.036,
                                       align=TextNode.ALeft, fg=(1, 0.95, 0.8, 1), shadow=(0, 0, 0, 0.8), mayChange=True)
            many = len(self.pack.dogs) > 1
            OnscreenText(text="type and Enter to speak" + (" (start with a name to speak to one: 'mal, sit')" if many
                                                           else "") +
                              "  |  F1 sit F2 down F3 come F4 stand F5 jump F6 shake  |  F8 shove" +
                              ("  F7 next dog" if many else "") + "  |  arrows: walk  |  Tab camera",
                         pos=(-1.75, -0.95), scale=0.033, align=TextNode.ALeft, fg=(1, 1, 1, 0.85),
                         shadow=(0, 0, 0, 0.8))
            if interactive:
                self.entry = DirectEntry(text="", scale=0.05, pos=(-1.75, 0, -0.86), width=30, numLines=1,
                                         focus=1, command=self.spoken, initialText="")
                for k, w in QUICK.items():
                    self.accept(k, self.quick, [w])
                self.accept("f8", self.shove)
                self.accept("f7", self.next_dog)
                self.keys = {k: False for k in ("arrow_up", "arrow_down", "arrow_left", "arrow_right")}
                for k in self.keys:
                    self.accept(k, self.keys.__setitem__, [k, True])
                    self.accept(k + "-up", self.keys.__setitem__, [k, False])
                self.accept("tab", self.switch_cam)
                self.accept("escape", sys.exit)

        def say(self, text, urgency=0.0):
            """To every dog, or to one when the sentence starts with its name ("mal, sit", "lab come")."""
            words = text.strip()
            low = words.lower()
            for rt in self.pack.dogs:
                for name in (rt.dog_name, {"mal": "malinois", "lab": "labrador"}[rt.dog_name]):
                    if low.startswith(name):
                        rt.say(words[len(name):].lstrip(" ,:;"), urgency)
                        return
            for rt in self.pack.dogs:
                rt.say(words, urgency)

        def spoken(self, text):
            if text.strip():
                self.say(text)
            self.entry.enterText("")
            self.entry["focus"] = 1

        def quick(self, word):
            self.say(word)

        def shove(self):
            self.shove_left = 8  # 0.16 s, on the dog the camera follows

        def next_dog(self):
            self.follow = (self.follow + 1) % len(self.pack.dogs)

        def switch_cam(self):
            self.cam_mode = (self.cam_mode + 1) % 4

        # -- the clock ---------------------------------------------------------------------------------------
        def sim_step(self):
            while self.says and self.pack.t >= self.says[0][0] - 1e-9:
                _, text, urg = self.says.pop(0)
                self.say(text, urg)
            if self.shove_left > 0:  # sideways to the dog, half its weight
                self.shove_left -= 1
                h = self.rt.hp + math.pi / 2
                f = 0.5 * self.rt.mg
                self.rt.push_force[0:2] = [math.cos(h) * f, math.sin(h) * f]
            self.pack.step()

        def tick(self, task):
            now = time.time()
            if offscreen:  # recording: exactly 2 physics steps per 25 fps frame
                steps = 2
            else:
                self.acc += 0.0 if self.last is None else min(now - self.last, 0.1)
                steps = int(self.acc / DT)
                self.acc -= steps * DT
            self.last = now
            for _ in range(steps):
                self.sim_step()
            self._move_person(steps * DT)
            self.redraw()
            return task.cont

        def redraw(self):
            for dd in self.draws:
                dd.place()
                dd.carry(self.things)
            self._camera()
            for lab, rt in zip(self.labels, self.pack.dogs):
                lab["text"] = ("> " if rt is self.rt and len(self.pack.dogs) > 1 else "") + rt.label
            self.events["text"] = "\n".join(self.rt.recent_events(4))

        def _move_person(self, dt):
            keys = getattr(self, "keys", None)
            if not keys or self.person_node is None or dt <= 0:
                return
            fwd = keys["arrow_up"] - keys["arrow_down"]
            side = keys["arrow_left"] - keys["arrow_right"]
            if not (fwd or side):
                return
            h = math.radians(self.camera.getH()) + math.pi / 2  # the camera's view direction on the ground
            v = 1.4 * dt  # a walking person, m/s
            step = np.array([math.cos(h) * fwd - math.sin(h) * side, math.sin(h) * fwd + math.cos(h) * side])
            self.person[:] = self.person + v * step / max(np.linalg.norm(step), 1e-9)  # in place: the brains see it
            self.person_node.setPos(float(self.person[0]), float(self.person[1]), 0.8)

        def _camera(self):
            d = self.rt.data
            hips = d.xpos[self.rt.hips_id]
            target = np.array([hips[0], hips[1], 0.35])
            if self.cam_mode == 0:  # behind and to the side of the dog, smoothed
                h = self.rt.hp
                want = target + np.array([-2.6 * math.cos(h) + 1.4 * math.cos(h + math.pi / 2),
                                          -2.6 * math.sin(h) + 1.4 * math.sin(h + math.pi / 2), 1.3])
            elif self.cam_mode == 1:  # the person's eyes
                want = np.array([self.person[0], self.person[1], 1.6])
            elif self.cam_mode == 3:  # all the dogs (and you), from the side of the group
                pts = np.array([rt.data.xpos[rt.hips_id][:2] for rt in self.pack.dogs] +
                               ([self.person.copy()] if self.person_node is not None else []))
                c = pts.mean(0)
                spread = max(1.5, float(np.linalg.norm(pts - c, axis=1).max()))
                target = np.array([c[0], c[1], 0.35])
                want = target + np.array([0.0, -1.0, 0.55]) * (2.2 + 1.6 * spread)
            else:  # in front of the dog, looking at its face
                h = self.rt.hp
                head = d.xpos[self.draws[self.follow].head_id]
                target = np.array([head[0], head[1], head[2]])
                want = target + np.array([1.1 * math.cos(h) + 0.45 * math.cos(h + math.pi / 2),
                                          1.1 * math.sin(h) + 0.45 * math.sin(h + math.pi / 2), 0.12])
            self.cam_pos = want if self.cam_pos is None else self.cam_pos + 0.06 * (want - self.cam_pos)
            self.camera.setPos(*[float(v) for v in self.cam_pos])
            self.camera.lookAt(float(target[0]), float(target[1]), float(target[2]))

    game = Game()
    if a.selftest:  # every control, as a player would use them
        game.spoken("platz")
        for _ in range(100):
            game.taskMgr.step()
        game.quick("come")
        game.shove()
        game.switch_cam()
        if len(game.pack.dogs) > 1:
            game.next_dog()
            game.spoken(f"{game.pack.dogs[1].dog_name}, sit")
        for _ in range(150):
            game.taskMgr.step()
        game.switch_cam()
        p0 = game.person.copy()
        game.keys["arrow_up"] = True  # the person walks off; the brains see it
        for _ in range(100):
            game.taskMgr.step()
        game.keys["arrow_up"] = False
        print("person walked", round(float(np.linalg.norm(game.person - p0)), 2), "m; brain sees",
              game.rt.brain.person.round(2), flush=True)
        game.quick("come")
        for _ in range(150):
            game.taskMgr.step()
        for rt in game.pack.dogs:
            hips = rt.data.xpos[rt.hips_id][:2]
            print(rt.dog_name, "to the person after 'come':", round(float(np.linalg.norm(hips - game.person)), 2), "m",
                  flush=True)
        game.spoken("go to the fridge door and fetch me a beer")
        for _ in range(100):
            game.taskMgr.step()
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, "selftest.png")
        game.win.saveScreenshot(Filename.fromOsSpecific(path))
        for rt in game.pack.dogs:
            heard = [e for e in rt.recent_events(40) if "person" in e or "yes" in e or "not" in e]
            print("SELFTEST", rt.dog_name, f"t={rt.t:.1f}s", heard[-6:], flush=True)
        print("SELFTEST ok, saved", path, flush=True)
        return
    if a.shots:
        os.makedirs(a.out, exist_ok=True)
        for ts in sorted(float(x) for x in a.shots.split(",")):
            while game.pack.t < ts:
                game.sim_step()
            game.cam_pos = None
            for _ in range(40):  # let the camera settle on the dog
                game.redraw()
            game.graphicsEngine.renderFrame()
            game.graphicsEngine.renderFrame()
            path = os.path.join(a.out, f"dog_{'_'.join(dogs)}_{ts:05.1f}s.png")
            game.win.saveScreenshot(Filename.fromOsSpecific(path))
            print("saved", path, flush=True)
        return
    if a.video:
        w, h = 1280, 720
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
                               f"{w}x{h}", "-r", "25", "-i", "-", "-vf", "vflip", "-c:v", "h264_nvenc", "-preset", "p4",
                               "-cq", "22", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
        while game.pack.t < a.skip:  # fast-forward, unseen
            game.sim_step()
        tex = Texture()
        game.win.addRenderTexture(tex, game.win.RTMCopyRam)
        n = int(a.video * 25)
        for _ in range(n):
            game.taskMgr.step()
            img = tex.getRamImageAs("RGBA")
            ff.stdin.write(bytes(img))
        ff.stdin.close()
        ff.wait()
        print("wrote", a.out, flush=True)
        return
    game.run()


if __name__ == "__main__":
    main()
