"""Motion matching over the labelled dog capture: regimes in, continuous motion out.

  .venv\\Scripts\\python motion_match.py demo [out.mp4]

The kinematic half of DReCon (Bergamin et al. 2019). Every 0.2 s the controller searches the whole
capture (refs/all, every frame named by catalog_mocap.py) for the frame that best continues the
current pose toward what is asked: a regime, a speed and a heading. It jumps there and blends.
Nothing is replayed whole; sequences are composed from pieces of 51 clips. A physics tracker
(train/) then makes the motion physical.
"""
import glob
import os
import subprocess
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUPPORT = ["stand", "walk", "trot", "run", "jump", "sit", "down", "side", "back", "crouch", "rear_up", "swim", "wade"]
SPEED = {"stand": 0.0, "walk": 1.0, "trot": 1.8, "run": 4.0, "jump": 3.0, "sit": 0.0, "down": 0.0, "side": 0.0,
         "back": 0.0, "crouch": 0.0, "rear_up": 0.0}
DT = 0.02
FUT = (10, 20, 30)  # future trajectory samples, frames (0.2, 0.4, 0.6 s)
REG_AHEAD = 15
PADS = ("LeftHand_pad", "RightHand_pad", "LeftFoot_pad", "RightFoot_pad")
# the capture the matcher plays from, and its cached database; DOG_MM_FLAT / DOG_MM_CACHE point them elsewhere
# (e.g. the capture re-expressed in anatomical joints, anat.py)
FLAT = os.environ.get("DOG_MM_FLAT", os.path.join(HERE, "refs", "flat"))
CACHE = os.environ.get("DOG_MM_CACHE", os.path.join(HERE, "refs", "mm_db.npz"))
W = {"pose": 1.0, "vel": 0.7, "traj": 1.5, "head": 1.0, "regime": 4.0}
LOW = {"down", "side", "sit", "crouch"}  # postures a dog settles into for a moment before changing again


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def yaw_quat(a):
    return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qslerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


# ---------- database ----------
def build_db(m):
    d = mujoco.MjData(m)
    hips = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Hips")
    neck = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Neck")
    pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in PADS]
    Q, R, H, N, P, CLIP = [], [], [], [], [], []
    for ci, f in enumerate(sorted(glob.glob(os.path.join(FLAT, "*.npz")))):
        z = np.load(f)
        for q in z["qpos"]:
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            H.append(d.xpos[hips].copy())
            N.append(d.xpos[neck].copy())
            P.append(np.array([d.geom_xpos[g] for g in pads]))
        Q.append(z["qpos"]), R.append(z["regime"]), CLIP.append(np.full(len(z["qpos"]), ci))
    db = {"qpos": np.concatenate(Q), "regime": np.concatenate(R).astype(int), "clip": np.concatenate(CLIP),
          "hips": np.array(H), "neck": np.array(N), "pads": np.array(P)}
    np.savez(CACHE, **db)
    return db


def load_db(m):
    if os.path.exists(CACHE):
        return dict(np.load(CACHE))
    return build_db(m)


def heading(db):
    v = db["neck"][:, :2] - db["hips"][:, :2]
    return np.arctan2(v[:, 1], v[:, 0])


def features(db):
    """Per frame, in that frame's own heading: paw positions and velocities, hip height and velocity,
    the trajectory 0.2-0.6 s ahead, and the regime 0.3 s ahead."""
    T = len(db["qpos"])
    h = heading(db)
    clip = db["clip"]
    end = np.zeros(T, int)
    for c in np.unique(clip):
        idx = np.where(clip == c)[0]
        end[idx] = idx[-1]
    hips, pads = db["hips"], db["pads"]
    nxt = np.minimum(np.arange(T) + 1, end)
    prv = np.where(clip[np.maximum(np.arange(T) - 1, 0)] == clip, np.maximum(np.arange(T) - 1, 0), np.arange(T))
    F = {k: [] for k in W}
    for i in range(T):
        Rm = rot2(-h[i])
        rel = (pads[i, :, :2] - hips[i, :2]) @ Rm.T
        F["pose"].append(np.concatenate([rel.ravel(), pads[i, :, 2], [hips[i, 2]]]))
        pv = (pads[nxt[i], :, :2] - pads[prv[i], :, :2]) @ Rm.T / (DT * max(1, nxt[i] - prv[i]))
        hv = (hips[nxt[i], :2] - hips[prv[i], :2]) @ Rm.T / (DT * max(1, nxt[i] - prv[i]))
        F["vel"].append(np.concatenate([pv.ravel(), hv]))
        fut = [min(i + k, end[i]) for k in FUT]
        F["traj"].append(np.concatenate([(hips[j, :2] - hips[i, :2]) @ Rm.T for j in fut]))
        F["head"].append(np.concatenate([[np.cos(h[j] - h[i]), np.sin(h[j] - h[i])] for j in fut]))
        oh = np.zeros(len(SUPPORT))
        oh[db["regime"][min(i + REG_AHEAD, end[i])]] = 1
        F["regime"].append(oh)
    F = {k: np.array(v, np.float32) for k, v in F.items()}
    scale = {k: (F[k].std(0).mean() + 1e-6) if k != "regime" else 1.0 for k in F}
    can = np.arange(T) <= end - max(FUT)  # a candidate needs its future inside its own clip
    return F, scale, end, can, h


class Matcher:
    def __init__(self, m):
        self.m = m
        self.db = load_db(m)
        self.F, self.scale, self.end, self.can, self.h = features(self.db)
        self.Z = np.concatenate([self.F[k] / self.scale[k] * W[k] for k in W], 1)
        self.cand = np.where(self.can)[0]
        self.graph = self._transitions()

    # ---------- which regimes lead to which (the capture's own posture graph) ----------
    def _transitions(self):
        """How often each regime is followed by each other one in the capture (between runs of 0.2 s or more,
        within a clip). A dog on its side, for one, only ever rolls onto its belly next: never straight up."""
        reg, clip = self.db["regime"], self.db["clip"]
        counts = np.zeros((len(SUPPORT), len(SUPPORT)))
        for c in np.unique(clip):
            r = reg[clip == c]
            runs, s = [], 0
            for t in range(1, len(r) + 1):
                if t == len(r) or r[t] != r[s]:
                    if t - s >= 10:
                        runs.append(int(r[s]))
                    s = t
            for a, b in zip(runs, runs[1:]):
                if a != b:
                    counts[a, b] += 1
        return counts

    def next_hop(self, cur, target):
        """The regime to ask for now so as to reach `target` from `cur` by the capture's most probable path
        (each step costs -log of how often the capture takes it). Unreachable or already there: `target`."""
        a, b = SUPPORT.index(cur), SUPPORT.index(target)
        if a == b or self.graph[a, b] >= 3:  # the capture goes there directly, often enough to find: let the search choose
            return target
        out = self.graph.sum(1, keepdims=True)
        with np.errstate(divide="ignore"):
            cost = np.where(self.graph > 0, -np.log(self.graph / np.maximum(out, 1)), np.inf)
        dist, first = np.full(len(SUPPORT), np.inf), [None] * len(SUPPORT)
        dist[a], todo = 0.0, set(range(len(SUPPORT)))
        while todo:
            u = min(todo, key=lambda i: dist[i])
            if not np.isfinite(dist[u]):
                break
            todo.remove(u)
            for v in range(len(SUPPORT)):
                if np.isfinite(cost[u, v]) and dist[u] + cost[u, v] < dist[v]:
                    dist[v] = dist[u] + cost[u, v]
                    first[v] = v if u == a else first[u]
        return SUPPORT[first[b]] if first[b] is not None else target

    def query(self, c, speed, dhead, regime):
        """Pose and velocity of the current frame c; desired trajectory from speed and heading change."""
        parts = []
        for k in W:
            if k in ("pose", "vel"):
                v = self.F[k][c]
            elif k == "traj":
                v = np.concatenate([speed * t * DT * np.array([np.cos(dhead * (1 - np.exp(-t * DT / 0.3))),
                                                                np.sin(dhead * (1 - np.exp(-t * DT / 0.3)))])
                                    for t in FUT])
            elif k == "head":
                v = np.concatenate([[np.cos(dhead * (1 - np.exp(-t * DT / 0.3))), np.sin(dhead * (1 - np.exp(-t * DT / 0.3)))]
                                    for t in FUT])
            else:
                v = np.zeros(len(SUPPORT))
                v[SUPPORT.index(regime)] = 1
            parts.append(v / self.scale[k] * W[k])
        return np.concatenate(parts).astype(np.float32)

    def best(self, q, regime=None):
        """Best frame for query q; with `regime`, only among frames in that regime, so a regime can change only by
        playing a transition the capture contains (a real roll, a real rise), never by blending between postures."""
        cand = self.cand
        if regime is not None:
            if not hasattr(self, "_by_reg"):
                self._by_reg = {r: self.cand[self.db["regime"][self.cand] == r] for r in range(len(SUPPORT))}
            sub = self._by_reg[SUPPORT.index(regime)]
            cand = sub if len(sub) else cand
        d = ((self.Z[cand] - q) ** 2).sum(1)
        j = int(np.argmin(d))
        return int(cand[j]), float(d[j])

    def cost(self, i, q):
        return float(((self.Z[i] - q) ** 2).sum())


class Player:
    """Plays the capture one 50 Hz frame at a time, steered live: ask for a regime, a speed and a heading."""

    def __init__(self, mm, start_regime, search_every=10, blend=10, margin=0.5, plan=True, same_regime=True, dwell=25):
        self.mm, self.db = mm, mm.db
        self.plan, self.same_regime = plan, same_regime
        self.dwell, self.since = dwell, dwell  # frames to stay in a regime just entered before starting another change
        self.c = int(mm.cand[np.argmax(self.db["regime"][mm.cand] == SUPPORT.index(start_regime))])
        self.pos, self.yaw = np.zeros(2), 0.0
        self.blend_t = 0  # inertialization: offsets from a jump decay to zero over `blend` frames
        self.f, self.last = 0, start_regime
        self.prev = None  # last output pose: hinge angles stay continuous with it
        self.search_every, self.blend, self.margin = search_every, blend, margin
        self.force_blend = False

    def resync(self, j, q_now, heading_now, regime, blend=None):
        """Continue from capture frame j, starting exactly at the pose q_now (for example the physics dog after a
        fall), placed at its position and heading; the difference fades over `blend` frames (inertialization)."""
        self.c = max(j - 1, 0)  # step() advances one frame
        self.pos, self.yaw = np.array(q_now[0:2], float), float(heading_now)
        self.prev = np.array(q_now, float)
        self.force_blend = True
        self.last, self.f = regime, 1  # no search on the very next frame
        if blend:
            self.blend = blend

    def step(self, reg, speed=None, want_yaw=None):
        mm, db = self.mm, self.db
        if self.plan:  # ask for the next regime on the capture's path to the one wanted (side -> down -> ... -> stand)
            hop = mm.next_hop(SUPPORT[db["regime"][self.c]], reg)
            if hop != reg:
                speed = None
            reg = hop
        speed = SPEED.get(reg, 0.0) if speed is None else speed
        dhead = 0.0 if want_yaw is None or speed == 0 else wrap(want_yaw - self.yaw)
        c = self.c
        jumped = False
        cur = SUPPORT[db["regime"][c]]
        # just entered this regime: let the clip carry on in it a moment before starting the next change (a dog
        # settles on its belly before standing), unless the clip itself is about to leave it
        settling = (self.same_regime and cur in LOW and reg != cur and self.since < self.dwell and c < mm.end[c] - 1
                    and db["regime"][c + 1] == db["regime"][c])
        if not settling and (self.f % self.search_every == 0 or c >= mm.end[c] - 1 or reg != self.last):
            q = mm.query(c, speed, dhead, reg)
            j, cj = mm.best(q, SUPPORT[db["regime"][c]] if self.same_regime else None)
            stay = c + 1 if c + 1 <= mm.end[c] - 1 and mm.can[min(c + 1, len(mm.can) - 1)] else None
            if stay is None or cj < mm.cost(stay, q) - self.margin:
                jumped = abs(j - c) > 2
                c = j - 1
        # advance one frame; root motion taken from the database, applied in the world heading
        c_prev, c = c, min(c + 1, mm.end[c])
        same = db["clip"][c] == db["clip"][c_prev]
        dh = wrap(mm.h[c] - mm.h[c_prev]) if same else 0.0
        step = db["hips"][c, :2] - db["hips"][c_prev, :2] if same else np.zeros(2)
        self.pos = self.pos + rot2(self.yaw) @ (rot2(-mm.h[c_prev]) @ step)
        self.yaw = wrap(self.yaw + dh)
        q = db["qpos"][c].copy()
        # place the frame in the world: rotate the database pose about z so its heading is the world yaw
        q[0:2] = self.pos
        q[3:7] = qmul(yaw_quat(self.yaw - mm.h[c]), db["qpos"][c, 3:7])
        if (jumped or self.force_blend) and self.prev is not None:
            # inertialization: start exactly from what was last shown, and let the difference fade
            self.force_blend = False
            self.off7 = wrap(self.prev[7:] - q[7:])
            self.offz = self.prev[2] - q[2]
            pq = self.prev[3:7] if np.dot(self.prev[3:7], q[3:7]) >= 0 else -self.prev[3:7]
            self.offq = qmul(pq, qconj(q[3:7]))
            self.blend_t = self.blend
        if self.blend_t > 0:
            x = self.blend_t / self.blend
            a = x * x * (3 - 2 * x)  # smooth fade from 1 to 0
            q[7:] = q[7:] + a * self.off7
            q[2] = q[2] + a * self.offz
            q[3:7] = qmul(qslerp(np.array([1.0, 0, 0, 0]), self.offq, a), q[3:7])
            q[3:7] /= np.linalg.norm(q[3:7])
            self.blend_t -= 1
        if self.prev is not None:  # different clips can sit a whole turn apart for the same pose
            q[7:] = self.prev[7:] + wrap(q[7:] - self.prev[7:])
        self.prev = q.copy()
        self.since = 0 if SUPPORT[db["regime"][c]] != cur else self.since + 1
        self.c, self.f, self.last = c, self.f + 1, reg
        return q, SUPPORT[db["regime"][c]]


def run(mm, plan, target_fn=None, **kw):
    """plan: list of (seconds, regime). target_fn(t, pos) -> desired world heading or None (keep course).
    Returns world qpos per frame, the regime played per frame, and the regime asked per frame."""
    sched = []
    for secs, reg in plan:
        sched += [reg] * int(round(secs / DT))
    p = Player(mm, plan[0][1], **kw)
    out_q, out_r = [], []
    for f, reg in enumerate(sched):
        q, played = p.step(reg, want_yaw=target_fn(f * DT, p.pos) if target_fn else None)
        out_q.append(q)
        out_r.append(played)
    return np.array(out_q), out_r, sched


def render(m, Q, labels, path, w=640, h=360, fps=50):
    from PIL import Image, ImageDraw

    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.0, 120, -18
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", path],
                          stdin=subprocess.PIPE)
    look = Q[0][:3].copy()
    for q, lab in zip(Q, labels):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        look = 0.9 * look + 0.1 * q[:3]
        cam.lookat[:] = [look[0], look[1], 0.3]
        r.update_scene(d, camera=cam)
        img = Image.fromarray(r.render())
        ImageDraw.Draw(img).text((12, 10), lab, fill=(255, 255, 255))
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()


def demo(path):
    """The first slice, scripted: at rest, shifting, asked to come, rising, walking over, sitting, lying down."""
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "dog_mocap_mjx.xml"))
    mm = Matcher(m)
    person = np.array([4.0, 2.0])
    plan = [(6, "down"), (4, "side"), (4, "down"), (2, "stand"), (5, "walk"), (4, "sit"), (5, "down")]
    target = lambda t, pos: np.arctan2(*(person - pos)[::-1])  # noqa: E731
    Q, played, asked = run(mm, plan, target)
    labels = [f"asked: {a:6s} playing: {p}" for a, p in zip(asked, played)]
    render(m, Q, labels, path)
    agree = np.mean([a == p for a, p in zip(asked, played)])
    print(f"wrote {path}: {len(Q) * DT:.0f} s; frames whose played regime matches the ask: {agree:.0%}; "
          f"final distance to the person {np.linalg.norm(Q[-1][:2] - person):.2f} m")


if __name__ == "__main__":
    if sys.argv[1] == "demo":
        demo(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "first_slice_mm.mp4"))
