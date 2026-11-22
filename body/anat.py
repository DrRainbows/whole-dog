"""Anatomical joints for the physics dog: each joint's rest pose, axes and range come from the capture itself.

The model built from the BVH (dog_mocap_mjx.xml) gives every joint three hinges about the capture's own axes,
measured from the BVH's straight-line rest pose. A standing dog is then a large rotation from zero, the middle
hinge comes near gimbal lock, and capture glitches spin the outer hinges through whole turns (in refs/flat the
right shoulder winds to -1065 deg and the right elbow to +617 deg). The joints have no limits at all.

The user's first notes ask for something else: "the only things you really have are a handful of joints with
known movement constraints, and sure you can break a joint but its a near zero out of plane motion until broken,
the muscles are what protect that from the ligaments which carry the force the muscles command."

So, per joint, from the real capture (refs/flat):
  rest   = the joint's mean standing pose (chordal mean of its rotations over standing frames);
  axes   = principal axes of its motion about rest (PCA of rotation vectors): a1 the main swing, a2 and a3 the
           out-of-plane ones, ordered by how much the joint uses them;
  hinges = a1, a2, a3 in that order, measured from rest, so typical angles are small and the middle hinge stays
           far from gimbal lock;
  range  = the capture's own extent per hinge (0.5th to 99.5th percentile) plus a margin: the ligaments.

The bones, masses, collision shapes and every pose are unchanged: only how each joint's rotation is written.

  .venv\\Scripts\\python anat.py fit                    -> anat.json
  .venv\\Scripts\\python anat.py build                  -> dog_anat.xml (position motors, as strong as before)
  .venv\\Scripts\\python anat.py convert IN_DIR OUT_DIR  references re-expressed in the new joints
  .venv\\Scripts\\python anat.py check REF.npz           the same poses on both models (paw positions agree)
"""
import glob
import json
import os
import re
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_XML = os.path.join(HERE, "dog_mocap_mjx.xml")
NEW_XML = os.path.join(HERE, "dog_anat.xml")
ANAT_JSON = os.path.join(HERE, "anat.json")
PASSIVE = {"Tail", "Tail1"}
STAND = 0  # regime id of standing in the references
CTRL_DT = 0.02
# ligament margin beyond the capture's extent (degrees): the main swing gets more room than the out-of-plane hinges
MARGIN = (15.0, 10.0, 10.0)
# the capture's extent per hinge, as percentiles of its frames: the main swing keeps nearly all of it (postures
# need it); the out-of-plane hinges keep their core, so rare faulty or singular frames are held at the edge
EXTENT = ((0.1, 99.9), (1.0, 99.0), (1.0, 99.0))
# clips left out of the fit and of anatomical training: in D1_ex03_KAN02_012 the right elbow bends backwards for
# 22 s of ordinary standing and walking (a fault in that clip's right foreleg, not a posture)
EXCLUDE = ("D1_ex03_KAN02_012",)
PADS = ("LeftHand_pad", "RightHand_pad", "LeftFoot_pad", "RightFoot_pad")


def hinge_bodies(m):
    """Bodies with actuated hinges: name -> [(joint id, axis, qpos address), ...] in the model's order."""
    out = {}
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        b = m.body(m.jnt_bodyid[j]).name
        if b in PASSIVE:
            continue
        out.setdefault(b, []).append((j, m.jnt_axis[j].copy(), int(m.jnt_qposadr[j])))
    return out


def body_rest_quat(m, b):
    q = m.body_quat[m.body(b).id]
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def rel_rots(m, Q):
    """Each actuated joint's rotation relative to its parent body, per frame: name -> (T, 3, 3)."""
    out = {}
    for b, hs in hinge_bodies(m).items():
        R = np.broadcast_to(body_rest_quat(m, b), (len(Q), 3, 3)).copy()
        for _, ax, a in hs:
            R = R @ Rot.from_rotvec(np.outer(Q[:, a], ax)).as_matrix()
        out[b] = R
    return out


def chordal_mean(R):
    U, _, Vt = np.linalg.svd(R.sum(0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


# --- twist: markers see where a bone points better than how it is turned about its own length -------------------
# Every limb bone here is a capsule whose child hangs on its axis, so turning a bone about its own length moves
# nothing but the plane its next hinge bends in. In the capture those twists are partly solver noise, and the
# hinge joints soak the noise up as sideways bending (the raw stifle bends 72 degrees out of its plane). Anatomy
# settles it: a hinge (elbow, carpus, stifle, hock) bends one way, in its own plane. So each bone above a hinge is
# turned about its length until the hinge below lies in plane, bending the same way as when the dog stands. The
# humerus and femur may turn freely (that is the shoulder's and the hip's rotation) and the tibia a little (it
# rotates on the femur up to about 15 degrees); whatever is left over stays as the lowest joints' own sideways
# play (carpus and hock). Turning the forearm too only moves that play back into the elbow. Bones with no hinge below turn as little as possible. Two twists are
# real and kept as captured: the thorax's roll (it carries the shoulders, off its axis) and the head's tilt.
TWIST_FOR = {  # bone -> (the hinge below it, largest turn in degrees)
    "LeftArm": ("LeftForeArm", 120), "RightArm": ("RightForeArm", 120),
    "LeftUpLeg": ("LeftLeg", 120), "RightUpLeg": ("RightLeg", 120),
    "LeftLeg": ("LeftFoot", 15), "RightLeg": ("RightFoot", 15)}
KEEP_TWIST = {"Spine1", "Head"}


def bone_axis(m, b):
    """Unit direction of a body's bone in its own frame: toward its first child, else its farthest geom."""
    bid = m.body(b).id
    kids = [i for i in range(1, m.nbody) if m.body_parentid[i] == bid and i != bid]
    if kids:
        v = m.body_pos[kids[0]].copy()
    else:
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        v = max((m.geom_pos[g].copy() for g in gs), key=np.linalg.norm)
    return v / np.linalg.norm(v)


def world_rots(m, Q):
    """World rotation of every body with hinges or the root, per frame: name -> (T, 3, 3)."""
    q = Q[:, 3:7]
    out = {"Hips": Rot.from_quat(np.c_[q[:, 1:4], q[:, 0]]).as_matrix()}
    rel = rel_rots(m, Q)
    for i in range(1, m.nbody):
        b = m.body(i).name
        if b == "Hips" or b in PASSIVE:
            continue
        out[b] = out[m.body(m.body_parentid[i]).name] @ rel[b]
    return out


def swing(D, u):
    """The part of rotations D (T,3,3) that moves axis u: D = swing @ twist, twist about u."""
    q = Rot.from_matrix(D).as_quat()  # x y z w
    p = (q[:, :3] @ u)[:, None] * u[None]
    tw = np.c_[p, q[:, 3]]
    n = np.linalg.norm(tw, axis=1, keepdims=True)
    tw = np.where(n > 1e-9, tw / np.maximum(n, 1e-12), np.array([0, 0, 0, 1.0]))
    return (Rot.from_quat(q) * Rot.from_quat(tw).inv()).as_matrix()


def about(d, phi):
    return Rot.from_rotvec(d * phi[:, None]).as_matrix()


def canonical_rel(m, Q, fitj):
    """Relative rotations with the invisible twists settled by anatomy (see TWIST_FOR). fitj: per joint 'rest'
    (3x3, parent->body) and 'a1' (main swing axis, body frame) from the previous fit; None = first pass."""
    W = world_rots(m, Q)
    C = {"Hips": W["Hips"]}
    rel = {}
    order = [m.body(i).name for i in range(1, m.nbody) if m.body(i).name not in PASSIVE and m.body(i).name != "Hips"]
    for b in order:
        par = m.body(m.body_parentid[m.body(b).id]).name
        if b in KEEP_TWIST or fitj is None:
            C[b] = W[b]
        else:
            u = bone_axis(m, b)
            rest = fitj[b]["rest"]
            D = np.einsum("ji,tjk->tik", rest, np.einsum("tji,tjk->tik", C[par], W[b]))
            Cb = C[par] @ rest @ swing(D, u)  # as little twist as possible, the bone pointing where captured
            if b in TWIST_FOR:
                c, lim = TWIST_FOR[b]
                uc, a1, rest_c = bone_axis(m, c), fitj[c]["a1"], fitj[c]["rest"]
                ds = Cb @ u
                dc = W[c] @ uc
                n0 = Cb @ (rest_c @ a1)
                n_par = np.sum(n0 * ds, 1, keepdims=True) * ds
                n_perp = n0 - n_par
                A = np.sum(dc * n_perp, 1)
                B = np.sum(dc * np.cross(ds, n_perp), 1)
                K = float(a1 @ uc) - np.sum(dc * n_par, 1)
                rho = np.hypot(A, B)
                psi = np.arctan2(B, A)
                delta = np.arccos(np.clip(K / np.maximum(rho, 1e-9), -1, 1))
                # two turns put the hinge in plane, bending one way or (mirrored) the other; keep the one that
                # bends the way the joint bends when the dog stands, else the smaller turn
                s0 = np.sign(np.cross(u, rest_c @ uc) @ (rest_c @ a1))
                cand = [(psi + d + np.pi) % (2 * np.pi) - np.pi for d in (delta, -delta)]
                good = []
                for phi in cand:
                    n = (about(ds, phi) @ Cb) @ (rest_c @ a1)
                    good.append(np.sign(np.sum(np.cross(ds, dc) * n, 1)) == s0)
                p1, p2 = cand
                pick1 = np.where(good[0] != good[1], good[0], np.abs(p1) <= np.abs(p2))
                best = np.clip(np.where(pick1, p1, p2), -np.radians(lim), np.radians(lim))
                # a nearly straight limb has no plane: fade the turn out below 6-17 degrees of bend
                sm = np.clip((rho - 0.1) / 0.2, 0, 1)
                Cb = about(ds, best * sm * sm * (3 - 2 * sm)) @ Cb
            C[b] = Cb
        rel[b] = np.einsum("tji,tjk->tik", C[par], C[b])
    return rel


def load(pattern):
    Q, REG = [], []
    for p in sorted(glob.glob(pattern)):
        if os.path.basename(p).startswith(EXCLUDE):
            continue
        z = np.load(p)
        Q.append(z["qpos"])
        REG.append(z["regime"] if "regime" in z.files else np.full(len(z["qpos"]), -1))
    return np.concatenate(Q), np.concatenate(REG)


def angles(rest, axes, R):
    """Rotations R (T,3,3) relative to the parent -> hinge angles (T,3) about axes a1, a2, a3 from rest.
    R = rest @ Rot(a1, t1) @ Rot(a2, t2) @ Rot(a3, t3), axes in the body's own frame (intrinsic, as MuJoCo
    composes hinges); with E = [a1 a2 a3], E^T (rest^T R) E = Rx(t1) Ry(t2) Rz(t3)."""
    D = np.einsum("ji,tjk->tik", rest, R)
    Dp = np.einsum("ji,tjk,kl->til", axes, D, axes)
    return Rot.from_matrix(Dp).as_euler("XYZ")


def fit_pass(rr, stand):
    """One fit: per joint, rest = mean standing rotation; axes = principal axes of its motion about rest."""
    F = {}
    for b, R in rr.items():
        rest = chordal_mean(R[stand] if stand.sum() > 100 else R)
        rv = Rot.from_matrix(np.einsum("ji,tjk->tik", rest, R)).as_rotvec()
        ok = np.linalg.norm(rv, axis=1) < np.radians(150)  # leave out flipped or glitched frames
        X = rv[ok] - rv[ok].mean(0)
        w, V = np.linalg.eigh(X.T @ X / len(X))
        w, V = w[::-1], V[:, ::-1]
        if np.linalg.det(V) < 0:
            V[:, 2] *= -1
        th = angles(rest, V, R)
        F[b] = {"rest": rest, "axes": V, "a1": V[:, 0], "var": w / w.sum(), "th": th, "ok": ok,
                "lo": np.array([np.percentile(th[ok, i], EXTENT[i][0]) for i in range(3)]),
                "hi": np.array([np.percentile(th[ok, i], EXTENT[i][1]) for i in range(3)]),
                "a1_range": (np.percentile(th[ok, 0], 1), np.percentile(th[ok, 0], 99))}
    return F


def fitj_from_json(anat):
    F = {}
    for b, j in anat["joints"].items():
        V = np.array(j["axes"]).T
        F[b] = {"rest": Rot.from_quat(np.roll(j["rest_quat_wxyz"], -1)).as_matrix(), "axes": V, "a1": V[:, 0],
                "a1_range": tuple(np.radians(j["a1_range_deg"]))}
    return F


def fit(pattern=os.path.join(HERE, "refs", "flat", "*.npz"), passes=4):
    m = mujoco.MjModel.from_xml_path(OLD_XML)
    Q, REG = load(pattern)
    stand = REG == STAND
    raw = fit_pass(rel_rots(m, Q), stand)
    F = raw
    for _ in range(passes):  # the twist rule needs each hinge's axis, and the axes are fitted on settled twists
        F = fit_pass(canonical_rel(m, Q, F), stand)
    d = mujoco.MjData(m)
    # the standing pose's frames, to say what each axis means on the dog (the hips' frame: x forward, z up)
    k_mid = np.where(stand)[0][len(np.where(stand)[0]) // 2] if stand.any() else 0
    d.qpos[:] = Q[k_mid]
    mujoco.mj_kinematics(m, d)
    R_hips = d.xmat[m.body("Hips").id].reshape(3, 3)
    res = {"source": pattern, "excluded": EXCLUDE, "extent_percentiles": EXTENT, "frames": int(len(Q)),
           "standing_frames": int(stand.sum()), "margin_deg": MARGIN,
           "twist_for": TWIST_FOR, "keep_twist": sorted(KEEP_TWIST), "joints": {}}
    mg = np.radians(MARGIN)
    for b, f in F.items():
        rest, V, th, ok = f["rest"], f["axes"], f["th"], f["ok"]
        rng = np.stack([f["lo"] - mg, f["hi"] + mg], 1)
        R_par = d.xmat[m.body_parentid[m.body(b).id]].reshape(3, 3)
        on_dog = (R_hips.T @ R_par @ rest @ V).T
        r0 = raw[b]
        res["joints"][b] = {
            "rest_quat_wxyz": np.roll(Rot.from_matrix(rest).as_quat(), 1).round(6).tolist(),
            "axes": V.T.round(6).tolist(),  # rows a1, a2, a3 in the body's own frame
            "variance_share": f["var"].round(4).tolist(),
            "range_deg": np.degrees(rng).round(1).tolist(),
            "capture_extent_deg": np.degrees(np.stack([f["lo"], f["hi"]], 1)).round(1).tolist(),
            "a1_range_deg": np.degrees(f["a1_range"]).round(2).tolist(),
            "middle_hinge_max_deg": round(float(np.degrees(np.abs(th[ok, 1]).max())), 1),
            "glitch_frames": int((~ok).sum()),
            "axes_in_hips_frame": on_dog.round(3).tolist(),
            "raw_capture_extent_deg": np.degrees(np.stack([r0["lo"], r0["hi"]], 1)).round(1).tolist(),
            "raw_variance_share": r0["var"].round(4).tolist(),
        }
    json.dump(res, open(ANAT_JSON, "w"), indent=1)
    print(f"wrote {ANAT_JSON}: {len(F)} joints from {len(Q)} frames ({stand.sum()} standing)")
    print(f"capture extent per hinge (percentiles {EXTENT}, degrees), raw twist -> anatomical twist")
    print(f"{'joint':14s} {'var a1/a2/a3':>15s}  {'a1':>14s} {'a2 (raw)':>24s} {'a3 (raw)':>24s}  mid-max glitch")
    for b, j in res["joints"].items():
        v = "/".join(f"{x:.2f}" for x in j["variance_share"])
        e, r = j["capture_extent_deg"], j["raw_capture_extent_deg"]
        cols = [f"[{e[0][0]:5.0f},{e[0][1]:5.0f}]"] + [f"[{e[k][0]:4.0f},{e[k][1]:4.0f}] ({r[k][1] - r[k][0]:4.0f})" for k in (1, 2)]
        print(f"{b:14s} {v:>15s}  {cols[0]:>14s} {cols[1]:>24s} {cols[2]:>24s}  {j['middle_hinge_max_deg']:6.1f} {j['glitch_frames']:5d}")
    return res


def build(out=NEW_XML, anat=None):
    """dog_anat.xml: the same bodies, each joint's rest pose in its body quat, hinges about a1, a2, a3 with
    ranges (the ligaments), actuators renamed to match. Position motors and gains as before."""
    anat = anat or json.load(open(ANAT_JSON))
    src = open(OLD_XML, encoding="utf-8").read()
    m = mujoco.MjModel.from_xml_path(OLD_XML)
    hb = hinge_bodies(m)
    for b, hs in hb.items():
        j = anat["joints"][b]
        cls = re.search(rf'<joint name="{b}_[xyz]" axis="[^"]+" class="(\w+)"/>', src).group(1)
        # body tag: add the rest rotation
        src = re.sub(rf'(<body name="{b}" pos="[^"]+")>', rf'\1 quat="{" ".join(f"{x:.6f}" for x in j["rest_quat_wxyz"])}">', src)
        # the three old hinge lines -> the three new ones (indentation kept)
        old = re.findall(rf'( *)<joint name="{b}_[xyz]" axis="[^"]+" class="\w+"/>\n', src)
        ind = old[0]
        block = re.compile(rf'(?: *<joint name="{b}_[xyz]" axis="[^"]+" class="\w+"/>\n){{3}}')
        new = "".join(f'{ind}<joint name="{b}_a{k + 1}" axis="{" ".join(f"{x:.6f}" for x in j["axes"][k])}" '
                      f'range="{j["range_deg"][k][0]:.1f} {j["range_deg"][k][1]:.1f}" class="{cls}"/>\n' for k in range(3))
        src, n = block.subn(new, src)
        assert n == 1, (b, n)
        acts = re.findall(rf' *<position name="{b}_[xyz]" joint="{b}_[xyz]" class="\w+"/>\n', src)
        assert len(acts) == 3, (b, len(acts))
        a_ind = re.match(r" *", acts[0]).group(0)
        a_cls = re.search(r'class="(\w+)"', acts[0]).group(1)
        src = src.replace(acts[0], "".join(f'{a_ind}<position name="{b}_a{k + 1}" joint="{b}_a{k + 1}" class="{a_cls}"/>\n'
                                           for k in range(3)), 1)
        src = src.replace(acts[1], "", 1).replace(acts[2], "", 1)
    src = src.replace('<mujoco model="dog_mocap">', '<mujoco model="dog_anat">')
    open(out, "w", encoding="utf-8").write(src)
    mm = mujoco.MjModel.from_xml_path(out)
    print(f"wrote {out}: nq {mm.nq} nv {mm.nv} nu {mm.nu}, limited hinges {int(mm.jnt_limited.sum())}, "
          f"mass {mm.body_mass.sum():.1f} kg")
    return out


def converter(anat=None):
    """old qpos (T, nq) -> new qpos: root and tail as they were, every actuated joint re-expressed."""
    anat = anat or json.load(open(ANAT_JSON))
    mo = mujoco.MjModel.from_xml_path(OLD_XML)
    mn = mujoco.MjModel.from_xml_path(NEW_XML)
    hb = hinge_bodies(mo)
    J = {}
    for b in hb:
        j = anat["joints"][b]
        rest = Rot.from_quat(np.roll(j["rest_quat_wxyz"], -1)).as_matrix()
        J[b] = (rest, np.array(j["axes"]).T, [int(mn.jnt_qposadr[mn.joint(f"{b}_a{k + 1}").id]) for k in range(3)])
    keep = [int(mo.jnt_qposadr[j]) for j in range(mo.njnt) if mo.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            and mo.body(mo.jnt_bodyid[j]).name in PASSIVE]
    keep_new = [int(mn.jnt_qposadr[mn.joint(mo.joint(j).name).id]) for j in range(mo.njnt)
                if mo.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and mo.body(mo.jnt_bodyid[j]).name in PASSIVE]

    F = fitj_from_json(anat)

    def conv(Q):
        Q = np.atleast_2d(Q)
        out = np.zeros((len(Q), mn.nq))
        out[:, :7] = Q[:, :7]
        out[:, keep_new] = Q[:, keep]
        rr = canonical_rel(mo, Q, F)
        for b, (rest, E, adr) in J.items():
            out[:, adr] = angles(rest, E, rr[b])
        return out

    return conv, mo, mn


def finish(mn, q):
    d = mujoco.MjData(mn)
    pad = [mn.geom(n).id for n in PADS]
    qv = np.zeros((len(q), mn.nv))
    for i in range(len(q) - 1):
        mujoco.mj_differentiatePos(mn, qv[i], CTRL_DT, q[i], q[i + 1])
    qv[-1] = qv[-2] if len(q) > 1 else 0
    paws, com = [], []
    for qi in q:
        d.qpos[:] = qi
        mujoco.mj_kinematics(mn, d)
        mujoco.mj_comPos(mn, d)
        paws.append(np.array([d.geom_xpos[g].copy() for g in pad]))
        com.append(d.subtree_com[1].copy())
    return qv, np.array(paws), np.array(com)


def convert(in_dir, out_dir):
    conv, mo, mn = converter()
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = mn.jnt_range[:, 0], mn.jnt_range[:, 1]
    lim = [(int(mn.jnt_qposadr[j]), lo[j], hi[j]) for j in range(mn.njnt) if mn.jnt_limited[j]]
    outside, total, moved = 0, 0, []
    files = [p for p in sorted(glob.glob(os.path.join(in_dir, "*.npz"))) if not os.path.basename(p).startswith(EXCLUDE)]
    for p in files:
        z = dict(np.load(p, allow_pickle=True))
        q = conv(z["qpos"])
        # a reference a dog with intact ligaments can do: poses past a joint's range (capture faults, or blends
        # in composed motion) are held at the range's edge
        n_out = 0
        for a, l, h in lim:
            o = (q[:, a] < l) | (q[:, a] > h)
            n_out += int(o.sum())
            q[:, a] = np.clip(q[:, a], l, h)
        outside += n_out
        total += len(q) * len(lim)
        qv, paws, com = finish(mn, q)
        if "paws" in z:
            moved.append(np.linalg.norm(paws - z["paws"], axis=2).max(1))
        z.update(qpos=q, qvel=qv, paws=paws, com=com, held_at_range=n_out)
        np.savez(os.path.join(out_dir, os.path.basename(p)), **z)
    mv = np.concatenate(moved) * 1000 if moved else np.zeros(1)
    print(f"converted {len(files)} references {in_dir} -> {out_dir}: hinge-frames held at a ligament range "
          f"{outside}/{total} ({outside / max(total, 1):.3%}); paw moved vs the original: median {np.median(mv):.2f} mm, "
          f"99th pct {np.percentile(mv, 99):.1f} mm, frames moved >2 cm {np.mean(mv > 20):.2%}")


def check(path):
    """Same motion on both models: every body's position (and so every bone's direction) and the paws, before any
    range is applied; and how fast the actuated hinges turn in each description."""
    conv, mo, mn = converter()
    z = np.load(path)
    q = conv(z["qpos"])
    do, dn = mujoco.MjData(mo), mujoco.MjData(mn)
    err = 0.0
    for qo, qn in zip(z["qpos"], q):
        do.qpos[:], dn.qpos[:] = qo, qn
        mujoco.mj_kinematics(mo, do)
        mujoco.mj_kinematics(mn, dn)
        err = max(err, float(np.abs(do.xpos - dn.xpos).max()))
    act_o = [int(mo.jnt_dofadr[mo.actuator_trnid[i, 0]]) for i in range(mo.nu)]
    act_n = [int(mn.jnt_dofadr[mn.actuator_trnid[i, 0]]) for i in range(mn.nu)]
    qv, _, _ = finish(mn, q)
    vo, vn = np.abs(np.degrees(z["qvel"][:, act_o])), np.abs(np.degrees(qv[:, act_n]))
    print(f"{os.path.basename(path)}: {len(q)} frames; largest body position difference {err * 1000:.2f} mm; "
          f"actuated hinge speed old p99 {np.percentile(vo, 99):.0f} max {vo.max():.0f} deg/s, "
          f"new p99 {np.percentile(vn, 99):.0f} max {vn.max():.0f} deg/s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(*sys.argv[2:3])
    elif cmd == "build":
        build()
    elif cmd == "convert":
        convert(sys.argv[2], sys.argv[3])
    elif cmd == "check":
        for p in sys.argv[2:]:
            check(p)
