"""Muscle specs for the physics dog (read by train/dog_track.py through config.muscles).

  .venv\\Scripts\\python make_muscles.py equiv  -> muscles_equiv.json  the old motors, through the muscle code (a check)
  .venv\\Scripts\\python make_muscles.py dog    -> muscles_dog.json    a 28 kg dog's strength, speed and activation lag

Each actuated hinge of dog_anat.xml gets:
  tau_max  the most torque its muscles produce, holding still (N m);
  w_max    the joint speed at which they can no longer pull (rad/s);
and the whole body: act_tau (activation lag, s), hill_k (curvature of force against speed), ecc (how much more a
muscle resists being stretched than it can pull).

Each hinge's role comes from anat.json: a1 is the joint's main swing; a2 and a3 are its out-of-plane hinges. At the
ball joints (hip, shoulder, the shoulder blade's sling, spine, neck, head) muscles also move the out-of-plane hinges;
at the hinge joints (elbow, carpus, stifle, hock) the ligaments hold them, so their muscles there are weak.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = {"Spine": "lumbar", "Spine1": "thoracic", "Neck": "neck", "Head": "head",
         "LeftShoulder": "scapula", "RightShoulder": "scapula", "LeftArm": "shoulder", "RightArm": "shoulder",
         "LeftForeArm": "elbow", "RightForeArm": "elbow", "LeftHand": "carpus", "RightHand": "carpus",
         "LeftUpLeg": "hip", "RightUpLeg": "hip", "LeftLeg": "stifle", "RightLeg": "stifle",
         "LeftFoot": "tarsus", "RightFoot": "tarsus"}
STRONG = {"lumbar", "thoracic", "scapula", "shoulder", "hip"}  # the old XML's classes: +-200 N m, else +-80
HINGE = {"elbow", "carpus", "stifle", "tarsus"}


def joints():
    anat = json.load(open(os.path.join(HERE, "anat.json")))
    for b in anat["joints"]:
        for k in range(3):
            yield b, k, f"{b}_a{k + 1}"


def equiv():
    """The old position motors exactly: same strength on every hinge, no speed limit, no lag."""
    tau = {n: (200.0 if GROUP[b] in STRONG else 80.0) for b, k, n in joints()}
    return {"about": "the old position motors through the muscle code (validation)", "act_tau": 1e-5,
            "hill_k": 0.25, "ecc": 1.0, "tau_max": tau, "w_max": {n: 1e6 for n in tau}}


def capture_speeds(pattern="runs/refs_anat/flat/*.npz", pct=99.9):
    """Per joint group: the fastest its main swing turns in the capture (rad/s, the given percentile)."""
    import glob

    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "dog_anat.xml"))
    V = np.concatenate([np.load(f)["qvel"] for f in sorted(glob.glob(pattern))])
    out = {}
    for b in GROUP:
        v = np.percentile(np.abs(V[:, m.jnt_dofadr[m.joint(f"{b}_a1").id]]), pct)
        out[GROUP[b]] = max(out.get(GROUP[b], 0.0), float(v))
    return out


def dog(table):
    """table: group -> {"tau": main-swing N m, "side": share of tau on a2, "twist": share on a3}; speeds from the
    capture (speed_rule)."""
    rule = table["speed_rule"]
    fast = capture_speeds()
    wg = {g: round(max(rule["min_rad_s"], rule["times_capture_p999"] * v), 1) for g, v in fast.items()}
    tau, w = {}, {}
    for b, k, n in joints():
        g = table["groups"][GROUP[b]]
        share = (1.0, g["side"], g["twist"])[k]
        tau[n] = round(g["tau"] * share, 2)
        w[n] = wg[GROUP[b]]
    return {"about": table["about"], "sources": table["sources"], "act_tau": table["act_tau"],
            "hill_k": table["hill_k"], "ecc": table["ecc"], "capture_p999_rad_s": {g: round(v, 1) for g, v in fast.items()},
            "w_max_by_group": wg, "tau_by_group": {g: v["tau"] for g, v in table["groups"].items()},
            "tau_max": tau, "w_max": w}


def main():
    which = sys.argv[1]
    if which == "equiv":
        spec, out = equiv(), "muscles_equiv.json"
    else:
        spec, out = dog(json.load(open(os.path.join(HERE, "dog_strength.json")))), "muscles_dog.json"
    path = os.path.join(HERE, out)
    json.dump(spec, open(path, "w"), indent=1)
    print(f"wrote {path}: {len(spec['tau_max'])} hinges, activation lag {spec['act_tau'] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
