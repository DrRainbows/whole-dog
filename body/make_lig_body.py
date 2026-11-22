"""The ligament body: hinge joints whose out-of-plane hinges are held by ligaments, not muscles.

  .venv\\Scripts\\python make_lig_body.py   -> dog_anat_lig.xml, muscles_dog_lig.json

Why (RESULTS.md, 21 Nov): at a dog's strength the muscles most often at their limit were the out-of-plane hinges of the
hinge joints (elbow side-bend 9.5% of steps, stifle twist 9.2%), not any main swing. In a dog those planes are held
by collateral ligaments: "a near zero out of plane motion until broken". In dog_anat.xml they are weak muscles (20%
of the main swing) in a range of the capture's spread plus 10 degrees, so the policy spends effort holding slack.

Here, at the elbow, carpus, stifle and hock (both sides), each out-of-plane hinge (a2, a3):
- range: the capture's spread (1-99 percentile) plus 2 degrees, not 10 (the ligament's slack);
- a spring toward the standing pose and damping (the ligament's give inside that slack);
- its muscle reduced to nothing (0.01 N m), keeping the actuator so the action and observation sizes stay those of
  anat_dog and its policy can continue from there.
"""
import json
import os
import re

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, OUT = os.path.join(HERE, "dog_anat.xml"), os.path.join(HERE, "dog_anat_lig.xml")
MUS_SRC, MUS_OUT = os.path.join(HERE, "muscles_dog.json"), os.path.join(HERE, "muscles_dog_lig.json")
HINGES = ["LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot"]
SLACK_DEG = 2.0
STIFFNESS, DAMPING, ARMATURE = 30.0, 1.5, 0.01  # N m/rad: about the old weak muscle's strength at 15 degrees


def main():
    anat = json.load(open(os.path.join(HERE, "anat.json")))
    src = open(SRC, encoding="utf-8").read()
    src = src.replace('<mujoco model="dog_anat">', '<mujoco model="dog_anat_lig">')
    lig = f'    <default class="lig"><joint stiffness="{STIFFNESS}" damping="{DAMPING}" armature="{ARMATURE}"/></default>\n'
    anchor = re.search(r' *<default class="passive">.*\n', src).group(0)
    src = src.replace(anchor, anchor + lig, 1)
    changed = []
    for b in HINGES:
        ext = anat["joints"][b]["capture_extent_deg"]
        for k in (1, 2):
            lo, hi = ext[k][0] - SLACK_DEG, ext[k][1] + SLACK_DEG
            pat = re.compile(rf'(<joint name="{b}_a{k + 1}" axis="[^"]+") range="[^"]+" class="\w+"/>')
            src, n = pat.subn(rf'\1 range="{lo:.1f} {hi:.1f}" class="lig"/>', src)
            assert n == 1, (b, k, n)
            changed.append(f"{b}_a{k + 1}")
    open(OUT, "w", encoding="utf-8").write(src)
    m = mujoco.MjModel.from_xml_path(OUT)
    m0 = mujoco.MjModel.from_xml_path(SRC)
    assert (m.nq, m.nv, m.nu) == (m0.nq, m0.nv, m0.nu)
    spec = json.load(open(MUS_SRC))
    for n in changed:
        spec["tau_max"][n] = 0.01
    spec["about"] = spec["about"] + "; out-of-plane hinges of elbow, carpus, stifle and hock held by ligaments (no muscle)"
    spec["ligament_held"] = changed
    json.dump(spec, open(MUS_OUT, "w"), indent=1)
    for n in changed:
        j = m.joint(n)
        print(f"  {n:16s} range {m.jnt_range[j.id][0] * 57.2958:6.1f} {m.jnt_range[j.id][1] * 57.2958:6.1f} deg, "
              f"stiffness {m.jnt_stiffness[j.id]:.0f}")
    print(f"wrote {OUT} (nq {m.nq}, nu {m.nu}) and {MUS_OUT}: {len(changed)} hinges ligament-held")


if __name__ == "__main__":
    main()
