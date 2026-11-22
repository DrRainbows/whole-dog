"""Render the first slice in physics: each dog's online run (body/train/online_dog.py --brain), side by side.

  body\\.venv\\Scripts\\python dog\\render_physics_slice.py LAB_online.npz MAL_online.npz [out.mp4] [--person 4,2]

Shows the simulated bodies only (no capture beside them), with what each brain was doing, as in
first_slice.py. Every second frame is drawn (25 fps).
"""
import os
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def model_with_person(person, xml_path=None):
    xml = open(xml_path or os.path.join(ROOT, "body", "dog_mocap_mjx.xml"), encoding="utf-8").read()
    body = (f'<body name="person" pos="{person[0]} {person[1]} 0"><geom type="capsule" fromto="0 0 0.15 0 0 1.55" '
            'size="0.17" rgba="0.35 0.45 0.75 0.35" contype="0" conaffinity="0"/></body>')
    m = mujoco.MjModel.from_xml_string(xml.replace("<worldbody>", "<worldbody>" + body, 1))
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    m.vis.headlight.diffuse[:] = [0.55, 0.55, 0.55]
    return m


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    person = np.array([4.0, 2.0])
    if "--person" in sys.argv:
        person = np.array([float(v) for v in sys.argv[sys.argv.index("--person") + 1].split(",")])
        args = [x for x in args if x != sys.argv[sys.argv.index("--person") + 1]]
    if "--until" in sys.argv:
        args = [x for x in args if x != sys.argv[sys.argv.index("--until") + 1]]
    xml_path = None  # --xml: the body the runs were simulated with (e.g. body/dog_anat.xml)
    if "--xml" in sys.argv:
        xml_path = sys.argv[sys.argv.index("--xml") + 1]
        args = [x for x in args if x != xml_path]
    runs = [np.load(p) for p in args[:2]]
    out = args[2] if len(args) > 2 else os.path.join(HERE, "first_slice_physics.mp4")
    m = model_with_person(person, xml_path)
    d = mujoco.MjData(m)
    w, h, every = 640, 400, 2
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.9, np.degrees(np.arctan2(person[1], person[0])) + 90, -20
    T = min(len(z["sim_0"]) for z in runs)
    if "--until" in sys.argv:  # only the first SECONDS
        T = min(T, int(float(sys.argv[sys.argv.index("--until") + 1]) * 50))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{w * len(runs)}x{h}", "-r", str(int(50 / every)), "-i", "-", "-c:v", "libx264",
                           "-preset", "veryfast", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    looks = [None] * len(runs)
    for f in range(0, T, every):
        tiles = []
        for i, z in enumerate(runs):
            q = z["sim_0"][f]
            nq = m.nq
            d.qpos[:] = q[:nq]
            mujoco.mj_forward(m, d)
            mid = 0.5 * (q[:2] + person)
            looks[i] = mid if looks[i] is None else 0.95 * looks[i] + 0.05 * mid
            cam.lookat[:] = [looks[i][0], looks[i][1], 0.4]
            r.update_scene(d, camera=cam)
            img = Image.fromarray(r.render())
            g = ImageDraw.Draw(img)
            lab = str(z["label_0"][f]) if "label_0" in z.files else ""
            g.rectangle([0, 0, w, 40], fill=(0, 0, 0))
            g.text((8, 5), f"t {f * 0.02:5.1f}s  physics", fill=(255, 255, 255))
            g.text((8, 21), lab[:100], fill=(255, 220, 120))
            tiles.append(np.asarray(img))
        ff.stdin.write(np.concatenate(tiles, 1).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
