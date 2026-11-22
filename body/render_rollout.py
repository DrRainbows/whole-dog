"""Render a policy rollout beside the capture it was tracking (run on Windows).

  .venv\\Scripts\\python render_rollout.py RUN_DIR|ROLLOUT.npz [out.mp4] [model.xml]

Left: the physics dog driven by the trained policy. Right: the motion capture at the same moments.
The same camera follows both. A red tint marks the moment tracking failed.
"""
import os
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    run = sys.argv[1]  # a run directory (reads its rollout.npz) or a rollout .npz file
    src = run if run.endswith(".npz") else os.path.join(run, "rollout.npz")
    path = sys.argv[2] if len(sys.argv) > 2 else src[:-4] + ".mp4"
    z = np.load(src)
    xml = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "dog_mocap_mjx.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]  # the scene light is fixed; a headlight keeps the dog lit
    m.vis.headlight.diffuse[:] = [0.55, 0.55, 0.55]
    d = mujoco.MjData(m)
    w, h = 480, 360
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.4, 100, -15
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w}x{h}",
                           "-r", "50", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", path],
                          stdin=subprocess.PIPE)
    for k in z["starts"]:
        sim, ref, fail = z[f"sim_{k}"], z[f"ref_{k}"], int(z[f"fail_{k}"])
        for i in range(len(sim)):
            tiles = []
            for q, label in ((sim[i], "physics"), (ref[i], "capture")):
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                cam.lookat[:] = [ref[i][0], ref[i][1], 0.3]  # both views follow the capture's position
                r.update_scene(d, camera=cam)
                img = Image.fromarray(r.render())
                txt = f"{label}  start {k}  t {i * 0.02:4.1f}s"
                if label == "physics" and f"label_{k}" in z.files:  # online runs: what the router said
                    txt = f"physics  t {i * 0.02:4.1f}s   {z[f'label_{k}'][i]}"
                if label == "physics" and 0 <= fail <= i:
                    txt += "  (off the capture)"
                    img = Image.blend(img, Image.new("RGB", img.size, (120, 0, 0)), 0.15)
                ImageDraw.Draw(img).text((10, 8), txt, fill=(255, 255, 255))
                tiles.append(np.asarray(img))
            ff.stdin.write(np.concatenate(tiles, 1).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", path)


if __name__ == "__main__":
    main()
