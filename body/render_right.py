"""Side-by-side video of the scramble_up reflex of two runs from the same falls (train/record_right.py output).

  .venv\\Scripts\\python render_right.py COMPARE.npz OUT.mp4

Left and right: each run's dog from the same fall, its clock, and when it is righted and on its feet. 25 fps.
"""
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

XML = __file__.replace("render_right.py", "dog_anat.xml")


def main():
    z = np.load(sys.argv[1])
    out = sys.argv[2]
    m = mujoco.MjModel.from_xml_path(XML)
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    d = mujoco.MjData(m)
    w, h = 560, 400
    r = mujoco.Renderer(m, h, w)
    n_runs = int(z["n_runs"])
    names = [str(z[f"name_{i}"]) for i in range(n_runs)]
    n_eps = z["qpos_0"].shape[0]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{w * n_runs}x{h}", "-r", "25", "-i", "-", "-c:v", "h264_nvenc", "-preset", "p4", "-cq",
                           "22", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for e in range(n_eps):
        first = {i: {"righted": None, "feet": None} for i in range(n_runs)}
        T = z["qpos_0"].shape[1]
        for t in range(0, T, 2):
            tiles = []
            for i in range(n_runs):
                q = z[f"qpos_{i}"][e, t]
                if t > 0:
                    if first[i]["righted"] is None and z[f"sternal_{i}"][e, t - 1]:
                        first[i]["righted"] = t * 0.02
                    if first[i]["feet"] is None and z[f"up_{i}"][e, t - 1]:
                        first[i]["feet"] = t * 0.02
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [q[0], q[1], 0.3]
                cam.distance, cam.azimuth, cam.elevation = 2.2, 120, -18
                r.update_scene(d, cam)
                img = Image.fromarray(r.render())
                dr = ImageDraw.Draw(img)
                fr = first[i]
                state = ("ON ITS FEET" if fr["feet"] is not None else "righted, on its belly" if fr["righted"] is not None
                         else "down")
                dr.rectangle([0, 0, w, 64], fill=(20, 20, 20))
                dr.text((10, 6), names[i], fill=(255, 255, 255))
                dr.text((10, 24), f"fall {e + 1} of {n_eps} ({'onto its back' if z['tilt'][e] > 135 else 'onto its side'})"
                                  f"   t = {t * 0.02:.2f} s", fill=(220, 220, 220))
                extra = (f"   righted {fr['righted']:.2f} s" if fr["righted"] is not None else "") + \
                        (f"   feet {fr['feet']:.2f} s" if fr["feet"] is not None else "")
                dr.text((10, 42), state + extra, fill=(120, 255, 120) if fr["feet"] is not None else (255, 200, 120))
                tiles.append(np.array(img))
            ff.stdin.write(np.hstack(tiles).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
