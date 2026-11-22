"""The first slice, decided rather than scripted: a dog at rest who answers a request.

  body\\.venv\\Scripts\\python dog\\first_slice.py [seconds] [seed]

Each dog lies at rest. Its brain (dog/brain.py) decides when to shift and how it breathes. At 20 s the
person calls "come", and repeats at 26 s and 32 s if the dog hasn't answered. The brain weighs each
call and may decline. Motion is composed live from the named capture (body/motion_match.py); the
brain's breathing and tail are laid over the spine and tail joints. Lab left, Malinois right.
Kinematic only: the physics tracker comes next.
"""
import os
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "body"))
sys.path.insert(0, HERE)
import motion_match as mmod  # noqa: E402
from brain import LAB, MAL, Brain, Router  # noqa: E402

DT = 0.02
PERSON = np.array([4.0, 2.0])
CALLS = (20.0, 26.0, 32.0)


def model_with_person():
    xml = open(os.path.join(ROOT, "body", "dog_mocap_mjx.xml"), encoding="utf-8").read()
    person = (f'<body name="person" pos="{PERSON[0]} {PERSON[1]} 0"><geom type="capsule" fromto="0 0 0.15 0 0 1.55" '
              'size="0.17" rgba="0.35 0.45 0.75 1" contype="0" conaffinity="0"/></body>')
    xml = xml.replace("<worldbody>", "<worldbody>" + person, 1)
    m = mujoco.MjModel.from_xml_string(xml)
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    m.vis.headlight.diffuse[:] = [0.55, 0.55, 0.55]
    return m


def simulate(m, mm, dog, router, seconds, seed):
    rng = np.random.default_rng(seed)
    brain = Brain(dog, router, rng)
    player = mmod.Player(mm, "down")
    j = {n: m.jnt_qposadr[m.joint(n).id] for n in ("Spine_x", "Spine1_x", "Tail_y")}
    frames, calls = [], list(CALLS)
    complied = False
    for f in range(int(seconds / DT)):
        t = f * DT
        if calls and t >= calls[0]:
            calls.pop(0)
            if not complied and brain.goal is None:
                brain.request(t, "come", PERSON)
                brain.events.append((t, "person: 'come!'"))
        reg, speed, want = brain.tick(t, DT, player.pos)
        complied = complied or brain.goal is not None
        q, played = player.step(reg, speed, want)
        b = brain.breath_offset()
        q[j["Spine_x"]] += b
        q[j["Spine1_x"]] -= 0.5 * b
        q[j["Tail_y"]] += brain.tail_wag(t)
        status = (f"{dog['name'].upper()}  {reg:6s} breath {brain.breath_rate():3.0f}/min{' panting' if brain.panting() else ''}  "
                  f"exertion {brain.E:.2f}  fatigue {brain.F:.2f}  arousal {brain.A:.2f}")
        frames.append((q, status, [e for e in brain.events if e[0] <= t][-3:]))
    return frames, brain


def render(m, runs, path, every=2, w=640, h=400):
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    # look across the line from the dog's bed to the person, so neither hides the other
    cam.distance, cam.azimuth, cam.elevation = 4.6, np.degrees(np.arctan2(PERSON[1], PERSON[0])) + 90, -20
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{w * len(runs)}x{h}", "-r", str(int(1 / (DT * every))), "-i", "-", "-c:v", "libx264",
                           "-preset", "veryfast", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    looks = [None] * len(runs)
    for f in range(0, min(len(x) for x in runs), every):
        tiles = []
        for i, frames in enumerate(runs):
            q, status, events = frames[f]
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            mid = 0.5 * (q[:2] + PERSON)
            looks[i] = mid if looks[i] is None else 0.95 * looks[i] + 0.05 * mid
            cam.lookat[:] = [looks[i][0], looks[i][1], 0.4]
            r.update_scene(d, camera=cam)
            img = Image.fromarray(r.render())
            g = ImageDraw.Draw(img)
            g.rectangle([0, 0, w, 22 + 16 * len(events)], fill=(0, 0, 0))
            g.text((8, 5), f"t {f * DT:5.1f}s  {status}", fill=(255, 255, 255))
            for k, (te, msg) in enumerate(events):
                g.text((8, 22 + 16 * k), f"{te:5.1f}s  {msg}", fill=(255, 220, 120))
            tiles.append(np.asarray(img))
        ff.stdin.write(np.concatenate(tiles, 1).tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    m = model_with_person()
    mm = mmod.Matcher(mujoco.MjModel.from_xml_path(os.path.join(ROOT, "body", "dog_mocap_mjx.xml")))
    router = Router()
    runs = []
    for dog in (LAB, MAL):
        frames, brain = simulate(m, mm, dog, router, seconds, seed)
        runs.append(frames)
        print(f"{dog['name']}: " + "; ".join(f"{t:.1f}s {msg}" for t, msg in brain.events)
              + f"; ends {np.hypot(*(frames[-1][0][:2] - PERSON)):.1f} m from the person")
    path = os.path.join(HERE, f"first_slice_seed{seed}.mp4")
    render(m, runs, path)
    print("wrote", path)
    # the same composed motion as tracking references, for the physics dog to follow (body/train/rollout.py)
    import make_reference as mr

    base = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "body", "dog_mocap_mjx.xml"))
    out_dir = os.path.join(ROOT, "body", "refs", "composed")
    os.makedirs(out_dir, exist_ok=True)
    for dog, frames in zip((LAB, MAL), runs):
        ref = mr.finish(base, np.array([q for q, _, _ in frames]))
        p = os.path.join(out_dir, f"first_slice_{dog['name']}_seed{seed}.npz")
        np.savez(p, **ref)
        print("wrote", p)


if __name__ == "__main__":
    main()
