"""Footage pipeline entry point.

  python run.py track VIDEO [--out DIR]     keypoints with DeepLabCut SuperAnimal-Quadruped -> DIR/NAME.tracks.npz
  python run.py measure TRACKS [--video V]  clean, normalize, dot clips, sound onsets, reactions, gait -> report
  python run.py all VIDEO [--out DIR]       both

Use the footage environment:  .venv\\Scripts\\python run.py ...
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import audio, dots, gait, normalize, react  # noqa: E402
from pipeline.tracks import Tracks, from_dlc_h5  # noqa: E402


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", video],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return float(num) / float(den), (int(s["width"]), int(s["height"]))


def has_audio(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", video], capture_output=True, text=True).stdout
    return bool(out.strip())


def track(video, out_dir):
    import deeplabcut

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video))[0]
    mp4 = os.path.join(out_dir, name + ".mp4")
    if not video.lower().endswith(".mp4"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", mp4], check=True)
    elif os.path.abspath(video) != os.path.abspath(mp4):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-c", "copy", mp4], check=True)
    deeplabcut.video_inference_superanimal(
        [mp4], superanimal_name="superanimal_quadruped", model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2", videotype=".mp4", dest_folder=out_dir,
        max_individuals=1, video_adapt=False, create_labeled_video=True)
    h5s = sorted(glob.glob(os.path.join(out_dir, name + "*superanimal*.h5")), key=os.path.getmtime)
    if not h5s:
        raise SystemExit("DeepLabCut wrote no .h5 file")
    fps, size = probe(mp4)
    tr = from_dlc_h5(h5s[-1], fps, size, source=mp4)
    path = os.path.join(out_dir, name + ".tracks.npz")
    tr.save(path)
    print(f"tracks: {path} ({tr.frames} frames, {len(tr.names)} keypoints)")
    return path


def coverage(tr, names, min_conf):
    idx = [tr.idx(n) for n in names if tr.has(n)]
    if not idx:
        return 0.0
    return float(np.mean(np.all(tr.conf[:, idx] >= min_conf, axis=1)))


def measure(tracks_path, video=None, min_conf=0.5):
    tr = Tracks.load(tracks_path)
    video = video or tr.source
    base = tracks_path.replace(".tracks.npz", "")
    clean = normalize.clean(tr, min_conf=min_conf)
    world, disp, meta = normalize.normalize(clean)
    report = {"tracks": os.path.basename(tracks_path), "frames": tr.frames, "fps": tr.fps, **meta}
    report["paws_all_tracked"] = coverage(tr, normalize.PAWS, min_conf)
    report["head_tracked"] = coverage(tr, ["left_eye", "nose"], min_conf)
    report["mean_confidence"] = {n: round(float(np.mean(tr.conf[:, tr.idx(n)])), 3) for n in tr.names}

    keep = [tr.idx(n) for _, n, _ in normalize.BONES if tr.has(n)] + [tr.idx(normalize.FRONT), tr.idx(normalize.REAR)]
    keep = sorted(set(keep))
    dots.render(disp, tr.fps, base + ".dots.mp4", keep=keep)
    dots.render(normalize.canonical(disp, tr.names), tr.fps, base + ".dots-canonical.mp4", keep=keep)
    report["clips"] = [os.path.basename(base + ".dots.mp4"), os.path.basename(base + ".dots-canonical.mp4")]

    stim = []
    if video and os.path.exists(video) and has_audio(video):
        wav = audio.extract(video, base + ".wav")
        y, sr = audio.load_wav(wav)
        stim = audio.onsets(y, sr)
        report["clips"].append(os.path.basename(wav))
    report["sound_onsets_s"] = [round(s, 3) for s in stim]

    if tr.has("nose"):
        ang = react.head_angle(disp, tr.names)
        report["head_turn_latency_s"] = [
            None if (lat := react.turn_latency(ang, tr.fps, s)) is None else round(lat, 3) for s in stim]

    paws = {}
    for p in normalize.PAWS:
        if tr.has(p):
            st = gait.stride_stats(gait.stance_mask(world[:, tr.idx(p), 0], tr.fps), tr.fps)
            if st:
                paws[p] = {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in st.items()}
    report["gait"] = paws
    if "back_left_paw" in paws and "front_right_paw" in paws:
        report["diagonal_dissociation"] = gait.dissociation(
            paws["back_left_paw"]["touchdowns"], paws["front_right_paw"]["touchdowns"], paws["back_left_paw"]["period_s"])
    out = base + ".report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "mean_confidence"}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["track", "measure", "all"])
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--min-conf", type=float, default=0.5)
    a = ap.parse_args()
    if a.cmd in ("track", "all"):
        out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.path)), "out")
        tracks = track(a.path, out_dir)
        if a.cmd == "all":
            measure(tracks, a.video, a.min_conf)
    else:
        measure(a.path, a.video, a.min_conf)


if __name__ == "__main__":
    main()
