"""Transcode the whole library and track both dogs in every clip.

  .venv\\Scripts\\python batch_track.py SOURCE_DIR

1. Transcode each video to library/work/NAME.mp4: upright, 30 fps, longest side 960 px (3 jobs at once).
2. Track up to two dogs per frame with DeepLabCut SuperAnimal-Quadruped.
3. Name the two tracks by coat: the brighter torso is the Labrador, the darker the Malinois.
4. Save library/work/tracks/NAME.dogs.npz: xy (T, 2, K, 2) ordered [lab, mal], conf (T, 2, K).

Safe to stop and rerun: finished steps are skipped. Progress goes to library/work/batch.log.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline.tracks import dlc_h5_all  # noqa: E402

LIB = os.path.join(HERE, "library")
WORK = os.path.join(LIB, "work")
DLC_OUT = os.path.join(WORK, "dlc")
TRACKS = os.path.join(WORK, "tracks")
LOG = os.path.join(WORK, "batch.log")
TORSO = ["back_middle", "back_base", "back_end", "body_middle_left", "body_middle_right", "belly_bottom", "neck_base"]


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


ROTATE = {-90: ",hwdownload,format=nv12,transpose=clock", 90: ",hwdownload,format=nv12,transpose=cclock",
          180: ",hwdownload,format=nv12,hflip,vflip", -180: ",hwdownload,format=nv12,hflip,vflip", 0: ""}


def transcode(src, dst, rotation=0):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return dst
    tmp = dst + ".part.mp4"
    # Decode, scale and encode on the GPU's video engines, so no CPU encoder starves the tracker.
    # iPhone video is 10-bit HDR, which the H.264 encoder can't take, so the scaler converts to 8-bit nv12.
    # The display rotation is applied by hand on the small scaled frames and the flag is cleared,
    # so the result matches the earlier CPU transcodes (upright, no rotation flag).
    # Any clip the GPU path rejects falls back to the CPU path.
    vf = "fps=30,scale_cuda='if(gt(iw,ih),960,-2)':'if(gt(iw,ih),-2,960)':format=nv12" + ROTATE.get(int(rotation or 0), "")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "0", "-hwaccel", "cuda",
                        "-hwaccel_output_format", "cuda", "-i", src, "-vf", vf,
                        "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", "-an", tmp], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf",
                        "fps=30,scale='if(gt(iw,ih),960,-2)':'if(gt(iw,ih),-2,960)'",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", tmp], check=True)
    os.replace(tmp, dst)
    return dst


def coat_brightness(video, names, xy, conf, min_conf=0.4, stride=3):
    """Mean brightness of small patches at torso keypoints, per frame and individual (NaN if unseen)."""
    import cv2

    idx = [names.index(n) for n in TORSO if n in names]
    T, I = xy.shape[0], xy.shape[1]
    out = np.full((T, I), np.nan)
    cap = cv2.VideoCapture(video)
    t = 0
    while t < T:
        if t % stride != 0:
            if not cap.grab():  # skip without decoding to colour
                break
            t += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if t % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            for i in range(I):
                vals = []
                for k in idx:
                    if conf[t, i, k] >= min_conf:
                        x, y = int(xy[t, i, k, 0]), int(xy[t, i, k, 1])
                        if 3 <= x < w - 3 and 3 <= y < h - 3:
                            vals.append(gray[y - 3:y + 4, x - 3:x + 4].mean())
                if vals:
                    out[t, i] = np.median(vals)
        t += 1
    cap.release()
    # carry each sample over its stride
    for i in range(I):
        col = out[:, i]
        for s in range(0, T, stride):
            col[s:s + stride] = col[s]
    return out


def assign(names, xy, conf, bright, split=None):
    """Order individuals per frame as [lab, mal] using torso brightness."""
    T, I = xy.shape[0], xy.shape[1]
    out_xy = np.full((T, 2, xy.shape[2], 2), np.nan)
    out_cf = np.zeros((T, 2, xy.shape[2]))
    seen = [i for i in range(I)]
    vals = bright[np.isfinite(bright)]
    if split is None:
        split = float(np.median(vals)) if vals.size else 128.0
    for t in range(T):
        cands = [(bright[t, i], i) for i in seen if np.isfinite(bright[t, i]) and conf[t, i].mean() > 0.2]
        if len(cands) >= 2:
            cands.sort(reverse=True)
            for slot, (_, i) in zip((0, 1), cands[:2]):
                out_xy[t, slot], out_cf[t, slot] = xy[t, i], conf[t, i]
        elif len(cands) == 1:
            b, i = cands[0]
            slot = 0 if b >= split else 1
            out_xy[t, slot], out_cf[t, slot] = xy[t, i], conf[t, i]
    return out_xy, out_cf, split


def track(work_mp4, name):
    import deeplabcut
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    out = os.path.join(TRACKS, name + ".dogs.npz")
    if os.path.exists(out):
        return out
    h5s = [f for f in os.listdir(DLC_OUT) if f.startswith(name + "_superanimal") and f.endswith(".h5")] if os.path.isdir(DLC_OUT) else []
    if not h5s:
        deeplabcut.video_inference_superanimal(
            [work_mp4], superanimal_name="superanimal_quadruped", model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2", videotype=".mp4", dest_folder=DLC_OUT,
            max_individuals=2, batch_size=64, detector_batch_size=32, video_adapt=False, create_labeled_video=False)
        h5s = [f for f in os.listdir(DLC_OUT) if f.startswith(name + "_superanimal") and f.endswith(".h5")]
    names, xy, conf = dlc_h5_all(os.path.join(DLC_OUT, h5s[0]))
    bright = coat_brightness(work_mp4, names, xy, conf)
    xy2, cf2, split = assign(names, xy, conf, bright)
    np.savez_compressed(out, names=np.array(names), xy=xy2, conf=cf2, fps=30.0, brightness=bright, split=split,
                        source=work_mp4)
    seen = (cf2.mean(axis=2) > 0.3).mean(axis=0)
    log(f"tracked {name}: {xy.shape[0]} frames; lab seen {seen[0]:.0%}, mal seen {seen[1]:.0%}; split {split:.0f}")
    return out


def main(src_dir):
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(TRACKS, exist_ok=True)
    inv = json.load(open(os.path.join(LIB, "inventory.json")))
    jobs = [(os.path.join(src_dir, r["file"]), os.path.join(WORK, os.path.splitext(r["file"])[0] + ".mp4"),
             r.get("rotation", 0)) for r in inv]
    from concurrent.futures import as_completed

    log(f"transcoding and tracking {len(jobs)} videos (each clip is tracked as soon as it is transcoded)")
    with ThreadPoolExecutor(2) as ex:
        futs = {ex.submit(transcode, *j): j for j in jobs}
        for fut in as_completed(futs):
            dst = futs[fut][1]
            name = os.path.splitext(os.path.basename(dst))[0]
            try:
                fut.result()
                track(dst, name)
            except Exception as e:  # noqa: BLE001
                log(f"FAILED {name}: {e!r}")
    log("all done")


if __name__ == "__main__":
    main(sys.argv[1])
