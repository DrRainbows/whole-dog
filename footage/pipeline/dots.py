"""Render movement-only clips: white dots on black, no body, no fur, no setting."""
import os
import subprocess
import tempfile

import numpy as np


def render(coords, fps, out_path, keep=None, size=(640, 480), body_px=None, radius=6, audio_from=None):
    """coords: (T, K, 2) in body lengths, y up (display or canonical). keep: indices of dots to draw."""
    import cv2

    W, H = size
    s = body_px or H * 0.42
    K = coords.shape[1]
    keep = list(range(K)) if keep is None else list(keep)
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.close()
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    for t in range(coords.shape[0]):
        frame = np.zeros((H, W, 3), np.uint8)
        for k in keep:
            x, y = coords[t, k]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(round(W / 2 + x * s)), int(round(H * 0.82 - y * s))), radius,
                           (255, 255, 255), -1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name]
    if audio_from:
        cmd += ["-i", audio_from, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", out_path]
    subprocess.run(cmd, check=True)
    os.unlink(tmp.name)
    return out_path
