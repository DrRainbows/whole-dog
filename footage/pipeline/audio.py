"""Sound: extract it from a clip and find the stimulus onsets (claps, whistles, a dropped toy).

A clap is a sudden broadband rise in high-frequency energy. Its onset time is the reference
for measuring how fast the dog reacts.
"""
import subprocess

import numpy as np
from scipy.signal import butter, sosfiltfilt


def extract(video, wav, sr=22050):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(sr), wav],
                   check=True)
    return wav


def load_wav(path):
    import soundfile as sf

    y, sr = sf.read(path, dtype="float64", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def onsets(y, sr, hp_hz=1500.0, rise=8.0, min_gap_s=0.25, frame_s=0.002, win_s=0.004, bg_s=0.5):
    """Times (s) where high-passed energy jumps to `rise` times its recent background level."""
    sos = butter(4, hp_hz, "hp", fs=sr, output="sos")
    hp = sosfiltfilt(sos, y)
    hop = max(1, int(round(frame_s * sr)))
    win = max(hop, int(round(win_s * sr)))
    power = np.convolve(hp * hp, np.ones(win) / win, mode="same")[::hop]
    env = np.sqrt(power)
    n_bg = max(5, int(round(bg_s / frame_s)))
    floor = np.percentile(env, 20) + 1e-12
    times, last = [], -1e9
    for i in range(1, len(env)):
        lo = max(0, i - n_bg)
        bg = max(np.median(env[lo:i]), floor)
        if env[i] > rise * bg and env[i - 1] <= rise * bg:
            # back up to where the rise began: first frame above twice the background
            j = i
            while j > 0 and env[j - 1] > 2 * bg and i - j < int(0.02 / frame_s):
                j -= 1
            t = j * hop / sr
            if t - last >= min_gap_s:
                times.append(t)
                last = t
    return times
