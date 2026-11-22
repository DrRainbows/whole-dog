"""Pipeline checks on synthetic data with known answers. Run: python -m unittest discover -s tests -v

Test names carry the requirement they verify (see ../REQUIREMENTS.md).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import audio, gait, normalize, react  # noqa: E402
from pipeline.tracks import Tracks  # noqa: E402


def clap_track(sr, dur, claps, rng):
    y = rng.normal(0, 0.003, int(sr * dur))
    t = np.arange(len(y)) / sr
    y += 0.05 * np.sin(2 * np.pi * 60 * t)  # mains hum, below the high-pass
    for c in claps:
        i = int(c * sr)
        n = int(0.08 * sr)
        burst = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.012 * sr)) * 0.5
        y[i:i + n] += burst[: len(y) - i]
    return y


class Audio(unittest.TestCase):
    def test_REQ_M2_clap_onsets_found_within_20ms(self):
        rng = np.random.default_rng(1)
        sr, truth = 22050, [0.53, 1.41, 2.27, 3.30]
        found = audio.onsets(clap_track(sr, 4.0, truth, rng), sr)
        self.assertEqual(len(found), len(truth), f"found {found}")
        for f, t in zip(found, truth):
            self.assertLessEqual(abs(f - t), 0.020, f"onset {f:.3f} vs truth {t:.3f}")

    def test_REQ_M2_whistle_onset_found_within_20ms(self):
        rng = np.random.default_rng(2)
        sr, start = 22050, 1.2
        y = rng.normal(0, 0.003, int(sr * 2.5))
        n = int(0.6 * sr)
        tone = 0.3 * np.sin(2 * np.pi * 2500 * np.arange(n) / sr)
        tone *= np.minimum(1, np.arange(n) / (0.03 * sr))  # 30 ms attack
        y[int(start * sr): int(start * sr) + n] += tone
        found = audio.onsets(y, sr)
        self.assertEqual(len(found), 1, f"found {found}")
        self.assertLessEqual(abs(found[0] - start), 0.020)

    def test_REQ_M2_no_onsets_in_steady_noise(self):
        rng = np.random.default_rng(3)
        self.assertEqual(audio.onsets(rng.normal(0, 0.01, 22050 * 3), 22050), [])


class Reaction(unittest.TestCase):
    def synthetic(self, fps, stim, latency, turn_deg=55.0, turn_s=0.25, seed=4):
        rng = np.random.default_rng(seed)
        t = np.arange(int(4 * fps)) / fps
        ang = 12 + rng.normal(0, 0.4, len(t))
        if latency is not None:
            s = (t - (stim + latency)) / turn_s
            ang += turn_deg * np.where(s <= 0, 0, np.where(s >= 1, 1, (1 - np.cos(np.pi * s)) / 2))
        return ang

    def test_REQ_M2_head_turn_latency_recovered_within_2_frames(self):
        for fps in (30, 60):
            for latency in (0.12, 0.30, 0.55):
                got = react.turn_latency(self.synthetic(fps, 2.0, latency), fps, 2.0)
                self.assertIsNotNone(got)
                self.assertLessEqual(abs(got - latency), 2 / fps + 1e-9, f"fps {fps} latency {latency}: got {got}")

    def test_REQ_M2_no_turn_gives_no_latency(self):
        self.assertIsNone(react.turn_latency(self.synthetic(30, 2.0, None), 30, 2.0))


class Gait(unittest.TestCase):
    def test_REQ_M1_footfalls_and_duty_from_a_synthetic_trot(self):
        fps, period, duty, speed = 60, 0.45, 0.45, 4.0  # speed in body lengths per second
        rng = np.random.default_rng(5)
        t = np.arange(int(5 * fps)) / fps
        x = np.zeros_like(t)
        stride = speed * period
        truth_td = []
        for i, ti in enumerate(t):
            k, ph = divmod(ti, period)
            if ph < duty * period:
                x[i] = k * stride
            else:
                u = (ph - duty * period) / ((1 - duty) * period)
                x[i] = k * stride + stride * (u ** 3 * (10 + u * (6 * u - 15)))
        truth_td = [k * period for k in range(1, int(5 / period) + 1) if k * period < t[-1]]
        x += rng.normal(0, 0.002, len(x))
        st = gait.stride_stats(gait.stance_mask(x, fps), fps)
        self.assertIsNotNone(st)
        self.assertLessEqual(abs(st["frequency_hz"] - 1 / period) / (1 / period), 0.03)
        # A minimum-jerk swing has zero speed at both ends, so any speed threshold counts the
        # ends of the swing as stance. That biases duty upward by about 0.07 here. Real footage
        # must be checked against hand-marked footfalls (REQ M1b).
        self.assertLessEqual(abs(st["duty"] - duty), 0.08)
        for td in st["touchdowns"]:
            nearest = min(truth_td, key=lambda v: abs(v - td))
            self.assertLessEqual(abs(td - nearest), 1.5 / fps + 1e-9, f"touchdown {td:.3f} vs {nearest:.3f}")


def toy_tracks(T=90, fps=30.0, seed=6):
    names = ["neck_base", "tail_base", "neck_end", "nose", "front_left_thai", "front_left_knee", "front_left_paw",
             "back_left_thai", "back_left_knee", "back_left_paw", "tail_end"]
    rng = np.random.default_rng(seed)
    base = np.array([[60, 40], [0, 40], [72, 52], [88, 46], [58, 30], [56, 15], [58, 0], [5, 32], [8, 16], [4, 0], [-30, 30]],
                    float)
    xy = np.repeat(base[None], T, 0) + rng.normal(0, 1.5, (T, len(names), 2))
    xy[:, :, 0] += np.arange(T)[:, None] * 2.0  # walking right
    xy[..., 1] = 300 - xy[..., 1]  # image coordinates, y down
    return Tracks(names, xy, np.ones((T, len(names))), fps, (640, 360))


class Normalize(unittest.TestCase):
    def test_REQ_N1_size_position_and_direction_removed(self):
        a = toy_tracks()
        b_xy = a.xy.copy()
        b_xy[..., 0] = 900 - 2.3 * b_xy[..., 0]  # bigger, elsewhere, walking left
        b_xy[..., 1] = 2.3 * b_xy[..., 1] + 77
        b = Tracks(a.names, b_xy, a.conf, a.fps, a.size)
        _, da, _ = normalize.normalize(a)
        _, db, meta = normalize.normalize(b)
        self.assertEqual(meta["facing"], "left (mirrored)")
        np.testing.assert_allclose(da, db, atol=1e-9)

    def test_REQ_N2_canonical_skeleton_has_fixed_bone_lengths(self):
        a = toy_tracks()
        _, disp, _ = normalize.normalize(a)
        can = normalize.canonical(disp, a.names)
        idx = {n: i for i, n in enumerate(a.names)}
        for parent, child, length in normalize.BONES:
            if parent in idx and child in idx:
                d = np.linalg.norm(can[:, idx[child]] - can[:, idx[parent]], axis=1)
                np.testing.assert_allclose(d, length, atol=1e-9, err_msg=f"{parent}->{child}")

    def test_REQ_N3_short_gaps_filled_long_gaps_kept(self):
        v = np.arange(20, dtype=float)
        v[3:5] = np.nan
        v[10:17] = np.nan
        out = normalize.fill_gaps(v, 3)
        np.testing.assert_allclose(out[3:5], [3, 4])
        self.assertTrue(np.isnan(out[10:17]).all())


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not installed")
class Dots(unittest.TestCase):
    def test_REQ_D1_dot_clip_has_every_frame_and_the_requested_size(self):
        from pipeline import dots

        a = toy_tracks(T=24)
        _, disp, _ = normalize.normalize(a)
        out = os.path.join(tempfile.mkdtemp(), "dots.mp4")
        dots.render(disp, a.fps, out, size=(320, 240))
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height,nb_read_frames", "-of", "csv=p=0", out],
                               capture_output=True, text=True, check=True).stdout.strip()
        w, h, n = (int(v) for v in probe.split(","))
        self.assertEqual((w, h, n), (320, 240, 24))


if __name__ == "__main__":
    unittest.main()
