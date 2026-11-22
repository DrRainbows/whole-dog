"""Keypoint tracks in one container, independent of the tracker that produced them."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tracks:
    names: list            # keypoint names
    xy: np.ndarray         # (T, K, 2) image pixels, y pointing down, NaN where missing
    conf: np.ndarray       # (T, K) tracker confidence, 0..1
    fps: float
    size: tuple            # (width, height) of the source frames
    source: str = ""
    meta: dict = field(default_factory=dict)

    def idx(self, name):
        return self.names.index(name)

    def has(self, name):
        return name in self.names

    def get(self, name):
        return self.xy[:, self.idx(name)]

    @property
    def frames(self):
        return self.xy.shape[0]

    def save(self, path):
        np.savez_compressed(path, names=np.array(self.names), xy=self.xy, conf=self.conf,
                            fps=self.fps, size=np.array(self.size), source=self.source)

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        return Tracks(list(d["names"]), d["xy"], d["conf"], float(d["fps"]),
                      tuple(int(v) for v in d["size"]), str(d["source"]))


def dlc_h5_all(path):
    """Every individual in a DeepLabCut file: names, xy (T, I, K, 2) and conf (T, I, K)."""
    import pandas as pd

    df = pd.read_hdf(path)
    cols = df.columns
    levels = list(cols.names)
    inds = list(dict.fromkeys(cols.get_level_values(levels.index("individuals")))) if "individuals" in levels else [None]
    names = list(dict.fromkeys(cols.get_level_values(levels.index("bodyparts"))))
    T, I, K = len(df), len(inds), len(names)
    xy = np.full((T, I, K, 2), np.nan)
    conf = np.zeros((T, I, K))
    for i, ind in enumerate(inds):
        sub = df.xs(ind, axis=1, level="individuals") if ind is not None else df
        for k, n in enumerate(names):
            part = sub.xs(n, axis=1, level="bodyparts")
            part.columns = part.columns.get_level_values(-1)
            xy[:, i, k, 0] = part["x"].to_numpy(dtype=float)
            xy[:, i, k, 1] = part["y"].to_numpy(dtype=float)
            conf[:, i, k] = part["likelihood"].to_numpy(dtype=float)
    return names, xy, conf


def from_dlc_h5(path, fps, size, source=""):
    """Read a DeepLabCut prediction file. With several individuals, keep the most confident one."""
    import pandas as pd

    df = pd.read_hdf(path)
    cols = df.columns
    levels = list(cols.names)
    bp_level = levels.index("bodyparts")
    co_level = levels.index("coords")
    if "individuals" in levels:
        ind_level = levels.index("individuals")
        best, best_score = None, -1.0
        for ind in cols.get_level_values(ind_level).unique():
            sub = df.xs(ind, axis=1, level=ind_level)
            lik = sub.xs("likelihood", axis=1, level="coords").to_numpy(dtype=float)
            score = np.nanmean(lik)
            if score > best_score:
                best, best_score = sub, score
        df = best
        cols = df.columns
        levels = list(cols.names)
        bp_level = levels.index("bodyparts")
    names = list(dict.fromkeys(cols.get_level_values(bp_level)))
    T, K = len(df), len(names)
    xy = np.full((T, K, 2), np.nan)
    conf = np.zeros((T, K))
    for k, n in enumerate(names):
        part = df.xs(n, axis=1, level="bodyparts")
        part.columns = part.columns.get_level_values(-1)
        xy[:, k, 0] = part["x"].to_numpy(dtype=float)
        xy[:, k, 1] = part["y"].to_numpy(dtype=float)
        conf[:, k] = part["likelihood"].to_numpy(dtype=float)
    return Tracks(names, xy, conf, fps, size, source)
