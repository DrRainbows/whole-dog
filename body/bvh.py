"""Minimal BVH reader: skeleton hierarchy, offsets, channels, and motion as Euler angles."""
import re

import numpy as np


class Joint:
    def __init__(self, name, parent):
        self.name, self.parent = name, parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []
        self.end = None  # end-site offset, if any


def read(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    head, motion = text.split("MOTION")
    tokens = re.findall(r"[^\s{}]+|[{}]", head)
    joints, stack, cur, i = [], [], None, 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            j = Joint(tokens[i + 1], cur)
            if cur:
                cur.children.append(j)
            joints.append(j)
            cur = j
            i += 2
        elif t == "End":
            # End Site { OFFSET x y z }
            k = tokens.index("OFFSET", i)
            cur.end = np.array([float(v) for v in tokens[k + 1:k + 4]])
            depth = 0
            while True:
                if tokens[i] == "{":
                    depth += 1
                elif tokens[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            i += 1
        elif t == "OFFSET":
            cur.offset = np.array([float(v) for v in tokens[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            cur.channels = tokens[i + 2:i + 2 + n]
            i += 2 + n
        elif t == "}":
            cur = cur.parent if cur else None
            i += 1
        else:
            i += 1
    lines = [ln for ln in motion.strip().splitlines() if ln.strip()]
    frames = int(re.search(r"Frames:\s*(\d+)", lines[0]).group(1))
    ft = float(re.search(r"Frame Time:\s*([\d.eE-]+)", lines[1]).group(1))
    data = np.array([[float(v) for v in ln.split()] for ln in lines[2:2 + frames]])
    return joints, data, ft


def channel_slices(joints):
    out, k = {}, 0
    for j in joints:
        out[j.name] = slice(k, k + len(j.channels))
        k += len(j.channels)
    return out
