"""CPU-only smoke tests: no GPU, no motion-capture data, no trained policies needed."""
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
mujoco = pytest.importorskip("mujoco")


@pytest.mark.parametrize("xml", ["body/dog_anat.xml", "body/dog_anat_lig.xml", "body/dog_mocap.xml"])
def test_body_compiles_and_steps(xml):
    m = mujoco.MjModel.from_xml_path(str(ROOT / xml))
    d = mujoco.MjData(m)
    assert m.nu == 54                       # 54 actuators
    assert abs(m.body_mass.sum() - 28.0) < 0.5   # a 28 kg dog
    for _ in range(500):                    # 1 s of passive physics
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all()


def test_lexicon_recipes_check_clean():
    out = subprocess.run([sys.executable, str(ROOT / "dog-lexicon/check_recipes.py")],
                         capture_output=True, text=True, check=True).stdout
    assert "problems: 0" in out


def test_grammar_generates_sentences():
    out = subprocess.run([sys.executable, str(ROOT / "dog/grammar.py"), "--dog", "mal", "--seed", "1", "generate"],
                         capture_output=True, text=True, check=True, cwd=ROOT).stdout
    lines = [l for l in out.splitlines() if "bits]" in l]
    assert lines, out
