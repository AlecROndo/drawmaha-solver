"""`scripts/deepcfr/distill_teacher.py`: the arms, their records, and the labels on the references.

What is pinned here, and why:

- Every arm name parses to the size it says, the headline is 4 × 256 at
  the Deep CFR arm's size, and the ablation and fraction arms differ from
  it in exactly the field their suffix names.
- The references label hands per seat from the files: the teacher's is
  iteration × workers read off the strategy's info, and Deep CFR's is read
  from the committed grades file, so the axis the figures share cannot be
  mislabelled by hand.
- An arm's record carries its parameter count, its config, its grade and
  its curve, and nothing in it is typed in.

Torch-free: the script imports torch only inside the fit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.answer.strategy import StrategyInfo, uniform_strategy

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "deepcfr" / "distill_teacher.py"

@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("distill_teacher_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_every_arm_parses_to_its_size_and_the_headline_is_the_deep_cfr_net(script):
    arms = script.ARMS
    headline = arms[script.HEADLINE]
    assert (headline.width, headline.depth) == (256, 4)
    assert headline.train_fraction == 1.0 and headline.weighted and headline.device == "mps"
    for name, config in arms.items():
        size, *suffix = name.split("-")
        depth, width = (int(part) for part in size.split("x"))
        assert (config.width, config.depth) == (width, depth), name
        if not suffix:
            assert (config.steps, config.train_fraction, config.weighted) == (headline.steps, 1.0, True)
        elif suffix == ["unweighted"]:
            assert not config.weighted and config.train_fraction == 1.0
        elif suffix[0].startswith("f"):
            assert config.train_fraction == float(suffix[0][1:]) and config.weighted
        elif suffix[0].endswith("k"):
            assert config.steps == 1_000 * int(suffix[0][:-1]) and config.train_fraction == 1.0
        else:
            raise AssertionError(f"unknown arm suffix in {name}")
    assert {"3x64", "3x128", "4x256", "4x512", "4x1024"} <= set(arms)

def test_references_label_hands_per_seat_from_the_files(script):
    teacher = uniform_strategy()
    info = StrategyInfo(rule="lcfr", column="linear", iteration=10_000_000, seed=0, workers=10)
    grade = {"br0": -0.08, "br1": 0.11, "exploitability": 0.015, "value_p0": -0.097}
    refs = script.references(info, grade, Path("iter-010000000.npz"), ROOT / script.DEEP_GRADES)
    assert refs["teacher"]["hands_per_seat"] == 100_000_000
    assert refs["teacher"]["iteration"] == 10_000_000 and refs["teacher"]["checkpoint"] == "iter-010000000.npz"
    assert refs["teacher"]["exploitability"] == 0.015
    deep = json.loads((ROOT / script.DEEP_GRADES).read_text())
    assert refs["deep_cfr"]["t"] == 2_000 and refs["deep_cfr"]["hands_per_seat"] == 20_000_000
    for column in ("exact", "pi", "current"):
        assert refs["deep_cfr"][column]["exploitability"] == deep[f"deep/02000/{column}"]["exploitability"]
    assert refs["deep_cfr"]["width"] == 256 and refs["deep_cfr"]["depth"] == 4
    assert teacher.info.rule == "uniform"  # the strategy itself is not what the references read

def test_parameter_count_is_the_mlp_formula(script):
    assert script.parameter_count(width=64, depth=3) == 17_543  # the net test's pin
    assert script.parameter_count(width=256, depth=4) == 136 * 256 + 256 + 3 * (256 * 256 + 256) + 256 * 7 + 7

def test_arm_record_carries_config_curve_and_grade_and_nothing_typed_in(script):
    config = script.ARMS["3x64"]
    curve = [{"step": 10, "loss": 0.5, "lr": 1e-3, "seconds": 1.0, "rw_error": 0.1}]
    grade = {"br0": 0.0, "br1": 0.02, "exploitability": 0.01, "value_p0": -0.1}
    record = script.arm_record(config, curve=curve, fit_s=12.0, grade=grade, grade_s=100.0,
                               strategy_file="runs/distill/3x64.npz", strategy_sha256="ab" * 32,
                               errors={"rw_error": 0.1})
    assert record["parameters"] == 17_543 and record["width"] == 64 and record["depth"] == 3
    assert record["curve"] == curve and record["exploitability"] == 0.01 and record["rw_error"] == 0.1
    assert record["strategy_sha256"] == "ab" * 32 and record["fit_s"] == 12.0
    assert set(np.unique(list(record))) >= {"steps", "batch", "lr", "train_fraction", "weighted", "seed", "device"}
