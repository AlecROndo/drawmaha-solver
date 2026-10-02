"""Rung 3's analysis: the grade bookkeeping and the strategy readouts.

The figures are drawn from a two-run stand-in for `grades.json` and a uniform
profile, and `grade_runs` is driven with a stand-in grade over a one-iteration
checkpoint, so nothing ungated here needs the release file or two minutes of
the exact walk. That the committed headline numbers — the frozen strategy's
and the uniform reference line's — are what the grader says is checked by
one test behind `MINIDRAWMAHA_FULL_GRADE=1`.
"""

import json
import os

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha import analysis
from drawmaha_solver.minidrawmaha.analysis import (
    GRADES,
    UNIFORM_RANDOM,
    answer_sheet,
    draw_frequencies,
    fig_columns,
    fig_draw,
    fig_race,
    fig_round_one,
    grade,
    grade_key,
    grade_runs,
    round_one_frequencies,
    series,
)
from drawmaha_solver.minidrawmaha.hands import InnerCategory
from drawmaha_solver.minidrawmaha.lockstep import new_lockstep, save_lockstep, train_lockstep
from drawmaha_solver.minidrawmaha.regret_rules import Average
from drawmaha_solver.minidrawmaha.strategy import (
    StrategyInfo,
    fetch_strategy,
    load_strategy,
    uniform_strategy,
)

FULL_GRADE = os.environ.get("MINIDRAWMAHA_FULL_GRADE") == "1"


class Uniform(dict):
    def __getitem__(self, key):
        width = len(key.legal_actions())
        return np.full(width, 1 / width)


def fake_grades():
    grades = {"uniform_random": {"exploitability": 4.84}}
    for rule, scale in (("vanilla", 1.0), ("lcfr", 0.7), ("cfrplus", 3.0), ("dcfr", 3.0)):
        for iteration in (10_000, 2_000_000):
            columns = ["linear", "uniform"] if rule in ("vanilla", "cfrplus") else ["linear"]
            for column in columns:
                grades[grade_key(rule, iteration, column)] = {
                    "exploitability": scale / np.sqrt(iteration), "br0": 0.0, "br1": 0.0,
                    "value_p0": -0.09, "workers": 10, "seed": 0,
                }
    return grades


def test_a_grade_key_sorts_by_iteration_and_names_its_column():
    assert grade_key("lcfr", 2_000_000, "linear") == "lcfr/002000000/linear"
    assert sorted([grade_key("dcfr", 20_000, "linear"), grade_key("dcfr", 100_000, "linear")])[0] \
        .endswith("000020000/linear")


def test_a_series_is_hands_per_seat_in_order_and_skips_reference_lines():
    points = series(fake_grades(), "vanilla")
    assert [hands for hands, _ in points] == [100_000, 20_000_000]
    assert points[0][1] > points[1][1]
    assert series(fake_grades(), "vanilla", "uniform") and not series(fake_grades(), "lcfr", "uniform")


@pytest.fixture(scope="module")
def uniform_readouts():
    return round_one_frequencies(Uniform()), draw_frequencies(Uniform())


def test_a_uniform_profile_reads_back_uniform_in_every_category(uniform_readouts):
    round_one, draw = uniform_readouts
    assert set(round_one["p0_open"]) == {c.name.lower() for c in InnerCategory}
    for spots in (round_one, draw):
        for by_category in spots.values():
            for row in by_category.values():
                np.testing.assert_allclose(row, np.full(len(row), 1 / len(row)))
    assert len(round_one["p1_facing_bet"]["pair"]) == 3
    assert len(draw["p1_after_draw"]["trips"]) == 4


def test_the_figures_and_the_sheet_are_written(tmp_path, uniform_readouts):
    round_one, draw = uniform_readouts
    grades = fake_grades()
    fig_race(grades, tmp_path / "race.png")
    fig_columns(grades, tmp_path / "columns.png")
    fig_round_one(round_one, tmp_path / "round_one.png")
    fig_draw(draw, tmp_path / "draw.png")
    assert all((tmp_path / name).stat().st_size > 0
               for name in ("race.png", "columns.png", "round_one.png", "draw.png"))

    class Frozen:
        info = StrategyInfo(rule="lcfr", column="linear", iteration=2_000_000, seed=0, workers=10)

    frozen = {"br0": -0.03, "br1": 0.15, "exploitability": 0.06, "value_p0": -0.09}
    sheet = answer_sheet(grades, Frozen(), frozen, round_one, draw)
    assert sheet["race_final_linear"]["LCFR"]["hands_per_seat"] == 20_000_000
    assert sheet["frozen_strategy"]["iteration"] == 2_000_000
    json.dumps(sheet)  # the sheet is plain JSON


def test_a_rule_missing_its_last_checkpoint_is_refused_not_drawn_short(tmp_path):
    grades = fake_grades()
    del grades[grade_key("dcfr", 2_000_000, "linear")]
    with pytest.raises(ValueError, match="dcfr"):
        fig_columns(grades, tmp_path / "columns.png")


def test_an_unknown_rule_is_named_rather_than_a_bare_key_error():
    grades = {**fake_grades(), grade_key("mystery", 10_000, "linear"): {"workers": 10}}
    with pytest.raises(ValueError, match="mystery"):
        series(grades, "vanilla")


def test_grading_runs_skips_what_is_held_and_banks_the_uniform_line(tmp_path, monkeypatch):
    # The real grade is two minutes a call; what is under test here is the
    # bookkeeping around it, so the grade is a stand-in that records its calls.
    graded = []

    def fake_grade(strategy):
        graded.append(strategy.info)
        return {"br0": 0.0, "br1": 0.0, "exploitability": float(len(graded)), "value_p0": 0.0}

    monkeypatch.setattr(analysis, "grade", fake_grade)
    runs = tmp_path / "runs"
    (runs / "lcfr").mkdir(parents=True)
    solve = new_lockstep(0, workers=1, rule="lcfr", averages=(Average.UNIFORM,))
    train_lockstep(solve, 1)
    save_lockstep(solve, runs / "lcfr" / "iter-000000001.npz")
    grades_path = tmp_path / "grades.json"
    held = grade_key("lcfr", 1, "uniform")
    grades_path.write_text(json.dumps({held: {"exploitability": -1.0}}))

    grade_runs(runs, grades_path)
    grades = json.loads(grades_path.read_text())
    assert grades[held] == {"exploitability": -1.0}, "a held grade is never redone"
    assert grades[UNIFORM_RANDOM]["exploitability"] == 1.0
    assert grades[grade_key("lcfr", 1, "linear")]["workers"] == 1
    assert [info.column for info in graded] == ["uniform", "linear"]

    grade_runs(runs, grades_path)
    assert len(graded) == 2, "a second pass finds nothing new"


@pytest.mark.skipif(
    not FULL_GRADE,
    reason="downloads the 18 MB release and grades it twice, ~5 min; set MINIDRAWMAHA_FULL_GRADE=1",
)
def test_the_committed_headline_numbers_are_what_the_grader_says():
    grades = json.loads(GRADES.read_text())
    frozen = grade(load_strategy(fetch_strategy()))
    for name in ("br0", "br1", "exploitability", "value_p0"):
        assert frozen[name] == pytest.approx(grades["lcfr/002000000/linear"][name], abs=1e-6)
    uniform = grade(uniform_strategy())
    for name in ("br0", "br1", "exploitability", "value_p0"):
        assert uniform[name] == pytest.approx(grades[UNIFORM_RANDOM][name], abs=1e-9)
