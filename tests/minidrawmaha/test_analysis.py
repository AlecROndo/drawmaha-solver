"""Rung 3's analysis: the grade bookkeeping and the strategy readouts.

The figures are drawn from a two-run stand-in for `grades.json` and a uniform
profile, so nothing here needs a checkpoint or the release file. That the
committed numbers are what the grader says is pinned where the grader is.
"""

import json

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.analysis import (
    answer_sheet,
    draw_frequencies,
    fig_columns,
    fig_draw,
    fig_race,
    fig_round_one,
    grade_key,
    round_one_frequencies,
    series,
)
from drawmaha_solver.minidrawmaha.hands import InnerCategory
from drawmaha_solver.minidrawmaha.strategy import StrategyInfo


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
