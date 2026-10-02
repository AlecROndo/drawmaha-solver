"""Rung 3's analysis: the grade bookkeeping, the strategy readouts and the figure titles.

The figures are drawn from a four-rule stand-in for `grades.json` and a
uniform profile, and `grade_runs` is driven with a stand-in grade over a
one-iteration checkpoint, so nothing ungated here needs the release file or
two minutes of the exact walk. The titles are captured as they are drawn,
because each one states a finding read off the data and must move with it.
That the committed headline numbers — the frozen strategy's and the uniform
reference line's — are what the grader says is checked by one test behind
`MINIDRAWMAHA_FULL_GRADE=1`.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from drawmaha_solver.minidrawmaha import analysis
from drawmaha_solver.minidrawmaha.analysis import (
    GRADES,
    RULES,
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
    """Every infoset's row uniform, built on demand instead of from the 18 MB table."""

    def __getitem__(self, key):
        width = len(key.legal_actions())
        return np.full(width, 1 / width)

def fake_grades():
    """Two checkpoints per rule, ordered the way the real race finishes."""
    grades = {"uniform_random": {"exploitability": 4.84}}
    for rule, scale in (("vanilla", 1.0), ("lcfr", 0.7), ("cfrplus", 3.0), ("dcfr", 3.0)):
        for iteration in (10_000, 2_000_000):
            # Uniform averaging lands a fifth worse than linear, as it does at 2M.
            columns = {"linear": 1.0, "uniform": 1.2} if rule in ("vanilla", "cfrplus") \
                else {"linear": 1.0}
            for column, penalty in columns.items():
                grades[grade_key(rule, iteration, column)] = {
                    "exploitability": penalty * scale / np.sqrt(iteration),
                    "br0": 0.0, "br1": 0.0,
                    "value_p0": -0.09, "workers": 10, "seed": 0,
                }
    return grades

@pytest.fixture(scope="module")
def uniform_readouts():
    return round_one_frequencies(Uniform()), draw_frequencies(Uniform())

@pytest.fixture
def titles(monkeypatch):
    """Every figure title drawn, in order, with nothing written to disk."""
    drawn = []
    real_new_axes = analysis.new_axes

    def recording_new_axes(title, **kwargs):
        drawn.append(title)
        return real_new_axes(title, **kwargs)

    monkeypatch.setattr(analysis, "new_axes", recording_new_axes)
    monkeypatch.setattr(analysis, "save", lambda fig, path: plt.close(fig))
    return drawn

# ---------------------------------------------------------------------------
# Reading grades.json
# ---------------------------------------------------------------------------

def test_a_grade_key_sorts_by_iteration_and_names_its_column():
    assert grade_key("lcfr", 2_000_000, "linear") == "lcfr/002000000/linear"
    assert sorted([grade_key("dcfr", 20_000, "linear"), grade_key("dcfr", 100_000, "linear")])[0] \
        .endswith("000020000/linear")

def test_a_series_is_hands_per_seat_in_order_and_skips_reference_lines():
    points = series(fake_grades(), "vanilla")
    assert [hands for hands, _ in points] == [100_000, 20_000_000]
    assert points[0][1] > points[1][1]
    assert series(fake_grades(), "vanilla", column=Average.UNIFORM)
    assert not series(fake_grades(), "lcfr", column=Average.UNIFORM)

def test_cfr_plus_is_drawn_in_its_own_shade_over_dcfr():
    # CFR+ is dashed on top of DCFR's wider line; in one shade its dashes vanish.
    cfr_plus, dcfr = RULES["cfrplus"], RULES["dcfr"]
    assert cfr_plus.colour != dcfr.colour
    assert cfr_plus.zorder > dcfr.zorder

def test_an_unknown_rule_is_named_rather_than_a_bare_key_error():
    grades = {**fake_grades(), grade_key("mystery", 10_000, "linear"): {"workers": 10}}
    with pytest.raises(ValueError, match="mystery"):
        series(grades, "vanilla")

def test_a_key_that_is_neither_a_run_nor_a_known_reference_line_is_refused():
    with pytest.raises(ValueError, match="mystery_line"):
        series({**fake_grades(), "mystery_line": {"exploitability": 1.0}}, "vanilla")

def test_a_key_naming_an_unknown_averaging_column_is_refused():
    grades = {**fake_grades(), "vanilla/000010000/cubic": {"exploitability": 1.0, "workers": 10}}
    with pytest.raises(ValueError, match="cubic"):
        series(grades, "vanilla")

# ---------------------------------------------------------------------------
# Grading checkpoints into grades.json
# ---------------------------------------------------------------------------

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

def test_grading_a_runs_directory_that_does_not_exist_is_refused_before_any_grading(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(analysis, "grade", lambda strategy: pytest.fail("graded something"))
    grades_path = tmp_path / "grades.json"
    with pytest.raises(FileNotFoundError, match="nowhere"):
        grade_runs(tmp_path / "nowhere", grades_path)
    assert not grades_path.exists()

def test_a_checkpoint_filed_under_another_rules_directory_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "grade", lambda strategy: {"exploitability": 0.0})
    (tmp_path / "dcfr").mkdir()
    solve = new_lockstep(0, workers=1, rule="lcfr")
    train_lockstep(solve, 1)
    save_lockstep(solve, tmp_path / "dcfr" / "iter-000000001.npz")
    with pytest.raises(ValueError, match="lcfr"):
        grade_runs(tmp_path, tmp_path / "grades.json")

# ---------------------------------------------------------------------------
# Reading the strategy
# ---------------------------------------------------------------------------

def test_a_uniform_profile_reads_back_uniform_in_every_category(uniform_readouts):
    round_one, draw = uniform_readouts
    assert set(round_one["p0_open"]) == {c.name.lower() for c in InnerCategory}
    for spots in (round_one, draw):
        for by_category in spots.values():
            for row in by_category.values():
                np.testing.assert_allclose(row, np.full(len(row), 1 / len(row)))
    assert len(round_one["p1_facing_bet"]["pair"]) == 3
    assert len(draw["p1_after_draw"]["trips"]) == 4

# ---------------------------------------------------------------------------
# Figures and the answer sheet
# ---------------------------------------------------------------------------

def test_the_figures_and_the_sheet_are_written(tmp_path, uniform_readouts):
    round_one, draw = uniform_readouts
    grades = fake_grades()
    fig_race(grades, tmp_path / "race.png")
    fig_columns(grades, tmp_path / "columns.png")
    fig_round_one(round_one, tmp_path / "round_one.png")
    fig_draw(draw, tmp_path / "draw.png")
    assert all((tmp_path / name).stat().st_size > 0
               for name in ("race.png", "columns.png", "round_one.png", "draw.png"))

    info = StrategyInfo(rule="lcfr", column="linear", iteration=2_000_000, seed=0, workers=10)
    frozen = {"br0": -0.03, "br1": 0.15, "exploitability": 0.06, "value_p0": -0.09}
    sheet = answer_sheet(grades=grades, info=info, frozen=frozen, round_one=round_one, draw=draw)
    assert sheet["race_final_linear"]["LCFR"]["hands_per_seat"] == 20_000_000
    assert sheet["frozen_strategy"]["iteration"] == 2_000_000
    json.dumps(sheet)  # the sheet is plain JSON

def test_the_race_needs_its_uniform_reference_line(tmp_path, titles):
    grades = fake_grades()
    del grades[UNIFORM_RANDOM]
    with pytest.raises(ValueError, match=UNIFORM_RANDOM):
        fig_race(grades, tmp_path / "race.png")

def test_the_race_title_claims_a_win_only_when_every_keeper_beats_every_flooring_rule(
    tmp_path, titles
):
    fig_race(fake_grades(), tmp_path / "race.png")
    assert titles[-1].startswith("Keeping negative regret wins")
    grades = fake_grades()
    grades[grade_key("cfrplus", 2_000_000, "linear")]["exploitability"] = 1e-6
    fig_race(grades, tmp_path / "race.png")
    assert "wins" not in titles[-1]

def test_a_rule_missing_its_last_checkpoint_is_refused_not_drawn_short(tmp_path):
    grades = fake_grades()
    del grades[grade_key("dcfr", 2_000_000, "linear")]
    with pytest.raises(ValueError, match="dcfr"):
        fig_columns(grades, tmp_path / "columns.png")

def test_the_column_title_names_the_better_column_per_rule(tmp_path, titles):
    fig_columns(fake_grades(), tmp_path / "columns.png")
    assert "linear beats uniform for vanilla and CFR+" in titles[-1]

def test_the_strategy_titles_read_their_numbers_off_the_bars(tmp_path, titles, uniform_readouts):
    round_one, draw = uniform_readouts
    fig_round_one(round_one, tmp_path / "round_one.png")
    assert "50%" in titles[-1] and "33%" in titles[-1]
    fig_draw(draw, tmp_path / "draw.png")
    assert "75%" in titles[-1]
    assert "almost always" not in titles[-1] and "stand pat" not in titles[-1]

# ---------------------------------------------------------------------------
# The committed numbers (gated: downloads and grades the release)
# ---------------------------------------------------------------------------

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
