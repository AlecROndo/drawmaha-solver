"""The grader-backed refresher: which profile it grades, what it hands the trainer, and when it skips the grader.

The script is loaded by path, as the other script tests do. The wiring is
pinned with stand-ins for the grader walk and the boost targets, so the
default suite runs no two-minute walk; one end-to-end refresh on the real
game sits behind MINIDRAWMAHA_FULL_GRADE.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.deepcfr.reservoir import Reservoir
from drawmaha_solver.deepcfr.targeting import TargetSpec, groups, n_rows, row_offsets
from drawmaha_solver.deepcfr.train import current_iterate
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deepcfr" / "targeting_refresh.py"

@pytest.fixture(scope="module")
def refresh():
    spec = importlib.util.spec_from_file_location("targeting_refresh", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope="module")
def table():
    table = PackedTable.whole_game()
    table.cumulative_regret[:] = np.random.default_rng(0).standard_normal(table.cumulative_regret.size)
    return table

class FakeAverage:
    """Records the freeze it was asked for and hands back a marker."""

    def __init__(self) -> None:
        self.asked = []

    def frozen(self, *, iteration: int, seed: int):
        self.asked.append((iteration, seed))
        return "the frozen average"

class FakeExact:
    """A stand-in for `BoostTargets`: σ is a marker, the arrival a weight per row from its number."""

    def __init__(self) -> None:
        self.asked = []

    def sigma(self, table):
        return ("sigma of", id(table))

    def arrivals(self, sigma, *, seat, rows):
        self.asked.append((sigma, seat, np.array(rows)))
        return (np.asarray(rows) % 5) + 1.0

def first_line(seat: int) -> np.ndarray:
    offsets = row_offsets()
    p = int(np.unique(groups("line", seat)[groups("line", seat) >= 0])[0])
    return np.arange(offsets[p], offsets[p + 1])

def test_the_average_is_required_only_when_the_spec_aims_at_it(refresh):
    with pytest.raises(ValueError, match="average"):
        refresh.GraderRefresher(TargetSpec(granularity="line", alpha=0.0), walks=10)
    refresh.GraderRefresher(TargetSpec(granularity="line", alpha=0.0, leak="current"), walks=10)  # grades the table
    refresh.GraderRefresher(TargetSpec(granularity="random", alpha=0.0), walks=10)  # grades nothing

@pytest.mark.parametrize("walks", [0, 2.5, True, None])
def test_a_bad_walk_count_is_refused_before_any_grader_walk(refresh, walks):
    with pytest.raises(ValueError, match="walks"):
        refresh.GraderRefresher(TargetSpec(granularity="line", alpha=0.0), walks=walks, average=FakeAverage())

def test_a_spec_that_is_not_a_spec_is_refused(refresh):
    with pytest.raises(ValueError, match="spec"):
        refresh.GraderRefresher({"granularity": "line"}, walks=10)

def test_an_infoset_refresh_never_reads_sigma(refresh, table, monkeypatch):
    class NoSigma(FakeExact):
        def sigma(self, table):
            raise AssertionError("an infoset set draws nothing, so it needs no arrival and no σ")

    def fake_gains(profile, *, game=None):
        leak = np.zeros(n_rows())
        leak[first_line(0)[:3]] = 1.0
        leak[first_line(1)[:3]] = 1.0
        return leak, np.ones(n_rows())

    monkeypatch.setattr(refresh, "gains_by_row", fake_gains)
    refresher = refresh.GraderRefresher(
        TargetSpec(granularity="infoset", alpha=0.0, budget=3), walks=10, average=FakeAverage(), exact=NoSigma()
    )
    targets = refresher(t=1, table=table, memories=(Reservoir(4), Reservoir(4)))
    assert targets.rows[0].tolist() == first_line(0)[:3].tolist()

def test_the_default_leak_is_the_exact_average(refresh, table):
    average = FakeAverage()
    refresher = refresh.GraderRefresher(TargetSpec(granularity="line", alpha=0.0), walks=10, average=average, seed=4)
    assert refresher.spec.leak == "average"
    assert refresher.profile(t=26, table=table) == "the frozen average"
    assert average.asked == [(26, 4)]

def test_the_current_leak_grades_the_tables_own_strategy(refresh, table):
    refresher = refresh.GraderRefresher(TargetSpec(granularity="line", alpha=0.0, leak="current"), walks=10)
    profile = refresher.profile(t=26, table=table)
    assert np.array_equal(profile.probabilities, current_iterate(table).probabilities)

def test_the_refresher_hands_the_grader_rows_and_the_walks_arrival_to_targeting(refresh, table, monkeypatch):
    graded = []

    def fake_gains(profile, *, game=None):
        graded.append(profile)
        leak, frequency = np.zeros(n_rows()), np.ones(n_rows())
        for seat in (0, 1):
            leak[first_line(seat)[:10]] = 1.0
        return leak, frequency

    monkeypatch.setattr(refresh, "gains_by_row", fake_gains)
    exact = FakeExact()
    spec = TargetSpec(granularity="line", alpha=0.0, budget=2, pool=1)
    refresher = refresh.GraderRefresher(spec, walks=10, average=FakeAverage(), exact=exact)
    targets = refresher(t=26, table=table, memories=(Reservoir(4), Reservoir(4)))
    assert graded == ["the frozen average"]
    assert [(sigma, seat) for sigma, seat, _ in exact.asked] == [(("sigma of", id(table)), 0), (("sigma of", id(table)), 1)]
    for seat in (0, 1):
        assert np.array_equal(targets.rows[seat], first_line(seat))
        assert np.array_equal(targets.weight[seat], (first_line(seat) % 5) + 1.0)
    assert targets.built_at == 26

def test_the_random_arm_never_runs_the_grader(refresh, table, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the random arm aims at nothing, so it grades nothing")

    monkeypatch.setattr(refresh, "gains_by_row", refuse)
    refresher = refresh.GraderRefresher(TargetSpec(granularity="random", alpha=0.0), walks=10, exact=FakeExact())
    targets = refresher(t=1, table=table, memories=(Reservoir(4), Reservoir(4)))
    assert targets.rows[0].size == int((groups("random", 0) >= 0).sum())

@pytest.mark.skipif(
    not os.environ.get("MINIDRAWMAHA_FULL_GRADE"),
    reason="two exact grader walks of the whole game, about two minutes; set MINIDRAWMAHA_FULL_GRADE=1",
)
def test_a_real_refresh_pools_reached_rows_and_draws_reproducibly(refresh, table):
    spec = TargetSpec(granularity="line", alpha=0.0, budget=50, pool=20, leak="current")
    refresher = refresh.GraderRefresher(spec, walks=10)
    targets = refresher(t=1, table=table, memories=(Reservoir(4), Reservoir(4)))
    _, frequency = refresh.gains_by_row(refresher.profile(t=1, table=table))
    for seat in (0, 1):
        assert (frequency[targets.rows[seat]] > 0).all()
        assert int((frequency[targets.rows[seat]] > 0).sum()) >= spec.pool * spec.budget
        assert (targets.weight[seat] > 0).any()
        drawn = targets.rows_for(1, seat, 0)
        assert drawn.size == spec.budget and np.array_equal(drawn, targets.rows_for(1, seat, 0))
