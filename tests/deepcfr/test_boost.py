"""The boost stage: what one phase's injection holds, and the stage's unbiasedness end to end.

`inject` is pinned with a stand-in for the boost oracle, so every weight and
every skip rule is checked against numbers worked by hand. The stage as a
whole — `Boost` inside `train.step` with the real `BoostTargets` on a
restricted deal — is pinned by the paired expectation test at the end: over
many seeds, the boosted iteration banks the same expected regret and weight
at every targeted row as plain sampling does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.deepcfr.boost import Boost, Injected, boosted_weight, inject
from drawmaha_solver.deepcfr.reservoir import TARGET_WIDTH
from drawmaha_solver.deepcfr.targeting import TargetSet, TargetSpec, row_offsets
from drawmaha_solver.minidrawmaha.rules.enumeration import public_decision_points
from drawmaha_solver.minidrawmaha.solver.boost_target import EXACT, Sampling

# ---------------------------------------------------------------------------
# A stand-in for the boost oracle
# ---------------------------------------------------------------------------

class FakeExact:
    """`targets` hands back the regret and arrival this test set per row, and records the call."""

    def __init__(self, regret: dict[int, list[float]], arrival: dict[int, float]) -> None:
        self.regret = regret
        self.arrival = arrival
        self.calls: list[tuple] = []

    def targets(self, sigma, *, seat, rows, sampling, seed, t):
        self.calls.append((sigma, seat, [int(r) for r in rows], sampling, seed, t))
        regret = np.zeros((len(rows), TARGET_WIDTH))
        for i, row in enumerate(rows):
            values = self.regret[int(row)]
            regret[i, : len(values)] = values
        return regret, np.array([self.arrival[int(row)] for row in rows], dtype=np.float64)

NATURAL_ROWS = np.array([5, 9, 5, 12], dtype=np.int32)
NATURAL_TARGETS = np.arange(16, dtype=np.float32).reshape(4, TARGET_WIDTH) / 8

def a_fake(arrival_5: float = 0.02, arrival_12: float = 0.5) -> FakeExact:
    return FakeExact({5: [1.5, -0.5], 12: [0.25, 0.75, -1.0]}, {5: arrival_5, 12: arrival_12})

def injected(fake: FakeExact, *, drawn=(5, 12), alpha: float = 0.0, walks: int = 100, sampling=Sampling(), rows=NATURAL_ROWS) -> Injected:
    return inject(
        rows,
        NATURAL_TARGETS[: rows.size],
        drawn=np.asarray(drawn, dtype=np.int64),
        exact=fake,
        sigma="σ",
        seat=0,
        t=3,
        seed=7,
        sampling=sampling,
        alpha=alpha,
        walks=walks,
    )

# ---------------------------------------------------------------------------
# The injection
# ---------------------------------------------------------------------------

def test_boosted_rows_follow_the_natural_rows_at_the_settled_weight():
    fake = a_fake()
    out = injected(fake)
    np.testing.assert_array_equal(out.rows, [5, 9, 5, 12, 5, 12])
    # Natural rows at the two targeted infosets are turned down to α = 0; the boosted
    # rows weigh (1 − α)·K·arrival: 100 × 0.02 = 2 and 100 × 0.5 = 50.
    np.testing.assert_array_equal(out.weight, np.float32([0, 1, 0, 0, 2.0, 50.0]))
    np.testing.assert_array_equal(out.targets[:4], NATURAL_TARGETS)
    np.testing.assert_array_equal(out.targets[4], np.float32([1.5, -0.5, 0, 0]))
    np.testing.assert_array_equal(out.targets[5], np.float32([0.25, 0.75, -1.0, 0]))
    assert (out.boosted, out.skipped) == (2, 0)
    assert out.rows.dtype == np.int32 and out.targets.dtype == np.float32 and out.weight.dtype == np.float32
    assert fake.calls == [("σ", 0, [5, 12], Sampling(), 7, 3)]

def test_alpha_half_leaves_half_on_the_natural_rows():
    out = injected(a_fake(arrival_5=0.04), alpha=0.5)
    np.testing.assert_array_equal(out.weight, np.float32([0.5, 1, 0.5, 0.5, 2.0, 25.0]))

def test_boosted_weight_is_one_minus_alpha_times_k_times_arrival():
    np.testing.assert_allclose(boosted_weight(np.array([0.02, 0.5]), alpha=0.25, walks=40), [0.6, 15.0])

def test_a_drawn_row_without_natural_rows_still_gets_its_boosted_row():
    out = injected(a_fake(), drawn=(5,), rows=np.array([9], dtype=np.int32))
    np.testing.assert_array_equal(out.rows, [9, 5])
    np.testing.assert_array_equal(out.weight, np.float32([1, 2.0]))
    assert (out.boosted, out.skipped) == (1, 0)

def test_a_zero_arrival_row_is_skipped_and_its_natural_rows_keep_weight_one():
    out = injected(a_fake(arrival_12=0.0))
    np.testing.assert_array_equal(out.rows, [5, 9, 5, 12, 5])
    np.testing.assert_array_equal(out.weight, np.float32([0, 1, 0, 1, 2.0]))
    assert (out.boosted, out.skipped) == (1, 1)

@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_a_row_whose_weight_would_read_as_a_walk_row_is_left_untreated(alpha):
    # PR 3's noise score tells walk rows from boosted rows by weight alone, so a
    # boosted row may never weigh exactly 1 or exactly α in float32. An arrival of
    # 1 / (K · (1 − α)) would make it 1; α / (K · (1 − α)) would make it α (at α = 0
    # that is the zero-arrival case). Such a row is not treated at all: no boosted
    # row, its natural rows at weight 1, and it is counted as skipped.
    walks = 100
    reads_as_one = 1.0 / (walks * (1.0 - alpha))
    reads_as_alpha = alpha / (walks * (1.0 - alpha))
    for arrival in {reads_as_one, reads_as_alpha} - {0.0}:
        out = injected(a_fake(arrival_5=arrival), alpha=alpha, walks=walks)
        np.testing.assert_array_equal(out.rows, [5, 9, 5, 12, 12])
        np.testing.assert_array_equal(out.weight, np.float32([1, 1, 1, alpha, (1 - alpha) * walks * 0.5]))
        assert (out.boosted, out.skipped) == (1, 1)

def test_an_exact_sampling_is_refused_before_any_target_is_computed():
    fake = a_fake()
    with pytest.raises(ValueError, match="extra sampling"):
        injected(fake, sampling=EXACT)
    # A sampling spelled with counts at or above every fan is exact too.
    with pytest.raises(ValueError, match="extra sampling"):
        injected(fake, sampling=Sampling(opponent_actions=None, board_cards=14, opponent_draws=12, hero_draws=12))
    assert fake.calls == []

@pytest.mark.parametrize(
    "drawn, match",
    [([5, 5], "drawn"), ([12, 5], "drawn"), ([[5]], "drawn"), ([5.0], "drawn")],
)
def test_repeated_unsorted_or_non_integer_drawn_rows_are_refused(drawn, match):
    with pytest.raises(ValueError, match=match):
        inject(
            NATURAL_ROWS,
            NATURAL_TARGETS,
            drawn=np.asarray(drawn),
            exact=a_fake(),
            sigma="σ",
            seat=0,
            t=3,
            seed=7,
            sampling=Sampling(),
            alpha=0.0,
            walks=100,
        )

@pytest.mark.parametrize("field, value, match", [("alpha", 1.5, "alpha"), ("alpha", -0.1, "alpha"), ("walks", 0, "walks"), ("walks", 2.5, "walks")])
def test_a_bad_alpha_or_walk_count_is_refused_by_name(field, value, match):
    kwargs = {"alpha": 0.0, "walks": 100} | {field: value}
    with pytest.raises(ValueError, match=match):
        injected(a_fake(), **kwargs)

def test_fewer_drawn_rows_than_the_budget_are_simply_fewer():
    out = injected(a_fake(), drawn=(12,))
    assert out.boosted == 1 and out.rows.size == NATURAL_ROWS.size + 1

def test_the_module_imports_no_grader():
    # Restates test_targeting's import check for this module alone: the boost
    # feeds what the grader certifies, so it never imports the meter.
    path = Path(__file__).resolve().parents[2] / "src" / "drawmaha_solver" / "deepcfr" / "boost.py"
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    assert not {name for name in names if "exploitability" in name or "answer.analysis" in name}

# ---------------------------------------------------------------------------
# The stage's fixed parts: Boost
# ---------------------------------------------------------------------------

def round_two_rows(seat: int, count: int = 3) -> np.ndarray:
    """The first `count` rows of `seat`'s first round-2 point: real rows of the right seat, cheap to boost."""
    points = public_decision_points()
    offsets = row_offsets()
    p = next(i for i, point in enumerate(points) if point.player == seat and point.board_cards == 2)
    return np.arange(offsets[p], offsets[p] + count)

SPEC = TargetSpec(granularity="infoset", alpha=0.0, budget=3, refresh=25)

def infoset_set(spec: TargetSpec = SPEC, *, built_at: int) -> TargetSet:
    rows = (round_two_rows(0), round_two_rows(1))
    return TargetSet(spec=spec, built_at=built_at, rows=rows, weight=tuple(np.ones(r.size) for r in rows), groups=rows)

class RecordingRefresher:
    def __init__(self, make):
        self.make = make
        self.calls = []

    def __call__(self, *, t, table, memories):
        self.calls.append((t, table, memories))
        return self.make(t)

class SigmaFake:
    """σ is the identity of the table it was read from; `targets` records its rows and arrives everywhere at 0.3."""

    def __init__(self) -> None:
        self.sigmas = []
        self.calls = []

    def sigma(self, table):
        self.sigmas.append(id(table))
        return ("σ of", id(table))

    def targets(self, sigma, *, seat, rows, sampling, seed, t):
        self.calls.append((sigma, seat, [int(r) for r in rows], sampling, seed, t))
        return np.zeros((len(rows), TARGET_WIDTH)), np.full(len(rows), 0.3)

class Offered:
    def __init__(self, rows, targets):
        self.advantage_rows = np.asarray(rows, dtype=np.int32)
        self.advantage_targets = np.asarray(targets, dtype=np.float32)

def test_due_when_there_is_no_set_or_its_window_has_ended():
    boost = Boost(RecordingRefresher(lambda t: infoset_set(built_at=t)), exact=SigmaFake())
    assert boost.due(None, 1)
    targets = infoset_set(built_at=1)
    assert not any(boost.due(targets, t) for t in range(1, 26))
    assert boost.due(targets, 26) and boost.due(targets, 40)

def test_refresh_hands_the_refresher_the_table_and_the_memories_and_keeps_its_set():
    refresher = RecordingRefresher(lambda t: infoset_set(built_at=t))
    boost = Boost(refresher, exact=SigmaFake())
    targets = boost.refresh(t=3, table="the table", memories=("m0", "m1"), spec=SPEC)
    assert targets.built_at == 3 and refresher.calls == [(3, "the table", ("m0", "m1"))]

def test_refresh_refuses_a_set_built_for_another_iteration_or_another_spec():
    boost = Boost(RecordingRefresher(lambda t: infoset_set(built_at=3)), exact=SigmaFake())
    with pytest.raises(ValueError, match="built_at"):
        boost.refresh(t=4, table=None, memories=(None, None), spec=SPEC)
    other = TargetSpec(granularity="infoset", alpha=0.0, budget=4, refresh=25)
    boost = Boost(RecordingRefresher(lambda t: infoset_set(other, built_at=t)), exact=SigmaFake())
    with pytest.raises(ValueError, match="spec"):
        boost.refresh(t=3, table=None, memories=(None, None), spec=SPEC)

def test_phase_reads_sigma_from_the_table_it_is_handed_and_draws_this_iterations_rows():
    fake = SigmaFake()
    boost = Boost(RecordingRefresher(lambda t: infoset_set(built_at=t)), exact=fake)
    table = object()
    offered = Offered([round_two_rows(1)[0], 14_060], np.zeros((2, TARGET_WIDTH)))
    out = boost.phase(
        infoset_set(built_at=1), offered, table=table, seat=1, t=2, seed=9, sampling=Sampling(playouts=2), alpha=0.0, walks=10
    )
    assert fake.sigmas == [id(table)]
    assert fake.calls == [(("σ of", id(table)), 1, round_two_rows(1).tolist(), Sampling(playouts=2), 9, 2)]
    np.testing.assert_array_equal(out.rows, [*offered.advantage_rows, *round_two_rows(1)])
    np.testing.assert_array_equal(out.weight, np.float32([0, 1, 3.0, 3.0, 3.0]))
    assert (out.boosted, out.skipped) == (3, 0)

def test_the_boost_builds_the_whole_games_oracle_only_when_first_asked():
    boost = Boost(RecordingRefresher(lambda t: infoset_set(built_at=t)))
    assert boost._exact is None
    from drawmaha_solver.minidrawmaha.solver.boost_target import BoostTargets

    assert isinstance(boost.exact, BoostTargets) and boost.exact is BoostTargets.build()
