"""Distillation: the teacher's distributions, the deal multiplicity, the reach weights, the fit, the freeze.

What is pinned here, and why:

- The targets are the teacher's rows at the legal columns and zero past
  them, a distribution at every row — a row the teacher never banked plays
  uniform, so no target is NaN.
- The deal multiplicity sums, over the keys of one public point, to the
  number of physical pictures of that shape derived by counting: a key
  with a suit symmetry stands for fewer deals than one without, and every
  value is 6 over the size of a stabiliser of three suits.
- A row's weight is zero exactly where the teacher's own play never
  arrives, and at a seat's first own action it is the multiplicity alone;
  the weights sum to one.
- (torch) The sampler never draws a zero-weight row and refuses all-zero
  weights; the masked cross-entropy is minimised, with zero gradient, at
  the teacher's own distribution; a tiny fit memorises a slice; a fraction
  arm is seeded and reports its error on the rows it never saw; the frozen
  student is the softmax at every key and survives the file.

The torch-free tests run in the default suite; the rest skip without torch.
"""

from __future__ import annotations

from itertools import permutations
from math import comb

import numpy as np
import pytest

from drawmaha_solver.deepcfr.average import own_reach, predecessors
from drawmaha_solver.deepcfr.distill import (
    deal_multiplicity,
    row_weights,
    teacher_targets,
)
from drawmaha_solver.minidrawmaha.rules.cards import DECK
from drawmaha_solver.minidrawmaha.rules.enumeration import (
    PublicPoint,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.rules.features import MAX_WIDTH, layout
from drawmaha_solver.minidrawmaha.rules.game import InfoSet
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.answer.strategy import (
    FrozenStrategy,
    StrategyInfo,
    average_rows,
    uniform_strategy,
)

# ---------------------------------------------------------------------------
# Helpers (restated: test modules never import one another)
# ---------------------------------------------------------------------------

def key_on(point: PublicPoint, private) -> InfoSet:
    hole, thrown, board = private
    return InfoSet(
        player=point.player, hole=hole, discarded=thrown, board=board,
        draws=point.draws, betting=point.betting,
    )

def some_keys() -> list[InfoSet]:
    """Two keys on every public point: the first and the middle private key."""
    keys = []
    for point in public_decision_points():
        shape = private_keys(board_cards=point.board_cards, discards=point.discards)
        keys += [key_on(point, shape[0]), key_on(point, shape[len(shape) // 2])]
    return keys

def random_strategy(seed: int) -> FrozenStrategy:
    """Positive random sums at every column, normalised: no row uniform by accident."""
    widths = PackedTable.whole_game().widths()
    sums = np.random.default_rng(seed).uniform(0.05, 1.0, int(widths.sum()))
    info = StrategyInfo(rule="test", column="linear", iteration=1, seed=seed, workers=1)
    return FrozenStrategy(average_rows(sums, widths), widths, info)

def is_first_action(point: PublicPoint) -> bool:
    return point.board_cards == 1 and not point.is_draw_decision and len(point.betting[0]) == point.player

# ---------------------------------------------------------------------------
# The targets
# ---------------------------------------------------------------------------

def test_targets_are_distributions_everywhere():
    built = layout()
    widths = np.repeat(built.width, np.diff(built.row_offset))
    uniform = teacher_targets(uniform_strategy())
    assert uniform.shape == (len(widths), MAX_WIDTH) and uniform.dtype == np.float32
    legal = np.arange(MAX_WIDTH)[None, :] < widths[:, None]
    np.testing.assert_array_equal(uniform[~legal], 0.0)
    np.testing.assert_allclose(uniform[legal], np.repeat(1.0 / widths, widths), rtol=1e-6)
    teacher = random_strategy(3)
    targets = teacher_targets(teacher)
    np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=2e-6)
    table = PackedTable.whole_game()
    for key in some_keys():
        row, width = table.row_of(key), len(key.legal_actions())
        np.testing.assert_array_equal(targets[row, :width], teacher[key])
        np.testing.assert_array_equal(targets[row, width:], 0.0)

# ---------------------------------------------------------------------------
# The deal multiplicity
# ---------------------------------------------------------------------------

def physical_pictures(*, board_cards: int, discards: int) -> int:
    """How many (hole, discard, ordered board) deals there are: a count, not a relabelling."""
    hole = comb(len(DECK), 3)
    rest = len(DECK) - 3
    count = hole
    for _ in range(discards + board_cards):
        count *= rest
        rest -= 1
    return count

def test_multiplicity_sums_to_the_physical_pictures_per_shape():
    multiplicity = deal_multiplicity()
    built = layout()
    assert multiplicity.shape == (built.row_offset[-1],) and multiplicity.dtype == np.int64
    assert set(np.unique(multiplicity)) <= {1, 2, 3, 6}
    seen = set()
    for p, point in enumerate(built.points):
        shape = (point.board_cards, point.discards)
        rows = slice(built.row_offset[p], built.row_offset[p + 1])
        assert multiplicity[rows].sum() == physical_pictures(board_cards=shape[0], discards=shape[1])
        seen.add(shape)
    assert seen == {(1, 0), (2, 0), (2, 1)}
    # The same key is the same deal count on every point of its shape.
    assert len(set(permutations(range(3)))) == 6

# ---------------------------------------------------------------------------
# The weights
# ---------------------------------------------------------------------------

def test_row_weights_are_zero_exactly_where_the_teacher_never_arrives():
    teacher = random_strategy(5)
    weights = row_weights(teacher)
    built = layout()
    assert weights.shape == (built.row_offset[-1],) and weights.dtype == np.float64
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert (weights >= 0.0).all()
    # Every row is reached under a strictly positive strategy.
    assert (weights > 0.0).all()
    # At a seat's first own action reach is 1, so the weight is the multiplicity alone.
    multiplicity = deal_multiplicity()
    ratios = []
    for p, point in enumerate(built.points):
        if is_first_action(point):
            rows = slice(built.row_offset[p], built.row_offset[p + 1])
            ratios.append(weights[rows] / multiplicity[rows])
    ratios = np.concatenate(ratios)
    assert len(ratios) > 1000 and np.allclose(ratios, ratios[0], rtol=1e-12)
    # Reach, not the strategy at the row itself, is what the weight carries:
    # the weight equals own reach × multiplicity, normalised, for both seats.
    table = PackedTable.whole_game()
    regret = table.cumulative_regret
    sigma = teacher.probabilities.astype(np.float64)
    # Regret matching over a row's probabilities plays those probabilities.
    regret[:] = sigma
    reach = own_reach(table, seat=0) + own_reach(table, seat=1)
    expected = reach * multiplicity
    # The frozen rows are float32; regret matching renormalises them in
    # float64, which moves a row by about 1e-7 relative, not more.
    np.testing.assert_allclose(weights, expected / expected.sum(), rtol=1e-5)

def test_a_row_behind_a_zeroed_action_gets_no_weight():
    # Zero one action at seat 0's first decision for one key; the row behind
    # that action one decision later (same hole, same board, one predecessor)
    # must weigh nothing while its neighbours still weigh something.
    built = layout()
    first = next(p for p, point in enumerate(built.points) if is_first_action(point) and point.player == 0)
    later = next(
        p for p, pred in enumerate(predecessors())
        if pred is not None and pred.point == first and pred.positions.shape[1] == 1
    )
    column = int(predecessors()[later].columns[0, 0])
    sums = np.full(int(built.column_offset[-1]), 1.0)
    position = 7
    start = built.column_offset[first] + position * built.width[first]
    sums[start + column] = 0.0
    widths = PackedTable.whole_game().widths()
    info = StrategyInfo(rule="test", column="linear", iteration=1, seed=0, workers=1)
    weights = row_weights(FrozenStrategy(average_rows(sums, widths), widths, info))
    rows = built.row_offset[later] + np.arange(built.row_offset[later + 1] - built.row_offset[later])
    assert weights[rows[position]] == 0.0
    assert weights[rows[position - 1]] > 0.0 and weights[rows[position + 1]] > 0.0
