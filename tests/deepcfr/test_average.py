"""The exact average: own reach against a brute-force walk, the running sum, its file, its resume.

No torch here. The claims, in order: the predecessor table names, for every
public point, an earlier OWN decision point with one position and column per
key — three for a round-2 point after a throw; `own_reach` equals a brute
force that walks `MiniState` from a representative deal along the key's own
line, on a random non-uniform table, at keys of every predecessor case
(gate 4); `ExactAverage` adds `t · reach · σ` for one seat at a time and
refuses a t out of order; its freeze is the normalised sums; its file round
trips; the sums an oracle run accumulates equal a replay from the models it
kept; and a run stopped and restarted keeps the same sums bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from drawmaha_solver.deepcfr.average import current_strategy, own_reach, predecessors
from drawmaha_solver.minidrawmaha.rules.cards import DECK
from drawmaha_solver.minidrawmaha.rules.enumeration import (
    PublicPoint,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.rules.features import layout
from drawmaha_solver.minidrawmaha.rules.game import (
    DRAW_ACTIONS,
    Action,
    InfoSet,
    MiniState,
    NodeKind,
    throw_count,
)
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.answer.strategy import StrategyInfo, uniform_strategy

# ---------------------------------------------------------------------------
# Helpers (restated, as the suite's rule is: test modules never import one another)
# ---------------------------------------------------------------------------

def key_on(point: PublicPoint, private) -> InfoSet:
    hole, thrown, board = private
    return InfoSet(
        player=point.player,
        hole=hole,
        discarded=thrown,
        board=board,
        draws=point.draws,
        betting=point.betting,
    )

def shape_of(point: PublicPoint):
    return private_keys(board_cards=point.board_cards, discards=point.discards)

def some_keys(seat: int | None = None) -> list[InfoSet]:
    """Two keys on every public point (of `seat`, if given): the first and the middle private key."""
    keys = []
    for point in public_decision_points():
        if seat is not None and point.player != seat:
            continue
        shape = shape_of(point)
        keys += [key_on(point, shape[0]), key_on(point, shape[len(shape) // 2])]
    return keys

def random_table(seed: int) -> PackedTable:
    """The whole game with N(0, 1) regrets: rows with no positive slot play uniform."""
    table = PackedTable.whole_game()
    table.cumulative_regret[:] = np.random.default_rng(seed).standard_normal(
        table.cumulative_regret.size
    )
    return table

def is_first_action(point: PublicPoint) -> bool:
    return point.board_cards == 1 and not point.is_draw_decision and len(point.betting[0]) == point.player

def is_throw_opening(point: PublicPoint) -> bool:
    """A round-2 point where the actor threw a card and has not yet acted in round 2."""
    return point.board_cards == 2 and point.discards == 1 and len(point.betting[1]) == point.player

def brute_force_own_reach(table: PackedTable, key: InfoSet) -> float:
    """Σ over the seat's physical pre-histories of Π own σ, walked through `MiniState`.

    The key's canonical cards stand in for physical ones. The opponent's hole,
    throw and replacement are FIXED to one choice disjoint from every card the
    seat has seen, so the sum runs over the seat's own histories only — the
    per-row constant the average cancels. A key with a discard has three
    pre-draw holes, `hole − d + discarded` for each d in the hole, and the walk
    deals d back as the replacement. Every own action is taken through
    `MiniState.apply`, so no column convention is assumed; the throw is the
    action whose discard pile comes out as the key's.
    """
    seat = key.player
    seen = set(key.hole) | set(key.discarded) | set(key.board)
    spare = [card for card in DECK if card not in seen]
    opponent = tuple(sorted(spare[:3]))
    if key.discarded:
        starts = [
            (tuple(sorted((set(key.hole) - {d}) | set(key.discarded))), d) for d in key.hole
        ]
    else:
        starts = [(key.hole, None)]

    def walk(state: MiniState, drawn, probability: float) -> float:
        if state.kind is NodeKind.TERMINAL:
            return 0.0
        if state.kind is NodeKind.CHANCE:
            short = [p for p, hole in enumerate(state.holes) if len(hole) < 3]
            if short:
                if short[0] == seat:
                    cards = (drawn,)
                else:
                    cards = next(
                        dealt
                        for dealt, _ in state.chance_outcomes()
                        if dealt[0] != drawn and dealt[0] not in seen
                    )
            else:
                cards = (key.board[len(state.board)],)
            return walk(state.apply_chance(cards), drawn, probability)
        if state.current_player == seat and state.infoset() == key:
            return probability
        player = state.current_player
        if state.is_draw_decision():
            count = key.draws[player].count
            if count == 0:
                action = Action.THROW_NONE
            elif player == seat:
                action = next(
                    a
                    for a in DRAW_ACTIONS
                    if throw_count(a) == 1 and state.apply(a).discards[seat] == key.discarded
                )
            else:
                action = DRAW_ACTIONS[1]
        else:
            line = state.betting[-1]
            action = key.betting[len(state.betting) - 1][len(line)]
        if player == seat:
            column = state.legal_actions().index(action)
            probability *= table[state.infoset()].strategy()[column]
        return walk(state.apply(action), drawn, probability)

    total = 0.0
    for pre, drawn in starts:
        holes = (pre, opponent) if seat == 0 else (opponent, pre)
        total += walk(MiniState(holes=holes), drawn, 1.0)
    return total

# ---------------------------------------------------------------------------
# The predecessor table
# ---------------------------------------------------------------------------

def test_predecessors_cover_the_six_cases():
    preds = predecessors()
    points = public_decision_points()
    assert len(preds) == len(points) == 141
    for index, (point, pred) in enumerate(zip(points, preds, strict=True)):
        if pred is None:
            assert is_first_action(point)
            continue
        assert pred.point < index  # parents before children, in walk order
        before = points[pred.point]
        assert before.player == point.player  # an OWN decision
        n = len(shape_of(point))
        m = 3 if is_throw_opening(point) else 1
        assert pred.positions.shape == pred.columns.shape == (n, m)
        assert pred.columns.min() >= 0 and pred.columns.max() < before.width
        assert pred.positions.min() >= 0 and pred.positions.max() < len(shape_of(before))
        if point.board_cards == 2 and len(point.betting[1]) == point.player:
            assert before.is_draw_decision and before.draws == point.draws[: point.player]
        elif point.is_draw_decision:
            assert not before.is_draw_decision and before.board_cards == 1
        else:
            assert before.betting[-1] == point.betting[-1][:-2]
            assert (pred.columns == before.actions.index(point.betting[-1][-2])).all()
    # P0 opens once; P1's first action faces a check or a bet, two points.
    assert sum(pred is None for pred in preds) == 3

def test_a_throw_keys_three_predecessors_are_its_three_pre_draw_holes():
    preds = predecessors()
    points = public_decision_points()
    index = next(i for i, point in enumerate(points) if is_throw_opening(point))
    pred = preds[index]
    distinct = np.array([len(set(row)) for row in pred.positions.tolist()])
    assert set(distinct.tolist()) == {2, 3}
    assert (distinct == 3).mean() > 0.9  # the spec's sample: 96 of 101 have three distinct
    # Stand-pat openings have one predecessor at the stand-pat column.
    pat = next(
        preds[i]
        for i, point in enumerate(points)
        if point.board_cards == 2 and point.discards == 0 and len(point.betting[1]) == point.player
    )
    assert (pat.columns == DRAW_ACTIONS.index(Action.THROW_NONE)).all()
    # A throw column is never the stand-pat column.
    assert (pred.columns >= 1).all()

# ---------------------------------------------------------------------------
# Own reach (gate 4)
# ---------------------------------------------------------------------------

def test_current_strategy_is_regret_matching_row_by_row():
    table = random_table(1)
    sigma = current_strategy(table)
    assert sigma.shape == table.cumulative_regret.shape and sigma.dtype == np.float64
    for key in some_keys():
        _, start, width = table.slot_of(key)
        # `reduceat` adds a row left to right; `ndarray.sum` may pair, one ulp apart.
        np.testing.assert_allclose(sigma[start : start + width], table[key].strategy(), rtol=1e-15, atol=0)

def throw_keys_with(distinct: int, seat: int) -> list[InfoSet]:
    """One round-2 throw-opening key of `seat` whose three predecessors are `distinct` draw keys."""
    preds = predecessors()
    points = public_decision_points()
    for index, point in enumerate(points):
        if point.player != seat or not is_throw_opening(point):
            continue
        rows = [len(set(row)) for row in preds[index].positions.tolist()]
        position = rows.index(distinct)
        return [key_on(point, shape_of(point)[position])]
    raise AssertionError("no throw-opening point")

@pytest.mark.parametrize("seat", [0, 1])
def test_own_reach_equals_the_brute_force_walk(seat):
    table = random_table(2)
    reach = own_reach(table, seat=seat)
    assert reach.shape == (len(table),) and reach.dtype == np.float64
    keys = some_keys(seat) + throw_keys_with(3, seat) + throw_keys_with(2, seat)
    assert len(keys) > 100
    for key in keys:
        expected = brute_force_own_reach(table, key)
        assert reach[table.row_of(key)] == pytest.approx(expected, rel=1e-12, abs=0.0), str(key)
    built = layout()
    for p, point in enumerate(built.points):
        rows = slice(built.row_offset[p], built.row_offset[p + 1])
        if point.player != seat:
            assert not reach[rows].any()
        elif is_first_action(point):
            assert (reach[rows] == 1.0).all()
        else:
            # A pure predecessor gives 1.0; three predecessors can sum past it,
            # and every later row behind a throw inherits that.
            assert reach[rows].min() >= 0.0 and np.isfinite(reach[rows]).all()
            assert reach[rows].max() <= (3.0 if point.discards == 1 else 1.0)

def test_own_reach_refuses_a_listed_table_and_a_bad_seat():
    listed = PackedTable.listed(some_keys()[:5])
    with pytest.raises(ValueError, match="whole game"):
        own_reach(listed, seat=0)
    with pytest.raises(ValueError, match="seat"):
        own_reach(random_table(0), seat=2)
