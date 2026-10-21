"""The exact average: own reach through a predecessor table, and the running sum it weights.

In CFR the strategy played at iteration t never settles; only the average
over iterations converges. Rung 3 got the average for free — every ledger
keeps a `strategy_sum`, banked by the walk at the opponent's visits — and
Deep CFR throws the table away every iteration, so nothing holds that sum
any more. This module rebuilds it exactly (Single Deep CFR): the loop
already materialises each iteration's whole strategy into the table before
each phase, and PR 4's observer hook hands it over right then. While it sits
there, every column of seat q's rows is added into a running sum,

    sums(I, a) += t · own_reach_t(I) · σ_t(I, a)

and the average at I is `sums(I, ·)` divided by its row total, uniform where
that is zero. The spec writes a denominator `Σ_t t · own_reach_t(I)` beside
it; that is the row total exactly, because every σ row sums to one, so it is
not stored and `strategy.average_rows` normalises, as it does for rung 3's
`strategy_sum`. One pass per seat per iteration, no replay, no fitting error.

**What own reach is, and why it is the weight.** Own reach of seat q at key
I under σ is the sum, over the physical histories the key stands for, of the
product of q's OWN action probabilities on the way there — the deck's and
the opponent's probabilities left out. It is the weight CFR's average
strategy carries (the textbook `π^q_t(I)`), and the reason is what it does
not contain: chance and the opponent's play are the same at I whichever t
it is, so they are a per-row constant that cancels when the row is
normalised, and only q's own play, which moves with t, decides how much
iteration t's σ(I) counts. Where q rarely arrives by its own choices, the
spot's strategy barely matters and barely counts.

**Why a post-draw key has three predecessors.** The key keeps the hole, the
card thrown and the board, and deliberately forgets which card was drawn
(`game.InfoSet`: merging those is lossless for the continuation). So three
pre-draw holes lead to one post-draw key — the hole minus each of its three
cards, plus the thrown card — and each arrived through its own draw key at
the throw column `draw_order` gives the thrown card there. All three count:
own reach at the post-draw key is the sum of the three draw keys' reach
times the throw probability at each. The deck's replacement probability is
the same for the three (one card from the same stub) and constant over t,
so it cancels. When two of the three pre-draw holes canonicalise to one draw
key, that key is listed twice: it still stands for two physical histories.

**The predecessor table.** Every key has an earlier own decision, or none:

    key                                   previous own decision                                rows
    round 1, first own action             none (reach 1)                                         0
    round 1, later                        same hole and board card, the line two actions shorter 1
    the draw                              the seat's last round-1 decision                       1
    round 2, stood pat                    the draw key (hole, board[:1]), stand-pat column       1
    round 2, threw a card                 the draw key of each pre-draw hole, its throw column   3
    round 2, later in the line            same key, the line two actions shorter                 1

The public half of a key is one of the 141 public points, so the table is
per point: the predecessor POINT (found by its public signature), and for
the point's keys a `(positions, columns)` map into the predecessor point's
shape of `private_keys`. Only two transitions change shape and need a real
map, `(2 board cards, 0 discards) → (1, 0)` and `(2, 1) → (1, 0)`, each
built once from `canonical_picture` and `draw_order`; every other
predecessor is the same private key at the same position. The 141 points
come in walk order (`enumeration.public_tree` is a depth-first walk), so a
predecessor always precedes its successors and the pass is 141 gathers in
that order, vectorised over the point's keys — the idea of the grader's
draw map (`exploitability.DrawMap`, one push for the three throws), moved
from its physical frame into key space, which is all the average needs.

**The whole game only.** The pass reads rows and columns at `features.layout()`'s
offsets, the same arithmetic `materialise` writes by, and refuses any other
table by the same fingerprint. A `listed` table's rows are in whatever
order the caller gave.

Measured on this Mac (2026-10-21): see `ExactAverage`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from drawmaha_solver.minidrawmaha.rules.enumeration import PublicPoint, private_keys
from drawmaha_solver.minidrawmaha.rules.features import Layout, layout
from drawmaha_solver.minidrawmaha.rules.game import (
    DRAW_ACTIONS,
    Action,
    canonical_picture,
    draw_order,
)
from drawmaha_solver.minidrawmaha.solver.lockstep import SEATS
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.answer.strategy import (
    FrozenStrategy,
    StrategyInfo,
    average_rows,
)

# ---------------------------------------------------------------------------
# The predecessor table
# ---------------------------------------------------------------------------

# eq=False: the fields are arrays; there is one table per process.
@dataclass(frozen=True, slots=True, eq=False)
class Pred:
    """Where one public point's keys came from: an earlier own point, and a position and column per key.

    `positions[k, j]` is the j-th predecessor of this point's k-th key as a
    position in `point`'s shape of `private_keys`; `columns[k, j]` the
    ledger column taken there. One column (j = 0) everywhere but a round-2
    point after a throw, which has three.
    """

    point: int
    positions: np.ndarray  # int64 (n_keys, m)
    columns: np.ndarray  # int64 (n_keys, m)

@cache
def predecessors() -> tuple[Pred | None, ...]:
    """One entry per public point in walk order; None at a seat's first own action."""
    built = layout()
    points = built.points
    by_signature = {
        (point.board_cards, point.draws, point.betting): index
        for index, point in enumerate(points)
    }
    table: list[Pred | None] = []
    for point in points:
        signature = _previous_own_decision(point)
        if signature is None:
            table.append(None)
            continue
        before, action = signature
        index = by_signature[before]
        shape = (point.board_cards, point.discards)
        if shape == (points[index].board_cards, points[index].discards):
            n = len(private_keys(board_cards=shape[0], discards=shape[1]))
            positions = np.arange(n, dtype=np.int64)[:, None]
            columns = np.broadcast_to(np.int64(points[index].actions.index(action)), (n, 1))
        elif point.discards == 0:
            positions = _pat_map()
            columns = np.broadcast_to(np.int64(DRAW_ACTIONS.index(Action.THROW_NONE)), positions.shape)
        else:
            positions, columns = _throw_map()
        table.append(Pred(point=index, positions=positions, columns=columns))
    return tuple(table)

def _previous_own_decision(point: PublicPoint) -> tuple[tuple, Action | None] | None:
    """The public signature `(board_cards, draws, betting)` of the actor's previous decision, and the action taken there.

    None at a seat's first own action. The action is None for the two
    draw-to-round-2 cases, where it depends on the key (stand pat, or which
    card was thrown).
    """
    player = point.player
    line = point.betting[-1]
    if point.is_draw_decision:
        # The seat's last round-1 action: the latest index of its parity.
        last = max(i for i in range(len(line)) if i % 2 == player)
        return (1, (), (line[:last],)), line[last]
    if point.board_cards == 1:
        if len(line) == player:
            return None
        return (1, (), (line[:-2],)), line[-2]
    if len(line) == player:
        # Round 2 opens for this seat right after its own draw.
        return (1, point.draws[:player], (point.betting[0],)), None
    return (2, point.draws, (point.betting[0], line[:-2])), line[-2]

@cache
def _pat_map() -> np.ndarray:
    """Each `(2, 0)` key's draw key, as a position in the `(1, 0)` shape; (n, 1)."""
    position = {key: k for k, key in enumerate(private_keys(board_cards=1, discards=0))}
    keys = private_keys(board_cards=2, discards=0)
    out = np.empty((len(keys), 1), dtype=np.int64)
    for k, (hole, _, board) in enumerate(keys):
        out[k, 0] = position[canonical_picture(hole, (), board[:1])]
    return out

@cache
def _throw_map() -> tuple[np.ndarray, np.ndarray]:
    """Each `(2, 1)` key's three draw keys and throw columns; both (n, 3).

    Pre-draw hole j is the hole minus its j-th card plus the thrown card.
    The column is the thrown card's place in `draw_order` of that pre-draw
    picture, after the stand-pat column — the same rule `MiniState.apply`
    throws by, so a key's three entries are exactly the three actions that
    could have produced it. Listed with multiplicity: two pre-draw holes
    that canonicalise to one draw key are two histories.
    """
    position = {key: k for k, key in enumerate(private_keys(board_cards=1, discards=0))}
    keys = private_keys(board_cards=2, discards=1)
    positions = np.empty((len(keys), 3), dtype=np.int64)
    columns = np.empty((len(keys), 3), dtype=np.int64)
    pat = DRAW_ACTIONS.index(Action.THROW_NONE)
    for k, (hole, thrown, board) in enumerate(keys):
        first = board[:1]
        for j, drawn in enumerate(hole):
            pre = tuple(sorted(card for card in hole if card != drawn) + list(thrown))
            positions[k, j] = position[canonical_picture(pre, (), first)]
            ordered = draw_order(hole=pre, discarded=(), board=first)
            columns[k, j] = pat + 1 + ordered.index(thrown[0])
    return positions, columns

# ---------------------------------------------------------------------------
# Own reach
# ---------------------------------------------------------------------------

def current_strategy(table: PackedTable) -> np.ndarray:
    """σ at every column, float64: regret matching row by row, as `RegretMatcher.strategy` reads it.

    The positive part of each row over its total; a row with no positive
    slot plays uniform. Vectorised over the whole table with one `reduceat`.
    """
    widths = table.widths()
    positive = np.maximum(table.cumulative_regret, 0.0)
    starts = np.cumsum(widths) - widths
    totals = np.add.reduceat(positive, starts)
    per_column = np.repeat(totals, widths)
    uniform = np.repeat(1.0 / widths, widths)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(per_column > 0.0, positive / per_column, uniform)

def own_reach(table: PackedTable, *, seat: int) -> np.ndarray:
    """Seat `seat`'s own reach at every row under the table's current σ; 0.0 at the other seat's rows.

    1.0 at the seat's first own action, then `Σ_pred reach[pred] · σ[pred,
    column]` point by point in walk order. Up to a per-row constant — the
    chance and opponent factors the average cancels — this is the textbook
    `π^seat(I)`; `tests/deepcfr/test_average.py` pins it against a walk
    through `MiniState`.
    """
    if seat not in SEATS:
        raise ValueError(f"seat must be one of {SEATS}, got {seat!r}")
    built = _require_whole_game(table)
    sigma = current_strategy(table)
    reach = np.zeros(len(table))
    for p, point in enumerate(built.points):
        if point.player != seat:
            continue
        rows = slice(built.row_offset[p], built.row_offset[p + 1])
        pred = predecessors()[p]
        if pred is None:
            reach[rows] = 1.0
            continue
        before = reach[built.row_offset[pred.point] + pred.positions]
        columns = (
            built.column_offset[pred.point]
            + pred.positions * built.width[pred.point]
            + pred.columns
        )
        reach[rows] = (before * sigma[columns]).sum(axis=1)
    return reach

def _require_whole_game(table: PackedTable) -> Layout:
    """The layout, once `table` is the whole game in the whole game's order (materialise's check, restated)."""
    built = layout()
    if (
        len(table) != built.row_offset[-1]
        or table.cumulative_regret.size != built.column_offset[-1]
        or not np.array_equal(table.widths(), np.repeat(built.width, np.diff(built.row_offset)))
    ):
        raise ValueError(
            "own reach reads the whole game's layout and was handed a table of "
            f"{len(table):,} rows that is not the whole game in the whole game's order"
        )
    return built
