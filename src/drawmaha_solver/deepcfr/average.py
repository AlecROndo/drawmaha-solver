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
strategy carries, the textbook `π^q_t(I)`, and the reason is Kuhn's
theorem: the average strategy is the one behavioural strategy that plays
like the uniform mixture of the σ_t, and the share of that mixture that
arrives at I under iteration t is how often q's OWN choices under σ_t lead
there. The opponent's choices are the opponent's, not part of q's strategy,
so they do not enter — the opponent's play does move with t, and weighting
σ_t(I) by it would average q's play by how the opponent happened to be
playing, a different average with no guarantee. The deck's choices are the
same at I whichever t it is, so they would be a per-row constant that
cancels in the normalisation anyway, and the pass leaves them out. Where q
rarely arrives by its own choices, the spot's strategy barely matters and
barely counts.

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

**Why a throw column can be two columns.** The ledger's throw columns name
a card by its place in `draw_order`, which sorts the hole by rank and then
by the CANONICAL suit label of the pre-draw picture. When that picture has
a suit symmetry that swaps the thrown card with an equal-rank twin — a
pair in the hole whose suits the board card does not tell apart — the
label is a tie the relabelling breaks one way in one physical instance of
the pre-draw key and the other way in another. The post-draw key stands
for every physical instance, so its histories split evenly: half arrived
through the thrown card's column, half through the twin's. Reading one
column in the post-draw key's own frame would charge the whole key to one
of them, and at a key where the net plays that column at zero the reach
would read zero where it is not. So each pre-draw hole contributes the
MEAN of σ over the orbit of the thrown card under the pre-draw picture's
stabiliser (a `weights` row beside `positions` and `columns`: 1 for a
lone card, 1/2 per twin, 1/3 for trips), which is the sum over physical
histories up to the per-row constant. The brute force in
`tests/deepcfr/test_average.py` walks the key's picture in every one of
the six suit frames for the same reason.

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

Measured on this Mac (2026-10-21): the predecessor table builds in 1.9 s,
once per process; one `own_reach` pass over the whole game takes 0.10–0.13 s
and one `accumulate` (reach, σ and the add) 0.20 s, against the spec's ~1 s
budget for the merges and this pass together; the sums are 114 MB. After
five oracle iterations the frozen average from float32 sums differs from
the float64 one by at most 6e-8, so precision is not why the sums are
float64 — the bitwise replay and resume pins are. Nearly every row is
reached (6,211,381 of 6,220,050 had positive weight after those five
iterations): own reach is positive wherever every own action on the way
has positive probability, which is almost everywhere, where rung 3's
sampled `strategy_sum` left 2.7 million rows unbanked after 2M iterations.

Graded (K = 1,000, W = 8, 1,000-step refits, seed 0; uniform plays at
4.842): the current iterate swings — 1.391 at t = 5, 0.865 at 10, 0.927 at
20, 0.998 at 40 — while the exact average falls through it, 1.522, 1.089,
0.869, 0.737, and Π follows the average at a distance, 1.512, 1.325, 0.975,
0.849. The average loses at t = 5 and 10 because iteration 1's uniform play
is still in it at weight 1/Σt and seat 1's share lags a refit; from t = 20
it leads, which is where gate 5 is pinned. Rung 3's LCFR average after 2M
iterations graded 0.061 (`figures/rung3/grades.json`). Every row of that
table was re-measured after the tied-column fix below and is unchanged to
three decimals: tied keys are few, and the fix moves only their share. The committed record of rung 4's
grades is PR 6's `figures/rung4/grades.json`; until it lands these are the
gate's own prints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from drawmaha_solver.minidrawmaha.rules.cards import SUIT_RELABELLINGS, relabel
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
    ledger column taken there; `weights[k, j]` that entry's share. One entry
    of weight 1 everywhere but a round-2 point after a throw, which has one
    entry per (pre-draw hole, twin of the thrown card), the three holes'
    shares each summing to 1, zero-weighted padding past the last.
    """

    point: int
    positions: np.ndarray  # int64 (n_keys, m)
    columns: np.ndarray  # int64 (n_keys, m)
    weights: np.ndarray  # float64 (n_keys, m)

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
            weights = np.broadcast_to(1.0, (n, 1))
        elif point.discards == 0:
            positions = _pat_map()
            columns = np.broadcast_to(np.int64(DRAW_ACTIONS.index(Action.THROW_NONE)), positions.shape)
            weights = np.broadcast_to(1.0, positions.shape)
        else:
            positions, columns, weights = _throw_map()
        table.append(Pred(point=index, positions=positions, columns=columns, weights=weights))
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
def _throw_map() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each `(2, 1)` key's draw keys, throw columns and shares; all (n, m), zero-weighted past the last entry.

    Pre-draw hole j is the hole minus its j-th card plus the thrown card.
    A column is a card's place in `draw_order` of that pre-draw picture,
    after the stand-pat column — the same rule `MiniState.apply` throws by.
    The cards charged are the thrown card's orbit under the pre-draw
    picture's stabiliser (the suit relabellings that fix the hole as a set
    and the board card), each at 1/|orbit|: the physical instances of the
    key arrive through each of them in equal number (module docstring).
    Listed with multiplicity: two pre-draw holes that canonicalise to one
    draw key are two histories.
    """
    position = {key: k for k, key in enumerate(private_keys(board_cards=1, discards=0))}
    keys = private_keys(board_cards=2, discards=1)
    pat = DRAW_ACTIONS.index(Action.THROW_NONE)
    entries: list[list[tuple[int, int, float]]] = []
    for hole, thrown, board in keys:
        first = board[:1]
        row = []
        for drawn in hole:
            pre = tuple(sorted(card for card in hole if card != drawn) + list(thrown))
            at = position[canonical_picture(pre, (), first)]
            ordered = draw_order(hole=pre, discarded=(), board=first)
            twins = {
                relabel(frame, thrown)[0]
                for frame in SUIT_RELABELLINGS
                if set(relabel(frame, pre)) == set(pre) and relabel(frame, first) == first
            }
            row += [(at, pat + 1 + ordered.index(twin), 1.0 / len(twins)) for twin in sorted(twins)]
        entries.append(row)
    width = max(len(row) for row in entries)
    positions = np.zeros((len(keys), width), dtype=np.int64)
    columns = np.zeros((len(keys), width), dtype=np.int64)
    weights = np.zeros((len(keys), width))
    for k, row in enumerate(entries):
        for j, (at, column, weight) in enumerate(row):
            positions[k, j], columns[k, j], weights[k, j] = at, column, weight
    return positions, columns, weights

# ---------------------------------------------------------------------------
# Own reach
# ---------------------------------------------------------------------------

def current_strategy(table: PackedTable) -> np.ndarray:
    """σ at every column, float64: regret matching row by row, as `RegretMatcher.strategy` reads it.

    The positive part of each row over its total; a row with no positive
    slot plays uniform. Vectorised over the whole table with one `reduceat`,
    which adds a row left to right where the ledger's `sum` may pair, so a
    row can differ from the ledger's by one ulp — the walks sample the
    ledger's; the average weights this one, and the replay and resume pins
    compare this one with itself.
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
    deck's factor, which the average cancels — this is the textbook
    `π^seat(I)`, the seat's own realization probability;
    `tests/deepcfr/test_average.py` pins it against a walk through
    `MiniState` in every suit frame.
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
        reach[rows] = (before * sigma[columns] * pred.weights).sum(axis=1)
    return reach

def _widths() -> np.ndarray:
    """Every row's width in the whole game's order, from the layout alone (no table built)."""
    built = layout()
    return np.repeat(built.width, np.diff(built.row_offset))

def _require_whole_game(table: PackedTable) -> Layout:
    """The layout, once `table` is the whole game in the whole game's order (materialise's check, restated)."""
    built = layout()
    if (
        len(table) != built.row_offset[-1]
        or table.cumulative_regret.size != built.column_offset[-1]
        or not np.array_equal(table.widths(), _widths())
    ):
        raise ValueError(
            "own reach reads the whole game's layout and was handed a table of "
            f"{len(table):,} rows that is not the whole game in the whole game's order"
        )
    return built

# ---------------------------------------------------------------------------
# The running sum
# ---------------------------------------------------------------------------

# What the file holds. `last` marks a file as an exact-average file: no other
# checkpoint in the repo records a last-accumulated t per seat.
_FILE_KEYS = ("sums", "last")

class ExactAverage:
    """The running `Σ_t t · own_reach_t · σ_t` per column, and its freeze: the spec's `accumulate_exact_average`.

    A PR 4 `Observer`: the loop calls `self(table, t, 1 − p)` right after
    each write, so `accumulate` sees exactly the σ the other seat's walks
    are about to sample — seat 1's from the previous refit at p = 0, seat
    0's freshly refit at p = 1 — and each seat is added once per iteration.
    Only `seat`'s rows move: the other seat's reach is 0.0 and adding 0.0
    leaves its sums bitwise as they were.

    Which refit each term is: seat 0's iteration-t term is its net refit at
    t, seat 1's is its net refit at t − 1 (uniform play at t = 1), because
    that is what the other seat's walks sample at iteration t. Over T
    iterations seat 1's share therefore runs one refit behind seat 0's.

    `last[seat]` is the last t accumulated for that seat, and a call must
    bring `last[seat] + 1`: a resume from an older mark, a skipped phase or
    a double call would weight the wrong σ by the wrong t and the average
    would come out plausible and wrong, so it is refused by name instead.

    The sums are float64, one per table column (114 MB for the whole game),
    the table's own dtype: a resumed run must add the same bits as an
    uninterrupted one, which an accumulation in float32 would not survive.
    `save` and `load` carry both arrays through one file, written atomically.
    """

    name = "exact"

    def __init__(self) -> None:
        built = layout()
        self.sums = np.zeros(int(built.column_offset[-1]))
        self.last = [0, 0]

    def __call__(self, table: PackedTable, t: int, seat: int) -> None:
        self.accumulate(table, t, seat=seat)

    def accumulate(self, table: PackedTable, t: int, *, seat: int) -> None:
        """Add iteration t's `t · own_reach · σ` for `seat` into the sums."""
        if seat not in SEATS:
            raise ValueError(f"seat must be one of {SEATS}, got {seat!r}")
        if t != self.last[seat] + 1:
            raise ValueError(
                f"the exact average for seat {seat} expected t = {self.last[seat] + 1} "
                f"next, got {t!r}: an iteration was skipped, repeated or resumed from "
                "the wrong save"
            )
        reach = own_reach(table, seat=seat)
        self.sums += t * np.repeat(reach, table.widths()) * current_strategy(table)
        self.last[seat] = t

    def frozen(self, *, iteration: int, seed: int) -> FrozenStrategy:
        """The average as a profile the grader reads: each row of the sums normalised, uniform where empty.

        `iteration` and `seed` only label the file.
        """
        widths = _widths()
        info = StrategyInfo(rule="deep", column="exact", iteration=iteration, seed=seed, workers=0)
        return FrozenStrategy(average_rows(self.sums, widths), widths, info)

    # -- the file ------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the sums and `last` to one `.npz`, atomically: a temp file renamed into place."""
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        # Through an open file, so NumPy does not append `.npz` to the name.
        with temporary.open("wb") as file:
            np.savez(file, sums=self.sums, last=np.asarray(self.last, dtype=np.int64))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)

    def load(self, path: Path) -> None:
        """Pour a saved file into this accumulator, replacing what it held.

        Refuses, naming the file and the field, a file without both arrays,
        sums of another length or dtype, and a `last` that is not two
        non-negative ints.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as saved:
            missing = [key for key in _FILE_KEYS if key not in saved.files]
            if missing:
                raise ValueError(f"{path} is not an exact-average file (no {missing[0]!r})")
            sums, last = saved["sums"], saved["last"]
        if sums.shape != self.sums.shape or sums.dtype != self.sums.dtype:
            raise ValueError(
                f"{path}: sums must be {self.sums.dtype.name} {self.sums.shape}, "
                f"got {sums.dtype.name} {sums.shape}"
            )
        if last.shape != (len(SEATS),) or not np.issubdtype(last.dtype, np.integer) or (last < 0).any():
            raise ValueError(f"{path}: last must be {len(SEATS)} non-negative ints, got {last!r}")
        self.sums[...] = sums
        self.last = [int(t) for t in last]
