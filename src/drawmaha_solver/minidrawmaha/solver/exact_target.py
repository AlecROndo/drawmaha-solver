"""The exact target: what the walk's regret sample at a table row is worth on average, computed.

Rung 4 learns every infoset from the regret samples the walks bank there,
`r = u − ⟨σ, u⟩` with one opponent hand and one line of opponent play drawn
per visit (`mccfr.traverse`, `lockstep.walk`). At a rare round-2 row a visit
comes once in thousands of iterations and its sample swings by chips, almost
all of it the opponent's hidden hand. This module computes, for any table
row, the quantity those samples estimate — the regret averaged over every
hand the opponent could hold, weighted by how often the deck and the
opponent's own play bring exactly that hand there — and the probability that
one walk arrives at the row at all. A later stage (the "boost") adds one row
with that regret at the weight the walk's samples would have carried, so a
targeted infoset is learned from an exact number instead of a handful of
noisy ones.

**What is computed.** For a row standing for the key I of seat `seat`:

    regret(I, a) = Σ_{b,h,o} w(b,h,o) · (v(a | b,h,o) − v(I | b,h,o))  /  Σ_{b,h,o} w(b,h,o)
    arrival(I)   = Σ_{b,h,o} w(b,h,o)
    w(b,h,o)     = deal(h) · r_b[o] · compat_{b,h}[o]

where (b, h) runs over every PHYSICAL entry of the key — a board row in the
order it came, and the seat's physical (hole, thrown card) — and o over the
opponent's physical states. `r_b[o]` is the reach of the deck and the
opponent at the row's public point on board b: the opponent's deal weight,
the chance probability of every card dealt along the path, and the
opponent's σ at each of its decisions, pushed through its draw. `compat`
is one where the two seats' cards are disjoint. `deal(h)` is the
traverser's own deal probability — seat 0 carries 1/100,100 per allowed
pair and seat 1 carries 1 (the grader's convention in `_weights`, restated
here) — and at a post-draw state it is the sum over the three pre-draw holes
that could have become it, because the key forgets which card was drawn and
the walk arrives through any of them. `v(a | b,h,o)` is the traverser's
expected chips below action a with both seats at σ and the deck averaged.

**Why the ratio, and why the traverser's own σ never enters the weight.**
The walk enumerates the traverser's own actions, so how often it arrives at
I is the deck's and the opponent's doing alone: `arrival` is exactly the
expected number of visits per walk, and `regret` is the conditional mean of
the banked sample given a visit — `E[r | reach]`. Normalising by Σ w is what
makes the exact row commensurable with the sampled ones: a sampled row's
target is one draw of r, so its expectation is this ratio, and the boost's
row weight, `K · arrival · t`, restores the total weight the K walks would
have put there. Dropping the denominator, or weighting every target 1,
would be a "teleport" that over-weights the rare rows it lands on, which is
the bias the targeted-sampling plan (kept, like the rung-3 plan, outside
the repo on Alec's machine under `~/Desktop/Claude/poker/`) exists to
avoid. The traverser's own σ enters only inside `v(I) = ⟨σ, v(·)⟩`, the
regret's baseline, as it does in the walk. One caveat on "visits": a walk
can bank a row twice (about one traversal in a thousand, through the suit
relabelling), so `arrival` is the expected number of visits per walk, a
probability but for those, and `regret` is the mean banked sample per
visit.

**Why every physical entry is summed.** A key is a suit orbit of one
picture — up to six physical (board, state) pairs, fewer when the picture
has a suit symmetry. In the full game a global suit relabelling maps the
whole game onto itself and σ is read in key space, so the six entries
carry identical numbers and one representative times the orbit size would
do. The module sums them anyway: a restricted deal set (`build(holes=)`,
which PR 4's oracle test and this module's own tests use) is not closed
under relabelling, and there the entries face different opponent ranges
and the sum is load-bearing. The orbit shortcut is an optimisation left for
the day the full game's cost matters, with the sum as its oracle.

**Card removal, applied once and in the same place.** The numerator's
weights and the denominator's are the same `compat` vector, computed by
`_compat` from the two seats' card sets. Below the row nothing further is
masked by hand: a replacement the deck could not deal — the board card, a
card the other seat holds — leads to a terminal where the two seats share a
card and pays zero through the same compat, and the chance node's public
1/|stub| is then exactly the probability of each live replacement, because
the stub's size is public (the walk relies on the same fact). The board-2
deal masks the opponent's states the same way the grader's levels do.

**The frame.** Every vector is indexed by a seat's physical state, 455
holes in deck order × 16 slots (slot 0 for nothing thrown, 1 + c for card
c), 7,280 entries, with the board as a batch axis: 15 rows after the first
card, 210 ORDERED pairs after the second. Canonical keys enter at one place,
a map from (board row, state) to a position in `private_keys(shape)`, used
to gather σ rows; the same map inverted lists each key's physical entries.
That is the grader's frame, re-derived here from `rules/` alone.

**Rules-only, on purpose.** This module imports `rules/` and the packed
table and nothing from `exploitability.py`, which grades what the boost
feeds; a tree, a key map or a showdown table shared between the two would
let one bug certify itself. The tests hold the two descriptions to each
other instead: the back-solved reach equals the grader's carried vector on
every board of every point to 1e-10, and the regret and arrival equal a
world-by-world walk through `MiniState` and the mean of thousands of real
walks.

**Cost, measured on this Mac (2026-11-05, a random table).** The description
is built once per process, 5–7 s — the key maps, the draw maps, the
public tree with parent links, the showdown ranks — and holds no showdown
matrix: shares and compatibility are computed per terminal from the holes'
ranks and card sets, folded over the opponent's 455 holes. `sigma` over the
whole table takes 0.12 s. A round-2 row costs 0.8 ms (1.5 ms when 2,240 rows
are spread over every round-2 point, where the reach memo warms less per
row), and the arrival alone 0.3 ms. A draw row costs 45–85 ms and a round-1
row 0.2–0.6 s, because the value below them expands over the traverser's own
replacements (36 states per throw) and the second board card (14 boards):
the opponent's vector is carried once per board and shared by every hero
state on it, which is what keeps those within a second. The opponent's reach
is memoised per call at every node with one board card out that lies on
the path to a decision (66: the 29 level-1 decision points and 37 chance
nodes, 15 boards each) and at every round-2 opening decision (20: one per
round-1 line and pair of draw counts, 210 boards each), 58 KB a (node,
board) pair, so the memo is at most 66 × 15 + 20 × 210 = 5,190 vectors,
302 MB — a count `tests/minidrawmaha/test_exact_target.py` re-derives
from the tree; a call over 2,240 round-2 rows peaked at about 260 MB. `tests/minidrawmaha/test_exact_target.py` prints the
round-2 figure on every run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import combinations, permutations
from typing import Any

import numpy as np

from drawmaha_solver.minidrawmaha.rules.cards import DECK, Card
from drawmaha_solver.minidrawmaha.rules.enumeration import (
    PublicPoint,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.rules.game import (
    BETTING_ACTIONS,
    DRAW_ACTIONS,
    Action,
    MiniState,
    NodeKind,
    canonical_picture,
    chip_state,
    draw_order,
    throw_count,
)
from drawmaha_solver.minidrawmaha.rules.hands import HOLE_CARDS, inner_score, outer_score
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable

Hole = tuple[Card, ...]
SEATS = (0, 1)

# The widest ledger: the draw's four throws. Regret rows are padded to it.
TARGET_WIDTH = max(len(DRAW_ACTIONS), len(BETTING_ACTIONS))

# ---------------------------------------------------------------------------
# The frame: physical states and board rows
# ---------------------------------------------------------------------------

HOLES: tuple[Hole, ...] = tuple(combinations(DECK, HOLE_CARDS))
SLOTS = 1 + len(DECK)
N_STATES = len(HOLES) * SLOTS

_HOLE_INDEX = {hole: index for index, hole in enumerate(HOLES)}
_BOARD_SETS = tuple(combinations(DECK, 2))
_SET_INDEX = {pair: index for index, pair in enumerate(_BOARD_SETS)}
_LEVEL_ROWS = (1, len(DECK), len(DECK) * (len(DECK) - 1))

def state_index(hole: Hole, thrown: Hole) -> int:
    """The row of `(hole, thrown)` in a seat's vector; `thrown` is empty or one card."""
    slot = 0 if not thrown else 1 + DECK.index(thrown[0])
    return _HOLE_INDEX[tuple(sorted(hole))] * SLOTS + slot

def board_row(board: tuple[Card, ...]) -> int:
    """A board's batch row: deck order for one card, `(first, second)` ordered for two.

    Ordered pairs, 210 rows rather than 105 sets, because round 1 was played
    against the first card alone and the two orders carry different reach
    into round 2; only the showdown, which cannot tell them apart, is per set.
    """
    if len(board) == 1:
        return DECK.index(board[0])
    first, second = DECK.index(board[0]), DECK.index(board[1])
    return first * (len(DECK) - 1) + (second if second < first else second - 1)

def _state_cards() -> tuple[np.ndarray, np.ndarray]:
    """Per state: which deck cards it accounts for (hole and thrown), and whether it is a real state."""
    holds = np.zeros((N_STATES, len(DECK)), dtype=bool)
    valid = np.zeros(N_STATES, dtype=bool)
    for h, hole in enumerate(HOLES):
        for slot in range(SLOTS):
            s = h * SLOTS + slot
            for card in hole:
                holds[s, DECK.index(card)] = True
            if slot == 0:
                valid[s] = True
            elif not holds[s, slot - 1]:
                holds[s, slot - 1] = True
                valid[s] = True
    return holds, valid

# ---------------------------------------------------------------------------
# The description: levels, key maps, draw maps, the public tree, the showdown ranks
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, eq=False)
class _Level:
    """The boards at one board count: `ok[b, s]` is 1 when state s avoids board b; `parent[b]` is the board above b and `children_of[p]` the boards of this level under p."""

    ok: np.ndarray
    parent: np.ndarray
    children_of: np.ndarray

@dataclass(frozen=True, slots=True, eq=False)
class _KeyMap:
    """(board row, state) → position in one shape's `private_keys`, −1 outside its support; and the inverse."""

    index: np.ndarray  # int32 (n_boards, N_STATES)
    n_keys: int
    entry_start: np.ndarray  # int64 (n_keys + 1,): entries of key k are entry_*[start[k]:start[k+1]]
    entry_board: np.ndarray  # int32
    entry_state: np.ndarray  # int32

@dataclass(frozen=True, slots=True, eq=False)
class _DrawMap:
    """One throw as a map from pre-draw holes to post-draw states: `targets[b, k, j]` on board b, hole k, replacement j."""

    source: np.ndarray  # int32 (455,): the pre-draw state of each hole
    targets: np.ndarray  # int32 (15, 455, 12)

@dataclass(frozen=True, slots=True, eq=False)
class _Edge:
    """The actions of a decision that lead to one child, with the draw map of each throw (None for a bet or standing pat)."""

    child: _Node
    actions: tuple[int, ...]
    draws: tuple[_DrawMap | None, ...]

@dataclass(frozen=True, slots=True, eq=False)
class _Node:
    """One public node. `point` is a decision's index among the 141 table points; `deals_board` tells a board deal from a replacement."""

    index: int
    kind: NodeKind
    level: int
    discards: tuple[int, int]
    player: int | None
    actions: tuple[Any, ...]
    point: int | None
    edges: tuple[_Edge, ...]
    child: _Node | None
    prob: float
    deals_board: bool
    fold_winner: int | None
    stake: float
    pot: float
    committed: tuple[float, float]

@dataclass(frozen=True, slots=True, eq=False)
class _Description:
    """Everything that does not depend on the deal set, built once per process."""

    points: tuple[PublicPoint, ...]
    row_offset: np.ndarray
    column_offset: np.ndarray
    width: np.ndarray
    holds: np.ndarray  # bool (N_STATES, 15)
    valid: np.ndarray  # bool (N_STATES,)
    hole_of: np.ndarray  # int32 (N_STATES,)
    hole_holds: np.ndarray  # float64 (455, 15): which cards each hole holds
    levels: tuple[_Level, _Level, _Level]
    set_of_row: np.ndarray  # int32 (210,): level-2 row → board set, for the showdown alone
    keys: dict[tuple[int, int], _KeyMap]
    root: _Node
    parent: dict[int, tuple[_Node, _Edge | None]]
    decision_of_point: tuple[_Node, ...]
    inner: np.ndarray  # int32 (455,): dense rank of each hole's inner hand
    outer: np.ndarray  # int32 (105, 455): dense rank of each hole's outer hand with each board set, 0 where they clash

def _levels(holds: np.ndarray, valid: np.ndarray) -> tuple[_Level, _Level, _Level]:
    n = len(DECK)
    # `parent` and `children_of` both live on the CHILD level: `parent[b]` is
    # the board above b, `children_of[p]` every board of this level under p.
    zero = _Level(
        ok=valid[None, :].astype(np.float64),
        parent=np.zeros(1, np.int32),
        children_of=np.zeros((0, 0), np.int32),
    )
    one = _Level(
        ok=(valid[None, :] & ~holds.T).astype(np.float64),
        parent=np.zeros(n, np.int32),
        children_of=np.arange(n, dtype=np.int32)[None, :],
    )
    ok = np.zeros((n * (n - 1), N_STATES))
    parent = np.zeros(n * (n - 1), np.int32)
    for first in range(n):
        for second in range(n):
            if first != second:
                row = board_row((DECK[first], DECK[second]))
                ok[row] = valid & ~holds[:, first] & ~holds[:, second]
                parent[row] = first
    two = _Level(
        ok=ok,
        parent=parent,
        children_of=np.stack([np.flatnonzero(parent == first) for first in range(n)]).astype(np.int32),
    )
    return zero, one, two

def _key_map(level: int, discards: int) -> _KeyMap:
    """(board row, state) → position in `private_keys(level, discards)`, built per ORDERED board row, and its inverse."""
    keys = private_keys(board_cards=level, discards=discards)
    position = {key: k for k, key in enumerate(keys)}
    boards = [(card,) for card in DECK] if level == 1 else list(permutations(DECK, 2))
    index = np.full((_LEVEL_ROWS[level], N_STATES), -1, np.int32)
    for board in boards:
        b = board_row(board)
        for h, hole in enumerate(HOLES):
            if set(hole) & set(board):
                continue
            if discards == 0:
                index[b, h * SLOTS] = position[canonical_picture(hole, (), board)]
                continue
            for card in DECK:
                if card in hole or card in board:
                    continue
                index[b, h * SLOTS + 1 + DECK.index(card)] = position[
                    canonical_picture(hole, (card,), board)
                ]
    flat = index.ravel()
    order = np.argsort(flat, kind="stable")
    order = order[flat[order] >= 0]
    sorted_keys = flat[order]
    start = np.searchsorted(sorted_keys, np.arange(len(keys) + 1)).astype(np.int64)
    return _KeyMap(
        index=index,
        n_keys=len(keys),
        entry_start=start,
        entry_board=(order // N_STATES).astype(np.int32),
        entry_state=(order % N_STATES).astype(np.int32),
    )

def _draw_maps() -> tuple[_DrawMap, _DrawMap, _DrawMap]:
    """The three one-card throws as maps, one board card out, every replacement listed.

    Which physical card `THROW_LOW` names depends on the board through the
    canonical relabelling, so the thrown card is read off `draw_order`.
    """
    n = len(DECK)
    replacements = n - HOLE_CARDS
    targets = np.zeros((n, len(HOLES), HOLE_CARDS, replacements), np.int32)
    for b, board_card in enumerate(DECK):
        for h, hole in enumerate(HOLES):
            ordered = draw_order(hole=hole, discarded=(), board=(board_card,))
            for k, thrown in enumerate(ordered):
                kept = tuple(card for card in hole if card != thrown)
                spare = [card for card in DECK if card not in hole]
                for j, drawn in enumerate(spare):
                    targets[b, h, k, j] = state_index(kept + (drawn,), (thrown,))
    source = (np.arange(len(HOLES)) * SLOTS).astype(np.int32)
    return tuple(_DrawMap(source=source, targets=targets[:, :, k, :]) for k in range(HOLE_CARDS))

def _public_tree(
    draws: tuple[_DrawMap, ...], points: tuple[PublicPoint, ...]
) -> tuple[_Node, dict[int, tuple[_Node, _Edge | None]], tuple[_Node, ...]]:
    """The public tree with edges and parent links, walked from one representative deal.

    Replacement deals are told apart from board deals (the census merges
    them), and a decision is tied to its table point by public signature.
    """
    by_signature = {
        (point.board_cards, point.draws, point.betting): index for index, point in enumerate(points)
    }
    root_state = MiniState(holes=(DECK[0:HOLE_CARDS], DECK[HOLE_CARDS : 2 * HOLE_CARDS]))
    nodes: dict[tuple, _Node] = {}
    order: list[_Node | None] = []
    parent: dict[int, tuple[_Node, _Edge | None]] = {}
    decision_of_point: dict[int, _Node] = {}

    def visit(state: MiniState) -> _Node:
        signature = (
            state.kind,
            len(state.board),
            state.draws,
            state.betting,
            tuple(len(hole) for hole in state.holes),
        )
        if signature in nodes:
            return nodes[signature]
        index = len(order)
        order.append(None)
        fields: dict[str, Any] = dict(
            index=index,
            kind=state.kind,
            level=len(state.board),
            discards=tuple(len(pile) for pile in state.discards),
            player=None,
            actions=(),
            point=None,
            edges=(),
            child=None,
            prob=0.0,
            deals_board=False,
            fold_winner=None,
            stake=0.0,
            pot=0.0,
            committed=(0.0, 0.0),
        )
        if state.kind is NodeKind.TERMINAL:
            chips = chip_state(state.betting)
            line = state.betting[-1]
            if line and line[-1] is Action.FOLD:
                loser = (len(line) - 1) % 2
                fields.update(fold_winner=1 - loser, stake=float(chips.committed[loser]))
            else:
                fields.update(
                    pot=float(chips.pot),
                    committed=(float(chips.committed[0]), float(chips.committed[1])),
                )
            node = _Node(**fields)
        elif state.kind is NodeKind.CHANCE:
            outcomes = state.chance_outcomes()
            child = visit(state.apply_chance(outcomes[0][0]))
            fields.update(
                child=child,
                prob=outcomes[0][1],
                # A short hole means the deck owes a replacement; otherwise a board card.
                deals_board=all(len(hole) == HOLE_CARDS for hole in state.holes),
            )
            node = _Node(**fields)
            parent[child.index] = (node, None)
        else:
            actions = state.legal_actions()
            grouped: dict[int, tuple[list[int], list[_DrawMap | None]]] = {}
            children: dict[int, _Node] = {}
            for k, action in enumerate(actions):
                child = visit(state.apply(action))
                children[child.index] = child
                columns, maps = grouped.setdefault(child.index, ([], []))
                columns.append(k)
                maps.append(
                    draws[DRAW_ACTIONS.index(action) - 1]
                    if state.is_draw_decision() and throw_count(action) > 0
                    else None
                )
            edges = tuple(
                _Edge(child=children[child_index], actions=tuple(columns), draws=tuple(maps))
                for child_index, (columns, maps) in grouped.items()
            )
            fields.update(
                player=state.current_player,
                actions=actions,
                point=by_signature[(len(state.board), state.draws, state.betting)],
                edges=edges,
            )
            node = _Node(**fields)
            for edge in edges:
                parent[edge.child.index] = (node, edge)
            decision_of_point[node.point] = node
        nodes[signature] = node
        order[index] = node
        return node

    root = visit(root_state)
    if len(decision_of_point) != len(points):
        raise AssertionError("the public tree and the enumerator disagree on the decision points")
    return root, parent, tuple(decision_of_point[p] for p in range(len(points)))

def _dense_rank(scores: Sequence[Any]) -> np.ndarray:
    """Scores as integers with the same order, so a sign of a difference is a comparison."""
    ladder = {score: rank for rank, score in enumerate(sorted(set(scores)))}
    return np.array([ladder[score] for score in scores], dtype=np.int32)

def _showdown_ranks() -> tuple[np.ndarray, np.ndarray]:
    """Per hole, the dense rank of its inner hand, and of its outer hand with each board set (0 where the hole clashes)."""
    inner = _dense_rank([inner_score(hole) for hole in HOLES])
    outer = np.zeros((len(_BOARD_SETS), len(HOLES)), np.int32)
    for s, board in enumerate(_BOARD_SETS):
        live = [h for h, hole in enumerate(HOLES) if not set(hole) & set(board)]
        outer[s, live] = _dense_rank([outer_score(HOLES[h], board) for h in live])
    return inner, outer

@cache
def _description() -> _Description:
    points = public_decision_points()
    sizes = [len(private_keys(board_cards=p.board_cards, discards=p.discards)) for p in points]
    widths = [p.width for p in points]
    holds, valid = _state_cards()
    hole_of = (np.arange(N_STATES) // SLOTS).astype(np.int32)
    set_of_row = np.zeros(_LEVEL_ROWS[2], np.int32)
    for first in DECK:
        for second in DECK:
            if first != second:
                set_of_row[board_row((first, second))] = _SET_INDEX[tuple(sorted((first, second)))]
    root, parent, decision_of_point = _public_tree(_draw_maps(), points)
    inner, outer = _showdown_ranks()
    return _Description(
        points=points,
        row_offset=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        column_offset=np.concatenate(
            [[0], np.cumsum([s * w for s, w in zip(sizes, widths, strict=True)])]
        ).astype(np.int64),
        width=np.asarray(widths, dtype=np.int64),
        holds=holds,
        valid=valid,
        hole_of=hole_of,
        hole_holds=holds[::SLOTS].astype(np.float64),
        levels=_levels(holds, valid),
        set_of_row=set_of_row,
        keys={shape: _key_map(*shape) for shape in ((1, 0), (2, 0), (2, 1))},
        root=root,
        parent=parent,
        decision_of_point=decision_of_point,
        inner=inner,
        outer=outer,
    )

# ---------------------------------------------------------------------------
# The deal set
# ---------------------------------------------------------------------------

def _weights(holes: tuple[Iterable[Hole], Iterable[Hole]] | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-seat deal probability at pre-draw states, zero elsewhere: uniform over the allowed disjoint pairs.

    P(r, h) = weight[0][r] · weight[1][h] · [r ∩ h = ∅]; 1/100,100 in the
    full game, carried by seat 0 with seat 1 at 1 — the walk's `random_deal`
    distribution, and the grader's convention.
    """
    allowed = (
        (set(HOLES), set(HOLES))
        if holes is None
        else (set(map(tuple, map(sorted, holes[0]))), set(map(tuple, map(sorted, holes[1]))))
    )
    for seat, chosen in enumerate(allowed):
        unknown = chosen - set(HOLES)
        if unknown:
            raise ValueError(f"seat {seat} cannot be dealt {sorted(unknown)}")
    pairs = sum(1 for r in allowed[0] for h in allowed[1] if not set(r) & set(h))
    if pairs == 0:
        raise ValueError("no deal gives both seats a hole from their allowed sets")
    out = []
    for seat, chosen in enumerate(allowed):
        weight = np.zeros(N_STATES)
        weight[::SLOTS] = [1.0 if hole in chosen else 0.0 for hole in HOLES]
        if seat == 0:
            weight /= pairs
        out.append(weight)
    return out[0], out[1]

def _own_deal(weight: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """The traverser's own deal factor per state, per shape: the pre-draw weight, or the sum over the three pre-draw holes of a post-draw state."""
    post = np.zeros(N_STATES)
    for h, hole in enumerate(HOLES):
        for slot in range(1, SLOTS):
            thrown = DECK[slot - 1]
            if thrown in hole:
                continue
            post[h * SLOTS + slot] = sum(
                weight[state_index(tuple(c for c in hole if c != drawn) + (thrown,), ())]
                for drawn in hole
            )
    return {(1, 0): weight, (2, 0): weight, (2, 1): post}

# ---------------------------------------------------------------------------
# The exact targets
# ---------------------------------------------------------------------------

_Sigma = Sequence[np.ndarray]

@dataclass(frozen=True, eq=False)
class ExactTargets:
    """Exact regret and arrival for any table row, under a strategy read from the materialised table.

        exact = ExactTargets.build()                 # once per process
        sigma = exact.sigma(table)                   # the walk's σ, per point
        regret, arrival = exact.targets(sigma, seat=0, rows=rows)

    `regret` is `(n, 4)` float64, zero past each row's width; `arrival` is
    `(n,)`, the expected number of visits one walk with traverser `seat`
    pays the row (a probability, but for the rare walk that reaches a row
    twice through two throws).
    `arrivals` is the second number alone, at a fraction of the cost.
    `build(holes=)` restricts what each seat may be dealt, uniformly over
    the pairs that remain, as `exploitability.minidrawmaha_game(holes=)` does
    and as a test's `deal` must then sample; the default is every deal.
    """

    description: _Description
    weight: tuple[np.ndarray, np.ndarray]
    own_deal: tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]

    @classmethod
    def build(cls, holes: tuple[Iterable[Hole], Iterable[Hole]] | None = None) -> ExactTargets:
        """The description (shared, built once) with this deal set's weights. The default is cached."""
        if holes is None:
            return _full_game()
        weights = _weights(holes)
        return cls(_description(), weights, (_own_deal(weights[0]), _own_deal(weights[1])))

    @property
    def points(self) -> tuple[PublicPoint, ...]:
        return self.description.points

    def entries_of(self, shape: tuple[int, int], position: int) -> tuple[np.ndarray, np.ndarray]:
        """The physical (board row, state) entries of key `position` in `shape`'s `private_keys`."""
        keys = self.description.keys[shape]
        span = slice(keys.entry_start[position], keys.entry_start[position + 1])
        return keys.entry_board[span], keys.entry_state[span]

    # -- σ ------------------------------------------------------------------

    def sigma(self, table: PackedTable) -> tuple[np.ndarray, ...]:
        """Regret matching over the whole-game table, one `(n_keys, width)` array per point, float64.

        The positive part of each row over its total, uniform where nothing
        is positive: `RegretMatcher.strategy`, vectorised with one `reduceat`,
        which adds a row left to right where the ledger's `sum` may pair, so
        a row can differ from the walk's by one ulp. Refuses a table that is
        not the whole game in the whole game's order.
        """
        d = self.description
        widths = table.widths()
        if (
            len(table) != d.row_offset[-1]
            or table.cumulative_regret.size != d.column_offset[-1]
            or not np.array_equal(widths, np.repeat(d.width, np.diff(d.row_offset)))
        ):
            raise ValueError(
                "the exact target reads the whole game's layout and was handed a table of "
                f"{len(table):,} rows that is not the whole game in the whole game's order"
            )
        positive = np.maximum(table.cumulative_regret, 0.0)
        starts = np.cumsum(widths) - widths
        totals = np.add.reduceat(positive, starts)
        per_column = np.repeat(totals, widths)
        uniform = np.repeat(1.0 / widths, widths)
        with np.errstate(invalid="ignore", divide="ignore"):
            flat = np.where(per_column > 0.0, positive / per_column, uniform)
        return tuple(
            flat[d.column_offset[p] : d.column_offset[p + 1]].reshape(-1, int(d.width[p]))
            for p in range(len(d.points))
        )

    # -- the targets --------------------------------------------------------

    def targets(
        self, sigma: _Sigma, *, seat: int, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact regret `(n, 4)` and arrival `(n,)` — expected visits per walk — for table `rows`, all of them `seat`'s, under `sigma`.

        A row the walk cannot reach (zero arrival) gets zero regret, not NaN;
        a row of the other seat raises, as does one outside the table.
        """
        rows = self._rows(seat, rows)
        regret = np.zeros((rows.size, TARGET_WIDTH))
        arrival = np.zeros(rows.size)
        memo: dict[tuple[int, int, int], np.ndarray] = {}
        for i, (node, position, boards, states, u, total) in enumerate(
            self._entries(sigma, seat, rows, memo)
        ):
            arrival[i] = total
            if total <= 0.0:
                continue
            values = self._action_values(sigma, seat, node, u, states[:, None], boards)[:, 0, :]
            mix = sigma[node.point][position]
            baseline = values @ mix
            regret[i, : len(mix)] = (values - baseline[:, None]).sum(axis=0) / total
        return regret, arrival

    def arrivals(self, sigma: _Sigma, *, seat: int, rows: np.ndarray) -> np.ndarray:
        """The arrival alone for table `rows`: the reach and the compatibility sum, no value below."""
        rows = self._rows(seat, rows)
        memo: dict[tuple[int, int, int], np.ndarray] = {}
        return np.fromiter(
            (total for *_, total in self._entries(sigma, seat, rows, memo)), dtype=np.float64, count=rows.size
        )

    def reach(self, sigma: _Sigma, *, carried: int, point: int, board: int) -> np.ndarray:
        """Seat `carried`'s reach at `point` on one board, over its physical states: deal weight × chance × its own σ along the path.

        What the grader carries down for the responder's opponent; here the
        row's opponent is `carried = 1 − seat`. Hero's actions never enter.
        """
        if carried not in SEATS:
            raise ValueError(f"carried must be one of {SEATS}, got {carried!r}")
        # A copy: at the root the vector is the deal weight itself.
        return self._reach(sigma, carried, self.description.decision_of_point[point], board, {}).copy()

    def _rows(self, seat: int, rows: np.ndarray) -> np.ndarray:
        if seat not in SEATS:
            raise ValueError(f"seat must be one of {SEATS}, got {seat!r}")
        d = self.description
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError(f"rows must be a 1-D array of table rows, got shape {rows.shape}")
        if rows.size and (rows.min() < 0 or rows.max() >= d.row_offset[-1]):
            raise IndexError(f"a row is outside the table's {d.row_offset[-1]:,} rows")
        return rows

    def _entries(self, sigma: _Sigma, seat: int, rows: np.ndarray, memo: dict):
        """Per row: its node and position, its physical entries, the weighted opponent vector at each, and the arrival."""
        d = self.description
        own = self.own_deal[seat]
        for row in rows.tolist():
            p = int(np.searchsorted(d.row_offset, row, side="right") - 1)
            position = row - int(d.row_offset[p])
            node = d.decision_of_point[p]
            if node.player != seat:
                raise ValueError(f"row {row} belongs to seat {node.player}, not seat {seat}")
            shape = (node.level, node.discards[seat])
            boards, states = self.entries_of(shape, position)
            reach = np.stack([self._reach(sigma, 1 - seat, node, int(b), memo) for b in boards])
            u = own[shape][states][:, None] * reach
            compat, _ = self._showdown(node.level, boards, states[:, None], u, shares=False)
            yield node, position, boards, states, u, float(compat.sum())

    # -- the back-solve ----------------------------------------------------

    def _reach(self, sigma: _Sigma, carried: int, node: _Node, b: int, memo: dict) -> np.ndarray:
        d = self.description
        key = (carried, node.index, b)
        if key in memo:
            return memo[key]
        if node is d.root:
            return self.weight[carried]
        above, edge = d.parent[node.index]
        if above.kind is NodeKind.CHANCE:
            level = d.levels[node.level]
            if above.deals_board:
                base = self._reach(sigma, carried, above, int(level.parent[b]), memo)
            else:
                base = self._reach(sigma, carried, above, b, memo)
            out = above.prob * base * level.ok[b]
        elif above.player != carried:
            out = self._reach(sigma, carried, above, b, memo)
        else:
            base = self._reach(sigma, carried, above, b, memo)
            index = d.keys[(above.level, above.discards[carried])].index[b]
            valid = index >= 0
            safe = np.where(valid, index, 0)
            table = sigma[above.point]
            out = np.zeros(N_STATES)
            for column, draw in zip(edge.actions, edge.draws, strict=True):
                played = base * np.where(valid, table[safe, column], 0.0)
                if draw is None:
                    out += played
                else:
                    out += _push(played[None, :], draw, np.array([b]))[0]
        # Memoised through the board-2 deal: the 66 level-1 nodes on the
        # paths (decisions and chance nodes) have 15 boards each and the 20
        # round-2 opening decisions 210 each; the rest of round 2 is at most
        # two opponent steps away and is recomputed, so a call's memo is at
        # most 5,190 vectors whatever rows it is handed.
        if node.level <= 1 or (above.kind is NodeKind.CHANCE and above.deals_board):
            memo[key] = out
        return out

    # -- the value below ----------------------------------------------------
    #
    # The recursion carries, per board b_i: the opponent's weighted vector
    # u_i (over its 7,280 states) and a SET of hero states h_i (the same
    # count on every board). Hero's own moves never touch u; a throw widens
    # the set twelvefold per throw, the board-2 deal widens the boards
    # fourteenfold, and values come back as one number per (board, hero
    # state). Keeping u per board rather than per hero state is what makes a
    # round-1 row affordable: the opponent's σ gathers and pushes below are
    # done once per board, not once per hero continuation.

    def _action_values(
        self, sigma: _Sigma, hero: int, node: _Node, u: np.ndarray, h: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Hero's weighted value below each action of `node`: `(nb, nh, width)` for boards b, hero states h (nb, nh), opponent vectors u (nb, S)."""
        d = self.description
        nb, nh = h.shape
        values = np.zeros((nb, nh, len(node.actions)))
        for edge in node.edges:
            if all(draw is None for draw in edge.draws):
                below = self._value(sigma, hero, edge.child, u, h, b)
                for column in edge.actions:
                    values[..., column] = below
                continue
            # The throw edge: the three throws' replacement states, stacked,
            # walk the one subtree below in one pass and are split after.
            hole = d.hole_of[h]
            fan = edge.draws[0].targets.shape[-1]
            stacked = np.concatenate(
                [draw.targets[b[:, None], hole] for draw in edge.draws], axis=-1
            )  # (nb, nh, 3 × 12)
            below = self._value(sigma, hero, edge.child, u, stacked.reshape(nb, -1), b)
            below = below.reshape(nb, nh, len(edge.draws), fan).sum(axis=-1)
            for j, column in enumerate(edge.actions):
                values[..., column] = below[..., j]
        return values

    def _value(
        self, sigma: _Sigma, hero: int, node: _Node, u: np.ndarray, h: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Hero's weighted expected chips from `node`, `(nb, nh)`, both seats at σ, the deck averaged."""
        d = self.description
        nb, nh = h.shape
        if node.kind is NodeKind.TERMINAL:
            compat, share = self._showdown(node.level, b, h, u, shares=node.fold_winner is None)
            if node.fold_winner is not None:
                sign = 1.0 if node.fold_winner == hero else -1.0
                return sign * node.stake * compat
            return node.pot * share - node.committed[hero] * compat
        if node.kind is NodeKind.CHANCE:
            if node.deals_board:
                level = d.levels[node.child.level]
                kids = level.children_of[b]  # (nb, 14)
                fan = kids.shape[1]
                spread = node.prob * u[:, None, :] * level.ok[kids]
                below = self._value(
                    sigma, hero, node.child,
                    spread.reshape(-1, N_STATES), np.repeat(h, fan, axis=0), kids.ravel(),
                )
                return below.reshape(nb, fan, nh).sum(axis=1)
            return self._value(sigma, hero, node.child, node.prob * u * d.levels[node.level].ok[b], h, b)
        if node.player == hero:
            index = d.keys[(node.level, node.discards[hero])].index[b[:, None], h]  # (nb, nh)
            valid = index >= 0
            mix = np.where(valid[..., None], sigma[node.point][np.where(valid, index, 0)], 0.0)
            return (self._action_values(sigma, hero, node, u, h, b) * mix).sum(axis=-1)
        index = d.keys[(node.level, node.discards[node.player])].index[b]  # (nb, S)
        valid = index >= 0
        safe = np.where(valid, index, 0)
        table = sigma[node.point]
        total = np.zeros((nb, nh))
        for edge in node.edges:
            pushed = np.zeros_like(u)
            for column, draw in zip(edge.actions, edge.draws, strict=True):
                played = u * np.where(valid, table[safe, column], 0.0)
                if draw is None:
                    pushed += played
                else:
                    pushed += _push(played, draw, b)
            total += self._value(sigma, hero, edge.child, pushed, h, b)
        return total

    # -- the showdown and compatibility ------------------------------------

    def _showdown(
        self, level: int, b: np.ndarray, h: np.ndarray, u: np.ndarray, *, shares: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Per (board i, hero state j): Σ_o u_i[o]·compat(h_ij, o) and, with `shares`, the same weighted by hero's pot share.

        Compatibility is the two card sets being disjoint — hero's hole and
        thrown card against the opponent's hole and thrown card — and a hero
        state that holds a card of its own board is no state and scores
        zero. Folded over the opponent's HOLES: u_i summed over the 16 slots
        of each hole gives the hole's total reach, and the slots of the cards
        hero holds are subtracted back out, because an opponent who threw one
        of hero's cards cannot be there. Shares come from the dense ranks of
        both halves: ¼(2 + sign(inner) + sign(outer)).
        """
        d = self.description
        nb, nh = h.shape
        folded = u.reshape(nb, len(HOLES), SLOTS)
        total = folded.sum(axis=-1)  # (nb, 455)
        mine = d.holds[h].astype(np.float64).transpose(0, 2, 1)  # (nb, 15, nh)
        shared = d.hole_holds @ mine  # (nb, 455, nh): cards an opponent hole shares with hero
        excluded = folded[:, :, 1:] @ mine  # (nb, 455, nh): reach of that hole's states that threw a card of hero's
        terms = np.where(shared == 0.0, total[:, :, None] - excluded, 0.0)
        live = (d.valid[h] & (d.levels[level].ok[b[:, None], h] > 0.0)).astype(np.float64)
        compat = terms.sum(axis=1) * live
        if not shares:
            return compat, None
        hole = d.hole_of[h]  # (nb, nh)
        sets = d.set_of_row[b]  # (nb,)
        inner = np.sign(d.inner[hole][:, None, :] - d.inner[None, :, None])
        outer = np.sign(d.outer[sets[:, None], hole][:, None, :] - d.outer[sets][:, :, None])
        share = 0.25 * (2.0 + inner + outer)
        return compat, (terms * share).sum(axis=1) * live

def _push(played: np.ndarray, draw: _DrawMap, b: np.ndarray) -> np.ndarray:
    """Each pair's vector through a throw on its level-1 board: every pre-draw hole spreads over its twelve replacements."""
    m = played.shape[0]
    targets = draw.targets[b]  # (m, 455, 12)
    offsets = (np.arange(m) * N_STATES)[:, None, None]
    weights = np.repeat(played[:, draw.source], targets.shape[2], axis=1).ravel()
    return np.bincount((offsets + targets).ravel(), weights=weights, minlength=m * N_STATES).reshape(
        m, N_STATES
    )

@cache
def _full_game() -> ExactTargets:
    weights = _weights(None)
    return ExactTargets(_description(), weights, (_own_deal(weights[0]), _own_deal(weights[1])))
