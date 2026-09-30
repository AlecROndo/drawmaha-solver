"""The solver's memory: one regret-matching ledger per mini-drawmaha infoset.

Still a mapping from `InfoSet` to `RegretMatcher`, still pre-allocated so no
code path can invent an entry, and still exactly two trained vectors per
ledger — accumulated regret and the banked strategy sum, with both strategies
derived at read time and never stored. What changes from rung 2 is the size,
**6,220,050 entries** against 288, and with it where the numbers live: a
`PackedTable` (`packed_table.py`) keeps them in four flat arrays, about 280 MB
for the whole game, and `table[key]` is a ledger *window* into them rather
than an object of its own. The first version of this rung held a
`RegretMatcher` per key instead — 3.4 GB and half a minute to build, 5.2 GB
once the ledger contract added its two slots — which no parallel run could
afford ten of.

Width is still per key, and this rung adds a third width. A ledger is
`len(infoset.legal_actions())` wide — 2 at an open or a shove, 3 facing a bet,
and **4 at the draw**, where the actions are the four throws. Entry k belongs to
`legal_actions()[k]`, NOT to the `Action` value k: column 0 is CHECK_CALL at the
open, FOLD facing a bet, and THROW_NONE at the draw. A uniform 4-wide table
would bank regret for throwing a card at a betting spot, and that phantom column
could win the normalization. The walk must build its utilities in the same
order; the ledger's shape check catches a wrong-width vector, but nothing
numerical catches a right-width one built in the wrong order.

**Three things do not survive the jump from 288 to 6.2 million, and this file is
mostly the answers to them.**

*Allocating.* `new_infoset_table()` takes the keys it should allocate, defaulting
to the whole game. The whole game is cheap now — the packed store's index is
built from the 141 public points and the cached private keys, never from the
6.2 million `InfoSet`s themselves — but a slice is still what a readout, a unit
test or a subtree experiment wants, because iterating a whole table is still
6.2 million keys. `ledger_widths()` answers what the full table costs without
building any of it, because the width of a ledger is a property of its public
point and the key count of a shape is already known — the table's size is a
141-term sum of products, not a walk.

*Reading strategies out.* `current_strategy` and `average_strategy` return a
live **view**, not a dict: materialising 6.2 million NumPy arrays to answer a
question about a handful of spots costs more than the table it reads. The
semantic distinction is rung 2's and it is the important half — `current_strategy`
is what the walk acts on, it cycles forever and is NOT the solution;
`average_strategy` is the object with the convergence guarantee and the only one
an exploitability meter or a dashboard should ever be shown.

*The readout.* Rung 2 printed all 288 spots as four grids. There is no printing
of 6.2 million, and the grid does not merely grow — it transposes. Leduc's
columns were the player's own rank, which survives canonicalisation because
Leduc's key has no suits; here the relabelling is joint over hole and board, so
two different holes against "the same" board land in different canonical frames
and cannot share a column. So the hand is fixed and the *spots* become the rows:
`format_strategy_tables` answers "holding these cards, what does the solve do
everywhere it could be asked?" — before the draw that is eleven rows for the
seat that draws first and eighteen for the seat that draws second, because P0's
count is public by the time P1 chooses; after the draw it is twenty-eight for
either seat.

Nothing here walks the tree or computes reach probabilities. The table only
stores and reports; `mccfr.py` reads sigma from it on the way down and banks
weighted increments into it on the way back up.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping

import numpy as np

from drawmaha_solver.minidrawmaha.cards import Card, hand_symbol
from drawmaha_solver.minidrawmaha.enumeration import (
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.game import (
    ACTION_SYMBOL,
    DRAW_ACTIONS,
    Action,
    InfoSet,
    action_label,
    canonical_picture,
    line_symbol,
)
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.regret_matching import RegretMatcher

# The walk keys this by `state.infoset()`. Every InfoSet a legal state can
# produce is one the enumerator yields, so a lookup into a fully allocated table
# can never miss — and a lookup that does miss is a walk that reached a position
# the census says does not exist.
InfoSetTable = PackedTable

# ---------------------------------------------------------------------------
# Allocating the table
# ---------------------------------------------------------------------------

def new_infoset_table(
    keys: Iterable[InfoSet] | None = None, *, extra_averages: int = 0
) -> InfoSetTable:
    """A fresh table: one independent ledger per key, all playing uniformly.

    Pre-allocated rather than filled on demand, so a key the walk should never
    produce raises `KeyError` instead of quietly gaining a ledger that nothing
    will ever train. Each ledger is exactly as wide as its spot's legal actions,
    for the reason in the module docstring: a phantom column is regret banked
    for a move the rules do not offer, and it competes in the normalization with
    the moves that are real.

    `keys` defaults to the whole game — 6,220,050 ledgers in four flat arrays,
    about 280 MB, a fraction of a second, and no `InfoSet` built or held:
    `PackedTable.whole_game` computes each key's row from the enumerator's own
    structure. Pass a slice to allocate one public point's worth, or a handful
    of spots, which is what a readout or a test wants; the keys are consumed in
    the order given, so a slice of `all_infosets()` and the full table agree
    entry for entry on the keys they share. Handing the whole enumerator in
    explicitly works but holds every key, which is the object table's cost
    back again — leave `keys` out for the whole game.

    A key arriving twice raises. A dict would keep the second row and throw
    the first away without a word, which is the same failure as two spots
    sharing one ledger — the thing this table exists to make impossible.
    `enumeration.py` refuses the equivalent collapse in `merged()` for the
    same reason, one layer up.

    `extra_averages` gives every ledger that many `extra_sums` rows beside its
    strategy sum — the averaging columns a regret rule banks (see
    `regret_matching`'s ledger contract). Zero unless a run asks.

    A one-wide spot would be refused by `RegretMatcher.over`, which is the
    wanted behaviour rather than an edge case to smooth over: a ledger with one
    legal action has no regret to accumulate, and the betting rules only
    produce one at a stack size this game does not use.
    """
    if keys is None:
        return PackedTable.whole_game(extra_averages=extra_averages)
    return PackedTable.listed(keys, extra_averages=extra_averages)

def ledger_widths() -> dict[int, int]:
    """How many ledgers of each width the full table holds — without allocating one.

    `{2: 4426680, 3: 1773000, 4: 20370}`, which is `census.json`'s
    `by_ledger_width`, and it is reachable without building a ledger because the
    width of one never depends on the cards. `legal_actions()` reads a key's
    betting and its draw signals, both of which are the public half, so every
    key standing on one public point has one width — and the table's whole shape
    is a 141-term sum over (that width, how many private keys of that shape
    exist).

    Costs about a second on a cold process and 0.02 ms after, because the sum
    needs the *sizes* of the four private-key shapes and `private_keys` caches
    them. That is the same second the table itself pays before its first ledger, so asking
    this question first is free against allocating anyway.

    Worth having as a function rather than a comment because it is the size
    question asked *without* a table in hand — a checkpoint's fingerprint, a
    memory budget — and because it is the cheap half of the only claim plan
    §9 makes about this module.
    """
    widths: dict[int, int] = {}
    for point in public_decision_points():
        keys = len(private_keys(board_cards=point.board_cards, discards=point.discards))
        widths[point.width] = widths.get(point.width, 0) + keys
    return dict(sorted(widths.items()))

def spots(
    *,
    player: int,
    hole: tuple[Card, ...],
    board: tuple[Card, ...],
    discarded: tuple[Card, ...] = (),
) -> tuple[InfoSet, ...]:
    """Every decision point this seat can stand on holding these cards.

    The private key is fixed and the public tree is walked for the points whose
    shape it fits: one board card and nothing thrown puts a seat at its round-1
    spots and its draw spots, two board cards at its round-2 spots. Which is
    why the count is asymmetric — P1 draws knowing whether P0 drew, so P1 has
    fourteen draw spots to P0's seven.

    The cards are canonicalised here. A human types physical suits and the table
    is keyed on canonical ones, so doing the relabelling at the call site is a
    step somebody will forget and be answered with `KeyError` at a spot that is
    perfectly well allocated. `board` is read in the order it was dealt: "5d
    then 6c" and "6c then 5d" are two positions with two sets of ledgers.

    A shape no decision point has — a discard while only one board card is out,
    which is a draw that has not happened yet — raises rather than returning
    nothing. An empty readout reads as a solve with no data in it.

    One card in two of the three groups is refused here and nowhere else. This
    is the first place in the rung where a *human* hands in raw cards rather
    than a state machine handing in cards it dealt itself, and neither
    `canonical` nor `InfoSet` checks across groups — so a fat-fingered duplicate
    would relabel into a perfectly ordinary key belonging to a different hand
    and read out somebody else's strategy.
    """
    dealt = hole + discarded + board
    if len(set(dealt)) != len(dealt):
        raise ValueError(f"one card cannot be in two places: {hand_symbol(dealt)}")
    hole, discarded, board = canonical_picture(hole, discarded, board)
    points = [
        point
        for point in public_decision_points()
        if point.player == player
        and point.board_cards == len(board)
        and point.discards == len(discarded)
    ]
    if not points:
        raise ValueError(
            f"no decision point has P{player} holding {len(discarded)} discard(s) "
            f"behind a {len(board)}-card board"
        )
    return tuple(
        InfoSet(
            player=player,
            hole=hole,
            discarded=discarded,
            board=board,
            draws=point.draws,
            betting=point.betting,
        )
        for point in points
    )

# ---------------------------------------------------------------------------
# Reading the strategies out
# ---------------------------------------------------------------------------

class StrategyView(Mapping[InfoSet, np.ndarray]):
    """One strategy per infoset, computed at the moment it is asked for.

    Rung 2 returned a `dict` because building 288 arrays cost nothing. Here the
    same dict is 6.2 million arrays — many times the table it reads — to
    answer a question every caller asks about a handful of spots. So this is a
    view: O(1) to make, backed by the table, and it computes exactly the entries
    somebody looks at.

    The consequence to know is that a view is **live**. Read it after another
    thousand iterations and the numbers have moved, which is what a dashboard
    polling one spot wants and is a trap for anything keeping a checkpoint. A
    caller that needs a snapshot says `dict(view)` and pays for it in the open.
    """

    def __init__(
        self, table: InfoSetTable, read: Callable[[RegretMatcher], np.ndarray]
    ):
        self._table = table
        self._read = read

    def __getitem__(self, infoset: InfoSet) -> np.ndarray:
        return self._read(self._table[infoset])

    def __iter__(self) -> Iterator[InfoSet]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)

def current_strategy(table: InfoSetTable) -> StrategyView:
    """What every infoset would play right now, from its positive regrets.

    This is what the walk acts on and what the reach weights are built from. It
    cycles forever under CFR and is NOT the solution — report `average_strategy`
    instead.
    """
    return StrategyView(table, RegretMatcher.strategy)

def average_strategy(table: InfoSetTable) -> StrategyView:
    """The answer: every infoset's strategy averaged over the whole run.

    This is the object with the convergence guarantee, and the only one an
    exploitability meter or a dashboard should ever be shown.
    """
    return StrategyView(table, RegretMatcher.average_strategy)

# ---------------------------------------------------------------------------
# The readout: one hand, every spot it can stand on
# ---------------------------------------------------------------------------

# The four throws as a cell can afford to print them: "pat", "low", "mid",
# "top". `ACTION_SYMBOL` cannot serve here and is not asked to: it is the
# BETTING shorthand, and a draw never appears inside a betting line, so it holds
# no throw at all. What the table records of a throw is its count, and a count
# cannot separate the three one-card throws — deliberately, since that is the
# whole of what the opponent learns. A readout is for the player, who knows
# which card went, so the cells name the position instead. The words come from
# `action_label` so that the two ways of naming a throw cannot drift apart, and
# a restored two-card cap renders as "low+mid" here without this file knowing
# the cap changed.
_THROW_SYMBOL = {
    action: action_label(action).removeprefix("throw ").removeprefix("stand ")
    for action in DRAW_ACTIONS
}

def _column_symbol(action: Action, actions: tuple[Action, ...]) -> str:
    """One action's letter inside a cell, in the context of its legal set.

    Two of the three branches exist because `ACTION_SYMBOL` covers only FOLD and
    POT. CHECK_CALL is absent from it because it is a check or a call depending
    on whether anything is owed, and the legal set answers that without needing
    the line: fold is offered exactly when there is something to fold against.
    The throws are absent because a throw never appears inside a betting line,
    which is the only thing that shorthand spells.
    """
    if action is Action.CHECK_CALL:
        return "c" if Action.FOLD in actions else "x"
    if action in _THROW_SYMBOL:
        return _THROW_SYMBOL[action]
    return ACTION_SYMBOL[action]

def _cell(infoset: InfoSet, probabilities: np.ndarray) -> str:
    """One spot's whole mixed strategy, e.g. "f0.10 c0.52 p0.38".

    Spelled out per legal action because no single-column convention survives
    rows whose legal sets differ in size AND in kind — a bare number would be
    ambiguous between a bet, a call and a throw.
    """
    actions = infoset.legal_actions()
    return " ".join(
        f"{_column_symbol(action, actions)}{p:.2f}"
        for action, p in zip(actions, probabilities, strict=True)
    )

def _row_label(infoset: InfoSet) -> str:
    """The public history that reached this spot: "xx/t0t1|p".

    Round 1, then both draw signals, then round 2 — the same order they
    happened in, and the same shorthand `line_symbol` writes the plan's line
    tables in. An empty line renders as `-`, so the round-2 open has a row label
    rather than a blank one.
    """
    label = line_symbol(infoset.betting[0]) or "-"
    if infoset.draws:
        label += "/" + "".join(f"t{signal.count}" for signal in infoset.draws)
    if len(infoset.betting) > 1:
        label += "|" + (line_symbol(infoset.betting[1]) or "-")
    return label

def _stage(infoset: InfoSet) -> str:
    return (
        "the draw"
        if infoset.is_draw_decision()
        else ("round 1" if len(infoset.betting) == 1 else "round 2")
    )

def format_strategy_tables(
    strategies: Mapping[InfoSet, np.ndarray],
    *,
    player: int,
    hole: tuple[Card, ...],
    board: tuple[Card, ...],
    discarded: tuple[Card, ...] = (),
) -> str:
    """One hand's whole strategy, as a block per stage, for eyeballing a solve.

    The question a human actually has in front of 6.2 million ledgers is not
    "show me the table" — it is "I hold these three cards on this board: what
    does the solve do?". So the hand is the argument and the rows are the spots
    it can be asked at: before the draw, eleven rows for P0 (four betting spots
    and seven draws) and eighteen for P1 (four and fourteen, since P0's draw
    count is public by then); after the draw, twenty-eight for either seat. Each
    cell spells the whole mix in legal-action order.

    Reading a converged solve: the `-` row of round 1 is the opening play with
    this hand, the draw block says which of the three cards it throws, and a
    hand that made a flush with the second board card should turn aggressive
    across the round-2 block.

    `strategies` is looked up per row, so a table allocated at only some of
    these spots raises `KeyError` on the first one it is missing rather than
    printing a hole.
    """
    keys = spots(player=player, hole=hole, board=board, discarded=discarded)
    # The heading reads its cards off the first key rather than relabelling the
    # arguments a second time, so it cannot name a hand the rows are not about.
    shown = keys[0]
    title = f"P{player} {hand_symbol(shown.hole)}"
    if shown.discarded:
        title += f" /{hand_symbol(shown.discarded)}"
    lines = [f"{title} on {hand_symbol(shown.board)}"]
    for stage in ("round 1", "the draw", "round 2"):
        rows = [key for key in keys if _stage(key) == stage]
        if not rows:
            continue
        width = max(len(_row_label(key)) for key in rows) + 2
        lines.append(stage)
        lines += [
            f"  {_row_label(key):<{width}}{_cell(key, strategies[key])}" for key in rows
        ]
    return "\n".join(lines)
