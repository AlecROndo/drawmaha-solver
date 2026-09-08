"""The solver's memory: one regret-matching ledger per Leduc infoset.

A plain `dict[InfoSet, RegretMatcher]` with 288 entries, pre-allocated so no
code path can invent a 289th. This is the whole persistent state of a Leduc
solve, and per ledger it is exactly TWO stored vectors — accumulated regret
and the banked strategy sum. The current strategy (positive regrets,
normalized) and the average strategy (the banked sum, normalized) are derived
from those on demand and never stored; normalization happens at read time,
so nothing in the table itself is ever "already averaged".

One thing changes from rung 1, and it is the point of this file: **width is
per key**. Kuhn allocated every ledger `N_ACTIONS` wide because both actions
were always legal. Leduc's spots offer {call, raise}, {fold, call, raise} or
{fold, call}, so each ledger is sized by `len(infoset.legal_actions())` — 192
two-wide and 96 three-wide. Entry k of a ledger belongs to
`legal_actions()[k]`, NOT to the `Action` value k: column 0 means FOLD at a
facing-a-raise spot and CALL at the open. The walk must build its utilities
vector in that same order; the ledger's own shape check catches a wrong-width
vector, but nothing numerical can catch a right-width one built in the wrong
order — which is why the order is pinned in `game.py` and tested there.

The 4x3 grid rung 1 printed does not survive 288 spots of varying width, so
the readout becomes per-round tables: one for round 1, one per board rank for
round 2, each cell spelling out the whole mixed strategy in legal-action
order rather than a single P(BET) column that no longer has one meaning.

Nothing here walks the tree or computes reach probabilities. The table only
stores and reports: the walk in `cfr.py` reads sigma from it on the way down
and banks weighted increments into it on the way back up.
"""

from __future__ import annotations

import numpy as np

from drawmaha_solver.leduc.game import (
    ACTION_SYMBOL,
    RANK_SYMBOL,
    RANKS,
    ROUND_CLOSING_LINES,
    ROUND_DECISION_LINES,
    Action,
    InfoSet,
    Rank,
    all_infosets,
)
from drawmaha_solver.regret_matching import RegretMatcher

# The walk keys this by `state.infoset()`; every InfoSet that can be
# constructed is one of the 288, so a lookup can never miss.
InfoSetTable = dict[InfoSet, RegretMatcher]

def new_infoset_table() -> InfoSetTable:
    """A fresh table: 288 independent ledgers, all playing uniformly.

    Pre-allocated rather than filled on demand, so a key the walk should
    never produce raises `KeyError` instead of quietly gaining a ledger.
    Each ledger is exactly as wide as its spot's legal actions — a uniform
    3-wide table would bank regret for folding at spots where folding is
    not a move, and that phantom column could win the normalization.
    """
    return {
        infoset: RegretMatcher(len(infoset.legal_actions()))
        for infoset in all_infosets()
    }

def current_strategy(table: InfoSetTable) -> dict[InfoSet, np.ndarray]:
    """What every infoset would play right now, from its positive regrets.

    This is what the walk acts on and what the reach weights are built from.
    It cycles forever under vanilla CFR and is NOT the solution — report
    `average_strategy` instead.
    """
    return {infoset: ledger.strategy() for infoset, ledger in table.items()}

def average_strategy(table: InfoSetTable) -> dict[InfoSet, np.ndarray]:
    """The answer: every infoset's strategy averaged over the whole run.

    This is the object with the convergence guarantee, and the only one an
    exploitability meter or a dashboard should ever be shown.
    """
    return {infoset: ledger.average_strategy() for infoset, ledger in table.items()}

# ---------------------------------------------------------------------------
# The readout tables
# ---------------------------------------------------------------------------

# Wide enough for a three-action cell ("f0.33 c0.33 r0.33" is 17 characters)
# plus a two-space gutter; the row label fits "crrc|crr", the longest key.
_CELL_WIDTH = 19
_LABEL_WIDTH = 10

def _line_label(line: tuple[Action, ...]) -> str:
    return "".join(ACTION_SYMBOL[a] for a in line) or "-"

def _cell(infoset: InfoSet, probabilities: np.ndarray) -> str:
    """One spot's whole mixed strategy, e.g. "f0.10 c0.52 r0.38".

    Spelled out per legal action because no single-column convention
    survives rows whose legal sets differ — rung 1 could print P(BET) and
    let the row explain itself; here a probability without its action letter
    is ambiguous between three different column meanings.
    """
    return " ".join(
        f"{ACTION_SYMBOL[action]}{p:.2f}"
        for action, p in zip(infoset.legal_actions(), probabilities, strict=True)
    )

def _table(
    title: str,
    row_keys: list[tuple[str, tuple[tuple[Action, ...], ...]]],
    board: Rank | None,
    strategies: dict[InfoSet, np.ndarray],
) -> list[str]:
    """One rendered block: a title, a rank header, one row per betting key."""
    lines = [title, " " * _LABEL_WIDTH + "".join(f"{RANK_SYMBOL[r]:<{_CELL_WIDTH}}" for r in RANKS)]
    for label, betting in row_keys:
        cells = "".join(
            f"{_cell(spot, strategies[spot]):<{_CELL_WIDTH}}"
            for rank in RANKS
            for spot in [InfoSet(rank=rank, board=board, betting=betting)]
        )
        lines.append(f"{label:<{_LABEL_WIDTH}}{cells}")
    return lines

def format_strategy_tables(strategies: dict[InfoSet, np.ndarray]) -> str:
    """The 288 spots as four tables, for eyeballing a solve.

    Round 1 is one 6x3 table — a row per betting line, a column per own
    rank, exactly rung 1's grid grown by two rows. Round 2 is one table per
    board rank, its rows keyed "round-1 line | round-2 line" (perfect
    recall: two routes to the same board are different spots). Reading a
    converged solve: the round-1 `-` row is the opening play, and on the
    board-J table the J column should turn aggressive — those spots hold
    the pair.
    """
    blocks = [
        _table(
            "round 1",
            [(_line_label(line), (line,)) for line in ROUND_DECISION_LINES],
            None,
            strategies,
        )
    ]
    for board in RANKS:
        rows = [
            (f"{_line_label(opening)}|{_line_label(line)}", (opening, line))
            for opening in ROUND_CLOSING_LINES
            for line in ROUND_DECISION_LINES
        ]
        blocks.append(
            _table(f"round 2, board {RANK_SYMBOL[board]}", rows, board, strategies)
        )
    return "\n\n".join("\n".join(block) for block in blocks)
