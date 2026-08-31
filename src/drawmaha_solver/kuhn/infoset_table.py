"""The solver's memory: one regret-matching ledger per Kuhn infoset.

A plain `dict[InfoSet, RegretMatcher]` with twelve entries, pre-allocated so
no code path can invent a thirteenth. This is the whole persistent state of a
Kuhn solve — twelve boxes of four numbers, 48 floats total.

Read the table as a 4x3 grid: one row per decision node (`()`, `(p)`, `(b)`,
`(p, b)`), one column per card the acting player might hold. Moving down a
column is the same hand in a different spot; moving across a row is the same
spot with a different hand, which is exactly what the opponent cannot see and
exactly why the twelve ledgers must stay separate.

Nothing here walks the tree or computes reach probabilities. The table only
stores and reports: the walk in `cfr.py` reads σ from it on the way down and
banks weighted increments into it on the way back up.
"""

from __future__ import annotations

import numpy as np

from drawmaha_solver.kuhn.game import (
    ACTION_SYMBOL,
    CARD_SYMBOL,
    DECISION_HISTORIES,
    DECK,
    N_ACTIONS,
    Action,
    InfoSet,
)
from drawmaha_solver.regret_matching import RegretMatcher

# The walk keys this by `state.infoset()`; every InfoSet that can be
# constructed is one of the twelve, so a lookup can never miss.
InfoSetTable = dict[InfoSet, RegretMatcher]

def new_infoset_table() -> InfoSetTable:
    """A fresh table: twelve independent ledgers, all playing uniformly.

    Pre-allocated rather than filled on demand, so a key the walk should
    never produce raises `KeyError` instead of quietly gaining a ledger.
    """
    return {
        InfoSet(card=card, history=history): RegretMatcher(N_ACTIONS)
        for history in DECISION_HISTORIES
        for card in DECK
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

def format_strategy_grid(strategies: dict[InfoSet, np.ndarray]) -> str:
    """The twelve infosets' P(BET) as the 4x3 grid, for eyeballing a solve.

    One row per decision node, one column per card. BET means "call" on the
    two rows that face a bet, so a converged grid reads as the answer sheet:
    row `-` is the opening bet frequency, row `pb` the calling frequency.
    """
    lines = [" " * 6 + "".join(f"{CARD_SYMBOL[card]:>8}" for card in DECK)]
    for history in DECISION_HISTORIES:
        path = "".join(ACTION_SYMBOL[action] for action in history) or "-"
        cells = "".join(
            f"{strategies[InfoSet(card=card, history=history)][Action.BET]:>8.3f}"
            for card in DECK
        )
        lines.append(f"{path:>6}{cells}")
    return "\n".join(lines)
