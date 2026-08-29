"""Regret matching (Hart & Mas-Colell 2000): the ledger at the heart of CFR.

The entire algorithm:

1. After each round, for every action a, record regret(a) = u(a) − u(played):
   how much better a would have done against the opponent's revealed action.
2. Next round, play each action with probability proportional to its
   accumulated POSITIVE regret (uniform when nothing is positive).
3. The running AVERAGE of the strategies played converges to equilibrium —
   for two-player zero-sum games, a Nash equilibrium. The current strategy
   never converges (it cycles forever); only the average does.

CFR (rung 1) is this same ledger run at every information set of a game tree
at once, with counterfactual values in place of raw payoffs.
"""

from __future__ import annotations

import numpy as np

class RegretMatcher:
    """One regret-matching ledger over a fixed set of actions."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.cumulative_regret = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions)

    def strategy(self) -> np.ndarray:
        """Current strategy: positive regrets normalized; uniform fallback."""
        positive = np.maximum(self.cumulative_regret, 0.0)
        total = positive.sum()
        if total <= 0.0:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return positive / total

    def average_strategy(self) -> np.ndarray:
        """Mean of all strategies played so far — the thing that converges."""
        total = self.strategy_sum.sum()
        if total <= 0.0:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return self.strategy_sum / total

    def update(self, utilities: np.ndarray, played: int) -> None:
        """Record one finished round.

        `utilities[a]` is the payoff action a would have earned against what
        the opponent actually did; `played` is the action we took. The
        strategy sum accumulates the strategy that was in effect for this
        round, so it must be called exactly once per round, before the regret
        ledger shifts the current strategy.
        """
        self.strategy_sum += self.strategy()
        self.cumulative_regret += utilities - utilities[played]
