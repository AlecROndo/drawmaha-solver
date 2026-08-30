"""Regret matching (Hart & Mas-Colell 2000): the ledger at the heart of CFR.

The entire algorithm:

1. After each round, for every action a, record regret(a) = u(a) − ⟨σ, u⟩:
   how much better always-a would have done than the mixed strategy σ we
   actually played, against the opponent's revealed action.
2. Next round, play each action with probability proportional to its
   accumulated POSITIVE regret (uniform when nothing is positive).
3. The running AVERAGE of the strategies played converges to equilibrium —
   for two-player zero-sum games, a Nash equilibrium. The current strategy
   never converges (it cycles forever); only the average does.

CFR (rung 1) is this same ledger run at every information set of a game tree
at once — and the u − ⟨σ, u⟩ form here is exactly its counterfactual regret,
so nothing about the update rule changes when the game grows a tree.
"""

from __future__ import annotations

import numpy as np

class RegretMatcher:
    """One regret-matching ledger over a fixed set of actions."""

    def __init__(self, n_actions: int):
        # A one-action "game" has no regret to ledger; letting it through
        # would silently make every strategy() call return [1.0] and hide a
        # caller bug.
        if n_actions < 2:
            raise ValueError(f"n_actions must be at least 2, got {n_actions}")
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

    def update(self, utilities: np.ndarray) -> None:
        """Record one finished round.

        `utilities[a]` is the payoff action a would have earned against what
        the opponent actually did. Regret is measured against the current
        strategy's expected utility ⟨σ, u⟩ — not against the sampled action —
        which is the counterfactual form CFR uses. The same σ is banked into
        the average, so call this exactly once per round, after acting.
        Sampling the action to actually play is the caller's job.
        """
        current = self.strategy()
        self.strategy_sum += current
        self.cumulative_regret += utilities - current @ utilities
