"""Regret matching (Hart & Mas-Colell 2000): the ledger at the heart of CFR.

Shared by every rung of the validation ladder — the learning rule does not
change as the games grow, only what feeds it does.

The entire algorithm:

1. After each visit, for every action a, record regret(a) = u(a) − ⟨σ, u⟩:
   how much better always-a would have done than the mixed strategy σ we
   actually played.
2. Next visit, play each action with probability proportional to its
   accumulated POSITIVE regret (uniform when nothing is positive).
3. The running AVERAGE of the strategies played converges to equilibrium —
   for two-player zero-sum games, a Nash equilibrium. The current strategy
   never converges (it cycles forever); only the average does.

Two weights make the same ledger serve a game tree. At rung 0 every round
counted the same, so both defaulted to 1. In an extensive-form game a visit
is only reached sometimes, and the two things `update` records want
different scales: regret is weighted by the COUNTERFACTUAL reach π₋ᵢ (chance
and the opponent, deliberately excluding your own choices, so a line you
currently avoid keeps learning at full strength), while the average is
weighted by your OWN reach πᵢ (because it must reproduce what you really
played). Passing them swapped is the classic CFR bug and both versions run
without complaint — hence `regret_weight` and `strategy_weight` are
keyword-only, so a call site cannot silently transpose them.

This module knows nothing about cards, trees, or reach probabilities.
Computing the two weights is the caller's job.
"""

from __future__ import annotations

import math

import numpy as np

class RegretMatcher:
    """One regret-matching ledger over a fixed set of actions.

    Holds exactly two persistent vectors — accumulated regret and the banked
    strategy sum. The current and average strategies are derived from them on
    demand and never stored.
    """

    def __init__(self, n_actions: int):
        # Caught here rather than left to np.zeros, which rejects a float with
        # "expected a sequence of integers" — a message that points at the
        # array shape instead of at the action count the caller got wrong.
        if not isinstance(n_actions, (int, np.integer)):
            raise ValueError(f"n_actions must be an int, got {n_actions!r}")
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

    def update(
        self,
        utilities: np.ndarray,
        *,
        regret_weight: float = 1.0,
        strategy_weight: float = 1.0,
    ) -> None:
        """Record one visit, banking both accumulators.

        `utilities[a]` is what action a would have earned — at rung 0, read
        from a payoff-matrix column; at rung 1 and beyond, computed by a walk
        of the subtree. Regret is measured against the current strategy's
        expected utility ⟨σ, u⟩ rather than against a sampled action, which
        is the counterfactual form CFR uses at every infoset.

            R += regret_weight   · (u − ⟨σ, u⟩)
            S += strategy_weight · σ

        Both weights default to 1, which is the rung-0 behavior. Weights of
        zero are legal and mean "this visit was unreachable, record nothing".
        Sampling an action to actually play is the caller's job, as is
        computing the two weights.

        Validates before mutating: a rejected call leaves the ledger exactly
        as it was, rather than half-advanced and permanently wrong.
        """
        utilities = np.asarray(utilities, dtype=np.float64)
        if utilities.shape != (self.n_actions,):
            raise ValueError(
                f"utilities must have shape ({self.n_actions},), got {utilities.shape}"
            )
        # A NaN utility poisons both accumulators irreversibly, and every
        # later strategy() call silently returns NaN rather than failing.
        if not np.all(np.isfinite(utilities)):
            raise ValueError(f"utilities must all be finite, got {utilities}")
        # A negative or non-finite weight means the caller computed a reach
        # wrong. Deliberately no upper bound: in vanilla CFR both weights are
        # reach probabilities in [0, 1], but the later rungs legitimately go
        # above 1 — linear CFR weights the strategy sum by the iteration
        # index, and sampling variants divide by a sampling probability.
        for name, weight in (
            ("regret_weight", regret_weight),
            ("strategy_weight", strategy_weight),
        ):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {weight}")

        current = self.strategy()
        self.cumulative_regret += regret_weight * (utilities - current @ utilities)
        self.strategy_sum += strategy_weight * current
