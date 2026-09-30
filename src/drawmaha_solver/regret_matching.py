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

**The ledger contract (rung 3).** Everything outside this file that touches a
ledger — the walk, the strategy views, the checkpoint, the regret rules, a
packed table — reads and writes exactly these slots, and nothing else:

- `n_actions`, `strategy()`, `average_strategy()`, `update(...)` — rung 0's
  ledger, unchanged. `update` is vanilla regret matching and touches only the
  first two arrays below.
- `cumulative_regret` — float64, `(n_actions,)`.
- `strategy_sum` — float64, `(n_actions,)`: the run's PRIMARY average, the one
  `average_strategy()` reads.
- `extra_sums` — float64, `(k, n_actions)`: k further averages of the same
  strategies under other weightings, banked beside the primary one. `k = 0`
  unless asked for. Only a regret rule writes them; which weighting row j
  holds is the run's business, not the ledger's.
- `stamp` — int64, `(1,)`: the last iteration a lazily applied discount has
  been brought up to. 0 until a rule that discounts sets it.

All four arrays may be VIEWS into a table's flat buffers (`RegretMatcher.over`),
so every writer changes them in place — `+=`, `[:] =` — and never rebinds
the attribute, which would silently cut the ledger loose from its table.
"""

from __future__ import annotations

import math

import numpy as np

class RegretMatcher:
    """One regret-matching ledger over a fixed set of actions.

    Holds two persistent vectors — accumulated regret and the banked strategy
    sum — plus the two slots rung 3's regret rules write (`extra_sums`,
    `stamp`; see the module docstring). The current and average strategies are
    derived on demand and never stored.
    """

    def __init__(self, n_actions: int, *, extra_averages: int = 0):
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
        self.extra_sums = np.zeros((extra_averages, n_actions))
        self.stamp = np.zeros(1, dtype=np.int64)

    @classmethod
    def over(
        cls,
        *,
        cumulative_regret: np.ndarray,
        strategy_sum: np.ndarray,
        extra_sums: np.ndarray,
        stamp: np.ndarray,
    ) -> RegretMatcher:
        """A ledger whose numbers live in arrays somebody else owns — no copy.

        What a packed table hands the walk: the four slots are views into the
        table's flat buffers, so an `update` lands in the table itself and the
        ledger object can be thrown away after the visit. Checked here rather
        than trusted, because a wrong-width view or a float32 buffer would not
        crash — it would bank into a neighbour's columns or lose precision
        without a word.
        """
        n = len(cumulative_regret)
        if n < 2:
            raise ValueError(f"n_actions must be at least 2, got {n}")
        for name, array, shape, dtype in (
            ("cumulative_regret", cumulative_regret, (n,), np.float64),
            ("strategy_sum", strategy_sum, (n,), np.float64),
            ("extra_sums", extra_sums, (len(extra_sums), n), np.float64),
            ("stamp", stamp, (1,), np.int64),
        ):
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(
                    f"{name} must be {np.dtype(dtype).name} {shape}, "
                    f"got {array.dtype.name} {array.shape}"
                )
        ledger = cls.__new__(cls)
        ledger.n_actions = n
        ledger.cumulative_regret = cumulative_regret
        ledger.strategy_sum = strategy_sum
        ledger.extra_sums = extra_sums
        ledger.stamp = stamp
        return ledger

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
