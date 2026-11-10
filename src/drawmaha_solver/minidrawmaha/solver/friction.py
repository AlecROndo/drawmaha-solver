"""Friction toward a reference strategy: RTCFR+'s reward transformation, under sampling.

Why it exists. A CFR run's CURRENT strategy need not converge (in the worst case
it circles the equilibrium); only its average is guaranteed to. RTCFR+ (Meng et al. 2023, arXiv 2308.11256, Eq. 7–8) changes
the game the regrets are learning so that the current strategy itself settles: each
player's loss gains μ·(x̂ − r̂), where x̂(I, a) is the player's own reach of the
sequence "arrive at I, then play a" and r̂(I, a) is the same for a REFERENCE
strategy. Playing a sequence more than the reference does is taxed; playing it less
earns a bonus. That pull makes the taxed game's equilibrium unique and the current
strategy spiral into it. The equilibrium is biased toward the reference, so every
so often the reference is reset to the current strategy (`Friction.reset`), and the
biased equilibria walk toward the real one. Regret totals are kept across a reset,
as in the paper's algorithm.

What the walk does with it. At each of the traverser's spots `mccfr.traverse`
subtracts `friction_adjustment(...)` from the action values before the regret is
formed, and passes each child its own reach σ(a)·x and the reference's
reference(a)·r. Values returned up the tree already carry the deeper spots'
friction, as the paper's counterfactual-value recursion does.

One departure from the paper, forced by sampling. In the paper the friction enters
every infoset's value with full weight, every iteration. External sampling banks a
row only when it is visited, and the payoff part of what it banks is weighted, in
expectation, by the chance-and-opponent reach π₋ᵢ(I) of the row; in the paper the
payoff is weighted the same way but the friction is not. The friction here is added
undivided on each visit, so in expectation it is scaled by π₋ᵢ(I) too: relative to
the paper the tax at row I is down-weighted by π₋ᵢ(I), i.e. μ acts as an effective
μ·π₋ᵢ(I), and the tax-to-payoff ratio is the same at every row instead of growing
as 1/π₋ᵢ at rare rows. Dividing by the visit probability would recover the paper's
weighting and hand rare rows enormous, noisy shoves. The paper's γ (a 1e-10 mix toward uniform)
is numerically inert at this scale and is left out.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import numpy as np

from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable

def friction_adjustment(
    sigma: np.ndarray,
    reference: np.ndarray,
    *,
    own_reach: float,
    reference_reach: float,
    mu: float,
) -> np.ndarray:
    """μ·(x̂ − r̂) at one spot: what to subtract from each action's value there.

    In: `sigma` and `reference`, the spot's current and reference strategies (one
    probability per legal action); `own_reach` and `reference_reach`, the chance
    that the traverser's own play (resp. the reference's) gets to this spot, its
    opponent and the deck not counted. Out: one tax per action, in chips.
    Positive where play out-reaches the reference, negative where it lags.
    """
    return mu * (own_reach * sigma - reference_reach * reference)

@dataclass(slots=True)
class Friction:
    """The tax's strength μ (chips) and the reference strategy it pulls toward.

    `table` is the run's table; `reference` holds one probability per column, a
    flat array aligned with a packed table's `cumulative_regret`, or a dict of
    arrays for a dict table (Leduc, tests). Built by `new`, never by hand.
    """

    mu: float
    table: object
    reference: np.ndarray | dict[Hashable, np.ndarray] = field(repr=False)

    @classmethod
    def new(cls, table, mu: float) -> Friction:
        """Friction of strength `mu` whose reference is the table's current play.

        On a fresh table that is uniform everywhere, which is RTCFR+'s starting
        reference. A negative or non-finite μ is refused by name.
        """
        if not np.isfinite(mu) or mu < 0:
            raise ValueError(f"friction mu must be a finite number ≥ 0, got {mu!r}")
        friction = cls(mu=float(mu), table=table, reference={})
        friction.reset()
        return friction

    def of(self, key: Hashable) -> np.ndarray:
        """The reference strategy at `key`, one probability per legal action."""
        if isinstance(self.table, PackedTable):
            _, start, width = self.table.slot_of(key)
            return self.reference[start : start + width]
        return self.reference[key]

    def reset(self) -> None:
        """Move the reference to what the table plays now (RTCFR+'s r ← x̂).

        Regrets are left alone. A packed table is copied in one vectorised pass;
        a dict table ledger by ledger.
        """
        if isinstance(self.table, PackedTable):
            self.reference = packed_current_strategy(self.table)
        else:
            self.reference = {key: ledger.strategy() for key, ledger in self.table.items()}

def packed_current_strategy(table: PackedTable) -> np.ndarray:
    """Every row's current strategy at once, flat and column-aligned with the table.

    In: a packed table. Out: an array the length of `cumulative_regret` holding,
    per row, the positive regrets normalised, or uniform where none is positive —
    the same numbers `RegretMatcher.strategy` gives row by row, without building
    6.2 million ledgers.
    """
    widths = table.widths()
    starts = np.concatenate(([0], np.cumsum(widths)[:-1]))
    positive = np.maximum(table.cumulative_regret, 0.0)
    totals = np.repeat(np.add.reduceat(positive, starts), widths)
    sizes = np.repeat(widths, widths).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0.0, positive / totals, 1.0 / sizes)
