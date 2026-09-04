"""Vanilla CFR on Kuhn poker: the recursive walk that feeds the twelve ledgers.

Rung 0's ledger learned from a utility vector the environment handed it — the
opponent revealed a move and the payoff matrix had a column ready. A tree
deletes that luxury: the value of "bet with the king" depends on what both
players do afterwards, so the utilities must be COMPUTED. That is this file,
and it is the only genuinely new idea at rung 1.

Two quantities travel through `walk`, in opposite directions:

- **Down: reach probabilities.** `reach[p]` is the product of player p's OWN
  action probabilities along the path so far — nothing else. Chance is not in
  either one.
- **Up: node values.** What the position is worth to each seat, as a plain
  conditional expectation. Values are NOT reach-weighted; the weights apply
  once, at the moment a ledger is banked.

At an infoset owned by player i, the two weights part ways. Regret is banked
at the COUNTERFACTUAL reach — the opponent's actions times the deal's
probability, with player i's own choices deliberately excluded, so a line the
player currently avoids keeps learning at full strength and can be
rediscovered. The average strategy is banked at player i's OWN reach, because
it has to reproduce what that player actually did. Swapping the two is the
classic CFR bug: both versions run, and one converges to nonsense.

The chance factor inside that regret weight is convergence-inert HERE: regret
matching reads only the RATIO of positive regrets, and Kuhn's six deals are
equally likely, so scaling every increment by the same 1/6 leaves σ, the reach
probabilities, and the average strategy unchanged — up to floating-point
rounding, measured at 9e-15 after 5,000 iterations rather than exactly zero,
since a/(a+b) and (a/6)/((a+b)/6) can differ in the last ulp. It is carried
because the counterfactual reach is DEFINED to include chance, not because
Kuhn's convergence needs it: the moment a game deals non-uniform chance the
factor stops being a common scale and starts mattering. It is also why these
ledgers hold numbers 6x smaller than the reference trace below, whose
pseudocode omits the factor and lets the loop over all six deals average
chance out instead.

Updates land in place as the recursion unwinds, which is the update ORDER
Neller & Lanctot's reference implementation uses. The textbook alternative
accumulates every increment under a frozen σᵗ and applies them at the end of
the iteration; both converge, and the in-place form is the one the published
traces are written against.

A subset of infosets can be LOCKED, which turns the same walk into a
best-response finder. At a locked infoset the walk plays the supplied
probabilities and banks nothing; everywhere else it learns as usual. Note what
does NOT change: a locked player's probabilities still advance the reach, so
they still enter the learner's counterfactual weight π₋ᵢ. That is the whole
mechanism — the learner is not told "the opponent is fixed", it is simply
weighted by how often that opponent actually brings it here, and maximizing
against a stationary opponent is what makes CFR converge to a best response
rather than to Nash.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from drawmaha_solver.kuhn.game import DEAL_PROBABILITY, DEALS, InfoSet, KuhnState
from drawmaha_solver.kuhn.infoset_table import InfoSetTable, new_infoset_table

# Probabilities the walk plays but does not learn, keyed by the infoset they
# belong to. A partial map on purpose: locking one node is as legal as locking
# a whole seat, and only the caller knows which it meant.
LockedStrategies = Mapping[InfoSet, np.ndarray]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    iterations: int,
    *,
    table: InfoSetTable | None = None,
    locked: LockedStrategies | None = None,
) -> InfoSetTable:
    """Run `iterations` full CFR iterations and return the table of ledgers.

    Pass an existing `table` to continue a run; omit it to start fresh. The
    answer is `average_strategy(table)` — never `current_strategy`, which
    cycles forever and is not the object with the convergence guarantee.

    With `locked`, the listed infosets play their given probabilities and never
    learn. Their ledgers stay at zero, so `average_strategy(table)` reports
    them as uniform — which is NOT what was played. Read a locked run through
    `exploiter.exploit`, which substitutes the locked probabilities back in.

    Deterministic: vanilla CFR enumerates the entire tree every iteration, so
    there is no sampling and no seed.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    if table is None:
        table = new_infoset_table()
    for _ in range(iterations):
        run_iteration(table, locked=locked)
    return table

def run_iteration(table: InfoSetTable, *, locked: LockedStrategies | None = None) -> None:
    """One iteration: walk the tree once per deal, mutating `table` in place.

    Every deal is walked, so each infoset is visited by the two deals
    consistent with it and its ledger accumulates a reach-weighted sum over
    both hidden worlds. That summation is the whole of the belief bookkeeping
    — there is no Bayes anywhere.
    """
    for deal in DEALS:
        walk(KuhnState(cards=deal), table, (1.0, 1.0), locked=locked)

# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def walk(
    state: KuhnState,
    table: InfoSetTable,
    reach: tuple[float, float],
    *,
    locked: LockedStrategies | None = None,
) -> tuple[float, float]:
    """Expected chips to (P0, P1) from `state`, banking a ledger update per node.

    `reach[p]` is player p's own contribution to the probability of arriving
    here — the product of p's action probabilities along the path, excluding
    chance and excluding the other player. Start a deal with `(1.0, 1.0)`.

    Returns the value to BOTH seats rather than to the player to act, so the
    caller indexes the seat it wants instead of negating on the way up; sign
    flips are where this recursion is usually gotten wrong. The returned value
    is unweighted — a plain conditional expectation given that play reached
    here.

    Mutates `table`: every decision node on the path banks one update, EXCEPT
    the ones named in `locked`, which are played and not learned.
    """
    if state.is_terminal():
        return state.returns()

    player = state.current_player
    infoset = state.infoset()
    # A locked node reads its σ from the caller instead of the ledger, and the
    # ledger is then never touched — not even looked up, so a locked run cannot
    # quietly bank into a spot it was told to hold still.
    pinned = None if locked is None else locked.get(infoset)
    ledger = None if pinned is not None else table[infoset]
    sigma = pinned if pinned is not None else ledger.strategy()

    # Descend once per action, handing each child the reach it was actually
    # played with. Only the acting player's own reach advances here — and a
    # locked player advances it exactly like a learning one, which is how its
    # probabilities reach the opponent's counterfactual weight.
    children = [
        walk(
            state.apply(action),
            table,
            _advance(reach, player, sigma[action]),
            locked=locked,
        )
        for action in state.legal_actions()
    ]

    if ledger is not None:
        # The utilities the ledger wants are this player's column of the
        # children's values: "what is each action worth to me from here".
        utilities = np.array([child[player] for child in children])
        ledger.update(
            utilities,
            # Chance rides with the opponent because the deck is not something
            # the player controls, and it is absent from the strategy weight.
            regret_weight=DEAL_PROBABILITY * reach[1 - player],
            strategy_weight=reach[player],
        )

    return (
        float(sigma @ [child[0] for child in children]),
        float(sigma @ [child[1] for child in children]),
    )

def _advance(reach: tuple[float, float], player: int, probability: float) -> tuple[float, float]:
    """`reach` with only `player`'s own contribution multiplied through."""
    if player == 0:
        return (reach[0] * probability, reach[1])
    return (reach[0], reach[1] * probability)
