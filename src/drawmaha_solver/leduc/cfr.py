"""Vanilla CFR on Leduc poker: the walk gains a chance branch mid-tree.

Rung 1's walk had two branches — terminal and decision — because chance
happened once, at the top: `run_iteration` looped over the deals and the walk
itself never met a chance node, so the deal probability could sit in a module
constant multiplied in at the bank. Leduc turns a card AFTER players have
acted, so the walk meets the deck in the middle of the tree and must handle
it from the inside. That third branch, and the receipt it leaves in the
weights, are the only new ideas in this file; the ledgers, the two-weight
update, and the shape of everything else are rung 1 unchanged.

The chance branch does two things, and they are different in kind:

- **For values, it averages.** A chance node has no strategy and no ledger;
  nobody chooses the board, so its value is the plain probability-weighted
  average of its children — not a sigma-weighted mix. This average feeds
  every utility above it, so it is load-bearing for what round-1 ledgers
  learn: skip it and take one board's value instead, and every opening-round
  regret is wrong.
- **For weights, it multiplies a receipt.** The counterfactual reach is
  defined to include chance, so the walk threads `chance` — the product of
  chance probabilities along the path, pi_c — down as it descends: 1/30 once
  the private cards are out, 1/30 * 1/4 = 1/120 once the board lands. It
  rides in the regret weight only. The strategy weight is the player's OWN
  reach and nothing else, because the average strategy must reproduce what
  the player did, not how often the deck cooperated.

Because this walk iterates CONCRETE cards (30 deals, 4 boards) and keys
ledgers by rank, every world of one ledger carries the same receipt — 1/30
for round-1 spots, 1/120 for round-2 — so the factor is convergence-inert
here exactly as Kuhn's 1/6 was: regret matching reads only the ratios inside
a ledger. Walk ranks instead and the receipts would differ across one
ledger's worlds and start steering the strategy. That is the paper's "walk
cards, key ranks" rule, enforced by construction.

One index discipline replaces rung 1's habit: sigma and the utilities vector
are indexed by POSITION in `legal_actions()`, never by `Action` value. Rung 1
could write `sigma[action]` because both actions were always legal and the
enum values doubled as columns; here a ledger is 2- or 3-wide and column 0
means CALL at the open but FOLD facing a raise. Every loop below pairs
`enumerate(legal_actions())` with the ledger's columns, and nothing else
keeps the two in step.

Updates land in place as the recursion unwinds, the order rung 1 uses and
the published traces are written against.

A subset of infosets can be LOCKED, which turns the same walk into a
best-response finder — rung 1's mechanism verbatim. At a locked infoset the
walk plays the supplied probabilities and banks nothing; everywhere else it
learns as usual. Note what does NOT change: a locked player's probabilities
still advance the reach, so they still enter the learner's counterfactual
weight pi_-i. The learner is never told "the opponent is fixed"; it is simply
weighted by how often that opponent actually brings it here, and maximizing
against a stationary opponent is what makes CFR converge to a best response
rather than to Nash.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from drawmaha_solver.leduc.game import DEAL_PROBABILITY, DEALS, InfoSet, LeducState
from drawmaha_solver.leduc.infoset_table import InfoSetTable, new_infoset_table

# Probabilities the walk plays but does not learn, keyed by the infoset they
# belong to. A partial map on purpose: locking one node is as legal as locking
# a whole seat, and only the caller knows which it meant. Rows follow the same
# index discipline as everything else at this rung: entry k belongs to
# `legal_actions()[k]`, and each row is as wide as its own spot.
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

    Deterministic: vanilla CFR enumerates the entire tree every iteration —
    all 30 deals, every betting line, all four boards — so there is no
    sampling and no seed.
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

    Every deal is walked, so each ledger accumulates a reach-weighted sum
    over every hidden world consistent with it — 10, 16 or 8 nodes' worth.
    That summation is the whole of the belief bookkeeping; there is no Bayes
    anywhere. The deal probability enters here as the walk's opening chance
    receipt, where rung 1 could leave it to the bank line.
    """
    for deal in DEALS:
        walk(LeducState(cards=deal), table, (1.0, 1.0), DEAL_PROBABILITY, locked=locked)

# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def walk(
    state: LeducState,
    table: InfoSetTable,
    reach: tuple[float, float],
    chance: float,
    *,
    locked: LockedStrategies | None = None,
) -> tuple[float, float]:
    """Expected chips to (P0, P1) from `state`, banking a ledger update per node.

    `reach[p]` is player p's own contribution to the probability of arriving
    here — the product of p's action probabilities along the path, excluding
    chance and excluding the other player. `chance` is pi_c, the product of
    every chance probability on the same path. Start a deal with
    `(1.0, 1.0)` and `DEAL_PROBABILITY`.

    Returns the value to BOTH seats rather than to the player to act, so the
    caller indexes the seat it wants instead of negating on the way up; sign
    flips are where this recursion is usually gotten wrong. The returned
    value is unweighted — a plain conditional expectation given that play
    reached here.

    Mutates `table`: every decision node on the path banks one update, EXCEPT
    the ones named in `locked`, which are played and not learned.
    """
    if state.is_terminal():
        return state.returns()

    # The new branch. The deck acts: average over what it might do, thread
    # its probability into the receipt, and touch no ledger — the deck has
    # no regrets. Both players' own reaches cross unchanged, which is why
    # the strategy weight below never smells the board.
    if state.is_chance_node():
        value = [0.0, 0.0]
        for card, probability in state.chance_outcomes():
            child = walk(
                state.apply_chance(card), table, reach, chance * probability, locked=locked
            )
            value[0] += probability * child[0]
            value[1] += probability * child[1]
        return (value[0], value[1])

    player = state.current_player
    infoset = state.infoset()
    # A locked node reads its sigma from the caller instead of the ledger, and
    # the ledger is then never touched — not even looked up, so a locked run
    # cannot quietly bank into a spot it was told to hold still.
    pinned = None if locked is None else locked.get(infoset)
    ledger = None if pinned is not None else table[infoset]
    sigma = pinned if pinned is not None else ledger.strategy()

    # Descend once per legal action, handing each child the reach it was
    # actually played with. sigma[k] belongs to legal_actions()[k] — position,
    # not Action value — which is the whole index discipline of this rung.
    # A locked player advances the reach exactly like a learning one, which is
    # how its pinned probabilities enter the opponent's counterfactual weight.
    children = [
        walk(
            state.apply(action),
            table,
            _advance(reach, player, sigma[k]),
            chance,
            locked=locked,
        )
        for k, action in enumerate(state.legal_actions())
    ]

    if ledger is not None:
        # The utilities the ledger wants are this player's column of the
        # children's values: "what is each action worth to me from here", in
        # the same position order sigma uses.
        utilities = np.array([child[player] for child in children])
        ledger.update(
            utilities,
            # Chance rides with the opponent because the deck is not something
            # the player controls; both are absent from the strategy weight.
            regret_weight=chance * reach[1 - player],
            strategy_weight=reach[player],
        )

    return (
        float(sigma @ [child[0] for child in children]),
        float(sigma @ [child[1] for child in children]),
    )

def _advance(
    reach: tuple[float, float], player: int, probability: float
) -> tuple[float, float]:
    """`reach` with only `player`'s own contribution multiplied through."""
    if player == 0:
        return (reach[0] * probability, reach[1])
    return (reach[0], reach[1] * probability)
