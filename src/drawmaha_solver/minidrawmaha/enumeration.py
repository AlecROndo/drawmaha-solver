"""Every infoset in mini-drawmaha, at the one grain small enough to enumerate.

A tabular solver has to know its own size before it allocates anything, and
rung 3 is the first rung where that question cannot be answered by walking the
game. The concrete tree — one node per (deal, board card, actions,
replacements) history — holds **8.7 x 10^11 decision nodes**. Rung 2's
`all_infosets()` could afford to be a literal list comprehension over 288
positions; the same approach here does not run slowly, it does not run.

What makes the count reachable anyway is that the position factorises into two
independent halves, and neither half is large:

* a **public point** — the betting lines so far, both draw counts, how many
  board cards are out, and whose turn it is. There are **288** of them, and not
  one depends on a single card, so ONE representative deal walks all of them.
  That is the same 288 whether the deck is 15 cards or 52.
* a **private key** — the acting player's own `(hole, discarded, board)` after
  the joint canonical relabelling. **970 / 10,170 / 50,450** while one board
  card is out and **5,160 / 50,450 / 226,200** once two are, each triple
  ordered by whether that player has thrown 0, 1 or 2 cards. The first triple
  sums to 61,590, which is the number §4.3 of the plan measured from the other
  side — by walking draw histories rather than by building keys directly.

An infoset is exactly one of each, so the enumerator is two nested loops and
the census is a 288-term sum of products: **23,706,960 ledgers, about 11.85
million per player.**

**Why every pairing is real, and why none is counted twice.** Reachability is
the claim that could quietly be wrong in either direction, so both halves of it
are argued here and tested in `tests/minidrawmaha/test_census.py`:

* *Nothing is missing.* A public point constrains the actor's own throw count
  (it is in `draws`) and nothing else about the cards; a private key constrains
  the cards and nothing about the betting. The deck has three cards of slack at
  the deepest line (§4.1), so the opponent's hand and both board cards always
  fit around any private key, and every subset of at most two of a hole is a
  legal throw — so card removal never blocks a pairing.
* *Nothing is doubled.* `player`, `draws` and `betting` are all carried inside
  the key, so two different public points can never produce the same `InfoSet`,
  and inside one point the private keys are distinct by construction. Counting
  what comes out is therefore exact, which matters: a `set` of 23.7 million
  keys would cost around twelve gigabytes to hold.

`all_infosets()` is deliberately the ONLY walk. The census script counts what
it yields, and the ledger table allocates one `RegretMatcher` per key it
yields; nothing re-derives the size a second way. It returns an **iterator**
rather than rung 2's tuple for the same reason the concrete tree is never
walked — materialising the result is the expensive thing, and neither caller
needs to.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import cache
from itertools import combinations
from math import comb

from drawmaha_solver.minidrawmaha.cards import DECK, Card, canonical
from drawmaha_solver.minidrawmaha.game import (
    Action,
    DrawSignal,
    InfoSet,
    MiniState,
    NodeKind,
)
from drawmaha_solver.minidrawmaha.hands import HOLE_CARDS

# The root chance event, which `MiniState` deliberately leaves outside the state
# machine: C(15,3) hands for P0 and C(12,3) for P1. It is not a node anybody
# walks — it is the weight the public tree carries to report what the concrete
# tree would have cost.
ROOT_DEALS = comb(len(DECK), HOLE_CARDS) * comb(len(DECK) - HOLE_CARDS, HOLE_CARDS)

# One private key: the acting player's (hole, discarded, board), canonical.
PrivateKey = tuple[tuple[Card, ...], tuple[Card, ...], tuple[Card, ...]]

# ---------------------------------------------------------------------------
# The public tree
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PublicPoint:
    """One node of the game as the table sees it — everything but the cards.

    The fields split in two. `kind`, `draws` and `betting` are what both
    players know; `player`, `actions` and `discards` describe the decision
    standing on it and are `None` at a chance node or a terminal. `board_cards`
    is a count rather than the cards themselves, because which cards are out is
    the private key's business — the same public point is paired with every key
    of its shape.

    `discards` is how many cards the player to act has already thrown, and it
    is the one field that ties the two halves together: it selects which shape
    of private key belongs here, and it is redundant with `draws` on purpose,
    because reading it off `draws` at the pairing site is exactly the place a
    seat mix-up would go unnoticed.
    """

    kind: NodeKind
    board_cards: int
    draws: tuple[DrawSignal, ...]
    betting: tuple[tuple[Action, ...], ...]
    player: int | None
    actions: tuple[Action, ...] | None
    discards: int | None
    is_draw_decision: bool
    concrete_nodes: int

    @property
    def width(self) -> int | None:
        """How wide a ledger allocated here is: `len(legal_actions())`."""
        return None if self.actions is None else len(self.actions)

def merged(seen: PublicPoint, arriving: PublicPoint) -> PublicPoint:
    """One public point out of two arrivals at it, with their weights added.

    Three of the seven throws take one card and the table cannot tell them
    apart afterwards, so the walk reaches the same public point by different
    routes and their concrete histories have to add up rather than overwrite.

    The merge is only honest if the two arrivals really are one spot, which is
    a property of the rules and not of this file: it holds because the public
    record — the betting, both draw counts and how many board cards are out —
    decides who acts and what they may do. Should that ever stop being true,
    the collapse would quietly count a smaller game, so it raises instead. A
    module-level function rather than a closure inside the walk precisely so
    the refusal can be tested without manufacturing a broken rule set.
    """
    if (seen.player, seen.actions, seen.discards) != (
        arriving.player,
        arriving.actions,
        arriving.discards,
    ):
        raise ValueError(
            f"one public signature, two different decisions: {seen}, {arriving}"
        )
    return replace(seen, concrete_nodes=seen.concrete_nodes + arriving.concrete_nodes)

@cache
def public_tree() -> tuple[PublicPoint, ...]:
    """Every public node of mini-drawmaha: 288 decisions, 78 chances, 393 terminals.

    Walked from a representative deal rather than derived, so that the betting
    rules, the draw's seven actions and the action-dependent chance branch are
    read off `game.py` itself — a change to any of them moves this tree without
    anybody having to remember to update a table. Which cards the deal holds
    cannot matter: the public tree is the same for all 100,100 of them, and
    that is precisely the property the factorisation rests on.

    Two things the walk does that a plain tree walk would not:

    * **one chance outcome stands in for all of them.** The deck's outcomes
      differ only in which cards come out, which is invisible at this grain, so
      the walk follows the first and multiplies the weight it carries by how
      many there were. That weight is `concrete_nodes`, and it is what lets the
      census report the ~10^12 concrete tree it refuses to walk.
    * **nodes are merged by signature.** Three of the seven throws take one
      card, and the table cannot tell them apart afterwards, so they arrive at
      the same public point and their weights add.
    """
    # Any deal walks the same tree; the lowest six cards make the trace in a
    # failure message easy to read against `cards.py`'s deck order.
    root = MiniState(holes=(DECK[0:3], DECK[3:6]))
    found: dict[tuple, PublicPoint] = {}

    def visit(state: MiniState, concrete: int) -> None:
        decision = state.kind is NodeKind.DECISION
        player = state.current_player if decision else None
        point = PublicPoint(
            kind=state.kind,
            board_cards=len(state.board),
            draws=state.draws,
            betting=state.betting,
            player=player,
            actions=state.legal_actions() if decision else None,
            discards=None if player is None else len(state.discards[player]),
            is_draw_decision=decision and state.is_draw_decision(),
            concrete_nodes=concrete,
        )
        signature = (point.kind, point.board_cards, point.draws, point.betting)
        if signature in found:
            point = merged(found[signature], point)
        found[signature] = point
        if state.kind is NodeKind.TERMINAL:
            return
        if state.kind is NodeKind.CHANCE:
            outcomes = state.chance_outcomes()
            visit(state.apply_chance(outcomes[0][0]), concrete * len(outcomes))
            return
        for action in point.actions:
            visit(state.apply(action), concrete)

    visit(root, ROOT_DEALS)
    return tuple(found.values())

@cache
def public_decision_points() -> tuple[PublicPoint, ...]:
    """The 288 public points a ledger can stand on, in walk order.

    Chance and terminal nodes are dropped here rather than never built: the
    census reports all three counts, and a terminal that the walk stopped
    calling terminal is the kind of rules change this file exists to notice.
    """
    return tuple(point for point in public_tree() if point.kind is NodeKind.DECISION)

# ---------------------------------------------------------------------------
# The private keys
# ---------------------------------------------------------------------------

@cache
def private_keys(*, board_cards: int, discards: int) -> tuple[PrivateKey, ...]:
    """Every canonical `(hole, discarded, board)` of this shape, sorted.

    Built in **key space**, not history space: three disjoint groups of the
    right sizes, relabelled jointly, deduped. The equivalent design-time script
    walked draw histories instead — dealt hand, throw, replacement — and got
    the same 61,590 at one board card, which is the cross-check that the
    shortcut is sound. It is sound because the throw cap allows *every* subset
    of at most two cards, so no split of a hole into kept and thrown is
    unreachable, and the deck's three cards of slack mean the opponent always
    fits around whatever is left.

    Sorted rather than left in set order so that two runs of the census, and
    two allocations of the ledger table, see the keys in the same order.
    Cached because the six shapes cost about twenty-five seconds together and
    every one of the 288 public points asks for one of them.
    """
    keys: set[PrivateKey] = set()
    for hole in combinations(DECK, HOLE_CARDS):
        after_hole = [card for card in DECK if card not in hole]
        for board in combinations(after_hole, board_cards):
            unseen = [card for card in after_hole if card not in board]
            for thrown in combinations(unseen, discards):
                keys.add(canonical(hole, thrown, board))
    return tuple(sorted(keys))

# ---------------------------------------------------------------------------
# The one enumerator
# ---------------------------------------------------------------------------

def all_infosets() -> Iterator[InfoSet]:
    """Every infoset in mini-drawmaha, each exactly once. 23,706,960 of them.

    A generator, and the only walk: `scripts/generate_minidrawmaha_census.py`
    counts what it yields and `infoset_table.py` allocates one ledger per key
    it yields, so the census and the table can never disagree about what the
    game is. Consuming the whole of it takes minutes — hold the two building
    blocks above instead if all you want is a count or a slice.
    """
    for point in public_decision_points():
        for hole, thrown, board in private_keys(
            board_cards=point.board_cards, discards=point.discards
        ):
            yield InfoSet(
                player=point.player,
                hole=hole,
                discarded=thrown,
                board=board,
                draws=point.draws,
                betting=point.betting,
            )
