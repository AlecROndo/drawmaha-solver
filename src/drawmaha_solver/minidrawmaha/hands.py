"""Both halves of the split pot, as comparable score tuples.

The pot pays twice on one deal, and the two halves read the same cards by
different rules — which is the whole idea of Drawmaha, shrunk:

* the **inner** half goes to the best 3-card hand made of your three private
  cards alone. Nothing public touches it, so it is decided the moment the draw
  is over.
* the **outer** half goes to the best 4-card hand made of **exactly two** of
  your private cards plus **both** board cards. That is Omaha's exactly-two rule
  at this scale, and it is what makes the two halves disagree: dealt trips, the
  inner half has its top category and the outer half can only bring a pair.

**Both rankings are measured rarity in this 15-card deck, not 52-card
convention**, with one stated exception — high card is the floor by definition,
because it is the absence of a made hand rather than a claim about rarity. Two
rows therefore look wrong and are correct:

* a four-card **straight** (25.11%) ranks BELOW two pair (16.38%) and trips
  (9.89%). A run of four out of only five ranks is easy to make. Ranking by
  measured rarity is the same rule that puts straights above both in a 52-card
  deck; the deck changed, so the order did.
* outer **high card** is the RAREST outer result at 0.74% — rarer than a
  straight flush — and still ranks last. The scorer takes the best of three
  two-card choices, so ending with nothing at all is a near-miracle.

Quads are not a category: a rank owns three cards because there are three suits.
Nor is there a wheel — rank 4 is the top and rank 0 the floor, and they are not
neighbours, which is why the ranks are numbered rather than given an ace's two
faces.

A score is `(category, ranks)`, and both fields are needed. `ranks` orders the
hand's ranks by count and then by rank, descending, so comparing two scores as
plain tuples settles the category first and then the kicker the way poker does:
trips of 3s with a 6 kicker is `(TRIPS, (3, 3, 3, 6))` — trip rank first, kicker
last, because "my trips are higher" outranks "my spare card is higher". Suits
appear nowhere in a score, so two hands tie exactly when nothing but their suits
differs.
"""

from __future__ import annotations

from collections import Counter
from enum import IntEnum
from itertools import combinations
from typing import NamedTuple

from drawmaha_solver.minidrawmaha.cards import Card

# How many cards each rule reads. The outer half is handed all three private
# cards and must choose two of them, which is the constraint rather than an
# implementation detail — see `outer_score`.
HOLE_CARDS = 3
BOARD_CARDS = 2
OUTER_HOLE_CARDS = 2

class InnerCategory(IntEnum):
    """The six things three cards can be, in this deck's rarity order.

    Two pair is missing because three cards cannot make it. The frequencies, out
    of all 455 dealt hands: trips 1.10%, straight flush 1.98%, flush 4.62%,
    straight 15.82%, pair 39.56%, high card 36.92% — so the single inversion is
    high card being marginally rarer than a pair and still pinned to the floor.
    """

    HIGH_CARD = 1
    PAIR = 2
    STRAIGHT = 3
    FLUSH = 4
    STRAIGHT_FLUSH = 5
    TRIPS = 6

class OuterCategory(IntEnum):
    """The seven things four cards can be, in this deck's rarity order.

    All seven occur, which is what the fifth rank was bought for: at four ranks
    "four distinct ranks" is the whole rank set, so every unpaired hand is a run
    of four and neither high card nor plain flush can exist at all. The
    frequencies, out of all 30,030 (hole, board) deals: straight flush 1.29%,
    flush 1.81%, trips 9.89%, two pair 16.38%, straight 25.11%, pair 44.78%,
    high card 0.74%.
    """

    HIGH_CARD = 1
    PAIR = 2
    STRAIGHT = 3
    TWO_PAIR = 4
    TRIPS = 5
    FLUSH = 6
    STRAIGHT_FLUSH = 7

class Score(NamedTuple):
    """A made hand as an orderable pair: what it is, then what it is made of.

    Compared as a plain tuple, so `<` between two scores is the showdown for one
    half of the pot. `ranks` is grouped by count first, so a category's own rank
    beats its kicker — and because suits are absent, equality is a genuine chop.
    """

    category: InnerCategory | OuterCategory
    ranks: tuple[int, ...]

def _shape(cards: tuple[Card, ...]) -> tuple[Counter[int], bool, bool, tuple[int, ...]]:
    """The four facts both rankings are built from.

    Counts by rank, whether one suit holds every card, whether the ranks are a
    run with no gaps, and the ranks ordered by count then rank. Written once
    because the two rankings differ only in how they *order* the categories
    these facts pick out, and a second copy would be a second place for the
    straight rule to go stale.
    """
    counts = Counter(card.rank for card in cards)
    flush = len({card.suit for card in cards}) == 1
    distinct = sorted(counts, reverse=True)
    # A straight is every card a distinct rank AND no gap between the ends. The
    # top rank does not wrap round to the bottom, so `4 0 1 2` is not a run.
    straight = len(distinct) == len(cards) and distinct[0] - distinct[-1] == len(cards) - 1
    by_count = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    ranks = tuple(rank for count, rank in by_count for _ in range(count))
    return counts, flush, straight, ranks

def inner_score(hole: tuple[Card, ...]) -> Score:
    """Score the inner half: the best 3-card hand these three cards make.

    There is nothing to choose — three cards are one hand — so this is a
    classification, and the board is not a parameter because the inner half
    never sees it.
    """
    if len(hole) != HOLE_CARDS:
        raise ValueError(f"the inner half scores three private cards, got {len(hole)}")
    counts, flush, straight, ranks = _shape(hole)
    top = max(counts.values())
    if top == 3:
        # Trips is never a flush: three of a rank is one card per suit.
        category = InnerCategory.TRIPS
    elif straight and flush:
        category = InnerCategory.STRAIGHT_FLUSH
    elif flush:
        category = InnerCategory.FLUSH
    elif straight:
        category = InnerCategory.STRAIGHT
    elif top == 2:
        category = InnerCategory.PAIR
    else:
        category = InnerCategory.HIGH_CARD
    return Score(category, ranks)

def four_card_score(cards: tuple[Card, ...]) -> Score:
    """Score one specific four-card hand: two chosen hole cards and both board cards.

    The outer *ranking*, applied to a hand that has already been chosen.
    `outer_score` is the maximum of this over the three legal pairings, and that
    split is what lets a caller — or a test — ask which pairing a player is
    actually playing.
    """
    if len(cards) != OUTER_HOLE_CARDS + BOARD_CARDS:
        raise ValueError(f"the outer half ranks four cards, got {len(cards)}")
    counts, flush, straight, ranks = _shape(cards)
    shape = sorted(counts.values(), reverse=True)
    if shape[0] == 3:
        category = OuterCategory.TRIPS
    elif straight and flush:
        category = OuterCategory.STRAIGHT_FLUSH
    elif flush:
        category = OuterCategory.FLUSH
    elif straight:
        category = OuterCategory.STRAIGHT
    elif shape[0] == 2 and shape[1] == 2:
        category = OuterCategory.TWO_PAIR
    elif shape[0] == 2:
        category = OuterCategory.PAIR
    else:
        category = OuterCategory.HIGH_CARD
    return Score(category, ranks)

def outer_score(hole: tuple[Card, ...], board: tuple[Card, ...]) -> Score:
    """Score the outer half: the best of exactly two hole cards plus both board cards.

    Three choices of pairing, and the best one is the hand — a player is never
    made worse off by holding a card they need not use. The exactly-two rule is
    what is being rehearsed here: it is why trips in the hand is worth a pair
    outside, and why the two halves of one pot can want opposite things.
    """
    if len(hole) != HOLE_CARDS:
        raise ValueError(f"the outer half chooses from three cards, got {len(hole)}")
    if len(board) != BOARD_CARDS:
        raise ValueError(f"the outer half uses both board cards, got {len(board)}")
    return max(
        four_card_score(chosen + tuple(board))
        for chosen in combinations(hole, OUTER_HOLE_CARDS)
    )
