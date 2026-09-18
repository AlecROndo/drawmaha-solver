"""The 15-card deck, and the one collapse this rung is allowed: relabelling suits.

Five ranks in three suits. Ranks are plain ints ordered 0 (low) to 4 (high) —
they have no faces to name, and `card.rank > other.rank` is the whole of "which
is higher". Three suits is forced: with two, a rank holds two cards and the
inner half loses trips entirely. The fifth rank is what gives the outer half a
complete ranking — at four ranks, four distinct ranks *is* the whole rank set,
so a run of four is the only unpaired hand there is and both high card and plain
flush have nowhere to exist.

**Suits carry information here, which is the break from rung 2.** Leduc's
infoset key discarded suits outright, losing nothing because Leduc has no
flushes. Both halves of this pot do, so the collapse that keeps the ledger count
down has to preserve the *relationships* between suits while forgetting their
names: `canonical()` relabels the three suits to the lexicographically smallest
image of the whole visible picture at once. That takes the 455 dealt hands down
to 95 classes — a 4.79x saving that costs nothing, because which physical suit
is which is not a fact about the position.

The word "jointly" in `canonical()`'s contract is load-bearing, and §7.3 of the
plan calls it the correctness trap of the rung. Canonicalising the hole and the
board separately forgets whether they *share* a suit, which is exactly what a
flush is; two genuinely different positions would then hash to one key and the
solve would be quietly wrong rather than loudly broken.

`canonical_relabelling()` returns the permutation behind the key. `game.py`
needs it, not just the key: a draw action names a card by position ("throw the
lowest"), and the positions have to mean the same card in every position that
shares an infoset — so the order is taken under the canonical labels, never
under the arbitrary physical ones.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from typing import NamedTuple

N_RANKS = 5
N_SUITS = 3

RANKS = tuple(range(N_RANKS))
SUITS = tuple(range(N_SUITS))

class Card(NamedTuple):
    """One physical card: a rank 0..4 and a suit 0..2.

    A tuple, so `sorted()` orders cards by rank and then by suit label. That
    order is a *labelling* convention and never a hand comparison — a single
    card is not a hand in this game, both halves score three or four cards at a
    time, and suits are unordered in the rules. `hands.py` owns every
    comparison that decides a pot.
    """

    rank: int
    suit: int

# Ordered by (rank, suit), so `DECK[:3]` is the whole of the lowest rank.
DECK = tuple(Card(rank, suit) for rank in RANKS for suit in SUITS)

# Deuce through six, clubs/diamonds/hearts: three suits with familiar letters
# and five ranks with no face cards, because rank 4 is the top of this deck and
# calling it an ace would invite a wheel that does not exist (see `hands.py`).
RANK_SYMBOL = "23456"
SUIT_SYMBOL = "cdh"

CARD_SYMBOL = {card: RANK_SYMBOL[card.rank] + SUIT_SYMBOL[card.suit] for card in DECK}

SYMBOL_CARD = {symbol: card for card, symbol in CARD_SYMBOL.items()}

def hand_symbol(cards: tuple[Card, ...]) -> str:
    """A group of cards as space-separated symbols, in the order given."""
    return " ".join(CARD_SYMBOL[card] for card in cards)

def parse_cards(text: str) -> tuple[Card, ...]:
    """`"6h 2c"` back into cards — for tests, fixtures and the play CLI.

    Refuses anything the deck does not hold rather than inventing a card: a
    typo in a fixture that silently parsed would pin the wrong answer.
    """
    cards = []
    for symbol in text.split():
        if symbol not in SYMBOL_CARD:
            raise ValueError(f"{symbol} is not a card in this 15-card deck")
        cards.append(SYMBOL_CARD[symbol])
    return tuple(cards)

# ---------------------------------------------------------------------------
# Canonical suit relabelling
# ---------------------------------------------------------------------------

# A relabelling reads `new_suit = relabelling[old_suit]`. All six of them, in a
# fixed order, so that `canonical_relabelling` breaks a tie between two
# permutations that produce the same image the same way every time it is asked.
# It has to be deterministic: `game.py` orders a hole by these labels, and an
# order that wobbled between calls would rename the draw actions mid-solve.
SUIT_RELABELLINGS = tuple(permutations(SUITS))

Group = tuple[Card, ...]

def relabel(relabelling: tuple[int, ...], cards: Group) -> Group:
    """`cards` with every suit renamed by `relabelling`, ranks untouched."""
    return tuple(Card(card.rank, relabelling[card.suit]) for card in cards)

@lru_cache(maxsize=1 << 18)
def _minimal_image(groups: tuple[Group, ...]) -> tuple[tuple[Group, ...], tuple[int, ...]]:
    """The smallest joint image of `groups`, and the relabelling that makes it.

    One search over all six permutations, scoring the *whole* tuple of groups.
    Every group is sorted after relabelling, so the image is a normal form: the
    order cards arrive in is not part of a position.

    Memoised, because the callers ask the same question repeatedly: a solver
    looks up one position's key at every visit, `canonical` and
    `canonical_relabelling` are two views of this one answer, and `game.py`
    asks for a hole's draw order and then its infoset key back to back. The
    cache is bounded rather than unlimited — a long run visits far more
    positions than a laptop wants to hold, and the ones it revisits are
    revisited soon.
    """
    best_image: tuple[Group, ...] | None = None
    best_relabelling = SUIT_RELABELLINGS[0]
    for relabelling in SUIT_RELABELLINGS:
        image = tuple(tuple(sorted(relabel(relabelling, group))) for group in groups)
        if best_image is None or image < best_image:
            best_image, best_relabelling = image, relabelling
    return best_image or (), best_relabelling

def canonical(*groups: Group) -> tuple[Group, ...]:
    """Relabel suits to the smallest image of ALL `groups` jointly.

    Called with the acting player's picture — `(hole, discards, board)` — as
    separate groups in a single call. One permutation across every group at
    once: the groups stay distinguishable in the answer, while the suit names
    that relate them are forgotten together.

    Canonicalising the groups one at a time instead destroys the suit
    relationship *between* them, which is what makes a flush. That merges
    distinct positions into one ledger and the solve comes out wrong with no
    traceback, so there is deliberately no single-group helper to reach for.
    """
    return _minimal_image(groups)[0]

def canonical_relabelling(*groups: Group) -> tuple[int, ...]:
    """The suit relabelling behind `canonical(*groups)`.

    `game.py` orders a hole by `(rank, relabelling[suit])` so that "the lowest
    of my three cards" names the same card in every position sharing an
    infoset. Under the physical suit numbers it would not: two positions that
    differ by a suit swap sort their equal-ranked cards in opposite orders, so
    one ledger's "throw the lowest" would mean the board-suited card in one
    position and the offsuit one in another.
    """
    return _minimal_image(groups)[1]
