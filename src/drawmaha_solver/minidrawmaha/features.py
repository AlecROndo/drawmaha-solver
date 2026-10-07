"""The feature encoder: every mini-drawmaha infoset as 136 floats, never a key id.

Rung 3's table holds one ledger per infoset and looks it up by key. A network
cannot be handed a key — a key is a name, and a net that learned names would
be a table with extra steps, which is exactly what rung 4 must not measure.
It is handed what the player can SEE: the three cards held, the card thrown,
the board in the order it came, both draw counts, the betting so far, the
chips, and whose turn it is. Two keys that share all of that share a vector,
and nothing in the vector says which row of the table the key sits in.

**One group of cards is 23 floats.** A multi-hot over the 15-card deck (which
cards), the count of each of the five ranks (a pair is a 2, trips a 3), and
the count of each of the three suits (two of a suit is a 2). The counts are
redundant with the multi-hot and deliberately so: they are the one-hot
analogue of Brown et al.'s rank and suit embeddings, and let the net carry
what it learns about "any pair" across cards instead of relearning it per
card. Suits are already canonical in the key, so suit 0 means the same thing
in every row.

**The vector splits where the key splits.** The enumerator builds a key from
a private half — `(hole, discarded, board)`, one of three cached shapes — and
a public half, one of 141 public points. The features split the same way:
four card groups (hole, discard, board card 1, board card 2; 92 floats) from
the private half, and the draw signals, the betting line, the chips and the
seat and stage (44 floats) from the public point. The private table is built
once per shape and the public one once for all 141 points; a key's row is the
two halves side by side, so no 6.2M x 136 matrix ever exists and a memory row
needs only the key's row number. `Layout` is the arithmetic from a row
number back to (point, position), the same arithmetic the packed table's
index uses forward.

**The gather.** The net's head is 7 wide, one output per `Action`, so an
output column means one thing at every spot. A ledger's columns are 2, 3 or 4
of them: `legal_gather()` says which, per public point.

Nothing here imports torch: the encoder is NumPy, the default suite runs
without the optional `deep` extra, and a net is somebody else's module.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from drawmaha_solver.minidrawmaha.cards import DECK, N_RANKS, N_SUITS, Card
from drawmaha_solver.minidrawmaha.enumeration import (
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.game import STACK, Action, DrawSignal, chip_state

# ---------------------------------------------------------------------------
# The widths
# ---------------------------------------------------------------------------

# One group of cards: which of the 15, how many of each rank, how many of each suit.
CARD_GROUP = len(DECK) + N_RANKS + N_SUITS  # 23
# Hole, discard, board card 1, board card 2.
PRIVATE_WIDTH = 4 * CARD_GROUP  # 92

# Per seat: has not drawn yet / threw 0 / threw 1.
_DRAW_STATES = 3
DRAW_WIDTH = 2 * _DRAW_STATES  # 6
# The longest betting line is five actions (bet, raise, raise, raise, call).
LINE_SLOTS = 5
_BETTING_ACTIONS = 3
BETTING_WIDTH = 2 * LINE_SLOTS * _BETTING_ACTIONS  # 30
# Pot, P0's chips behind, P1's chips behind, each over the stack.
CHIP_WIDTH = 3
# Seat to act (2) and stage: round 1, the draw, round 2 (3).
SEAT_STAGE_WIDTH = 5
PUBLIC_WIDTH = DRAW_WIDTH + BETTING_WIDTH + CHIP_WIDTH + SEAT_STAGE_WIDTH  # 44

FEATURE_WIDTH = PRIVATE_WIDTH + PUBLIC_WIDTH  # 136

# The net's head: one output per Action, indexed by the Action's value.
HEAD_WIDTH = len(Action)  # 7
# No ledger is wider than the draw's four throws.
MAX_WIDTH = 4
# The gather's padding for a ledger narrower than MAX_WIDTH.
NO_ACTION = -1

# ---------------------------------------------------------------------------
# The private half: four groups of cards
# ---------------------------------------------------------------------------

def card_group(cards: tuple[Card, ...]) -> np.ndarray:
    """One group of cards as 23 floats: multi-hot, rank counts, suit counts. Zeros for none."""
    group = np.zeros(CARD_GROUP, dtype=np.float32)
    for card in cards:
        group[DECK.index(card)] += 1.0
        group[len(DECK) + card.rank] += 1.0
        group[len(DECK) + N_RANKS + card.suit] += 1.0
    return group

def private_row(
    hole: tuple[Card, ...], discarded: tuple[Card, ...], board: tuple[Card, ...]
) -> np.ndarray:
    """A private key's 92 floats: hole, discard, board card 1, board card 2 (zeros before it comes)."""
    return np.concatenate(
        [card_group(hole), card_group(discarded), card_group(board[:1]), card_group(board[1:])]
    )

@cache
def private_features(*, board_cards: int, discards: int) -> np.ndarray:
    """Every private key of this shape as a row, in `private_keys` order; read-only, built once."""
    keys = private_keys(board_cards=board_cards, discards=discards)
    table = np.empty((len(keys), PRIVATE_WIDTH), dtype=np.float32)
    for position, (hole, discarded, board) in enumerate(keys):
        table[position] = private_row(hole, discarded, board)
    table.setflags(write=False)
    return table

# ---------------------------------------------------------------------------
# The public half: draws, betting, chips, seat and stage
# ---------------------------------------------------------------------------

def public_row(
    *,
    draws: tuple[DrawSignal, ...],
    betting: tuple[tuple[Action, ...], ...],
    player: int,
    is_draw: bool,
) -> np.ndarray:
    """A public point's 44 floats: draw signals, betting slots, chips, seat, stage.

    Draw signals: per seat, one-hot over {not drawn yet, threw 0, threw 1}.
    Betting: per round, five slots, each a one-hot over {fold, check/call,
    pot} for the action in that slot and zeros past the line's end — so the
    ORDER of the line is in the vector, not just which actions it holds.
    Chips: the pot and each seat's chips behind, over the 26-chip stack.
    Stage: round 1, the draw, round 2.
    """
    row = np.zeros(PUBLIC_WIDTH, dtype=np.float32)
    for seat in (0, 1):
        state = 1 + draws[seat].count if len(draws) > seat else 0
        row[seat * _DRAW_STATES + state] = 1.0
    at = DRAW_WIDTH
    for round_index in range(2):
        line = betting[round_index] if round_index < len(betting) else ()
        for slot, action in enumerate(line):
            row[at + (round_index * LINE_SLOTS + slot) * _BETTING_ACTIONS + int(action)] = 1.0
    at += BETTING_WIDTH
    chips = chip_state(betting)
    row[at : at + CHIP_WIDTH] = (
        chips.pot / STACK,
        chips.behind[0] / STACK,
        chips.behind[1] / STACK,
    )
    at += CHIP_WIDTH
    row[at + player] = 1.0
    stage = 1 if is_draw else (len(betting) - 1) * 2
    row[at + 2 + stage] = 1.0
    return row

@cache
def public_features() -> np.ndarray:
    """Every public decision point as a row, in walk order; read-only, built once."""
    points = public_decision_points()
    table = np.empty((len(points), PUBLIC_WIDTH), dtype=np.float32)
    for index, point in enumerate(points):
        table[index] = public_row(
            draws=point.draws,
            betting=point.betting,
            player=point.player,
            is_draw=point.is_draw_decision,
        )
    table.setflags(write=False)
    return table

# ---------------------------------------------------------------------------
# The gather: which head columns a ledger's columns are
# ---------------------------------------------------------------------------

@cache
def legal_gather() -> np.ndarray:
    """Per public point, the head column of each ledger column, −1 past the ledger's width."""
    points = public_decision_points()
    gather = np.full((len(points), MAX_WIDTH), NO_ACTION, dtype=np.int8)
    for index, point in enumerate(points):
        for column, action in enumerate(point.actions):
            gather[index, column] = int(action)
    gather.setflags(write=False)
    return gather
