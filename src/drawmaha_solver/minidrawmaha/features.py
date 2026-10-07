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
redundant with the multi-hot and deliberately so: they are a hand-built,
unlearned stand-in for Brown et al.'s rank and suit embeddings (which Deep
CFR learns and sums per card), and they hand the net "any pair" as a feature
a single unit can read directly, where the multi-hot alone would need a
nonlinearity to find it card by card. Suits are already canonical in the key,
so suit 0 means the same thing in every row.

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

from dataclasses import dataclass
from functools import cache

import numpy as np

from drawmaha_solver.minidrawmaha.cards import DECK, N_RANKS, N_SUITS, Card
from drawmaha_solver.minidrawmaha.enumeration import (
    PublicPoint,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.game import (
    BETTING_ACTIONS,
    DRAW_ACTIONS,
    N_ROUNDS,
    STACK,
    THROW_CAP,
    Action,
    DrawSignal,
    InfoSet,
    chip_state,
)

# ---------------------------------------------------------------------------
# The widths
# ---------------------------------------------------------------------------

# One group of cards: which of the 15, how many of each rank, how many of each suit.
CARD_GROUP = len(DECK) + N_RANKS + N_SUITS  # 23
# Hole, discard, board card 1, board card 2.
PRIVATE_WIDTH = 4 * CARD_GROUP  # 92

# Per seat: has not drawn yet, then one state per throw count 0..THROW_CAP.
_DRAW_STATES = THROW_CAP + 2  # 3
DRAW_WIDTH = 2 * _DRAW_STATES  # 6
# The longest betting line is five actions — check, bet, raise, raise (the
# all-in), call — which the stack fixes, not a raise cap. Each pot-sized bet
# or raise puts the actor's own in-round commitment at 2, then 8, then what
# would be 26; with 25 behind the ante the third truncates to the shove and
# nothing deeper exists (`chip_state().in_round` reads 2, 8, 25 along the
# line; the test pins it). `public_row` refuses a longer line rather than
# spilling it into the next round's slots.
LINE_SLOTS = 5
_BETTING_ACTIONS = len(BETTING_ACTIONS)  # 3
BETTING_WIDTH = N_ROUNDS * LINE_SLOTS * _BETTING_ACTIONS  # 30
# Pot, P0's chips behind, P1's chips behind, each over the stack.
CHIP_WIDTH = 3
# Seat to act (2) and stage: round 1, the draw, round 2 (3).
SEAT_STAGE_WIDTH = 5
PUBLIC_WIDTH = DRAW_WIDTH + BETTING_WIDTH + CHIP_WIDTH + SEAT_STAGE_WIDTH  # 44

FEATURE_WIDTH = PRIVATE_WIDTH + PUBLIC_WIDTH  # 136

# The net's head: one output per Action, indexed by the Action's value.
HEAD_WIDTH = len(Action)  # 7
# No ledger is wider than the draw's four throws.
MAX_WIDTH = max(len(BETTING_ACTIONS), len(DRAW_ACTIONS))  # 4
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

    Draw signals: per seat, one-hot over {not drawn yet, then one state per
    throw count up to `THROW_CAP`}. Betting: per round, `LINE_SLOTS` slots,
    each a one-hot over `BETTING_ACTIONS` for the action in that slot and
    zeros past the line's end — so the ORDER of the line is in the vector,
    not just which actions it holds. Chips: the pot and each seat's chips
    behind, over the stack. Stage: round 1, the draw, round 2.

    Refuses, rather than corrupting the row, a line longer than the slots,
    a third draw signal, or an empty betting history: round 1's line exists
    from the first deal, so a key without one is not a key of this game.
    """
    if not betting:
        raise ValueError("a public point has at least round 1's betting line, got none")
    if any(len(line) > LINE_SLOTS for line in betting):
        raise ValueError(f"a betting line is longer than {LINE_SLOTS} actions: {betting}")
    if len(draws) > 2:
        raise ValueError(f"two seats draw, got {len(draws)} draw signals")
    row = np.zeros(PUBLIC_WIDTH, dtype=np.float32)
    # The draws come in seat order, P0 then P1, so a missing signal is
    # always the later seat's and means "not drawn yet"; `DrawSignal` has
    # already bounded the count to THROW_CAP.
    for seat in (0, 1):
        state = 1 + draws[seat].count if len(draws) > seat else 0
        row[seat * _DRAW_STATES + state] = 1.0
    at = DRAW_WIDTH
    for round_index in range(N_ROUNDS):
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
    # The line count says which round a betting decision is in: round 2's
    # line is opened (empty) the moment the draw ends, so one line means
    # round 1 and two means round 2, and the draw sits between with one line.
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

# ---------------------------------------------------------------------------
# One key, for reference
# ---------------------------------------------------------------------------

def encode(key: InfoSet) -> np.ndarray:
    """One infoset's 136 floats, built from the key alone: what the batch path must equal."""
    return np.concatenate(
        [
            private_row(key.hole, key.discarded, key.board),
            public_row(
                draws=key.draws,
                betting=key.betting,
                player=key.player,
                is_draw=key.is_draw_decision(),
            ),
        ]
    )

# ---------------------------------------------------------------------------
# From a row number to its two halves
# ---------------------------------------------------------------------------

# eq=False: the fields are arrays, so the generated __eq__ would be ambiguous;
# there is one layout per process and it is compared by identity.
@dataclass(frozen=True, slots=True, eq=False)
class Layout:
    """Where every row of the whole-game table sits: which point, which position.

    The same arithmetic `PackedTable`'s whole-game index runs forward, run
    backward: rows are laid out point by point in walk order, each point's
    rows in its shape's `private_keys` order, so a row's point is the last
    `row_offset` at or below it and its position is the remainder. Columns
    the same way, each row `width[point]` wide. The private table is the
    three shapes end to end (`private_offset[point]` is where a point's
    shape starts in it), so a batch is one gather on each half.
    """

    points: tuple[PublicPoint, ...]
    row_offset: np.ndarray  # (142,) int64: where each point's rows start, then the end
    column_offset: np.ndarray  # (142,) int64: where each point's columns start, then the end
    width: np.ndarray  # (141,) int64
    private_offset: np.ndarray  # (141,) int64: where each point's shape starts in `private`
    private: np.ndarray  # (sum of the shapes' key counts, 92) float32: the three shapes end to end
    public: np.ndarray  # (141, 44) float32

    def locate_rows(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each row's public point and its position in that point's shape.

        Raises `IndexError` for a row outside the table: past the end would
        land on a phantom 142nd point, and a negative one on the last point's
        tail, both silently.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= self.row_offset[-1]):
            raise IndexError(f"a row is outside the table's {self.row_offset[-1]:,} rows")
        point = np.searchsorted(self.row_offset, rows, side="right") - 1
        return point, rows - self.row_offset[point]

    def rows_of_columns(self, columns: np.ndarray) -> np.ndarray:
        """The row that owns each column start — what a walk's strategy record carries.

        Raises `IndexError` for a column outside the table, for the same
        reason `locate_rows` does: past the end would read a phantom point's
        width, and a negative column would wrap onto the last point's tail.
        """
        columns = np.asarray(columns, dtype=np.int64)
        if columns.size and (columns.min() < 0 or columns.max() >= self.column_offset[-1]):
            raise IndexError(f"a column is outside the table's {self.column_offset[-1]:,} columns")
        point = np.searchsorted(self.column_offset, columns, side="right") - 1
        return self.row_offset[point] + (columns - self.column_offset[point]) // self.width[point]

    def features(self, rows: np.ndarray) -> np.ndarray:
        """The 136 floats of each row, gathered from the two halves; (n, 136) float32.

        `rows` is a 1-D array of integer row numbers, as `locate_rows` takes.
        """
        point, position = self.locate_rows(rows)
        private = self.private[self.private_offset[point] + position]
        return np.concatenate([private, self.public[point]], axis=1)

@cache
def layout() -> Layout:
    """The whole game's layout, built once from the enumerator's two halves."""
    points = public_decision_points()
    shapes = sorted({(point.board_cards, point.discards) for point in points})
    tables = {shape: private_features(board_cards=shape[0], discards=shape[1]) for shape in shapes}
    private = np.concatenate([tables[shape] for shape in shapes])
    private.setflags(write=False)
    shape_start = {}
    start = 0
    for shape in shapes:
        shape_start[shape] = start
        start += len(tables[shape])
    sizes = [len(tables[(point.board_cards, point.discards)]) for point in points]
    widths = [point.width for point in points]
    return Layout(
        points=points,
        row_offset=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        column_offset=np.concatenate(
            [[0], np.cumsum([size * width for size, width in zip(sizes, widths, strict=True)])]
        ).astype(np.int64),
        width=np.asarray(widths, dtype=np.int64),
        private_offset=np.asarray(
            [shape_start[(point.board_cards, point.discards)] for point in points], dtype=np.int64
        ),
        private=private,
        public=public_features(),
    )
