# Deep CFR M1, PR 1: the feature encoder — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every mini-drawmaha infoset becomes a fixed 136-float vector of card, discard, board, draw, betting, chip and seat features — never a key id — built per private-key shape and per public point so that any batch of table rows can be encoded on the fly, plus the per-point gather from the net's 7-wide head to a ledger's columns, and a `deep` optional dependency group carrying torch.

**Architecture:** one new module, `minidrawmaha/features.py`, with no torch in it. Two cached tables — the private half per shape (970 / 10,170 / 100,400 rows × 92) and the public half per point (141 × 44) — and a `Layout` that turns a packed-table row number into (point, position) with the same arithmetic the enumerator uses, so a memory row needs only the key's row number. A per-key reference encoder, `encode(key)`, exists so tests can pin the batch path against it on real keys.

**Tech Stack:** Python 3.12+, NumPy; torch only as an optional extra in `pyproject.toml`.

**Spec:** `docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`, sections "Inputs" and "The network", and PR 1 of the PR table.

## Global Constraints

- Inputs are "card/discard/board/draw-count/betting/chip/seat features (136, in the spec), NEVER a per-key id or embedding".
- `PackedTable`, `enumeration.py`, `game.py` are graded, merged rung-3 code: extend around them, do not rewrite them. `features.py` imports from them and changes nothing in them.
- The default suite passes WITHOUT torch: `features.py` never imports torch; any torch test uses `pytest.importorskip("torch")`.
- Repo conventions: long explanatory module docstring, 75-hyphen section banners, one blank line between top-level defs, WHY-only comments, `from __future__ import annotations`.
- Row order is the enumerator's: `all_infosets()` = for each public point in walk order, that point's shape of `private_keys` in sorted order. The `Layout` must agree with `PackedTable.whole_game().row_of` on every key.
- The 7-wide head is indexed by `int(Action)`: fold 0, check/call 1, pot 2, throw-none 3, throw-low 4, throw-mid 5, throw-top 6.

## Review Focus

1. A key whose board has one card must get zeros for the board-2 group, not garbage: pinned in Task 1 (`test_card_group_of_nothing_is_zeros`) and Task 3 (`encode` on a round-1 key).
2. A row number past the table's end must raise, not silently index another point: pinned in Task 3 (`test_locate_rows_refuses_rows_past_the_end`).
3. Two keys in the same shape must never share a private row (otherwise the net sees two spots as one): pinned in Task 1 (`test_private_rows_are_distinct_within_a_shape`).
4. The betting slots must encode the line's ORDER, not just its multiset — `(pot, check)` and `(check, pot)` are different spots: pinned in Task 2 (`test_betting_slots_keep_the_order`).
5. The gather for a draw point must be exactly the four throw columns, and a 2-wide betting point must pad with −1: pinned in Task 2 (`test_legal_gather_matches_each_points_actions`).

---

### Task 1: card groups and the private half

**Files:**
- Create: `src/drawmaha_solver/minidrawmaha/features.py`
- Test: `tests/minidrawmaha/test_features.py`

**Interfaces:**
- Consumes: `cards.DECK`, `cards.N_RANKS`, `cards.N_SUITS`, `cards.Card`; `enumeration.private_keys(board_cards=, discards=)`.
- Produces: `CARD_GROUP = 23`, `PRIVATE_WIDTH = 92`, `card_group(cards: tuple[Card, ...]) -> np.ndarray` (23,) float32, `private_row(hole, discarded, board) -> np.ndarray` (92,) float32, `private_features(*, board_cards: int, discards: int) -> np.ndarray` (n, 92) float32 read-only, cached, rows in `private_keys` order.

- [ ] **Step 1: Write the failing tests**

```python
"""The feature encoder: every infoset as 136 floats, never a key id."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.cards import DECK, Card
from drawmaha_solver.minidrawmaha.enumeration import private_keys
from drawmaha_solver.minidrawmaha import features
from drawmaha_solver.minidrawmaha.features import (
    CARD_GROUP,
    PRIVATE_WIDTH,
    card_group,
    private_features,
    private_row,
)

# ---------------------------------------------------------------------------
# One group of cards: multi-hot, rank counts, suit counts
# ---------------------------------------------------------------------------

def test_card_group_is_multi_hot_then_rank_counts_then_suit_counts():
    pair_and_one = (Card(2, 0), Card(2, 1), Card(4, 1))
    got = card_group(pair_and_one)
    assert got.shape == (CARD_GROUP,) and got.dtype == np.float32
    multi_hot, ranks, suits = got[:15], got[15:20], got[20:23]
    assert [DECK[i] for i in np.flatnonzero(multi_hot)] == list(pair_and_one)
    assert ranks.tolist() == [0, 0, 2, 0, 1]
    assert suits.tolist() == [1, 2, 0]

def test_card_group_of_nothing_is_zeros():
    assert not card_group(()).any()

# ---------------------------------------------------------------------------
# The private half: hole, discard, board 1, board 2
# ---------------------------------------------------------------------------

def test_private_row_lays_the_four_groups_end_to_end():
    hole = (Card(0, 0), Card(1, 1), Card(3, 2))
    thrown = (Card(2, 0),)
    board = (Card(4, 0), Card(0, 1))
    row = private_row(hole, thrown, board)
    assert row.shape == (PRIVATE_WIDTH,)
    assert np.array_equal(row[0:23], card_group(hole))
    assert np.array_equal(row[23:46], card_group(thrown))
    assert np.array_equal(row[46:69], card_group(board[:1]))
    assert np.array_equal(row[69:92], card_group(board[1:]))

def test_private_row_before_the_second_board_card_has_zeros_there():
    row = private_row((Card(0, 0), Card(1, 1), Card(3, 2)), (), (Card(4, 0),))
    assert not row[23:46].any() and not row[69:92].any()

@pytest.mark.parametrize("shape", [(1, 0), (2, 0), (2, 1)])
def test_private_features_follow_private_keys_order(shape):
    table = private_features(board_cards=shape[0], discards=shape[1])
    keys = private_keys(board_cards=shape[0], discards=shape[1])
    assert table.shape == (len(keys), PRIVATE_WIDTH) and table.dtype == np.float32
    assert not table.flags.writeable
    for position in (0, len(keys) // 2, len(keys) - 1):
        assert np.array_equal(table[position], private_row(*keys[position]))

def test_private_rows_are_distinct_within_a_shape():
    table = private_features(board_cards=2, discards=1)
    assert len(np.unique(table, axis=0)) == len(table)

def test_the_encoder_never_imports_torch():
    assert "torch" not in sys.modules
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/minidrawmaha/test_features.py -q`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'drawmaha_solver.minidrawmaha.features'`.

- [ ] **Step 3: Write the module with the private half**

```python
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
    Action,
    DrawSignal,
    InfoSet,
    chip_state,
    STACK,
)

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/minidrawmaha/test_features.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/minidrawmaha/features.py tests/minidrawmaha/test_features.py
git commit -m "feat(rung4): the private half of the feature encoder — four card groups of 23 floats, per shape"
```

---

### Task 2: the public half and the legal gather

**Files:**
- Modify: `src/drawmaha_solver/minidrawmaha/features.py` (append after the private half)
- Test: `tests/minidrawmaha/test_features.py` (append)

**Interfaces:**
- Consumes: `enumeration.public_decision_points()` (141 `PublicPoint`s with `board_cards`, `draws`, `betting`, `player`, `actions`, `is_draw_decision`); `game.chip_state(betting)` (`.pot`, `.behind`), `game.STACK` (26).
- Produces: `PUBLIC_WIDTH = 44`, `public_row(*, draws, betting, player, is_draw) -> np.ndarray` (44,) float32, `public_features() -> np.ndarray` (141, 44) read-only cached, `legal_gather() -> np.ndarray` (141, 4) int8 read-only cached, −1 padded.

- [ ] **Step 1: Write the failing tests**

```python
from drawmaha_solver.minidrawmaha.enumeration import public_decision_points
from drawmaha_solver.minidrawmaha.game import Action, DrawSignal, chip_state
from drawmaha_solver.minidrawmaha.features import (
    HEAD_WIDTH,
    MAX_WIDTH,
    NO_ACTION,
    PUBLIC_WIDTH,
    legal_gather,
    public_features,
    public_row,
)

# ---------------------------------------------------------------------------
# The public half: draws, betting, chips, seat and stage
# ---------------------------------------------------------------------------

def test_public_row_after_a_bet_and_a_raise():
    # Ante 1 each (pot 2); P0 bets the pot (2), P1 raises (2 to call + 6 = 8):
    # pot 12, P0 has 23 behind, P1 has 17. P0 to act in round 1.
    betting = ((Action.POT, Action.POT),)
    row = public_row(draws=(), betting=betting, player=0, is_draw=False)
    assert row.shape == (PUBLIC_WIDTH,) and row.dtype == np.float32
    draws, lines, chips, seat_stage = row[:6], row[6:36], row[36:39], row[39:44]
    assert draws.tolist() == [1, 0, 0, 1, 0, 0]  # neither seat has drawn yet
    slots = lines.reshape(2, 5, 3)
    assert slots[0, 0].tolist() == [0, 0, 1] and slots[0, 1].tolist() == [0, 0, 1]
    assert not slots[0, 2:].any() and not slots[1].any()
    assert np.allclose(chips, [12 / 26, 23 / 26, 17 / 26])
    assert seat_stage.tolist() == [1, 0, 1, 0, 0]

def test_betting_slots_keep_the_order():
    bet_then_call = public_row(
        draws=(), betting=((Action.POT, Action.CHECK_CALL, ), ()), player=0, is_draw=False
    )
    check_then_bet = public_row(
        draws=(), betting=((Action.CHECK_CALL, Action.POT),), player=0, is_draw=False
    )
    assert not np.array_equal(bet_then_call[6:36], check_then_bet[6:36])

def test_public_row_at_p1s_draw_after_p0_stood_pat():
    row = public_row(
        draws=(DrawSignal(0),),
        betting=((Action.CHECK_CALL, Action.CHECK_CALL),),
        player=1,
        is_draw=True,
    )
    assert row[:6].tolist() == [0, 1, 0, 1, 0, 0]
    assert row[39:44].tolist() == [0, 1, 0, 1, 0]

def test_public_row_in_round_two_marks_the_stage_and_both_draws():
    row = public_row(
        draws=(DrawSignal(1), DrawSignal(0)),
        betting=((Action.CHECK_CALL, Action.CHECK_CALL), (Action.POT,)),
        player=1,
        is_draw=False,
    )
    assert row[:6].tolist() == [0, 0, 1, 0, 1, 0]
    assert row[39:44].tolist() == [0, 1, 0, 0, 1]
    assert row[6:36].reshape(2, 5, 3)[1, 0].tolist() == [0, 0, 1]

def test_public_features_follow_the_public_points():
    table = public_features()
    points = public_decision_points()
    assert table.shape == (len(points), PUBLIC_WIDTH) and not table.flags.writeable
    for index in (0, 70, len(points) - 1):
        point = points[index]
        expected = public_row(
            draws=point.draws,
            betting=point.betting,
            player=point.player,
            is_draw=point.is_draw_decision,
        )
        assert np.array_equal(table[index], expected)
        assert np.isclose(table[index, 36], chip_state(point.betting).pot / 26)

def test_public_rows_are_distinct_across_points():
    assert len(np.unique(public_features(), axis=0)) == len(public_decision_points())

# ---------------------------------------------------------------------------
# The gather: a ledger's columns inside the 7-wide head
# ---------------------------------------------------------------------------

def test_legal_gather_matches_each_points_actions():
    gather = legal_gather()
    points = public_decision_points()
    assert gather.shape == (len(points), MAX_WIDTH) and gather.dtype == np.int8
    assert not gather.flags.writeable
    for index, point in enumerate(points):
        head = [int(action) for action in point.actions]
        assert gather[index, : len(head)].tolist() == head
        assert (gather[index, len(head):] == NO_ACTION).all()
        if point.is_draw_decision:
            assert head == [3, 4, 5, 6]
        else:
            assert all(0 <= column < 3 for column in head)
    assert gather.max() == HEAD_WIDTH - 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/minidrawmaha/test_features.py -q`
Expected: FAIL at import: `ImportError: cannot import name 'public_row'`.

- [ ] **Step 3: Append the public half and the gather**

```python
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
    row[at : at + CHIP_WIDTH] = (chips.pot / STACK, chips.behind[0] / STACK, chips.behind[1] / STACK)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/minidrawmaha/test_features.py -q`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/minidrawmaha/features.py tests/minidrawmaha/test_features.py
git commit -m "feat(rung4): the public half of the feature encoder and the per-point legal gather"
```

---

### Task 3: the reference encoder and the row layout

**Files:**
- Modify: `src/drawmaha_solver/minidrawmaha/features.py` (append)
- Test: `tests/minidrawmaha/test_features.py` (append)

**Interfaces:**
- Consumes: `PackedTable.whole_game()`, `.row_of(key)`, `.slot_of(key)`, `.widths()`; `enumeration.all_infosets()`.
- Produces: `encode(key: InfoSet) -> np.ndarray` (136,) float32; `class Layout` with `points`, `row_offset` (142,) int64, `column_offset` (142,) int64, `width` (141,) int64, `shape_of_point`, `locate_rows(rows) -> (point, position)` int64 arrays, `rows_of_columns(columns) -> rows`, `features(rows) -> np.ndarray` (n, 136) float32; `layout() -> Layout` cached.

- [ ] **Step 1: Write the failing tests**

```python
from drawmaha_solver.minidrawmaha.enumeration import all_infosets
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.features import FEATURE_WIDTH, Layout, encode, layout

# ---------------------------------------------------------------------------
# The reference encoder and the row layout
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sampled_keys():
    # 4,000 keys spread over the whole enumeration, every public point touched.
    return [key for index, key in enumerate(all_infosets()) if index % 1555 == 0]

def test_encode_is_the_private_row_beside_the_public_row(sampled_keys):
    key = sampled_keys[3]
    vector = encode(key)
    assert vector.shape == (FEATURE_WIDTH,) and vector.dtype == np.float32
    assert np.array_equal(vector[:92], private_row(key.hole, key.discarded, key.board))
    assert np.array_equal(
        vector[92:],
        public_row(
            draws=key.draws, betting=key.betting, player=key.player, is_draw=key.is_draw_decision()
        ),
    )

def test_layout_counts_the_whole_game():
    built = layout()
    table = PackedTable.whole_game()
    assert isinstance(built, Layout)
    assert built.row_offset[-1] == len(table) == 6_220_050
    assert built.column_offset[-1] == int(table.widths().sum())
    assert len(built.points) == 141

def test_layout_locates_every_sampled_row_where_the_table_puts_it(sampled_keys):
    built, table = layout(), PackedTable.whole_game()
    rows = np.array([table.row_of(key) for key in sampled_keys])
    point, position = built.locate_rows(rows)
    for key, p, q in zip(sampled_keys, point, position, strict=True):
        at = built.points[p]
        assert (at.board_cards, at.draws, at.betting) == (len(key.board), key.draws, key.betting)
        shape = private_keys(board_cards=at.board_cards, discards=at.discards)
        assert shape[q] == (key.hole, key.discarded, key.board)

def test_layout_recovers_rows_from_column_starts(sampled_keys):
    built, table = layout(), PackedTable.whole_game()
    slots = [table.slot_of(key) for key in sampled_keys]
    starts = np.array([start for _, start, _ in slots])
    rows = np.array([row for row, _, _ in slots])
    assert np.array_equal(built.rows_of_columns(starts), rows)

def test_features_of_rows_equal_the_reference_encoder(sampled_keys):
    built, table = layout(), PackedTable.whole_game()
    rows = np.array([table.row_of(key) for key in sampled_keys])
    batch = built.features(rows)
    assert batch.shape == (len(rows), FEATURE_WIDTH) and batch.dtype == np.float32
    expected = np.stack([encode(key) for key in sampled_keys])
    assert np.array_equal(batch, expected)

def test_locate_rows_refuses_rows_past_the_end():
    built = layout()
    with pytest.raises(IndexError):
        built.locate_rows(np.array([built.row_offset[-1]]))
    with pytest.raises(IndexError):
        built.locate_rows(np.array([-1]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/minidrawmaha/test_features.py -q`
Expected: FAIL at import: `ImportError: cannot import name 'Layout'`.

- [ ] **Step 3: Append the encoder and the layout**

```python
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

@dataclass(frozen=True, slots=True)
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
    private: np.ndarray  # (111,540, 92) float32: the three shapes end to end
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
        """The row that owns each column start — what a walk's strategy record carries."""
        columns = np.asarray(columns, dtype=np.int64)
        point = np.searchsorted(self.column_offset, columns, side="right") - 1
        return self.row_offset[point] + (columns - self.column_offset[point]) // self.width[point]

    def features(self, rows: np.ndarray) -> np.ndarray:
        """The 136 floats of each row, gathered from the two halves; (n, 136) float32."""
        point, position = self.locate_rows(rows)
        private = self.private[self.private_offset[point] + position]
        return np.concatenate([private, self.public[point]], axis=1)

@cache
def layout() -> Layout:
    """The whole game's layout, built once from the enumerator's two halves."""
    points = public_decision_points()
    shapes = sorted({(point.board_cards, point.discards) for point in points})
    private = np.concatenate(
        [private_features(board_cards=b, discards=d) for b, d in shapes]
    )
    private.setflags(write=False)
    shape_start = dict(
        zip(shapes, np.cumsum([0] + [len(private_features(board_cards=b, discards=d)) for b, d in shapes[:-1]]), strict=True)
    )
    sizes = [len(private_features(board_cards=p.board_cards, discards=p.discards)) for p in points]
    widths = [point.width for point in points]
    return Layout(
        points=points,
        row_offset=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        column_offset=np.concatenate(
            [[0], np.cumsum([size * width for size, width in zip(sizes, widths, strict=True)])]
        ).astype(np.int64),
        width=np.asarray(widths, dtype=np.int64),
        private_offset=np.asarray(
            [shape_start[(p.board_cards, p.discards)] for p in points], dtype=np.int64
        ),
        private=private,
        public=public_features(),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/minidrawmaha/test_features.py -q`
Expected: 22 passed (the whole-game table costs a second or two).

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/minidrawmaha/features.py tests/minidrawmaha/test_features.py
git commit -m "feat(rung4): encode(key) and the row Layout — a batch of rows to features on the fly, pinned to the packed table"
```

---

### Task 4: torch as an optional `deep` extra

**Files:**
- Modify: `pyproject.toml` (add `[project.optional-dependencies]`), `uv.lock` (regenerated)
- Test: `tests/minidrawmaha/test_features.py` (append)

**Interfaces:**
- Produces: `uv sync --extra deep` installs torch; `uv run pytest` without it still passes.

- [ ] **Step 1: Write the failing test**

```python
import tomllib
from pathlib import Path

def test_torch_is_an_optional_extra_named_deep():
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    deep = pyproject["project"]["optional-dependencies"]["deep"]
    assert any(spec.startswith("torch") for spec in deep)
    assert not any(spec.startswith("torch") for spec in pyproject["project"]["dependencies"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/minidrawmaha/test_features.py::test_torch_is_an_optional_extra_named_deep -q`
Expected: FAIL with `KeyError: 'optional-dependencies'`.

- [ ] **Step 3: Add the extra and relock**

In `pyproject.toml`, after `dependencies = [...]`:

```toml
[project.optional-dependencies]
# Rung 4's nets. Optional so the tabular rungs, and the default test suite,
# never install torch; `uv sync --extra deep` brings it in.
deep = [
    "torch>=2.4",
]
```

Then: `uv lock` (regenerates `uv.lock` with torch resolved but not installed).

- [ ] **Step 4: Run the test and the import check**

Run: `uv run pytest tests/minidrawmaha/test_features.py -q`
Expected: 23 passed; `uv run python -c "import torch"` still fails (not installed by default); `uv run --extra deep python -c "import torch; print(torch.__version__)"` prints a version.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/minidrawmaha/test_features.py
git commit -m "build: torch as the optional 'deep' extra, never a default dependency"
```

---

### Task 5: the whole suite, and the PR

- [ ] **Step 1:** `uv run pytest -q > .context/pytest-pr1.log 2>&1 &` and wait; expect the baseline's 1244 + 23 passed, 14 skipped.
- [ ] **Step 2:** `/open-pr` on this branch against `main`, title `feat(rung4): the 136-feature encoder for mini-drawmaha, and torch as an optional extra`, body from the spec's "Inputs" section, then drive it to merge-ready.
- [ ] **Step 3:** The walkthrough PDF under `/Users/alec/Desktop/Claude/poker/code-walkthroughs/pr52-features/` in the SPEC.md format, and the intuition-first chat explanation.
