"""The feature encoder: every infoset as 136 floats, never a key id.

What is pinned here, and why:

- One card group is multi-hot, then rank counts, then suit counts, in that
  order, and an empty group is zeros — the net reads positions, so the order
  is the contract.
- The private half is the four groups end to end, per shape, in
  `private_keys` order, with no two keys of a shape sharing a row.
- The public half keeps the betting line's ORDER (bet-then-call differs from
  check-then-bet), the chips are read off `chip_state`, and the stage and seat
  are one-hot.
- The gather names, per point, exactly the head columns of that point's
  `legal_actions()`, padded with −1.
- `Layout` agrees with `PackedTable.whole_game()` on where every sampled key
  sits, forward (row → point, position) and back (column start → row), and
  the batch encoder equals the per-key reference on real keys.
- The module never imports torch: the default suite runs without it.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.cards import DECK, Card
from drawmaha_solver.minidrawmaha.enumeration import private_keys
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
    # In a fresh interpreter, so another test module that did import torch
    # cannot mask an import added here by mistake.
    probe = (
        "import sys; import drawmaha_solver.minidrawmaha.features; "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0

# ---------------------------------------------------------------------------
# The public half: draws, betting, chips, seat and stage
# ---------------------------------------------------------------------------

from drawmaha_solver.minidrawmaha.enumeration import public_decision_points  # noqa: E402
from drawmaha_solver.minidrawmaha.features import (  # noqa: E402
    HEAD_WIDTH,
    MAX_WIDTH,
    NO_ACTION,
    PUBLIC_WIDTH,
    legal_gather,
    public_features,
    public_row,
)
from drawmaha_solver.minidrawmaha.game import Action, DrawSignal, chip_state  # noqa: E402

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
        draws=(), betting=((Action.POT, Action.CHECK_CALL),), player=0, is_draw=False
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
        assert (gather[index, len(head) :] == NO_ACTION).all()
        if point.is_draw_decision:
            assert head == [3, 4, 5, 6]
        else:
            assert all(0 <= column < 3 for column in head)
    assert gather.max() == HEAD_WIDTH - 1
