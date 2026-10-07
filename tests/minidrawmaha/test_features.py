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
import tomllib
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.cards import DECK, Card
from drawmaha_solver.minidrawmaha.enumeration import (
    all_infosets,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.features import (
    CARD_GROUP,
    FEATURE_WIDTH,
    HEAD_WIDTH,
    LINE_SLOTS,
    MAX_WIDTH,
    NO_ACTION,
    PRIVATE_WIDTH,
    PUBLIC_WIDTH,
    Layout,
    card_group,
    encode,
    layout,
    legal_gather,
    private_features,
    private_row,
    public_features,
    public_row,
)
from drawmaha_solver.minidrawmaha.game import Action, DrawSignal, chip_state
from drawmaha_solver.minidrawmaha.packed_table import PackedTable

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
    # Every row for the two small shapes; the 100,400-row shape is sampled.
    positions = range(len(keys)) if len(keys) < 20_000 else (0, len(keys) // 2, len(keys) - 1)
    for position in positions:
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

def test_the_longest_line_in_the_game_fills_the_slots_exactly():
    # The widths are pinned to the game, not assumed: a longer line would
    # spill into the next round's slots, so the encoder refuses it instead.
    points = public_decision_points()
    assert max(len(line) for point in points for line in point.betting) == LINE_SLOTS
    assert max(len(point.betting) for point in points) == 2
    assert max(point.width for point in points) == MAX_WIDTH
    # The stack, not a raise cap, is what ends the longest line: it is
    # check, bet, raise, raise, call, and the last raise is the all-in.
    longest = {line for point in points for line in point.betting if len(line) == LINE_SLOTS}
    cpppc = (Action.CHECK_CALL, Action.POT, Action.POT, Action.POT, Action.CHECK_CALL)
    assert longest == {cpppc}
    assert chip_state((cpppc[:4],)).in_round == (8, 25)
    assert chip_state((cpppc,)).behind == (0, 0)
    six = (cpppc + (Action.CHECK_CALL,),)
    with pytest.raises(ValueError):
        public_row(draws=(), betting=six, player=0, is_draw=False)

def test_public_row_refuses_an_empty_betting_history():
    with pytest.raises(ValueError):
        public_row(draws=(), betting=(), player=0, is_draw=False)

def test_public_row_refuses_a_third_draw_signal():
    three = (DrawSignal(0), DrawSignal(0), DrawSignal(0))
    with pytest.raises(ValueError):
        public_row(draws=three, betting=((Action.CHECK_CALL,),), player=0, is_draw=False)

def test_the_widths_are_the_numbers_the_comments_say():
    # The widths are derived from the game's constants; the numbers in the
    # comments beside them, and the slices the tests above read, are these.
    assert (CARD_GROUP, PRIVATE_WIDTH, PUBLIC_WIDTH, FEATURE_WIDTH) == (23, 92, 44, 136)
    assert (LINE_SLOTS, MAX_WIDTH, HEAD_WIDTH) == (5, 4, 7)

def test_the_stage_follows_the_draws_at_every_point():
    # Round 1 is before either draw, round 2 after both; the stage is read
    # off the line count, so this pins that the two agree everywhere.
    table = public_features()
    for index, point in enumerate(public_decision_points()):
        stage = table[index, 41:44].tolist()
        if point.is_draw_decision:
            assert stage == [0, 1, 0]
        elif len(point.draws) == 0:
            assert stage == [1, 0, 0]
        else:
            assert len(point.draws) == 2 and stage == [0, 0, 1]

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

# ---------------------------------------------------------------------------
# The reference encoder and the row layout
# ---------------------------------------------------------------------------

def point_of(key):
    return (len(key.board), key.draws, key.betting)

@pytest.fixture(scope="module")
def sampled_keys():
    # 4,000 keys spread over the whole enumeration, plus the first key of
    # every public point, so the small points the stride skips are covered.
    seen = set()
    keys = []
    for index, key in enumerate(all_infosets()):
        if index % 1555 == 0 or point_of(key) not in seen:
            seen.add(point_of(key))
            keys.append(key)
    return keys

@pytest.fixture(scope="module")
def whole_table():
    return PackedTable.whole_game()

def test_sampled_keys_touch_every_public_point(sampled_keys):
    assert len({point_of(key) for key in sampled_keys}) == len(public_decision_points()) == 141

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

def test_layout_counts_the_whole_game(whole_table):
    built = layout()
    assert isinstance(built, Layout)
    assert built.row_offset[-1] == len(whole_table) == 6_220_050
    assert built.column_offset[-1] == int(whole_table.widths().sum())
    assert len(built.points) == 141

def test_layout_locates_every_sampled_row_where_the_table_puts_it(sampled_keys, whole_table):
    built = layout()
    rows = np.array([whole_table.row_of(key) for key in sampled_keys])
    point, position = built.locate_rows(rows)
    for key, p, q in zip(sampled_keys, point, position, strict=True):
        at = built.points[p]
        assert (at.board_cards, at.draws, at.betting) == (len(key.board), key.draws, key.betting)
        shape = private_keys(board_cards=at.board_cards, discards=at.discards)
        assert shape[q] == (key.hole, key.discarded, key.board)

def test_layout_recovers_rows_from_column_starts(sampled_keys, whole_table):
    built = layout()
    slots = [whole_table.slot_of(key) for key in sampled_keys]
    starts = np.array([start for _, start, _ in slots])
    rows = np.array([row for row, _, _ in slots])
    assert np.array_equal(built.rows_of_columns(starts), rows)

def test_features_of_rows_equal_the_reference_encoder(sampled_keys, whole_table):
    built = layout()
    rows = np.array([whole_table.row_of(key) for key in sampled_keys])
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

def test_rows_of_columns_refuses_columns_past_the_end():
    built = layout()
    with pytest.raises(IndexError):
        built.rows_of_columns(np.array([built.column_offset[-1]]))
    with pytest.raises(IndexError):
        built.rows_of_columns(np.array([-1]))

# ---------------------------------------------------------------------------
# Torch is an optional extra
# ---------------------------------------------------------------------------

def test_torch_is_an_optional_extra_named_deep():
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    deep = pyproject["project"]["optional-dependencies"]["deep"]
    assert any(spec.startswith("torch") for spec in deep)
    assert not any(spec.startswith("torch") for spec in pyproject["project"]["dependencies"])
