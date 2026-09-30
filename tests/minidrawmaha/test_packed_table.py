"""The packed table: 6.2 million ledgers as four flat arrays, pinned cheaply.

The object table could only be tested in slices, because 6.2 million
`RegretMatcher`s cost gigabytes and half a minute. The packed table costs a few
hundred megabytes of zeros the kernel hands out lazily, and its index is built
from the 141 public points and the four cached private-key shapes, so the WHOLE
table is allocatable in a fraction of a second and most of this file uses it.
What stays gated behind `MINIDRAWMAHA_FULL_TABLE=1` is walking all 6.2 million
keys through `all_infosets()`, which is the slow half: the keys, not the table.

Three claims are what the module has to make good on:

* **every key finds its own row, and the rows are `all_infosets()` order** —
  so a checkpoint's flat arrays are the table's flat arrays, and two lookups
  never share a row;
* **a ledger is a window, not a copy** — an `update` through `table[key]` lands
  in the flat buffers, and nothing else does;
* **the learner cannot tell the difference** — a fixed-seed run on the packed
  table and on a dict of `RegretMatcher`s bank identical numbers, bit for bit.
"""

import os
from itertools import islice

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.enumeration import (
    all_infosets,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.game import Action, InfoSet, random_deal
from drawmaha_solver.minidrawmaha.mccfr import new_solve, train
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.regret_matching import RegretMatcher

FULL_RUN = os.environ.get("MINIDRAWMAHA_FULL_TABLE") == "1"

TOTAL_LEDGERS = 6_220_050
TOTAL_WIDTH = 2 * 4_426_680 + 3 * 1_773_000 + 4 * 20_370

P = Action.POT

def key_at(point, index: int) -> InfoSet:
    hole, thrown, board = private_keys(
        board_cards=point.board_cards, discards=point.discards
    )[index]
    return InfoSet(
        player=point.player,
        hole=hole,
        discarded=thrown,
        board=board,
        draws=point.draws,
        betting=point.betting,
    )

def keys_at(point, count: int) -> list[InfoSet]:
    return [key_at(point, index) for index in range(count)]

class ObjectTable(dict):
    """The table this rung replaced: a dict that grows a `RegretMatcher` per key.

    The walk's reference implementation for the bit-for-bit gate below. Grows
    on first touch so it never pays for the 6.2 million keys a short run does
    not reach.
    """

    def __missing__(self, infoset: InfoSet) -> RegretMatcher:
        ledger = self[infoset] = RegretMatcher(len(infoset.legal_actions()))
        return ledger

# ---------------------------------------------------------------------------
# The whole game as one table, and where each key lives in it
# ---------------------------------------------------------------------------

def test_the_whole_game_packs_into_two_arrays_of_the_censused_width():
    table = PackedTable.whole_game()
    assert len(table) == TOTAL_LEDGERS
    assert table.cumulative_regret.shape == table.strategy_sum.shape == (TOTAL_WIDTH,)
    assert table.cumulative_regret.dtype == table.strategy_sum.dtype == np.float64
    assert table.stamp.shape == (TOTAL_LEDGERS,) and table.stamp.dtype == np.int64
    assert table.extra_sums.shape == (0, TOTAL_WIDTH)

def test_the_first_keys_of_the_enumerator_sit_in_the_first_rows():
    # Row k of the table is key k of `all_infosets()`, and its columns start
    # where the k − 1 ledgers before it end: the checkpoint's flat layout.
    table = PackedTable.whole_game()
    prefix = list(islice(all_infosets(), 2_000))
    assert [table.row_of(key) for key in prefix] == list(range(2_000))
    widths = [len(key.legal_actions()) for key in prefix]
    assert table.widths()[:2_000].tolist() == widths

def test_the_last_key_of_the_game_sits_in_the_last_row():
    # Reached by arithmetic, not by streaming 6.2 million keys: the last point
    # and the last private key of its shape.
    table = PackedTable.whole_game()
    last_point = public_decision_points()[-1]
    held = len(
        private_keys(board_cards=last_point.board_cards, discards=last_point.discards)
    )
    assert table.row_of(key_at(last_point, held - 1)) == TOTAL_LEDGERS - 1

def test_rows_are_public_point_offset_plus_private_key_position():
    # The index in the open: a key's row is the number of ledgers standing on
    # earlier points, plus its position in its shape's sorted private keys.
    table = PackedTable.whole_game()
    offset = 0
    for point in public_decision_points():
        shape = private_keys(board_cards=point.board_cards, discards=point.discards)
        for index in (0, 17, len(shape) - 1):
            assert table.row_of(key_at(point, index)) == offset + index
        offset += len(shape)
    assert offset == TOTAL_LEDGERS

def test_a_key_of_a_position_the_game_does_not_have_raises_key_error():
    # A betting line no rule produces: `InfoSet` accepts it (it validates the
    # cards and the actor, not the line), so the table is the last line of
    # defence against a walk that reached a position the census lacks.
    table = PackedTable.whole_game()
    real = key_at(public_decision_points()[0], 0)
    impossible = InfoSet(
        player=real.player,
        hole=real.hole,
        discarded=real.discarded,
        board=real.board,
        draws=real.draws,
        betting=((P, P, P, P, P, P),),
    )
    with pytest.raises(KeyError):
        table[impossible]
    assert impossible not in table

def test_iteration_is_the_enumerator_in_its_own_order():
    table = PackedTable.whole_game()
    assert list(islice(iter(table), 500)) == list(islice(all_infosets(), 500))

# ---------------------------------------------------------------------------
# A ledger is a window into the table
# ---------------------------------------------------------------------------

def test_an_update_through_a_ledger_lands_in_the_flat_buffers():
    table = PackedTable.whole_game()
    point = public_decision_points()[3]
    key = key_at(point, 5)
    row, width = table.row_of(key), point.width
    start = int(table.widths()[:row].sum())
    table[key].update(np.arange(width, dtype=float), regret_weight=1.0, strategy_weight=2.0)
    # Only this ledger's columns moved: its regret is u − mean(u) at a uniform
    # sigma, its strategy sum is 2 · (1/width), and everything else is zero.
    assert np.array_equal(
        table.cumulative_regret[start : start + width],
        np.arange(width) - np.arange(width).mean(),
    )
    assert np.array_equal(table.strategy_sum[start : start + width], np.full(width, 2 / width))
    assert np.count_nonzero(table.cumulative_regret) == np.count_nonzero(
        np.arange(width) - np.arange(width).mean()
    )
    assert np.count_nonzero(table.strategy_sum) == width

def test_the_ledger_is_built_over_the_table_and_reads_back_what_was_written():
    table = PackedTable.whole_game(extra_averages=2)
    key = key_at(public_decision_points()[10], 42)
    ledger = table[key]
    ledger.cumulative_regret[:] = [1.0, 2.0, 3.0][: ledger.n_actions]
    ledger.extra_sums[1, :] = 7.0
    ledger.stamp[0] = 99
    again = table[key]
    assert again is not ledger  # a fresh window each time, over the same cells
    assert np.array_equal(again.cumulative_regret, ledger.cumulative_regret)
    assert np.all(again.extra_sums[1] == 7.0) and np.all(again.extra_sums[0] == 0.0)
    assert again.stamp[0] == 99
    assert table.stamp[table.row_of(key)] == 99

def test_rebinding_a_slot_does_not_reach_the_table():
    # The contract's one rule, seen from the table's side: `ledger.x = array`
    # replaces the window and leaves the table untouched; `ledger.x[:] = ...`
    # writes through. A writer that rebinds is silently training nothing.
    table = PackedTable.whole_game()
    key = key_at(public_decision_points()[0], 0)
    ledger = table[key]
    ledger.strategy_sum = np.array([5.0, 5.0])
    assert not table[key].strategy_sum.any()
    table[key].strategy_sum[:] = [5.0, 5.0]
    assert table[key].strategy_sum.tolist() == [5.0, 5.0]

def test_extra_averaging_rows_are_a_view_as_wide_as_the_ledger():
    table = PackedTable.whole_game(extra_averages=3)
    assert table.extra_sums.shape == (3, TOTAL_WIDTH)
    draw = next(point for point in public_decision_points() if point.is_draw_decision)
    assert table[key_at(draw, 0)].extra_sums.shape == (3, 4)
    assert table[key_at(public_decision_points()[0], 0)].extra_sums.shape == (3, 2)

def test_values_and_items_walk_the_rows_in_order_without_a_lookup_each():
    table = PackedTable.listed(keys_at(public_decision_points()[0], 40))
    for row, ((key, ledger), value) in enumerate(zip(table.items(), table.values(), strict=True)):
        assert table.row_of(key) == row
        assert np.shares_memory(ledger.cumulative_regret, table[key].cumulative_regret)
        assert np.shares_memory(value.strategy_sum, table[key].strategy_sum)

# ---------------------------------------------------------------------------
# A listed slice: the same store over the keys a test or a readout names
# ---------------------------------------------------------------------------

def test_a_listed_table_holds_exactly_its_keys_in_the_order_given():
    points = public_decision_points()
    keys = [key_at(points[7], 3), key_at(points[0], 0), key_at(points[140], 9)]
    table = PackedTable.listed(keys)
    assert list(table) == keys
    assert [table.row_of(key) for key in keys] == [0, 1, 2]
    assert table.widths().tolist() == [len(key.legal_actions()) for key in keys]
    assert len(table.cumulative_regret) == sum(table.widths())

def test_a_listed_table_refuses_a_key_handed_in_twice():
    key = key_at(public_decision_points()[0], 0)
    with pytest.raises(ValueError, match="twice"):
        PackedTable.listed([key, key])

def test_a_listed_table_raises_on_a_key_it_was_not_given():
    points = public_decision_points()
    table = PackedTable.listed(keys_at(points[0], 10))
    with pytest.raises(KeyError):
        table[key_at(points[1], 0)]

def test_a_negative_or_fractional_number_of_extra_averages_is_refused():
    # Caught by name here rather than left to NumPy, whose "negative dimensions
    # are not allowed" points at an array shape and not at the argument.
    with pytest.raises(ValueError, match="extra_averages"):
        PackedTable.whole_game(extra_averages=-1)
    with pytest.raises(ValueError, match="extra_averages"):
        PackedTable.listed([], extra_averages=1.5)

def test_a_listed_table_of_no_keys_is_empty():
    table = PackedTable.listed([])
    assert len(table) == 0 and list(table) == []
    assert table.cumulative_regret.shape == (0,)

# ---------------------------------------------------------------------------
# The gate: the learner banks the same numbers on either table
# ---------------------------------------------------------------------------

def test_a_fixed_seed_run_banks_identical_numbers_on_the_packed_and_object_tables():
    # Sixty iterations from one seed on the whole packed table and on a dict
    # that grows a `RegretMatcher` per key: every ledger the walk touched holds
    # the same bits on both, and the packed table touched nothing else.
    packed = train(new_solve(11, table=PackedTable.whole_game()), 60).table
    objects = train(new_solve(11, table=ObjectTable()), 60).table
    assert len(objects) > 100
    for key, ledger in objects.items():
        assert np.array_equal(packed[key].cumulative_regret, ledger.cumulative_regret)
        assert np.array_equal(packed[key].strategy_sum, ledger.strategy_sum)
    touched = {packed.row_of(key) for key in objects}
    rows = np.repeat(np.arange(len(packed)), packed.widths())
    moved = set(rows[packed.cumulative_regret != 0.0]) | set(rows[packed.strategy_sum != 0.0])
    assert moved <= touched

def test_a_listed_slice_of_the_visited_keys_banks_the_same_numbers_too():
    # The listed index under the walk: record which keys sixty iterations
    # reach, allocate exactly those, and rerun the seed.
    objects = train(new_solve(11, table=ObjectTable()), 60).table
    sliced = train(new_solve(11, table=PackedTable.listed(list(objects))), 60).table
    assert list(sliced) == list(objects)
    for key, ledger in objects.items():
        assert np.array_equal(sliced[key].cumulative_regret, ledger.cumulative_regret)
        assert np.array_equal(sliced[key].strategy_sum, ledger.strategy_sum)

@pytest.mark.skipif(
    not FULL_RUN,
    reason="streams all 6.2M keys through all_infosets(); set MINIDRAWMAHA_FULL_TABLE=1",
)
def test_every_key_of_the_game_locates_to_its_enumeration_row():
    # The non-circular check: the arithmetic index against the one walk.
    table = PackedTable.whole_game()
    for row, key in enumerate(all_infosets()):
        assert table.row_of(key) == row
    assert row == TOTAL_LEDGERS - 1

@pytest.mark.skipif(
    not FULL_RUN,
    reason="allocates the 5 GB object table beside the packed one; set MINIDRAWMAHA_FULL_TABLE=1",
)
def test_a_full_object_table_and_the_packed_table_agree_after_a_seeded_run():
    objects = {key: RegretMatcher(len(key.legal_actions())) for key in all_infosets()}
    train(new_solve(7, table=objects, deal=random_deal), 200)
    packed = train(new_solve(7, table=PackedTable.whole_game(), deal=random_deal), 200).table
    assert np.array_equal(
        packed.cumulative_regret,
        np.concatenate([ledger.cumulative_regret for ledger in objects.values()]),
    )
    assert np.array_equal(
        packed.strategy_sum,
        np.concatenate([ledger.strategy_sum for ledger in objects.values()]),
    )
