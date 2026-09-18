"""The ledger table, pinned without ever allocating all 3.1 million of it.

Rung 2 could assert `len(table) == 288` and be done. Here the real table is
3,142,290 ledgers, **1.77 GB and fourteen seconds**, which is affordable once per
solve and ruinous once per test — so this file is organised around what can be
proved *structurally* instead of counted.

Two claims are what plan §9 asks of this module, and neither is spot-checkable:

* **ledger width equals `len(legal_actions())` at every key.** Made total by
  proving the premise underneath it — `legal_actions()` reads only the public
  half of a key, so every key standing on one public point has one width. That
  is checked at all 141 points, which is every width the table can hold; the
  allocator is then a one-liner over those widths.
* **no key collides across genuinely different positions.** Made total the same
  way — the first key of every shape, allocated at all 141 points at once, must
  come back as 141 distinct ledgers. `test_census.py` proves the enumerator's
  public signatures are distinct; this proves the *table* keeps them apart,
  which is the failure that would silently share one strategy between two spots.

The whole 3.1M allocation is here too, behind `MINIDRAWMAHA_FULL_TABLE=1`, so
the default run stays short — and the fast tests already move whenever the slow
one would, because both read the same census fixture.
"""

import json
import os
from itertools import islice
from pathlib import Path
from random import Random

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.cards import canonical, parse_cards
from drawmaha_solver.minidrawmaha.enumeration import (
    all_infosets,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.game import DRAW_ACTIONS, Action, InfoSet
from drawmaha_solver.minidrawmaha.infoset_table import (
    average_strategy,
    current_strategy,
    format_strategy_tables,
    ledger_widths,
    new_infoset_table,
    spots,
)

CENSUS = json.loads((Path(__file__).parent / "census.json").read_text())

FULL_RUN = os.environ.get("MINIDRAWMAHA_FULL_TABLE") == "1"

F, C, P = Action.FOLD, Action.CHECK_CALL, Action.POT

def point_where(**fields):
    return next(
        point
        for point in public_decision_points()
        if all(getattr(point, name) == value for name, value in fields.items())
    )

# One public point per stage, which is also one per ledger width: 2 at the open,
# 3 facing a bet, 4 at the draw. Round 2 holds no width of its own and is here
# because it is the only stage whose keys carry a discard.
THE_OPEN = point_where(betting=((),))  # P0 opens round 1: check or bet
FACING_A_BET = point_where(betting=((P,),))  # P1 answers it: fold, call or raise
THE_DRAW = point_where(is_draw_decision=True)
ROUND_TWO = point_where(board_cards=2)

def keys_at(point, indices=None):
    """The infosets standing on `point` — all of them, or the ones `indices` picks.

    The enumerator's inner loop restated, exactly as `test_census.py` restates
    it: the point supplies the public half and `private_keys` the private one,
    so this reaches the same keys `all_infosets()` yields without streaming past
    the millions in front of them.
    """
    keys = private_keys(board_cards=point.board_cards, discards=point.discards)
    chosen = keys if indices is None else [keys[index] for index in indices]
    return [
        InfoSet(
            player=point.player,
            hole=hole,
            discarded=thrown,
            board=board,
            draws=point.draws,
            betting=point.betting,
        )
        for hole, thrown, board in chosen
    ]

# ---------------------------------------------------------------------------
# Allocation: one ledger per key, at that key's own width
# ---------------------------------------------------------------------------

def test_a_table_allocated_at_one_public_point_holds_exactly_its_private_keys():
    keys = keys_at(THE_OPEN)
    table = new_infoset_table(keys)
    assert len(table) == len(private_keys(board_cards=1, discards=0)) == 970
    assert set(table) == set(keys)

def test_every_ledger_is_exactly_as_wide_as_its_spots_legal_actions():
    # Rung 2's claim, restated at every width this rung has. A uniform 4-wide
    # table would bank regret for throwing a card at a betting spot, and that
    # phantom column could win the normalization.
    for point in (THE_OPEN, FACING_A_BET, THE_DRAW, ROUND_TWO):
        for infoset, ledger in new_infoset_table(keys_at(point, range(50))).items():
            assert ledger.n_actions == len(infoset.legal_actions())

def test_width_is_a_property_of_the_public_point_and_never_of_the_cards():
    # The premise that makes the claim above total rather than sampled, and the
    # reason the histogram below can be computed without allocating: a key's
    # legal actions are read off its betting and its draws, both of which are
    # public, so the cards a key holds cannot change its width. Checked at all
    # 141 points, which is every width in the game.
    rng = Random(20260918)
    for point in public_decision_points():
        held = len(private_keys(board_cards=point.board_cards, discards=point.discards))
        sample = keys_at(point, (0, held - 1, rng.randrange(held)))
        assert {len(key.legal_actions()) for key in sample} == {point.width}

def test_the_width_histogram_matches_the_census_and_allocates_nothing():
    recorded = CENSUS["infosets"]["by_ledger_width"]
    assert ledger_widths() == {int(width): count for width, count in recorded.items()}
    assert sum(ledger_widths().values()) == CENSUS["infosets"]["total"] == 3_142_290
    # Width 4 exists only at the draw, and only before anybody has drawn, so it
    # is the 21 draw points times the 970 keys with one board card and nothing
    # thrown — the smallest slice of the table and the one rung 2 had no
    # equivalent of.
    assert ledger_widths()[4] == 21 * 970

def test_ledgers_are_distinct_objects():
    # A single shared RegretMatcher would still run and still converge — to one
    # strategy played at every spot in the game, which is not poker.
    ledgers = list(new_infoset_table(keys_at(THE_DRAW, range(200))).values())
    assert len({id(ledger) for ledger in ledgers}) == len(ledgers)

def test_no_two_public_points_share_a_ledger():
    # The collision probe, run across the whole public tree at once for the
    # price of 141 keys: one key per point, all of the same few shapes, so any
    # public field the table failed to keep in the key would collapse them.
    points = public_decision_points()
    first_keys = [key for point in points for key in keys_at(point, [0])]
    assert len(new_infoset_table(first_keys)) == len(points) == 141

def test_one_private_key_at_two_different_points_is_two_ledgers():
    # The same three cards and the same board, one seat, two betting histories:
    # distinct spots that a key missing its public half would merge. Named
    # explicitly because it is the exact shape of the bug the probe above
    # catches in bulk.
    opening, drawing = (keys_at(point, [0])[0] for point in (THE_OPEN, THE_DRAW))
    assert (opening.player, opening.hole, opening.board) == (
        drawing.player,
        drawing.hole,
        drawing.board,
    )
    assert opening != drawing
    assert len(new_infoset_table([opening, drawing])) == 2

def test_a_key_the_walk_should_never_produce_has_no_ledger():
    # Pre-allocated rather than filled on demand: a lookup that misses raises
    # instead of quietly gaining a ledger nothing will ever train.
    table = new_infoset_table(keys_at(THE_OPEN, range(10)))
    with pytest.raises(KeyError):
        table[keys_at(THE_DRAW, [0])[0]]

def test_entry_k_of_a_ledger_means_legal_actions_k():
    # The pairing the whole rung leans on: ledger columns are positions in
    # legal_actions(), not Action values. Column 0 is CHECK_CALL at the open,
    # FOLD facing a bet, and THROW_NONE at the draw.
    assert keys_at(THE_OPEN, [0])[0].legal_actions() == (C, P)
    assert keys_at(FACING_A_BET, [0])[0].legal_actions() == (F, C, P)
    assert keys_at(THE_DRAW, [0])[0].legal_actions() == DRAW_ACTIONS

@pytest.mark.skipif(
    not FULL_RUN,
    reason="the whole 3.1M-ledger table is 1.77 GB; set MINIDRAWMAHA_FULL_TABLE=1",
)
def test_the_default_table_is_the_whole_census():
    # What the solver actually allocates, counted the expensive way once: the
    # default argument really is every key the enumerator yields, and the
    # widths really do come out as the product argument predicts.
    table = new_infoset_table()
    assert len(table) == CENSUS["infosets"]["total"]
    widths: dict[int, int] = {}
    for ledger in table.values():
        widths[ledger.n_actions] = widths.get(ledger.n_actions, 0) + 1
    assert widths == ledger_widths()

def test_the_default_argument_is_the_enumerator_itself():
    # The cheap half of the test above, run every time: the first keys of the
    # default table are the first keys of `all_infosets()`, so the slice
    # argument cannot have quietly become the only path anybody exercises.
    prefix = list(islice(all_infosets(), 2_000))
    assert list(new_infoset_table(prefix)) == prefix

# ---------------------------------------------------------------------------
# Reading the strategies out, without materialising 3.1 million arrays
# ---------------------------------------------------------------------------

def test_a_fresh_table_plays_uniformly_at_all_three_widths():
    for point in (THE_OPEN, FACING_A_BET, THE_DRAW):
        table = new_infoset_table(keys_at(point, range(20)))
        for strategies in (current_strategy(table), average_strategy(table)):
            assert len(strategies) == 20
            for infoset, probabilities in strategies.items():
                width = len(infoset.legal_actions())
                assert np.allclose(probabilities, np.full(width, 1 / width))

def test_the_readouts_are_views_and_never_a_dict_of_three_million_arrays():
    # The scale decision, pinned by its one observable consequence: a view is
    # LIVE, so a number read after training is the trained number and not the
    # one that was true when the view was made. A caller that wants a snapshot
    # has to say so — `dict(view)` — and pay for it in the open.
    table = new_infoset_table(keys_at(THE_OPEN, range(5)))
    spot = keys_at(THE_OPEN, [0])[0]
    average = average_strategy(table)
    snapshot = dict(average)
    table[spot].strategy_sum = np.array([3.0, 1.0])
    assert np.allclose(average[spot], [0.75, 0.25])
    assert np.allclose(snapshot[spot], [0.5, 0.5])
    assert not isinstance(average, dict)

def test_current_strategy_follows_positive_regret():
    table = new_infoset_table(keys_at(FACING_A_BET, range(5)))
    spot, untouched = keys_at(FACING_A_BET, range(2))
    table[spot].cumulative_regret = np.array([-1.0, 2.0, 5.0])
    # Negative regret contributes nothing; the rest normalize: [0, 2/7, 5/7].
    assert np.allclose(current_strategy(table)[spot], [0.0, 2 / 7, 5 / 7])
    assert np.allclose(current_strategy(table)[untouched], np.full(3, 1 / 3))

def test_average_strategy_follows_the_banked_strategy_sum():
    table = new_infoset_table(keys_at(THE_OPEN, range(5)))
    spot = keys_at(THE_OPEN, [0])[0]
    table[spot].strategy_sum = np.array([3.0, 1.0])
    assert np.allclose(average_strategy(table)[spot], [0.75, 0.25])

def test_the_two_readouts_are_independent():
    # Regret drives what is played next; the strategy sum is the answer. A
    # solver that reports the wrong one of these looks convergent and is not.
    table = new_infoset_table(keys_at(THE_OPEN, range(5)))
    spot = keys_at(THE_OPEN, [0])[0]
    table[spot].cumulative_regret = np.array([0.0, 5.0])
    table[spot].strategy_sum = np.array([1.0, 1.0])
    assert np.allclose(current_strategy(table)[spot], [0.0, 1.0])
    assert np.allclose(average_strategy(table)[spot], [0.5, 0.5])

def test_one_banked_visit_at_the_draw_matches_the_hand_computation():
    # A fresh 4-wide draw ledger visited once, at the width rung 2 never had.
    #   sigma = [1/4]*4, u = [-1, 2, 3, 0], <sigma, u> = 1
    #   regret += 1/30 * (u - 1) = [-2, 1, 2, -1]/30
    #   sum    += 1/2  * sigma   = [1/8]*4
    table = new_infoset_table(keys_at(THE_DRAW, range(5)))
    spot = keys_at(THE_DRAW, [0])[0]
    table[spot].update(
        np.array([-1.0, 2.0, 3.0, 0.0]), regret_weight=1 / 30, strategy_weight=1 / 2
    )
    assert np.allclose(table[spot].cumulative_regret, np.array([-2, 1, 2, -1]) / 30)
    assert np.allclose(table[spot].strategy_sum, np.full(4, 1 / 8))
    # Positive regrets [0, 1, 2, 0] normalize to [0, 1/3, 2/3, 0]: standing pat
    # and throwing the top card both regretted negative, so neither is played.
    assert np.allclose(current_strategy(table)[spot], [0.0, 1 / 3, 2 / 3, 0.0])

def test_a_utilities_vector_of_the_wrong_width_is_rejected():
    # The silent trap: a 3-long vector at a 4-wide draw spot means the walk
    # built utilities over the betting actions instead of the throws. The
    # ledger's own shape check is what catches it.
    table = new_infoset_table(keys_at(THE_DRAW, range(2)))
    with pytest.raises(ValueError, match="shape"):
        table[keys_at(THE_DRAW, [0])[0]].update(np.array([0.0, 1.0, 2.0]))

# ---------------------------------------------------------------------------
# The readout: the spots one hand can stand on
# ---------------------------------------------------------------------------

# Deliberately NOT in canonical form — the suits relabel to `2c 3c 4d` on `5h`
# — so every readout test below also exercises the relabelling the caller would
# otherwise have to remember.
HOLE = parse_cards("2h 3h 4c")
BOARD_ONE = parse_cards("5d")
BOARD_TWO = parse_cards("5d 6c")
THROWN = parse_cards("6h")

def test_one_hand_before_the_draw_stands_on_its_betting_and_its_draw_spots():
    # P0 with one board card out: the four round-1 points where P0 acts, and
    # the seven draw points — one per closing round-1 line. Every ledger the
    # hand can reach before a card is thrown, and none of anybody else's.
    keys = spots(player=0, hole=HOLE, board=BOARD_ONE)
    assert len(keys) == 4 + 7
    assert {key.player for key in keys} == {0}
    assert sum(1 for key in keys if key.is_draw_decision()) == 7

def test_the_seat_that_acts_second_at_the_draw_has_twice_as_many_draw_spots():
    # P1 draws knowing whether P0 drew, so P0's signal doubles P1's draw points
    # — the cap pulling its second time, now visible in a single hand's readout.
    keys = spots(player=1, hole=HOLE, board=BOARD_ONE)
    assert len(keys) == 4 + 14

def test_a_two_card_board_reads_out_the_round_two_spots_for_that_discard_count():
    # 112 round-2 points split four ways — two seats times two discard counts —
    # and a hand fits exactly one of the four.
    assert len(spots(player=0, hole=HOLE, board=BOARD_TWO)) == 28
    assert len(spots(player=0, hole=HOLE, board=BOARD_TWO, discarded=THROWN)) == 28

def test_a_hand_shape_no_decision_point_has_is_refused_rather_than_empty():
    # Nobody has drawn while one board card is out, so this question has no
    # spots. An empty readout would read as a solve with no data in it.
    with pytest.raises(ValueError, match="no decision point"):
        spots(player=0, hole=HOLE, board=BOARD_ONE, discarded=THROWN)

def test_the_readout_canonicalises_the_hand_it_is_given():
    # A human types physical suits; the table is keyed on canonical ones. Doing
    # the relabelling here rather than at the call site is what stops a readout
    # from raising KeyError at a spot that is perfectly well allocated.
    hole, _, board = canonical(HOLE, (), BOARD_ONE)
    assert spots(player=0, hole=HOLE, board=BOARD_ONE) == spots(
        player=0, hole=hole, board=board
    )
    assert hole != HOLE or board != BOARD_ONE  # the fixture really is relabelled

def test_the_readout_titles_every_stage_the_hand_reaches():
    keys = spots(player=0, hole=HOLE, board=BOARD_ONE)
    text = format_strategy_tables(
        current_strategy(new_infoset_table(keys)), player=0, hole=HOLE, board=BOARD_ONE
    )
    assert "round 1" in text
    assert "the draw" in text
    assert "round 2" not in text
    # One row per spot, plus a title line per block and the header.
    assert sum(1 for line in text.splitlines() if line.startswith("  ")) == 11

def test_cells_print_the_whole_mixed_strategy_in_legal_action_order():
    # No single P(BET) column survives rows whose legal sets differ in size and
    # in meaning, so a cell spells out every legal action with its probability.
    keys = spots(player=0, hole=HOLE, board=BOARD_ONE)
    table = new_infoset_table(keys)
    opening = next(key for key in keys if key.betting == ((),))
    table[opening].strategy_sum = np.array([0.0, 1.0])  # pure bet
    text = format_strategy_tables(
        average_strategy(table), player=0, hole=HOLE, board=BOARD_ONE
    )
    assert "x0.00 p1.00" in text
    assert "f0.33 c0.33 p0.33" in text  # a three-wide spot, still uniform

def test_the_draw_cells_name_the_card_each_throw_takes():
    # `ACTION_SYMBOL` is the PUBLIC record, where all three one-card throws read
    # `t1` because that is exactly what the table sees. A readout is for the
    # player, who knows which card went, so the cells name the position.
    keys = spots(player=0, hole=HOLE, board=BOARD_ONE)
    text = format_strategy_tables(
        current_strategy(new_infoset_table(keys)), player=0, hole=HOLE, board=BOARD_ONE
    )
    assert "pat0.25 low0.25 mid0.25 top0.25" in text

def test_the_readout_labels_a_row_with_the_public_history_that_reached_it():
    keys = spots(player=1, hole=HOLE, board=BOARD_TWO, discarded=THROWN)
    text = format_strategy_tables(
        current_strategy(new_infoset_table(keys)),
        player=1,
        hole=HOLE,
        board=BOARD_TWO,
        discarded=THROWN,
    )
    labels = {line.split()[0] for line in text.splitlines() if line.startswith("  ")}
    # A round-2 label carries all three parts of the history that reached it, in
    # the order they happened: "xx/t0t1|p" is check-check, P0 stood pat while P1
    # drew one, and P0 has bet the second board card. Every one of these rows
    # ends on P1 to act, which is what makes them one seat's readout — a label
    # whose round-2 line had even length would be P0's spot in P1's table.
    assert "xx/t0t1|p" in labels
    assert "xx/t1t1|x" in labels
    assert all(len(label.split("|")[1].replace("-", "")) % 2 == 1 for label in labels)

def test_the_readout_names_the_hand_it_is_reading():
    keys = spots(player=0, hole=HOLE, board=BOARD_ONE)
    text = format_strategy_tables(
        current_strategy(new_infoset_table(keys)),
        player=0,
        hole=HOLE,
        board=BOARD_ONE,
    )
    assert text.splitlines()[0].startswith("P0 ")
