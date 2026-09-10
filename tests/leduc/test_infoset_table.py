import numpy as np
import pytest

from drawmaha_solver.leduc.game import (
    Action,
    InfoSet,
    Rank,
    all_infosets,
)
from drawmaha_solver.leduc.infoset_table import (
    average_strategy,
    current_strategy,
    format_strategy_tables,
    new_infoset_table,
)

J, Q, K = Rank.JACK, Rank.QUEEN, Rank.KING
F, C, R = Action.FOLD, Action.CALL, Action.RAISE

UNIFORM2 = np.full(2, 1 / 2)
UNIFORM3 = np.full(3, 1 / 3)

# A 3-wide spot (facing the opening raise) and a 2-wide spot (the open),
# used throughout: the two ledger shapes this rung introduces.
FACING_RAISE = InfoSet(rank=J, board=None, betting=((R,),))
THE_OPEN = InfoSet(rank=K, board=None, betting=((),))

# ---------------------------------------------------------------------------
# The table: one ledger per infoset, sized by that infoset's legal actions
# ---------------------------------------------------------------------------

def test_a_fresh_table_has_one_ledger_for_every_infoset():
    table = new_infoset_table()
    assert set(table) == set(all_infosets())
    assert len(table) == 288

def test_every_ledger_is_exactly_as_wide_as_its_spots_legal_actions():
    # The rung-2 delta: no uniform N_ACTIONS width. A 3-wide ledger at a
    # 2-wide spot would bank regret for an action the rules forbid, and the
    # phantom column could win the normalization.
    for infoset, ledger in new_infoset_table().items():
        assert ledger.n_actions == len(infoset.legal_actions())

def test_the_widths_split_ninety_six_three_wide_and_the_rest_two_wide():
    widths = [ledger.n_actions for ledger in new_infoset_table().values()]
    assert widths.count(3) == 96
    assert widths.count(2) == 192

def test_ledgers_are_distinct_objects():
    # A single shared RegretMatcher would still run and still converge — to
    # one strategy played at all 288 spots, which is not poker.
    ledgers = list(new_infoset_table().values())
    assert len({id(ledger) for ledger in ledgers}) == 288

def test_entry_k_of_a_ledger_means_legal_actions_k():
    # The pairing the whole rung leans on: ledger columns are positions in
    # legal_actions(), not Action values. At a (fold, call, raise) spot,
    # column 0 is FOLD; at a (call, raise) spot, column 0 is CALL.
    assert FACING_RAISE.legal_actions() == (F, C, R)
    assert THE_OPEN.legal_actions() == (C, R)

# ---------------------------------------------------------------------------
# Reading the strategies out
# ---------------------------------------------------------------------------

def test_a_fresh_table_plays_uniformly_everywhere_at_both_widths():
    table = new_infoset_table()
    for strategies in (current_strategy(table), average_strategy(table)):
        assert len(strategies) == 288
        for infoset, probabilities in strategies.items():
            expected = UNIFORM3 if len(infoset.legal_actions()) == 3 else UNIFORM2
            assert np.allclose(probabilities, expected)

def test_current_strategy_follows_positive_regret():
    table = new_infoset_table()
    table[FACING_RAISE].cumulative_regret = np.array([-1.0, 2.0, 5.0])
    # Negative regret contributes nothing; the rest normalize: [0, 2/7, 5/7].
    assert np.allclose(current_strategy(table)[FACING_RAISE], [0.0, 2 / 7, 5 / 7])
    # Untouched spots keep their uniform fallback.
    assert np.allclose(current_strategy(table)[THE_OPEN], UNIFORM2)

def test_average_strategy_follows_the_banked_strategy_sum():
    table = new_infoset_table()
    table[THE_OPEN].strategy_sum = np.array([3.0, 1.0])
    assert np.allclose(average_strategy(table)[THE_OPEN], [0.75, 0.25])

def test_the_two_readouts_are_independent():
    # Regret drives what is played next; the strategy sum is the answer. A
    # solver that reports the wrong one of these looks convergent and is not.
    table = new_infoset_table()
    table[THE_OPEN].cumulative_regret = np.array([0.0, 5.0])
    table[THE_OPEN].strategy_sum = np.array([1.0, 1.0])
    assert np.allclose(current_strategy(table)[THE_OPEN], [0.0, 1.0])
    assert np.allclose(average_strategy(table)[THE_OPEN], UNIFORM2)

# ---------------------------------------------------------------------------
# One banked visit, hand-computed at the new width and the new chance weight
# ---------------------------------------------------------------------------

def test_one_banked_visit_matches_the_hand_computation():
    # A fresh 3-wide ledger visited once with the weights a rung-2 walk will
    # pass: regret_weight carries the 1/30 deal probability, strategy_weight
    # the player's own half-probability reach. Worked by hand:
    #   sigma = [1/3, 1/3, 1/3], u = [-1, 2, 3], <sigma, u> = 4/3
    #   regret += 1/30 * (u - 4/3)  = [-7/90, 2/90, 5/90]
    #   sum    += 1/2 * sigma       = [1/6, 1/6, 1/6]
    table = new_infoset_table()
    table[FACING_RAISE].update(
        np.array([-1.0, 2.0, 3.0]), regret_weight=1 / 30, strategy_weight=1 / 2
    )
    assert np.allclose(
        table[FACING_RAISE].cumulative_regret, [-7 / 90, 2 / 90, 5 / 90]
    )
    assert np.allclose(table[FACING_RAISE].strategy_sum, [1 / 6, 1 / 6, 1 / 6])
    # The next read: positive regrets [0, 2, 5]/90 normalize to [0, 2/7, 5/7],
    # so folding — which regretted negative — is not played at all.
    assert np.allclose(current_strategy(table)[FACING_RAISE], [0.0, 2 / 7, 5 / 7])

def test_a_utilities_vector_of_the_wrong_width_is_rejected():
    # The silent trap the paper names: a 3-long vector at a 2-wide spot means
    # the walk built utilities in Action-value order instead of legal-action
    # order. The ledger's own shape check is what catches it.
    table = new_infoset_table()
    with pytest.raises(ValueError, match="shape"):
        table[THE_OPEN].update(np.array([0.0, 1.0, 2.0]))

# ---------------------------------------------------------------------------
# The readout tables (the 4x3 grid does not survive 288 spots)
# ---------------------------------------------------------------------------

def test_the_readout_has_one_round_one_table_and_one_per_board():
    text = format_strategy_tables(current_strategy(new_infoset_table()))
    assert text.count("round 1") == 1
    assert text.count("board") == 3
    for symbol in ("J", "Q", "K"):
        assert f"board {symbol}" in text

def test_round_one_renders_a_row_per_line_and_a_column_per_rank():
    lines = format_strategy_tables(current_strategy(new_infoset_table())).splitlines()
    header = lines[lines.index("round 1") + 1]
    assert header.split() == ["J", "Q", "K"]
    labels = [line.split()[0] for line in lines[lines.index("round 1") + 2 :][:6]]
    assert labels == ["-", "c", "r", "cr", "rr", "crr"]

def test_cells_print_the_whole_mixed_strategy_in_legal_action_order():
    # No single P(BET) convention survives rows whose legal sets differ, so a
    # cell spells out every legal action with its probability.
    table = new_infoset_table()
    table[THE_OPEN].strategy_sum = np.array([0.0, 1.0])  # pure raise with the K
    lines = format_strategy_tables(average_strategy(table)).splitlines()
    open_row = next(line for line in lines if line.split()[0:1] == ["-"])
    assert open_row.count("c0.50 r0.50") == 2  # J and Q still uniform
    assert "c0.00 r1.00" in open_row  # the K column moved

def test_round_two_rows_carry_both_lines_in_the_label():
    lines = format_strategy_tables(current_strategy(new_infoset_table())).splitlines()
    labels = {line.split()[0] for line in lines if "|" in line}
    # 5 surviving round-1 lines x 6 round-2 decision lines.
    assert len(labels) == 30
    assert "cc|-" in labels
    assert "crrc|crr" in labels
