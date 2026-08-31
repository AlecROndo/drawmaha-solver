import numpy as np
import pytest

from drawmaha_solver.kuhn.game import (
    DECISION_HISTORIES,
    DECK,
    Action,
    Card,
    InfoSet,
    all_infosets,
)
from drawmaha_solver.kuhn.infoset_table import (
    average_strategy,
    current_strategy,
    format_strategy_grid,
    new_infoset_table,
)

J, Q, K = Card.JACK, Card.QUEEN, Card.KING
P, B = Action.PASS, Action.BET

UNIFORM = np.full(2, 0.5)

# ---------------------------------------------------------------------------
# The table: one ledger per infoset
# ---------------------------------------------------------------------------

def test_a_fresh_table_has_one_ledger_for_every_infoset():
    table = new_infoset_table()
    assert set(table) == set(all_infosets())
    assert len(table) == 12

def test_every_ledger_is_two_wide():
    # Two actions at every Kuhn decision node, so every regret vector is the
    # same shape — the property that lets one ledger class serve all 12.
    for ledger in new_infoset_table().values():
        assert ledger.n_actions == 2

def test_ledgers_are_distinct_objects():
    # A single shared RegretMatcher would still run and still converge — to
    # one strategy played at all twelve spots, which is not poker.
    ledgers = list(new_infoset_table().values())
    assert len({id(ledger) for ledger in ledgers}) == 12

def test_the_table_cannot_miss_a_key():
    # InfoSet validates its own card and history at construction, so every
    # infoset that can be built is one of the twelve the table pre-allocates.
    # There is no key the walk can produce that the table does not hold.
    table = new_infoset_table()
    for card in DECK:
        for history in DECISION_HISTORIES:
            assert InfoSet(card=card, history=history) in table

def test_a_terminal_history_is_not_a_key():
    # Nobody acts at a terminal node, so it owns no ledger. InfoSet rejects it
    # at construction rather than letting the walk bank a phantom thirteenth.
    with pytest.raises(ValueError, match="not a decision node"):
        InfoSet(card=K, history=(P, P))

# ---------------------------------------------------------------------------
# Reading the strategies out
# ---------------------------------------------------------------------------

def test_a_fresh_table_plays_uniformly_everywhere():
    table = new_infoset_table()
    for strategies in (current_strategy(table), average_strategy(table)):
        assert len(strategies) == 12
        for probabilities in strategies.values():
            assert np.allclose(probabilities, UNIFORM)

def test_current_strategy_follows_positive_regret():
    table = new_infoset_table()
    spot = InfoSet(card=K, history=())
    table[spot].cumulative_regret = np.array([-1.0, 3.0])
    assert np.allclose(current_strategy(table)[spot], [0.0, 1.0])
    # Untouched spots keep their uniform fallback.
    assert np.allclose(current_strategy(table)[InfoSet(card=J, history=())], UNIFORM)

def test_average_strategy_follows_the_banked_strategy_sum():
    table = new_infoset_table()
    spot = InfoSet(card=Q, history=(B,))
    table[spot].strategy_sum = np.array([3.0, 1.0])
    assert np.allclose(average_strategy(table)[spot], [0.75, 0.25])

def test_the_two_readouts_are_independent():
    # Regret drives what is played next; the strategy sum is the answer. A
    # solver that reports the wrong one of these looks convergent and is not.
    table = new_infoset_table()
    spot = InfoSet(card=K, history=())
    table[spot].cumulative_regret = np.array([0.0, 5.0])
    table[spot].strategy_sum = np.array([1.0, 1.0])
    assert np.allclose(current_strategy(table)[spot], [0.0, 1.0])
    assert np.allclose(average_strategy(table)[spot], UNIFORM)

# ---------------------------------------------------------------------------
# One traversal's worth of banking, from the rung-1 delta's worked trace
# ---------------------------------------------------------------------------

# The four ledger calls a single walk of the deal (K to P0, Q to P1) makes,
# with every ledger fresh. Utilities and weights are the report's hand-computed
# values; the expected accumulators are what the ledger must produce from them.
# No tree walk is involved — this pins the table-and-ledger interface the walk
# will call, so cfr.py can be diffed against a known-correct first iteration.
# (The report numbers its players P1/P2; ours are P0/P1.)
TRACE = [
    # infoset, utilities, w_r, w_s, expected regret, expected strategy sum
    (InfoSet(K, ()), [0.75, 1.5], 1 / 6, 1.0, [-0.0625, 0.0625], [0.5, 0.5]),
    (InfoSet(Q, (P,)), [-1.0, -0.5], 1 / 12, 1.0, [-1 / 48, 1 / 48], [0.5, 0.5]),
    (InfoSet(Q, (B,)), [-1.0, -2.0], 1 / 12, 1.0, [1 / 24, -1 / 24], [0.5, 0.5]),
    (InfoSet(K, (P, B)), [-1.0, 2.0], 1 / 12, 0.5, [-0.125, 0.125], [0.25, 0.25]),
]

@pytest.mark.parametrize("infoset,utilities,w_r,w_s,regret,strategy_sum", TRACE)
def test_one_banked_visit_matches_the_worked_trace(
    infoset, utilities, w_r, w_s, regret, strategy_sum
):
    table = new_infoset_table()
    table[infoset].update(
        np.array(utilities), regret_weight=w_r, strategy_weight=w_s
    )
    assert np.allclose(table[infoset].cumulative_regret, regret)
    assert np.allclose(table[infoset].strategy_sum, strategy_sum)

def test_the_trace_visits_four_infosets_of_the_twelve():
    # One deal reaches four of the twelve spots, so a full iteration over all
    # six deals makes 24 calls and every infoset is banked exactly twice —
    # once per hidden world consistent with it.
    assert len({infoset for infoset, *_ in TRACE}) == 4
    assert all(infoset in new_infoset_table() for infoset, *_ in TRACE)

def test_swapping_the_two_weights_gives_a_different_answer():
    # The classic CFR bug is silent precisely because both versions run. Pin
    # that the trace's one asymmetric row can actually tell them apart.
    infoset, utilities, w_r, w_s, _, _ = TRACE[-1]
    right, wrong = new_infoset_table(), new_infoset_table()
    right[infoset].update(np.array(utilities), regret_weight=w_r, strategy_weight=w_s)
    wrong[infoset].update(np.array(utilities), regret_weight=w_s, strategy_weight=w_r)
    assert not np.allclose(
        right[infoset].cumulative_regret, wrong[infoset].cumulative_regret
    )
    assert not np.allclose(right[infoset].strategy_sum, wrong[infoset].strategy_sum)

# ---------------------------------------------------------------------------
# The 4x3 display grid
# ---------------------------------------------------------------------------

def test_the_grid_renders_a_row_per_decision_node_and_a_column_per_card():
    grid = format_strategy_grid(current_strategy(new_infoset_table())).splitlines()
    assert len(grid) == 5  # header plus four decision nodes
    assert grid[0].split() == ["J", "Q", "K"]
    assert [line.split()[0] for line in grid[1:]] == ["-", "p", "b", "pb"]
    for line in grid[1:]:
        assert line.split()[1:] == ["0.500", "0.500", "0.500"]

def test_the_grid_reports_probability_of_bet():
    table = new_infoset_table()
    # Pure BET with the king at the open; index 1 is BET.
    table[InfoSet(card=K, history=())].strategy_sum = np.array([0.0, 1.0])
    grid = format_strategy_grid(average_strategy(table)).splitlines()
    assert grid[1].split() == ["-", "0.500", "0.500", "1.000"]
