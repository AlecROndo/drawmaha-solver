import numpy as np
import pytest

from drawmaha_solver.kuhn.cfr import run_iteration, train, walk
from drawmaha_solver.kuhn.game import (
    DEAL_PROBABILITY,
    DEALS,
    Action,
    Card,
    InfoSet,
    KuhnState,
)
from drawmaha_solver.kuhn.infoset_table import average_strategy, new_infoset_table

J, Q, K = Card.JACK, Card.QUEEN, Card.KING
P, B = Action.PASS, Action.BET

def independent_game_value(strategies: dict[InfoSet, np.ndarray]) -> float:
    """P0's expected chips under `strategies`, by a walk that is NOT cfr.walk.

    Deliberately a second implementation: it grades the solver, so it must not
    be able to share a bug with it. Only `game.py` is trusted here.
    """
    def value(state: KuhnState) -> float:
        if state.is_terminal():
            return state.returns()[0]
        sigma = strategies[state.infoset()]
        return sum(
            sigma[action] * value(state.apply(action))
            for action in state.legal_actions()
        )

    return sum(DEAL_PROBABILITY * value(KuhnState(cards=deal)) for deal in DEALS)

# ---------------------------------------------------------------------------
# One traversal, against the hand-worked trace
# ---------------------------------------------------------------------------

# The four ledgers a single walk of the deal (K to P0, Q to P1) must leave
# behind, starting from a fresh table. Same numbers the rung-1 delta report
# computed by hand; here the WALK has to produce them, weights and all.
TRACE = [
    (InfoSet(K, ()), [-0.0625, 0.0625], [0.5, 0.5]),
    (InfoSet(Q, (P,)), [-1 / 48, 1 / 48], [0.5, 0.5]),
    (InfoSet(Q, (B,)), [1 / 24, -1 / 24], [0.5, 0.5]),
    (InfoSet(K, (P, B)), [-0.125, 0.125], [0.25, 0.25]),
]

@pytest.mark.parametrize("infoset,regret,strategy_sum", TRACE)
def test_one_traversal_reproduces_the_worked_trace(infoset, regret, strategy_sum):
    table = new_infoset_table()
    walk(KuhnState(cards=(K, Q)), table, (1.0, 1.0))
    assert np.allclose(table[infoset].cumulative_regret, regret)
    assert np.allclose(table[infoset].strategy_sum, strategy_sum)

def test_a_traversal_returns_the_uniform_node_value_to_both_seats():
    # Every ledger fresh, so both players play (1/2, 1/2) everywhere: the deal
    # (K, Q) is worth +1.125 to P0. Zero-sum, so P1's value is its negation.
    table = new_infoset_table()
    assert walk(KuhnState(cards=(K, Q)), table, (1.0, 1.0)) == pytest.approx(
        (1.125, -1.125)
    )

def test_a_traversal_only_touches_the_infosets_on_its_path():
    # One deal reaches four of the twelve spots. A walk that banks anywhere
    # else is reading a card it cannot see.
    table = new_infoset_table()
    walk(KuhnState(cards=(K, Q)), table, (1.0, 1.0))
    touched = {
        infoset
        for infoset, ledger in table.items()
        if ledger.strategy_sum.any() or ledger.cumulative_regret.any()
    }
    assert touched == {infoset for infoset, *_ in TRACE}

def test_a_full_iteration_visits_every_infoset():
    # Six deals x four spots = 24 visits, so all twelve are banked exactly
    # twice — once per hidden world consistent with them.
    table = new_infoset_table()
    run_iteration(table)
    assert all(ledger.strategy_sum.any() for ledger in table.values())

def test_the_walk_leaves_the_state_alone():
    # KuhnState is frozen; pin that the walk really does descend by building
    # children rather than mutating a cursor.
    root = KuhnState(cards=(K, Q))
    walk(root, new_infoset_table(), (1.0, 1.0))
    assert root.history == ()
    assert root.cards == (K, Q)

# ---------------------------------------------------------------------------
# Convergence: the answer sheet
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solved():
    return average_strategy(train(20_000))

def bet(strategies, card, history):
    return strategies[InfoSet(card=card, history=history)][Action.BET]

def test_game_value_converges_to_minus_one_eighteenth(solved):
    # Kuhn's closed-form value to the first player. Checked with an evaluator
    # written independently of the solver.
    assert independent_game_value(solved) == pytest.approx(-1 / 18, abs=2e-3)

def test_the_bluff_is_one_third_of_the_value_bet(solved):
    # The one fact everyone quotes about Kuhn: opening with the jack is a pure
    # bluff, and its frequency is chained to the king's value bet at 1:3.
    alpha = bet(solved, J, ())
    assert bet(solved, K, ()) == pytest.approx(3 * alpha, abs=0.05)

def test_alpha_is_not_pinned_to_one_value(solved):
    # The equilibrium is a FAMILY over alpha in [0, 1/3], so the right check is
    # the invariants, never a hardcoded alpha-hat. Pin only the legal range.
    alpha = bet(solved, J, ())
    assert 0.0 <= alpha <= 1 / 3 + 0.02

@pytest.mark.parametrize(
    "card,history,expected",
    [
        (Q, (), 0.0),  # never open-bet the queen
        (J, (P,), 1 / 3),  # bluff the jack after a check
        (Q, (P,), 0.0),
        (K, (P,), 1.0),  # always bet the king when checked to
        (J, (B,), 0.0),  # fold the jack to a bet
        (Q, (B,), 1 / 3),  # call a third of the time with the queen
        (K, (B,), 1.0),  # never fold the king
        (J, (P, B), 0.0),
        (K, (P, B), 1.0),
    ],
)
def test_the_nine_pinned_infosets(solved, card, history, expected):
    assert bet(solved, card, history) == pytest.approx(expected, abs=0.05)

def test_the_queens_call_tracks_alpha(solved):
    # The third free spot: calling with the queen after checking is alpha+1/3,
    # so it moves with the opening bluff rather than sitting at a fixed value.
    alpha = bet(solved, J, ())
    assert bet(solved, Q, (P, B)) == pytest.approx(alpha + 1 / 3, abs=0.05)

# ---------------------------------------------------------------------------
# The average is the answer; the current strategy is not
# ---------------------------------------------------------------------------

def test_training_is_deterministic():
    # Vanilla CFR enumerates the whole tree, so there is no sampling and no
    # seed: two runs must agree to the bit.
    first, second = average_strategy(train(200)), average_strategy(train(200))
    for infoset, probabilities in first.items():
        assert np.array_equal(probabilities, second[infoset])

def test_more_iterations_move_the_game_value_closer():
    short = independent_game_value(average_strategy(train(100)))
    long = independent_game_value(average_strategy(train(5_000)))
    assert abs(long - (-1 / 18)) < abs(short - (-1 / 18))

def test_train_can_continue_an_existing_table():
    table = train(100)
    banked = table[InfoSet(K, ())].strategy_sum.sum()
    train(100, table=table)
    assert table[InfoSet(K, ())].strategy_sum.sum() > banked
