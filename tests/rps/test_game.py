import numpy as np
import pytest

from drawmaha_solver.rps.game import (
    PAYOFF,
    Action,
    action_values,
    best_response_value,
    payoff,
    winner,
)

R, P, S = Action.ROCK, Action.PAPER, Action.SCISSORS

# (p0 action, p1 action, expected winner, expected p0 payoff)
OUTCOMES = [
    (R, R, None, 0.0),
    (R, P, 1, -1.0),
    (R, S, 0, 1.0),
    (P, R, 0, 1.0),
    (P, P, None, 0.0),
    (P, S, 1, -1.0),
    (S, R, 1, -1.0),
    (S, P, 0, 1.0),
    (S, S, None, 0.0),
]

@pytest.mark.parametrize("a,b,expected_winner,expected_payoff", OUTCOMES)
def test_all_nine_outcomes(a, b, expected_winner, expected_payoff):
    assert winner(a, b) == expected_winner
    assert payoff(a, b) == expected_payoff

def test_payoff_matrix_is_zero_sum():
    assert np.array_equal(PAYOFF, -PAYOFF.T)

def test_action_values_against_pure_rock():
    values = action_values(np.array([1.0, 0.0, 0.0]))
    assert values[R] == 0.0
    assert values[P] == 1.0
    assert values[S] == -1.0

def test_uniform_strategy_is_unexploitable():
    assert best_response_value(np.full(3, 1 / 3)) == pytest.approx(0.0)

def test_pure_strategy_is_fully_exploitable():
    assert best_response_value(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)

def test_slightly_biased_strategy_is_slightly_exploitable():
    # 40/30/30 rock-heavy: the counter (paper) earns 0.4*1 + 0.3*0 + 0.3*(-1) = 0.1
    assert best_response_value(np.array([0.4, 0.3, 0.3])) == pytest.approx(0.1)
