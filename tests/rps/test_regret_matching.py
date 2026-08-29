import numpy as np
import pytest

from drawmaha_solver.rps.game import PAYOFF, best_response_value
from drawmaha_solver.rps.players import FixedStrategyPlayer, RegretMatchingPlayer, play_match
from drawmaha_solver.rps.regret_matching import RegretMatcher

UNIFORM = np.full(3, 1 / 3)

def test_initial_strategy_is_uniform():
    assert np.allclose(RegretMatcher(3).strategy(), UNIFORM)
    assert np.allclose(RegretMatcher(3).average_strategy(), UNIFORM)

def test_all_negative_regret_falls_back_to_uniform():
    m = RegretMatcher(3)
    m.cumulative_regret = np.array([-1.0, -2.0, -0.5])
    assert np.allclose(m.strategy(), UNIFORM)

def test_strategy_proportional_to_positive_regret_only():
    m = RegretMatcher(3)
    m.cumulative_regret = np.array([3.0, 1.0, -5.0])
    assert np.allclose(m.strategy(), [0.75, 0.25, 0.0])

def test_update_accumulates_regret_relative_to_played_action():
    m = RegretMatcher(3)
    # We played rock (0) into the opponent's paper: utilities are what each of
    # our actions would have earned against paper.
    utilities = PAYOFF[:, 1]  # rock -1, paper 0, scissors +1
    m.update(utilities, played=0)
    assert np.allclose(m.cumulative_regret, [0.0, 1.0, 2.0])
    # The uniform strategy in effect for that round was banked in the average.
    assert np.allclose(m.strategy_sum, UNIFORM)

def test_self_play_average_strategy_converges_to_nash():
    rng = np.random.default_rng(0)
    p0, p1 = RegretMatchingPlayer(), RegretMatchingPlayer()
    play_match(p0, p1, 50_000, rng)
    for player in (p0, p1):
        avg = player.learner.average_strategy()
        assert np.allclose(avg, UNIFORM, atol=0.02), avg
        assert best_response_value(avg) < 0.02

def test_vs_biased_opponent_average_converges_to_best_response():
    # Against 50/25/25 rock-heavy, the unique best response is pure paper
    # (worth +0.25/round).
    rng = np.random.default_rng(0)
    learner = RegretMatchingPlayer()
    payoffs = play_match(learner, FixedStrategyPlayer([0.5, 0.25, 0.25]), 20_000, rng)
    avg = learner.learner.average_strategy()
    assert avg[1] > 0.9, avg
    assert payoffs.mean() > 0.1

def test_match_is_deterministic_given_seed():
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        runs.append(play_match(RegretMatchingPlayer(), RegretMatchingPlayer(), 1_000, rng))
    assert np.array_equal(runs[0], runs[1])
