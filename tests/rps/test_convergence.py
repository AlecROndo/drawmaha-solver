"""Regret matching actually converging, on RPS.

The ledger's own arithmetic is pinned in tests/test_regret_matching.py; these
are the end-to-end runs that prove the rule reaches equilibrium.
"""

import numpy as np

from drawmaha_solver.rps.game import best_response_value
from drawmaha_solver.rps.players import FixedStrategyPlayer, RegretMatchingPlayer, play_match

UNIFORM = np.full(3, 1 / 3)

def test_self_play_average_strategy_converges_to_nash():
    rng = np.random.default_rng(0)
    p0, p1 = RegretMatchingPlayer(), RegretMatchingPlayer()
    play_match(p0, p1, n_rounds=50_000, rng=rng)
    for player in (p0, p1):
        avg = player.learner.average_strategy()
        assert np.allclose(avg, UNIFORM, atol=0.02), avg
        assert best_response_value(avg) < 0.02

def test_vs_biased_opponent_average_converges_to_best_response():
    # Against 50/25/25 rock-heavy, the unique best response is pure paper
    # (worth +0.25/round).
    rng = np.random.default_rng(0)
    learner = RegretMatchingPlayer()
    payoffs = play_match(
        learner, FixedStrategyPlayer([0.5, 0.25, 0.25]), n_rounds=20_000, rng=rng
    )
    avg = learner.learner.average_strategy()
    assert avg[1] > 0.9, avg
    assert payoffs.mean() > 0.1

def test_match_is_deterministic_given_seed():
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        runs.append(
            play_match(RegretMatchingPlayer(), RegretMatchingPlayer(), n_rounds=1_000, rng=rng)
        )
    assert np.array_equal(runs[0], runs[1])
