import numpy as np
import pytest

from drawmaha_solver.regret_matching import RegretMatcher

UNIFORM = np.full(3, 1 / 3)

# ---------------------------------------------------------------------------
# The rung-0 ledger math, unchanged
# ---------------------------------------------------------------------------

def test_initial_strategy_is_uniform():
    assert np.allclose(RegretMatcher(3).strategy(), UNIFORM)
    assert np.allclose(RegretMatcher(3).average_strategy(), UNIFORM)

def test_rejects_fewer_than_two_actions():
    with pytest.raises(ValueError):
        RegretMatcher(1)

def test_rejects_a_non_integer_action_count():
    # np.zeros would reject a float anyway, but as a TypeError reading
    # "expected a sequence of integers" — which sends the reader looking at
    # the array shape rather than at the action count.
    with pytest.raises(ValueError, match="n_actions"):
        RegretMatcher(3.0)

def test_all_negative_regret_falls_back_to_uniform():
    m = RegretMatcher(3)
    m.cumulative_regret = np.array([-1.0, -2.0, -0.5])
    assert np.allclose(m.strategy(), UNIFORM)

def test_strategy_proportional_to_positive_regret_only():
    m = RegretMatcher(3)
    m.cumulative_regret = np.array([3.0, 1.0, -5.0])
    assert np.allclose(m.strategy(), [0.75, 0.25, 0.0])

def test_update_accumulates_regret_against_expected_utility():
    m = RegretMatcher(3)
    # Opponent showed paper: utilities (rock -1, paper 0, scissors +1).
    # Current strategy is uniform, so expected utility = 0 and the regret
    # increment is the utility vector itself.
    m.update(np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(m.cumulative_regret, [-1.0, 0.0, 1.0])
    # The strategy in effect for the round was banked into the average.
    assert np.allclose(m.strategy_sum, UNIFORM)

def test_update_measures_regret_relative_to_current_strategy():
    m = RegretMatcher(3)
    m.cumulative_regret = np.array([1.0, 1.0, 0.0])  # current strategy (.5, .5, 0)
    # Utilities (2, 0, 1): expected utility = .5*2 + .5*0 = 1,
    # so increments are (2-1, 0-1, 1-1) = (1, -1, 0).
    m.update(np.array([2.0, 0.0, 1.0]))
    assert np.allclose(m.cumulative_regret, [2.0, 0.0, 0.0])
    assert np.allclose(m.strategy_sum, [0.5, 0.5, 0.0])

def test_average_strategy_is_mean_of_played_strategies():
    m = RegretMatcher(3)
    m.strategy_sum = np.array([1.0, 3.0, 0.0])
    assert np.allclose(m.average_strategy(), [0.25, 0.75, 0.0])

# ---------------------------------------------------------------------------
# The rung-1 addition: one call, two independent weights
# ---------------------------------------------------------------------------

def test_unweighted_update_is_the_rung_zero_behavior():
    # Rung 0 calls update(u) with no weights and all 39 of its tests must keep
    # passing, so the defaults have to be exactly weight 1.
    weighted, plain = RegretMatcher(3), RegretMatcher(3)
    weighted.update(np.array([2.0, 0.0, 1.0]), regret_weight=1.0, strategy_weight=1.0)
    plain.update(np.array([2.0, 0.0, 1.0]))
    assert np.allclose(weighted.cumulative_regret, plain.cumulative_regret)
    assert np.allclose(weighted.strategy_sum, plain.strategy_sum)

def test_regret_weight_scales_regret_and_leaves_the_average_alone():
    m = RegretMatcher(3)
    m.update(np.array([-1.0, 0.0, 1.0]), regret_weight=0.25)
    assert np.allclose(m.cumulative_regret, [-0.25, 0.0, 0.25])
    assert np.allclose(m.strategy_sum, UNIFORM)

def test_strategy_weight_scales_the_average_and_leaves_regret_alone():
    m = RegretMatcher(3)
    m.update(np.array([-1.0, 0.0, 1.0]), strategy_weight=0.5)
    assert np.allclose(m.cumulative_regret, [-1.0, 0.0, 1.0])
    assert np.allclose(m.strategy_sum, UNIFORM * 0.5)

def test_the_two_weights_are_independent():
    # The classic CFR bug is swapping them, which is only detectable if the
    # two scales can differ — pin that they do.
    m = RegretMatcher(3)
    m.update(np.array([-1.0, 0.0, 1.0]), regret_weight=0.25, strategy_weight=0.5)
    assert np.allclose(m.cumulative_regret, [-0.25, 0.0, 0.25])
    assert np.allclose(m.strategy_sum, UNIFORM * 0.5)

def test_zero_reach_is_a_legal_no_op():
    # A line the walk reaches with probability zero contributes nothing; that
    # is a real position in the tree, not a caller error.
    m = RegretMatcher(3)
    m.update(np.array([-1.0, 0.0, 1.0]), regret_weight=0.0, strategy_weight=0.0)
    assert np.allclose(m.cumulative_regret, np.zeros(3))
    assert np.allclose(m.strategy_sum, np.zeros(3))

def test_negative_weights_are_rejected():
    # Every weight the ladder produces is non-negative, so a negative one
    # means the walk computed a reach wrong.
    m = RegretMatcher(3)
    with pytest.raises(ValueError, match="regret_weight"):
        m.update(np.array([1.0, 0.0, 0.0]), regret_weight=-0.5)
    with pytest.raises(ValueError, match="strategy_weight"):
        m.update(np.array([1.0, 0.0, 0.0]), strategy_weight=-0.5)

def test_weights_above_one_are_allowed():
    # No upper bound on purpose: vanilla CFR passes reaches in [0, 1], but
    # linear CFR weights the strategy sum by the iteration index and sampling
    # variants divide by a sampling probability. Capping at 1 here would have
    # to be ripped back out at the next rung.
    m = RegretMatcher(3)
    m.update(np.array([-1.0, 0.0, 1.0]), regret_weight=7.0, strategy_weight=4.0)
    assert np.allclose(m.cumulative_regret, [-7.0, 0.0, 7.0])
    assert np.allclose(m.strategy_sum, UNIFORM * 4.0)

# ---------------------------------------------------------------------------
# Validate before mutating
# ---------------------------------------------------------------------------

def test_wrong_length_utilities_leave_the_ledger_untouched():
    # The hardening the rung-0 writeup deferred: the old order banked the
    # strategy first, so numpy's broadcast error arrived with strategy_sum
    # already advanced — a half-updated ledger that stays wrong forever.
    m = RegretMatcher(3)
    with pytest.raises(ValueError, match="3"):
        m.update(np.array([1.0, 2.0]))
    assert np.allclose(m.cumulative_regret, np.zeros(3))
    assert np.allclose(m.strategy_sum, np.zeros(3))

def test_non_finite_utilities_leave_the_ledger_untouched():
    m = RegretMatcher(3)
    with pytest.raises(ValueError, match="finite"):
        m.update(np.array([1.0, np.nan, 0.0]))
    assert np.allclose(m.cumulative_regret, np.zeros(3))
    assert np.allclose(m.strategy_sum, np.zeros(3))

def test_bad_weight_leaves_the_ledger_untouched():
    m = RegretMatcher(3)
    with pytest.raises(ValueError):
        m.update(np.array([1.0, 0.0, 0.0]), regret_weight=float("nan"))
    assert np.allclose(m.cumulative_regret, np.zeros(3))
    assert np.allclose(m.strategy_sum, np.zeros(3))
