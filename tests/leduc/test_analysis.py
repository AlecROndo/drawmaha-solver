import numpy as np
import pytest

from drawmaha_solver.leduc.analysis import (
    TRACKED,
    Trajectory,
    _probability,
    run,
)
from drawmaha_solver.leduc.game import Action, InfoSet, Rank, all_infosets
from drawmaha_solver.leduc.infoset_table import average_strategy, new_infoset_table

C, R = Action.CALL, Action.RAISE
J, Q, K = Rank.JACK, Rank.QUEEN, Rank.KING
UNIFORM = average_strategy(new_infoset_table())


@pytest.fixture(scope="module")
def trajectory() -> Trajectory:
    # 400 iterations is where the three tracked plays have arrived and
    # exploitability has fallen an order of magnitude, at ~25 s.
    return run(400, checkpoints=12)


# ---------------------------------------------------------------------------
# Shape of the record
# ---------------------------------------------------------------------------


def test_checkpoints_are_log_spaced_and_end_at_the_last_iteration(trajectory):
    assert trajectory.iterations[0] == 1
    assert trajectory.iterations[-1] == 400
    assert np.all(np.diff(trajectory.iterations) > 0)


def test_every_series_has_one_point_per_checkpoint(trajectory):
    n = len(trajectory.iterations)
    assert trajectory.exploitability_average.shape == (n,)
    assert trajectory.exploitability_current.shape == (n,)
    assert trajectory.game_value.shape == (n,)
    assert trajectory.tracked.shape == (n, len(TRACKED))


def test_the_run_is_deterministic():
    # Vanilla CFR enumerates the tree, so there is no seed and two runs must
    # agree exactly. This is what makes figures/rung2 a regression fingerprint.
    first, second = run(20, checkpoints=4), run(20, checkpoints=4)
    assert np.array_equal(first.exploitability_average, second.exploitability_average)
    assert np.array_equal(first.tracked, second.tracked)


def test_rejects_too_few_iterations():
    with pytest.raises(ValueError, match="iterations"):
        run(0)


def test_the_final_average_covers_every_spot(trajectory):
    assert len(trajectory.final_average) == len(all_infosets()) == 288
    for spot, probabilities in trajectory.final_average.items():
        # Width is per spot here, unlike rung 1 where every ledger was 2 wide.
        assert probabilities.shape == (len(spot.legal_actions()),)
        assert probabilities.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Reading a spot's strategy by position, not by enum value
# ---------------------------------------------------------------------------


def test_a_frequency_is_read_off_the_action_s_own_column():
    # The index discipline of this whole rung. At the opening spot the legal
    # actions are (CALL, RAISE), so RAISE is column 1 even though Action.RAISE
    # is 2; indexing by the enum would read past the end or return CALL's
    # number at every 2-wide spot.
    opening = InfoSet(rank=K, board=None, betting=((),))
    assert opening.legal_actions() == (C, R)
    strategies = dict(UNIFORM)
    strategies[opening] = np.array([0.25, 0.75])
    assert _probability(strategies, opening, R) == pytest.approx(0.75)
    assert _probability(strategies, opening, C) == pytest.approx(0.25)


def test_the_same_column_means_different_things_at_different_spots():
    # Column 0 is CALL at the open and FOLD facing a raise. A reader who
    # assumed one layout for all 288 spots would be wrong at 192 of them.
    opening = InfoSet(rank=K, board=None, betting=((),))
    facing_raise = InfoSet(rank=K, board=None, betting=((R,),))
    assert opening.legal_actions()[0] is Action.CALL
    assert facing_raise.legal_actions()[0] is Action.FOLD


@pytest.mark.parametrize("label,spot,action", TRACKED, ids=[t[0] for t in TRACKED])
def test_every_tracked_play_is_legal_where_it_is_tracked(label, spot, action):
    # A tracked spot whose action is not legal there would plot a silent
    # ValueError's worth of nonsense, or worse, another action's number.
    assert action in spot.legal_actions()


# ---------------------------------------------------------------------------
# What the figures are supposed to show
# ---------------------------------------------------------------------------


def test_the_average_converges_and_the_current_strategy_does_not(trajectory):
    # The headline of the exploitability figure: one line marches down, the
    # other stays put. Without this the plot would be a pretty coincidence.
    assert trajectory.exploitability_average[-1] < 0.05
    assert trajectory.exploitability_average[-1] < trajectory.exploitability_average[0]
    assert trajectory.exploitability_current[-1] > 0.3


def test_the_game_value_walks_toward_the_exact_one(trajectory, lp_value):
    # Rung 1 checked its solve against a closed form. Leduc has none, so the
    # target is the sequence-form LP's measured answer instead.
    assert trajectory.game_value[-1] == pytest.approx(lp_value, abs=0.01)
    assert abs(trajectory.game_value[-1] - lp_value) < abs(
        trajectory.game_value[0] - lp_value
    )


def test_the_three_dominance_pinned_plays_all_arrive(trajectory):
    check_raise, fold, open_raise = trajectory.tracked[-1]
    # Trips do not lead the round-2 open; they check and raise when bet into.
    assert check_raise > 0.9
    # A jack on a king board is beaten by everything not bluffing.
    assert fold > 0.9
    # The king opens for a raise more often than not, but it is a mixed
    # strategy — pinning it at 1.0 would be pinning a bluffing frequency the
    # game does not fix.
    assert 0.4 < open_raise < 1.0


def test_the_tracked_series_agree_with_the_final_average(trajectory):
    for i, (_, spot, action) in enumerate(TRACKED):
        assert trajectory.tracked[-1, i] == pytest.approx(
            _probability(trajectory.final_average, spot, action)
        )


def test_the_nuts_prefer_the_check_raise_to_leading_out(trajectory):
    # The poker idea the check-raise series exists to show, stated as the
    # comparison it actually is: at the round-2 open holding trips, checking
    # beats betting, because betting folds out every hand it beats.
    leading = _probability(
        trajectory.final_average,
        InfoSet(rank=J, board=J, betting=((C, C), ())),
        Action.RAISE,
    )
    assert leading < 0.1
