import numpy as np
import pytest

from drawmaha_solver.kuhn.analysis import Trajectory, run
from drawmaha_solver.kuhn.game import Action, Card, InfoSet

J, Q, K = Card.JACK, Card.QUEEN, Card.KING

@pytest.fixture(scope="module")
def trajectory() -> Trajectory:
    return run(2_000)

# ---------------------------------------------------------------------------
# Shape of the record
# ---------------------------------------------------------------------------

def test_checkpoints_are_log_spaced_and_end_at_the_last_iteration(trajectory):
    assert trajectory.iterations[0] == 1
    assert trajectory.iterations[-1] == 2_000
    assert np.all(np.diff(trajectory.iterations) > 0)

def test_every_series_has_one_point_per_checkpoint(trajectory):
    n = len(trajectory.iterations)
    assert trajectory.exploitability_average.shape == (n,)
    assert trajectory.exploitability_current.shape == (n,)
    assert trajectory.opening.shape == (n, 3)

def test_the_run_is_deterministic():
    # Vanilla CFR enumerates the tree, so there is no seed and two runs must
    # agree exactly. This is what makes figures/rung1 a regression fingerprint.
    first, second = run(200), run(200)
    assert np.array_equal(first.exploitability_average, second.exploitability_average)
    assert np.array_equal(first.opening, second.opening)

def test_rejects_too_few_iterations():
    with pytest.raises(ValueError, match="iterations"):
        run(0)

# ---------------------------------------------------------------------------
# What the figures are supposed to show
# ---------------------------------------------------------------------------

def test_the_average_converges_and_the_current_strategy_does_not(trajectory):
    # The headline of the exploitability figure: one line marches down, the
    # other stays put. Without this the plot would be a pretty coincidence.
    assert trajectory.exploitability_average[-1] < 0.01
    assert trajectory.exploitability_average[-1] < trajectory.exploitability_average[0]
    assert trajectory.exploitability_current[-1] > 0.1

def test_the_solve_reaches_the_closed_form(trajectory):
    assert trajectory.game_value == pytest.approx(-1 / 18, abs=2e-3)
    assert trajectory.exploitability_average[-1] == pytest.approx(0.0, abs=0.01)

def test_the_opening_series_track_the_three_open_infosets(trajectory):
    # Columns are J, Q, K opening bet frequency — the three cells of the top
    # row of the strategy grid.
    final = {
        card: trajectory.final_average[InfoSet(card, ())][Action.BET]
        for card in (J, Q, K)
    }
    assert trajectory.opening[-1] == pytest.approx([final[J], final[Q], final[K]])

def test_the_bluff_settles_at_a_third_of_the_value_bet(trajectory):
    # What the strategy-convergence figure exists to show: alpha is free, the
    # ratio is not.
    alpha, queen, king = trajectory.opening[-1]
    assert king == pytest.approx(3 * alpha, abs=0.05)
    assert queen == pytest.approx(0.0, abs=0.05)

def test_the_final_average_covers_every_infoset(trajectory):
    assert len(trajectory.final_average) == 12
    for probabilities in trajectory.final_average.values():
        assert probabilities.sum() == pytest.approx(1.0)
