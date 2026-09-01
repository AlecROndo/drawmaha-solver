import json

import numpy as np
import pytest

from drawmaha_solver.kuhn.analysis import Trajectory, run, to_json, write_json
from drawmaha_solver.kuhn.game import Action, Card, InfoSet, all_infosets

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

# ---------------------------------------------------------------------------
# Export for the visualizer
# ---------------------------------------------------------------------------

def test_the_export_keys_every_infoset_by_its_literature_label(trajectory):
    payload = to_json(trajectory)
    assert set(payload["bet"]) == {str(spot) for spot in all_infosets()}
    assert set(payload["closedForm"]) == set(payload["bet"])
    assert "Kpb" in payload["bet"] and "K" in payload["bet"]

def test_the_export_is_plain_json(trajectory, tmp_path):
    # numpy scalars are neither int nor float to json.dumps, so the payload has
    # to be cast at the boundary or it raises on the first one.
    json.dumps(to_json(trajectory))
    path = tmp_path / "nested" / "solve.json"
    write_json(trajectory, path)
    assert json.loads(path.read_text())["gameValue"] == pytest.approx(-1 / 18, abs=2e-3)

def test_the_exported_series_all_run_the_length_of_the_run(trajectory):
    payload = to_json(trajectory)
    n = len(payload["iterations"])
    assert len(payload["exploitabilityAverage"]) == n
    assert len(payload["exploitabilityCurrent"]) == n
    assert all(len(series) == n for series in payload["bet"].values())

def test_the_exported_closed_form_uses_the_alpha_the_solver_found(trajectory):
    # The page draws solved against exact side by side, so the exact side must
    # be the family member this run landed on, not an arbitrary one.
    payload = to_json(trajectory)
    assert payload["closedForm"]["J"] == pytest.approx(payload["alpha"])
    assert payload["closedForm"]["K"] == pytest.approx(3 * payload["alpha"])
    assert payload["closedForm"]["Kpb"] == 1.0
    assert payload["closedForm"]["Q"] == 0.0

def test_the_exported_bet_series_agree_with_the_final_average(trajectory):
    payload = to_json(trajectory)
    for spot in all_infosets():
        assert payload["bet"][str(spot)][-1] == pytest.approx(
            trajectory.final_average[spot][Action.BET]
        )
