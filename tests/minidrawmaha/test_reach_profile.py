"""`reach_profile` bins every spot's share of the exploitability by how often it is reached.

Checked on the Leduc oracle description the grader's own tests build: the
bins' shares sum to the exploitability the grader reports, every spot lands
in exactly one bin, the per-stage totals agree with the bins, and the check
the file carries (each seat's best-response gain from two further walks) is
what the gains sum to.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_exploitability import LEDUC, PROFILE_IDS, PROFILE_VALUES  # noqa: E402

from drawmaha_solver.minidrawmaha.answer.analysis import REACH_FLOORS, STAGES, reach_profile, stage_of  # noqa: E402
from drawmaha_solver.minidrawmaha.exploitability import exploitability  # noqa: E402


@pytest.mark.parametrize("profile", PROFILE_VALUES, ids=PROFILE_IDS)
def test_the_bins_partition_the_spots_and_sum_to_the_exploitability(profile):
    result = reach_profile(profile, game=LEDUC)
    assert result["exploitability"] == pytest.approx(exploitability(profile, game=LEDUC), abs=1e-12)
    assert result["gain"] == pytest.approx(result["check"]["best_response_minus_value"], abs=1e-12)
    assert sum(row["infosets"] for row in result["bins"]) == result["infosets"]
    assert sum(row["exploitability"] for row in result["bins"]) == pytest.approx(result["exploitability"], abs=1e-12)
    assert sum(row["frequency"] for row in result["bins"]) == pytest.approx(
        sum(stage["frequency"] for stage in result["stages"].values()), abs=1e-12
    )
    for stage in STAGES:
        assert sum(row["stages"][stage]["infosets"] for row in result["bins"]) == result["stages"][stage]["infosets"]
    assert result["infosets"] == sum(len(LEDUC.spot_keys(node)) for node in LEDUC.decisions)


def test_the_bins_are_the_reach_floors_in_descending_order():
    result = reach_profile(PROFILE_VALUES[0], game=LEDUC)
    assert [row["floor"] for row in result["bins"]] == list(REACH_FLOORS)
    assert result["bins"][0]["ceiling"] is None
    assert [row["ceiling"] for row in result["bins"][1:]] == list(REACH_FLOORS[:-1])


def test_mini_drawmahas_decisions_fall_into_the_three_stages():
    from drawmaha_solver.minidrawmaha.exploitability import minidrawmaha_game

    game = minidrawmaha_game()
    stages = {stage_of(node) for node in game.decisions}
    assert stages == set(STAGES)
    for node in game.decisions:
        if stage_of(node) == "draw":
            assert node.level == 1 and node.discards == (0, 0) or node.discards[node.player] == 0
        if node.level == 2:
            assert stage_of(node) == "round 2"
