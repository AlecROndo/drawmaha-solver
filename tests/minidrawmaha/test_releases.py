"""The catalogue of released strategies: what both exporters read off it, pinned once."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from drawmaha_solver.minidrawmaha.answer.releases import (
    PUBLIC,
    RELEASES,
    REPO,
    Grade,
    provenance_json,
    read_grade,
    release_of,
)
from drawmaha_solver.minidrawmaha.answer.strategy import RELEASE_URL, STRATEGY_SHA256, StrategyInfo


def test_the_catalogue_names_five_strategies_three_of_them_behind_solver_s_switch():
    assert set(RELEASES) == {"tabular", "deep-pi", "deep-exact", "deep-pot-exact", "deep-pot-pi"}
    assert len({release.sha256 for release in RELEASES.values()}) == 5
    exported = [release for release in RELEASES.values() if release.exported]
    assert sorted(release.name for release in exported) == ["deep-exact", "deep-pi", "tabular"]
    assert len({release.directory for release in exported}) == 3
    assert RELEASES["tabular"].sha256 == STRATEGY_SHA256
    assert RELEASES["tabular"].url == RELEASE_URL
    for name, release in RELEASES.items():
        assert release.name == name
        assert release.url.endswith(f"/{release.tag}/{release.asset}")
        assert (REPO / release.grade_file).exists(), release.grade_file
        if release.exported:
            assert release.out == PUBLIC / release.directory
        else:
            with pytest.raises(ValueError, match=name):
                release.out


def test_the_pot_arm_s_two_averages_share_a_release_and_a_grades_file():
    exact, pi = RELEASES["deep-pot-exact"], RELEASES["deep-pot-pi"]
    assert exact.tag == pi.tag == "rung4-deep-pot0.5-t200"
    assert exact.grade_file == pi.grade_file == "figures/rung4/pot_0.5-256x4-s16000-lr5e-4.json"
    assert (exact.grade_key, pi.grade_key) == ("deep/00200/exact", "deep/00200/pi")
    assert exact.asset.endswith("-t200-exact.npz") and pi.asset.endswith("-t200-pi.npz")


def test_release_of_finds_a_release_by_digest_in_the_catalogue_it_is_given():
    assert release_of(STRATEGY_SHA256) is RELEASES["tabular"]
    assert release_of(RELEASES["deep-pot-exact"].sha256) is RELEASES["deep-pot-exact"]
    assert release_of("0" * 64) is None
    only = {"tabular": RELEASES["tabular"]}
    assert release_of(RELEASES["deep-pot-exact"].sha256, only) is None


def test_provenance_is_one_block_for_both_manifests():
    info = StrategyInfo(rule="deep", column="exact", iteration=200, seed=0, workers=0)
    block = provenance_json(info, sha256="ab" * 32, asset="x.npz", release=RELEASES["deep-pot-exact"])
    assert block == {
        "sha256": "ab" * 32,
        "release": "rung4-deep-pot0.5-t200",
        "asset": "x.npz",
        "rule": "deep",
        "column": "exact",
        "iteration": 200,
        "workers": 0,
        "seed": 0,
    }
    assert provenance_json(info, sha256="ab" * 32, asset="x.npz", release=None)["release"] is None


def test_read_grade_quotes_the_pot_arm_s_exact_average_with_its_self_play_value():
    info = StrategyInfo(rule="deep", column="exact", iteration=200, seed=0, workers=0)
    grade = read_grade(RELEASES["deep-pot-exact"], info)
    recorded = json.loads((REPO / RELEASES["deep-pot-exact"].grade_file).read_text())["deep/00200/exact"]
    assert grade == Grade(
        exploitability=recorded["exploitability"],
        value_p0=recorded["value_p0"],
        hands_per_seat=2_000_000,
        file="figures/rung4/pot_0.5-256x4-s16000-lr5e-4.json",
        key="deep/00200/exact",
    )
    assert grade.as_json() == {
        "exploitability": recorded["exploitability"],
        "valueP0": recorded["value_p0"],
        "handsPerSeat": 2_000_000,
        "file": "figures/rung4/pot_0.5-256x4-s16000-lr5e-4.json",
        "key": "deep/00200/exact",
    }
    # The best graded Deep CFR object so far, and what the pack deals against: under the t = 500 release's.
    assert grade.exploitability < read_grade(RELEASES["deep-exact"], replace(info, iteration=500)).exploitability


def test_read_grade_refuses_an_entry_without_a_self_play_value(tmp_path):
    """The Play tab quotes what each seat is worth; a grade without `value_p0` would print nothing there."""
    (tmp_path / "grades.json").write_text(json.dumps({"deep/00200/exact": {"exploitability": 0.1, "hands_per_seat": 5}}))
    bare = replace(RELEASES["deep-pot-exact"], grade_file="grades.json")
    info = StrategyInfo(rule="deep", column="exact", iteration=200, seed=0, workers=0)
    with pytest.raises(KeyError, match="value_p0"):
        read_grade(bare, info, root=tmp_path)
    assert isinstance(Path(bare.grade_file), Path)
