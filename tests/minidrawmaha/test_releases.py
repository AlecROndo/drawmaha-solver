"""The catalogue of released strategies: what both exporters read off it, pinned once."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from drawmaha_solver.minidrawmaha.answer.releases import (
    PUBLIC,
    RELEASES,
    REPO,
    DEEP_KEY,
    Grade,
    entry_mismatch,
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


# ---------------------------------------------------------------------------
# The entry must be the strategy's own
# ---------------------------------------------------------------------------

DEEP = StrategyInfo(rule="deep", column="exact", iteration=200, seed=0, workers=0)


def test_every_catalogue_entry_s_grade_is_the_grade_of_the_strategy_its_key_names():
    """Each key read back as a strategy: the rung-4 keys spell mark and column, rung 3's entry records its run."""
    for release in RELEASES.values():
        recorded = json.loads((REPO / release.grade_file).read_text())[release.grade_key]
        if release.grade_key == "frozen_strategy":
            info = StrategyInfo(rule=recorded["rule"], column=recorded["column"], iteration=recorded["iteration"], seed=0, workers=10)
            # The answer sheet also records the digest it graded, which is the release's.
            assert recorded["sha256"] == release.sha256
        else:
            _, mark, column = release.grade_key.split("/")
            info = StrategyInfo(rule="deep", column=column, iteration=int(mark), seed=0, workers=0)
        assert entry_mismatch(release, recorded, info) is None, release.name
        assert read_grade(release, info).key == release.grade_key


def test_every_catalogue_entry_is_checkable():
    """An entry the key check cannot read back would be accepted unchecked; no catalogue entry may be one."""
    for release in RELEASES.values():
        recorded = json.loads((REPO / release.grade_file).read_text())[release.grade_key]
        spelled = DEEP_KEY.match(release.grade_key) is not None
        recorded_run = {"rule", "column", "iteration"} <= set(recorded)
        assert spelled or recorded_run, f"{release.name}'s grade entry says nothing the strategy can be held to"
        # And a strategy that is not this one is refused, whichever way the entry is checkable.
        wrong = StrategyInfo(rule="deep", column="current", iteration=1, seed=0, workers=0)
        assert entry_mismatch(release, recorded, wrong) is not None, release.name


def test_a_rung4_key_is_held_to_the_strategy_s_mark_and_column():
    release = RELEASES["deep-pot-exact"]
    entry = {"exploitability": 0.1, "value_p0": 0.0, "hands_per_seat": 5}
    assert entry_mismatch(release, entry, DEEP) is None
    assert entry_mismatch(release, entry, replace(DEEP, iteration=500)) == "it grades iteration = 200 and the strategy's is 500"
    assert entry_mismatch(release, entry, replace(DEEP, column="pi")) == "it grades column = 'exact' and the strategy's is 'pi'"
    assert entry_mismatch(release, entry, replace(DEEP, rule="lcfr")) == "it grades rule = 'deep' and the strategy's is 'lcfr'"


def test_an_entry_that_records_its_run_is_held_to_it_and_a_silent_one_is_not():
    sheet = replace(RELEASES["tabular"], grade_key="some_key")
    lcfr = StrategyInfo(rule="lcfr", column="linear", iteration=2_000_000, seed=0, workers=10)
    recorded = {"exploitability": 0.1, "value_p0": 0.0, "rule": "lcfr", "column": "linear", "iteration": 2_000_000}
    assert entry_mismatch(sheet, recorded, lcfr) is None
    assert entry_mismatch(sheet, recorded, replace(lcfr, iteration=1_000_000)) == (
        "it grades iteration = 2000000 and the strategy's is 1000000"
    )
    assert entry_mismatch(sheet, {"exploitability": 0.1, "value_p0": 0.0}, replace(lcfr, iteration=7)) is None


def test_read_grade_refuses_another_mark_s_grade_under_a_release_s_digest(tmp_path):
    """A catalogue entry whose key and digest drifted apart would quote the wrong strategy's grade; the key is read back."""
    (tmp_path / "grades.json").write_text(
        json.dumps({"deep/00500/exact": {"exploitability": 0.1, "value_p0": 0.0, "hands_per_seat": 5}})
    )
    drifted = replace(RELEASES["deep-pot-exact"], grade_file="grades.json", grade_key="deep/00500/exact")
    with pytest.raises(ValueError, match="iteration = 500 and the strategy's is 200"):
        read_grade(drifted, DEEP, root=tmp_path)


def test_read_grade_refuses_an_entry_that_is_no_grade_by_name(tmp_path):
    """Not an object, a missing number, or a number that is null, a string or a bool: each named, none a bare error."""
    info = StrategyInfo(rule="deep", column="exact", iteration=200, seed=0, workers=0)
    bare = replace(RELEASES["deep-pot-exact"], grade_file="grades.json")
    good = {"exploitability": 0.1, "value_p0": -0.05, "hands_per_seat": 5}

    def refused(entry, error, message):
        (tmp_path / "grades.json").write_text(json.dumps({"deep/00200/exact": entry}))
        with pytest.raises(error, match=message):
            read_grade(bare, info, root=tmp_path)

    refused(0.1, TypeError, "not a grade object but 0.1")
    refused("0.1", TypeError, "not a grade object")
    refused({k: v for k, v in good.items() if k != "exploitability"}, KeyError, "records no exploitability")
    refused({k: v for k, v in good.items() if k != "value_p0"}, KeyError, "records no value_p0")
    for not_a_number in (None, "0.1", True, float("nan"), float("inf")):
        refused({**good, "exploitability": not_a_number}, ValueError, "exploitability = .*not a finite number")
        refused({**good, "value_p0": not_a_number}, ValueError, "value_p0 = .*not a finite number")
    (tmp_path / "grades.json").write_text(json.dumps({"deep/00200/exact": good}))
    assert read_grade(bare, info, root=tmp_path).exploitability == 0.1
