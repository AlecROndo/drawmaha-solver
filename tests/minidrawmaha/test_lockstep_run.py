"""The run driver: kept checkpoints on the grid, a resume file, and nothing lost.

A small listed slice stands in for the whole game so a run takes seconds.
"""

from __future__ import annotations

import json
import signal

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.lockstep import load_lockstep, new_lockstep, read_lockstep, train_lockstep
from drawmaha_solver.minidrawmaha.lockstep_run import RESUME, kept_name, run_to
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.regret_rules import Average, RegretRule
from drawmaha_solver.regret_matching import RegretMatcher

BOTH = (Average.UNIFORM, Average.QUADRATIC)
SLOTS = ("cumulative_regret", "strategy_sum", "extra_sums", "stamp")

class GrowingTable(dict):
    def __missing__(self, key) -> RegretMatcher:
        ledger = self[key] = RegretMatcher(len(key.legal_actions()), extra_averages=2)
        return ledger

@pytest.fixture(scope="module")
def keys():
    grown = GrowingTable()
    train_lockstep(new_lockstep(0, workers=2, table=grown, rule="dcfr", averages=BOTH), 30)
    return list(grown)

def straight(keys, iterations):
    solve = new_lockstep(
        0, workers=2, table=PackedTable.listed(keys, extra_averages=2), rule="dcfr", averages=BOTH
    )
    return train_lockstep(solve, iterations).table

def same(first, second) -> bool:
    return all(np.array_equal(getattr(first, s), getattr(second, s)) for s in SLOTS)

def run(out, keys, target, **kw):
    return run_to(
        out, rule="dcfr", target=target, workers=2, keys=keys, grid=(10, 20), log=lambda _: None, **kw
    )

def test_a_run_keeps_its_grid_and_ends_at_its_target(tmp_path, keys):
    commits = []
    solve = run(tmp_path, keys, 30, commit=lambda: commits.append(1))
    assert solve.iteration == 30
    for mark in (10, 20):
        assert read_lockstep(tmp_path / kept_name(mark))["iteration"] == mark
    assert read_lockstep(tmp_path / RESUME)["iteration"] == 30
    assert same(solve.table, straight(keys, 30))
    kept = load_lockstep(
        tmp_path / kept_name(20),
        table=PackedTable.listed(keys, extra_averages=2),
        workers=2,
        rule="dcfr",
        averages=BOTH,
    )
    assert same(kept.table, straight(keys, 20))
    lines = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert lines[-1]["iteration"] == 30 and lines[-1]["stopped"] == "done"
    assert commits and not list(tmp_path.glob("*.tmp"))

def test_a_run_stopped_and_restarted_equals_one_straight_run(tmp_path, keys):
    assert run(tmp_path, keys, 13).iteration == 13
    solve = run(tmp_path, keys, 30)
    assert same(solve.table, straight(keys, 30))
    assert run(tmp_path, keys, 30).iteration == 30  # already there: nothing to do

def test_sigterm_finishes_the_iteration_saves_and_resumes_exactly(tmp_path, keys):
    raised = []

    def log(line):
        # The first progress record, i.e. after the first iteration's save.
        if line.startswith("{") and not raised:
            raised.append(1)
            signal.raise_signal(signal.SIGTERM)

    solve = run_to(
        tmp_path, rule="dcfr", target=30, workers=2, keys=keys, grid=(10, 20),
        save_every_s=0.0, log=log,
    )
    stopped_at = solve.iteration
    assert 0 < stopped_at < 30
    assert read_lockstep(tmp_path / RESUME)["iteration"] == stopped_at
    last = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[-1])
    assert last["stopped"] == "SIGTERM"
    assert same(run(tmp_path, keys, 30).table, straight(keys, 30))

def test_a_deadline_stops_the_run_with_its_state_saved(tmp_path, keys):
    solve = run(tmp_path, keys, 30, deadline_s=0.0)
    assert solve.iteration < 30
    assert read_lockstep(tmp_path / RESUME)["iteration"] == solve.iteration
