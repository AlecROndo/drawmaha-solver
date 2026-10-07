"""W processes over one shared table equal the in-process lockstep reference, bit for bit.

That equality is the whole claim: the workers only read during a phase, the
parent banks their records in worker order, so scheduling cannot change a bit.
Plus: a worker that raises surfaces its traceback, one that dies is noticed
in seconds, keys that do not match the table are refused, and no shared memory
or worker process outlives the run however it ends — a worker raising, dying
while starting or killed mid-run, or the caller raising.
"""

from __future__ import annotations

import os
import signal
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.lockstep import new_lockstep, train_lockstep
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.parallel import ParallelLockstep
from drawmaha_solver.minidrawmaha.regret_rules import Average, RegretRule
from drawmaha_solver.regret_matching import RegretMatcher

BOTH_COLUMNS = (Average.UNIFORM, Average.QUADRATIC)
SLOTS = ("cumulative_regret", "strategy_sum", "extra_sums", "stamp")

@pytest.fixture(autouse=True)
def importable_tests(monkeypatch):
    # Spawned workers unpickle a test-defined deal by module name, which
    # resolves only with the repo root on the path they inherit.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))

class GrowingTable(dict):
    def __missing__(self, key) -> RegretMatcher:
        ledger = self[key] = RegretMatcher(len(key.legal_actions()), extra_averages=2)
        return ledger

def keys_met(seed: int, workers: int, iterations: int, rule: RegretRule) -> list:
    grown = GrowingTable()
    solve = new_lockstep(seed, workers=workers, table=grown, rule=rule, averages=BOTH_COLUMNS)
    train_lockstep(solve, iterations)
    return list(grown)

def same(first: PackedTable, second: PackedTable) -> bool:
    return all(np.array_equal(getattr(first, s), getattr(second, s)) for s in SLOTS)

@pytest.mark.parametrize("rule", [RegretRule.DCFR, RegretRule.CFR_PLUS])
def test_three_processes_equal_three_walks_in_one_process(rule):
    seed, workers, iterations = 5, 3, 15
    keys = keys_met(seed, workers, iterations, rule)
    run = {"workers": workers, "rule": rule, "averages": BOTH_COLUMNS}
    reference = train_lockstep(
        new_lockstep(seed, table=PackedTable.listed(keys, extra_averages=2), **run), iterations
    )
    solve = new_lockstep(seed, table=PackedTable.listed(keys, extra_averages=2), **run)
    with ParallelLockstep(solve, keys=keys) as parallel:
        parallel.train(7)
        parallel.train(iterations - 7)
    assert solve.iteration == iterations
    assert same(solve.table, reference.table)
    assert solve.table.strategy_sum.any() and solve.table.extra_sums.any()

def test_the_whole_game_runs_in_parallel():
    run = {"workers": 2, "rule": "lcfr", "averages": BOTH_COLUMNS}
    reference = train_lockstep(new_lockstep(9, **run), 4)
    solve = new_lockstep(9, **run)
    with ParallelLockstep(solve) as parallel:
        parallel.train(4)
    assert same(solve.table, reference.table)

def test_a_parallel_run_resumes_in_process_and_back():
    # The parallel run is only a scheduler: its table after exit continues
    # in-process (and in parallel again) exactly as one straight run.
    seed, workers = 2, 2
    keys = keys_met(seed, workers, 12, RegretRule.VANILLA)
    run = {"workers": workers, "averages": BOTH_COLUMNS}
    straight = train_lockstep(
        new_lockstep(seed, table=PackedTable.listed(keys, extra_averages=2), **run), 12
    )
    solve = new_lockstep(seed, table=PackedTable.listed(keys, extra_averages=2), **run)
    with ParallelLockstep(solve, keys=keys) as parallel:
        parallel.train(4)
    train_lockstep(solve, 4)
    with ParallelLockstep(solve, keys=keys) as parallel:
        parallel.train(4)
    assert same(solve.table, straight.table)

def assert_unlinked(names: list[str]) -> None:
    assert names
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)

def broken_deal(rng):
    raise RuntimeError("the deck is on fire")

def test_a_worker_that_raises_surfaces_and_leaves_no_shared_memory():
    solve = new_lockstep(0, workers=2, deal=broken_deal)
    parallel = ParallelLockstep(solve)
    with pytest.raises(RuntimeError, match="the deck is on fire"):
        with parallel:
            names = [block.name for block in parallel._blocks]
            parallel.train(1)
    assert_unlinked(names)
    assert isinstance(solve.table, PackedTable) and not solve.table.cumulative_regret.any()

def test_a_parallel_run_needs_a_packed_table():
    with pytest.raises(TypeError, match="PackedTable"):
        ParallelLockstep(new_lockstep(0, workers=2, table={}))

class DiesInTheWorker:
    """Pickles in the parent; unpickling it in the worker raises."""

    def __reduce__(self):
        return (broken_deal, (None,))

@pytest.fixture
def created(monkeypatch) -> list[str]:
    """The names of every shared block the parent creates during the test."""
    names: list[str] = []

    class Recording(shared_memory.SharedMemory):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if kwargs.get("create"):
                names.append(self.name)

    monkeypatch.setattr(shared_memory, "SharedMemory", Recording)
    return names

def test_a_worker_that_dies_while_starting_fails_fast(created):
    # The keys are handed to the workers; one they cannot rebuild kills them
    # before they report ready. The parent must say so at once, not after the
    # startup timeout.
    solve = new_lockstep(0, workers=2, table=PackedTable.listed([]))
    parallel = ParallelLockstep(solve, keys=[DiesInTheWorker()])
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exited while starting"):
        with parallel:
            pass
    assert time.monotonic() - started < 60
    assert_unlinked(created)
    assert not parallel._processes

def test_a_worker_killed_mid_run_is_noticed_within_seconds():
    solve = new_lockstep(0, workers=2)
    started = None
    with pytest.raises(RuntimeError, match="dead workers"):
        with ParallelLockstep(solve) as parallel:
            parallel.train(2)
            names = [block.name for block in parallel._blocks]
            processes = list(parallel._processes)
            os.kill(processes[1].pid, signal.SIGKILL)
            started = time.monotonic()
            parallel.train(1_000)
    assert time.monotonic() - started < 10
    assert_unlinked(names)
    assert not any(process.is_alive() for process in processes)
    assert solve.iteration == 2 and solve.table.cumulative_regret.any()

def test_a_caller_that_raises_stops_the_workers_and_leaves_no_shared_memory():
    solve = new_lockstep(0, workers=2)
    with pytest.raises(KeyError, match="the caller"):
        with ParallelLockstep(solve) as parallel:
            parallel.train(1)
            names = [block.name for block in parallel._blocks]
            processes = list(parallel._processes)
            raise KeyError("the caller")
    assert_unlinked(names)
    assert not any(process.is_alive() for process in processes)
    assert solve.iteration == 1 and solve.table.cumulative_regret.any()

def test_keys_that_do_not_match_the_table_are_refused_at_startup():
    # The workers rebuild the table's index from `keys`. A key list that
    # lays out fewer rows than the parent's table would fit inside its
    # shared buffers and silently bank the walks in the wrong columns.
    keys = keys_met(1, 2, 3, RegretRule.VANILLA)
    table = PackedTable.listed(keys, extra_averages=2)
    solve = new_lockstep(1, workers=2, table=table, averages=BOTH_COLUMNS)
    with pytest.raises(RuntimeError, match="cumulative_regret must be"):
        with ParallelLockstep(solve, keys=keys[:-1]):
            pass
