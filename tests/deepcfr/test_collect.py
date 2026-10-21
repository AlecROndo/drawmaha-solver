"""The collector: K walks for one seat on W processes, the same rows for any W.

The claim is that `Collected` is a function of (seed, t, seat, k) and nothing
else: the in-process collector equals walking k by k by hand, and W = 1 equals
W = 3 bytewise, chunks included. Beside it: the workers read the regrets as they
stand at `collect` time (the parent writes between phases and the next phase
walks the new σ), a worker that raises surfaces its message in the parent,
bad arguments are refused by name, and the shared block is unlinked however
the `with` ends.
"""

from __future__ import annotations

import os
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

import drawmaha_solver.deepcfr.collect as collect
from drawmaha_solver.deepcfr.collect import Collected, Collector, joined, rows_of
from drawmaha_solver.deepcfr.reservoir import TARGET_WIDTH
from drawmaha_solver.minidrawmaha.rules import features
from drawmaha_solver.minidrawmaha.solver import lockstep
from drawmaha_solver.minidrawmaha.rules.game import random_deal
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.solver.regret_rules import Average

UNIFORM = (Average.UNIFORM,)

@pytest.fixture(autouse=True)
def importable_tests(monkeypatch):
    # Spawned workers unpickle a test-defined deal by module name, which
    # resolves only with the repo root on the path they inherit.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))

def exploding_deal(rng: np.random.Generator):
    raise RuntimeError("boom")

# How many deals this process has made; `first_deal_explodes` fails the first
# and deals the rest, so a worker survives its own failure.
_deals = 0

def first_deal_explodes(rng: np.random.Generator):
    global _deals
    _deals += 1
    if _deals == 1:
        raise RuntimeError("boom")
    return random_deal(rng)

def dying_deal(rng: np.random.Generator):
    os._exit(3)

def write_some_regrets(table: PackedTable) -> None:
    """A fixed, non-uniform σ at every row, the same whichever table it is written into."""
    table.cumulative_regret[:] = np.random.default_rng(0).standard_normal(
        table.cumulative_regret.size
    )

def fields(collected: Collected) -> tuple[np.ndarray, ...]:
    return (
        collected.advantage_rows,
        collected.advantage_targets,
        collected.policy_rows,
        collected.policy_targets,
    )

def collected_equal(first: Collected, second: Collected) -> bool:
    return first.walks == second.walks and all(
        a.dtype == b.dtype and np.array_equal(a, b)
        for a, b in zip(fields(first), fields(second), strict=True)
    )

def assert_collected_equal(first: Collected, second: Collected) -> None:
    assert first.walks == second.walks
    for a, b in zip(fields(first), fields(second), strict=True):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)

def walked_by_hand(table: PackedTable, seed: int, t: int, seat: int, walks: int) -> Collected:
    return joined(
        [
            rows_of(lockstep.walk(table, random_deal, seed, t, seat, k, UNIFORM))
            for k in range(walks)
        ]
    )

# ---------------------------------------------------------------------------
# One walk as rows
# ---------------------------------------------------------------------------

def test_rows_of_one_walk_are_its_visits():
    table = PackedTable.whole_game(extra_averages=1)
    write_some_regrets(table)
    recorded = lockstep.walk(table, random_deal, 5, 1, 0, 0, UNIFORM)
    got = rows_of(recorded)
    assert got.walks == 1
    assert got.advantage_rows.dtype == np.int32 and got.policy_rows.dtype == np.int32
    assert got.advantage_targets.dtype == np.float32 and got.policy_targets.dtype == np.float32
    assert got.advantage_targets.shape == (len(recorded.regret), TARGET_WIDTH)
    assert got.policy_targets.shape == (len(recorded.strategy), TARGET_WIDTH)
    assert got.advantage_rows.tolist() == [slot[0] for _, slot, _ in recorded.regret]
    assert got.policy_rows.tolist() == [slot[0] for _, slot, _, _ in recorded.strategy]
    for i, (_, (_, _, width), r) in enumerate(recorded.regret):
        np.testing.assert_array_equal(got.advantage_targets[i, :width], r.astype(np.float32))
        assert not got.advantage_targets[i, width:].any()
    for i, (key, (_, _, width), _, extra) in enumerate(recorded.strategy):
        # The uniform column is σ exactly, read once more from the table.
        sigma = table[key].strategy().astype(np.float32)
        np.testing.assert_array_equal(got.policy_targets[i, :width], sigma)
        np.testing.assert_array_equal(got.policy_targets[i, :width], extra[0].astype(np.float32))
        assert not got.policy_targets[i, width:].any()

def test_rows_of_refuses_a_walk_without_slots_or_without_the_uniform_column():
    table = PackedTable.whole_game(extra_averages=1)
    recorded = lockstep.walk(table, random_deal, 5, 1, 0, 0, UNIFORM)
    unslotted = lockstep.Recorded(
        regret=[(key, None, r) for key, _, r in recorded.regret],
        strategy=[(key, None, p, e) for key, _, p, e in recorded.strategy],
    )
    with pytest.raises(ValueError, match="packed"):
        rows_of(unslotted)
    no_column = lockstep.walk(PackedTable.whole_game(), random_deal, 5, 1, 0, 0, ())
    with pytest.raises(ValueError, match="uniform"):
        rows_of(no_column)
    # One extra column, but the quadratic one: its row is t²·σ, which at t = 3
    # sums to 9, and would be filed as σ by a check on the row count alone.
    quadratic = lockstep.walk(table, random_deal, 5, 3, 0, 0, (Average.QUADRATIC,))
    with pytest.raises(ValueError, match="uniform"):
        rows_of(quadratic)
    # At t = 1 every column weighs 1, so the quadratic row IS σ: the check is
    # on the row's content and rightly takes it, bit for bit the uniform rows.
    write_some_regrets(table)
    at_one = lockstep.walk(table, random_deal, 5, 1, 0, 0, (Average.QUADRATIC,))
    uniform = lockstep.walk(table, random_deal, 5, 1, 0, 0, UNIFORM)
    assert_collected_equal(rows_of(at_one), rows_of(uniform))

def test_joined_keeps_walk_order_and_counts_walks():
    table = PackedTable.whole_game(extra_averages=1)
    parts = [rows_of(lockstep.walk(table, random_deal, 5, 1, 0, k, UNIFORM)) for k in range(3)]
    whole = joined(parts)
    assert whole.walks == 3
    np.testing.assert_array_equal(
        whole.advantage_rows, np.concatenate([p.advantage_rows for p in parts])
    )
    np.testing.assert_array_equal(
        whole.policy_targets, np.concatenate([p.policy_targets for p in parts])
    )
    assert_collected_equal(joined([whole]), whole)

def test_joined_of_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing"):
        joined([])

# ---------------------------------------------------------------------------
# Whose rows they are
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def collected_seat_1() -> Collected:
    with Collector(seed=7, workers=0) as run:
        write_some_regrets(run.table)
        return run.collect(seat=1, t=1, walks=10)

def test_advantage_rows_are_the_seats_and_policy_rows_the_opponents(collected_seat_1):
    layout = features.layout()
    points = layout.locate_rows(collected_seat_1.advantage_rows)[0]
    assert points.size and all(layout.points[p].player == 1 for p in points)
    points = layout.locate_rows(collected_seat_1.policy_rows)[0]
    assert points.size and all(layout.points[p].player == 0 for p in points)

def test_policy_targets_are_distributions_over_the_legal_columns(collected_seat_1):
    layout = features.layout()
    points = layout.locate_rows(collected_seat_1.policy_rows)[0]
    widths = layout.width[points]
    for target, width in zip(collected_seat_1.policy_targets, widths, strict=True):
        assert target[:width].sum() == pytest.approx(1.0, rel=1e-6)
        assert (target[:width] >= 0).all() and not target[width:].any()

# ---------------------------------------------------------------------------
# The same rows for any W
# ---------------------------------------------------------------------------

def test_collect_in_process_equals_walking_k_by_k():
    with Collector(seed=11, workers=0) as run:
        write_some_regrets(run.table)
        got = run.collect(seat=0, t=3, walks=20)
        expect = walked_by_hand(run.table, 11, 3, 0, 20)
    assert_collected_equal(got, expect)
    assert got.walks == 20

def test_collect_is_the_same_on_one_and_three_workers():
    results = []
    for workers in (1, 3):
        with Collector(seed=11, workers=workers) as run:
            write_some_regrets(run.table)
            results.append(run.collect(seat=1, t=2, walks=37))  # 37: chunks don't divide evenly
    assert_collected_equal(*results)
    with Collector(seed=11, workers=0) as reference:
        write_some_regrets(reference.table)
        assert_collected_equal(results[0], reference.collect(seat=1, t=2, walks=37))

def test_a_collector_runs_more_than_one_phase():
    with Collector(seed=2, workers=2) as run:
        first = run.collect(seat=0, t=1, walks=6)
        second = run.collect(seat=1, t=1, walks=6)
        again = run.collect(seat=0, t=1, walks=6)
    assert_collected_equal(first, again)
    assert not collected_equal(first, second)

def test_workers_see_regrets_written_after_start():
    with Collector(seed=4, workers=2) as run:
        before = run.collect(seat=0, t=1, walks=10)
        write_some_regrets(run.table)
        after = run.collect(seat=0, t=1, walks=10)
    with Collector(seed=4, workers=0) as reference:
        write_some_regrets(reference.table)
        assert_collected_equal(after, reference.collect(seat=0, t=1, walks=10))
    assert not collected_equal(before, after)

# ---------------------------------------------------------------------------
# Refusals and failures
# ---------------------------------------------------------------------------

def test_collect_refuses_bad_arguments():
    with Collector(seed=1, workers=0) as run:
        with pytest.raises(ValueError, match="walks"):
            run.collect(seat=0, t=1, walks=0)
        with pytest.raises(ValueError, match="seat"):
            run.collect(seat=2, t=1, walks=1)
        with pytest.raises(ValueError, match="t"):
            run.collect(seat=0, t=0, walks=1)
    with pytest.raises(ValueError, match="workers"):
        Collector(seed=1, workers=-1)
    with pytest.raises(ValueError, match="workers"):
        Collector(seed=1, workers=True)
    with pytest.raises(ValueError, match="seed"):
        Collector(seed=-1, workers=0)
    # A lambda cannot reach a spawned worker by pickle; refused at W = 0 too.
    with pytest.raises(ValueError, match="deal"):
        Collector(seed=1, workers=0, deal=lambda rng: random_deal(rng))

def test_a_collector_must_be_open_to_collect():
    run = Collector(seed=1, workers=0)
    with pytest.raises(RuntimeError, match="open"):
        run.collect(seat=0, t=1, walks=1)
    with pytest.raises(RuntimeError, match="open"):
        run.table

def test_a_failing_deal_raises_in_the_parent():
    with Collector(seed=1, workers=2, deal=exploding_deal) as run:
        with pytest.raises(RuntimeError, match="boom"):
            run.collect(seat=0, t=1, walks=4)

def test_a_collector_is_usable_after_a_raised_phase():
    # The worker's first deal fails and the phase raises; the chunks queued
    # behind it are cancelled, and the next phase walks as if nothing happened.
    with Collector(seed=1, workers=1, deal=first_deal_explodes) as run:
        write_some_regrets(run.table)
        with pytest.raises(RuntimeError, match="boom"):
            run.collect(seat=0, t=1, walks=8)
        again = run.collect(seat=1, t=2, walks=4)
    expected = PackedTable.whole_game(extra_averages=1)
    write_some_regrets(expected)
    assert_collected_equal(again, walked_by_hand(expected, 1, 2, 1, 4))

def test_a_dead_worker_is_named_and_breaks_the_collector_for_the_run():
    with Collector(seed=1, workers=1, deal=dying_deal) as run:
        with pytest.raises(RuntimeError, match="died"):
            run.collect(seat=0, t=1, walks=4)
        # The pool is broken for good; the next phase says so by name too,
        # rather than leaking the pool's own exception.
        with pytest.raises(RuntimeError, match="died"):
            run.collect(seat=0, t=1, walks=4)

def test_a_failing_deal_raises_in_process_too():
    with Collector(seed=1, workers=0, deal=exploding_deal) as run:
        with pytest.raises(RuntimeError, match="boom"):
            run.collect(seat=0, t=1, walks=4)

def test_close_unlinks_the_block_even_after_an_error():
    with pytest.raises(KeyError):
        with Collector(seed=1, workers=2) as run:
            name = run._block.name
            raise KeyError("caller failed")
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)

def test_a_table_kept_past_the_with_still_reads():
    with Collector(seed=1, workers=1) as run:
        write_some_regrets(run.table)
        kept = run.table
        first = float(kept.cumulative_regret[:3].sum())
    # The block is gone; the kept table was moved onto private memory first.
    assert float(kept.cumulative_regret[:3].sum()) == first
    assert kept.cumulative_regret.base is None or not isinstance(
        kept.cumulative_regret.base, memoryview
    )

def test_an_attach_failure_is_reported_by_the_first_job(monkeypatch):
    # The mechanism, in this process: attaching to a block that does not exist
    # is kept as a traceback, and the first job raises it instead of walking.
    monkeypatch.setattr(collect, "_worker", None)
    monkeypatch.setattr(collect, "_attach_failure", "")
    collect._attach_worker("no-such-block-" + "x" * 8, 1, random_deal)
    assert collect._worker is None and "FileNotFoundError" in collect._attach_failure
    with pytest.raises(RuntimeError, match="failed to start(.|\n)*FileNotFoundError"):
        collect._walk_chunk((0, 1, 0, 1))

def test_close_unlinks_the_block_after_a_phase():
    with Collector(seed=1, workers=1) as run:
        name = run._block.name
        run.collect(seat=0, t=1, walks=2)
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)
    with pytest.raises(RuntimeError, match="open"):
        run.table
