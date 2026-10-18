"""The collector: K lockstep walks for one seat on W processes, returned as memory rows.

One Deep CFR phase is "seat p walks K hands against the materialised table at
iteration t". The walk is rung 3's `lockstep.walk`, unchanged: it reads σₜ
from `cumulative_regret`, writes nothing, and records per traverser visit
`(key, slot, r)` with r = u − ⟨σ, u⟩ and per opponent visit `(key, slot, t·σ,
extra)`. Those records are the memories' rows, and this module only reshapes
them: `rows_of` turns one walk's record into `Collected` — the ADVANTAGE rows
(the traverser's table rows and their r, zero-padded to 4 columns) and the
POLICY rows (the opponent's table rows and their σ) — and `joined` lays K of
them end to end in walk order.

Two facts the loop leans on:

- **The row is the slot's.** `Recorded` carries each visit's packed slot
  `(row, first column, width)`, so the table row a memory row names is
  `slot[0]`; nothing is searched for.
- **σ comes from the uniform column.** The walk is asked for the `uniform`
  averaging column, whose weight at every t is t⁰ = 1, so the recorded
  `extra[0]` is σ bit for bit (`test_collect.py` pins it against
  `table[key].strategy()`); dividing the primary `t·σ` by t would not be.

**Why the sample is a function of (seed, t, seat, k), never of W.** Walk k's
stream is `lockstep.stream(seed, t, seat, k)` and the K walks are indexed
k = 0..K−1 whatever process runs them. The W workers are handed contiguous
chunks of k in order and the chunks are joined in order, so W = 1 and W = 32
return the same bytes. Chunks are ⌈K / 4W⌉ walks, so the phase's tail is one
quarter of a worker's share and a slow chunk cannot stall it.

**Spawn everywhere, one shared block.** The run's whole-game `PackedTable`
keeps its `cumulative_regret` in a `multiprocessing.shared_memory` block that
the parent owns; each worker, spawned (not forked — the loop that calls this
holds torch threads, which a fork would copy mid-flight) and started once for
the run, rebuilds the whole-game index (a second of arithmetic) and lays it
over that block. The worker's `strategy_sum`, `extra_sums` and `stamp` are its
own untouched zeros: a lockstep walk writes only into scratch. The parent
materialises the next iteration's net into `Collector.table.cumulative_regret`
between phases, and the workers read whatever stands there when `collect` is
called — the table is only ever written while nobody walks. The block is
unlinked when the `with` ends, also on an exception, and the table object
handed out as `run.table` is given private memory first, so a reference kept
past the `with` still reads (a bare array taken from it does not: it dangles).
A worker that hangs while alive is not detected, as `parallel.py` says of its
own wait; the phase then lasts as long as the hang. `workers=0` walks in this
process over a private table with no block and no pool: the reference the
workers must equal.

Measured on this Mac (18 cores, 2026-10-17) over the whole-game table with
random regrets: 10,000 walks for one seat on 8 workers take 4.2 s (2.9 ms a
walk in one process, so 6.9× on 8), returning 171,810 advantage rows and
293,315 policy rows — 17.2 and 29.3 a walk; the design report measured 14 and
22 on a trained table, which is sparser. One spawned worker takes 1.1 s to
import, build its index and attach, and peaks at 237 MB resident (most of it
the index and the interpreter; the regret block is shared), so W = 32 on
Modal's 16 GB is under 8 GB of workers beside a parent of about 0.5 GB.
"""

from __future__ import annotations

import math
import pickle
import sys
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context, shared_memory

import numpy as np

from drawmaha_solver.deepcfr.reservoir import TARGET_WIDTH, _is_int
from drawmaha_solver.minidrawmaha.game import random_deal
from drawmaha_solver.minidrawmaha.lockstep import SEATS, Recorded, walk
from drawmaha_solver.minidrawmaha.mccfr import RootSampler
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.regret_rules import Average

# The one averaging column the walk is asked for: weight 1 at every t, so the
# recorded extra row is σ itself.
UNIFORM_COLUMN = (Average.UNIFORM,)

# Chunks per worker per phase. More chunks mean a finer tail at the phase's
# end; each costs one pickled result, which at a few hundred rows is nothing.
CHUNKS_PER_WORKER = 4

# ---------------------------------------------------------------------------
# A phase's rows
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Collected:
    """The rows one phase offers the memories, in walk order and visit order within a walk.

    `advantage_rows[i]` is the table row of the traverser's i-th visit and
    `advantage_targets[i]` its regret sample r, zero-padded from the ledger's
    width to `TARGET_WIDTH`; `policy_*` the same for the opponent's visits
    with σ. `walks` is how many walks the rows came from.
    """

    advantage_rows: np.ndarray  # int32 (a,)
    advantage_targets: np.ndarray  # float32 (a, TARGET_WIDTH)
    policy_rows: np.ndarray  # int32 (p,)
    policy_targets: np.ndarray  # float32 (p, TARGET_WIDTH)
    walks: int

def rows_of(recorded: Recorded) -> Collected:
    """One walk's record as memory rows, visits in the order the walk made them.

    The walk must have run over a packed table (so every visit has a slot) and
    recorded one extra row that is a distribution, which is σ: the uniform
    column weighs 1 at every t, and at t = 1 so does every column (t⁰ = t² = 1),
    so a row that sums to 1 is σ whichever column wrote it. Any other record is
    refused rather than misread.
    """
    if any(slot is None for _, slot, _ in recorded.regret) or any(
        slot is None for _, slot, _, _ in recorded.strategy
    ):
        raise ValueError("rows_of needs a walk over a packed table; a visit has no slot")
    # The check is on the row, not the column's name (`rows_of` is not told
    # t): a quadratic row at t ≥ 2 sums to t² and is refused; at t = 1 it is σ
    # and passes, rightly. The rows are float64 — the worst |Σ − 1| seen over
    # 200 walks is 2e-16 — so 1e-6 admits no σ of width 4 that is not one.
    if any(
        extra.shape[0] != 1 or abs(float(extra[0].sum()) - 1.0) > 1e-6
        for _, _, _, extra in recorded.strategy
    ):
        raise ValueError("rows_of needs a walk asked for the uniform column alone")
    return Collected(
        advantage_rows=np.asarray([slot[0] for _, slot, _ in recorded.regret], dtype=np.int32),
        advantage_targets=_padded([r for _, _, r in recorded.regret]),
        policy_rows=np.asarray([slot[0] for _, slot, _, _ in recorded.strategy], dtype=np.int32),
        policy_targets=_padded([extra[0] for _, _, _, extra in recorded.strategy]),
        walks=1,
    )

def joined(parts: Sequence[Collected]) -> Collected:
    """The rows of several `Collected` end to end, in the order given; walks are summed."""
    if not parts:
        raise ValueError("joined was handed nothing; a phase has at least one walk")
    return Collected(
        advantage_rows=np.concatenate([p.advantage_rows for p in parts]),
        advantage_targets=np.concatenate([p.advantage_targets for p in parts]),
        policy_rows=np.concatenate([p.policy_rows for p in parts]),
        policy_targets=np.concatenate([p.policy_targets for p in parts]),
        walks=sum(p.walks for p in parts),
    )

def _padded(values: list[np.ndarray]) -> np.ndarray:
    """Vectors of width 2 to 4 as rows of an `(n, TARGET_WIDTH)` float32 array, zeros past each."""
    out = np.zeros((len(values), TARGET_WIDTH), dtype=np.float32)
    for row, value in zip(out, values, strict=True):
        row[: value.shape[0]] = value
    return out

# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------

class Collector:
    """A run's table and its W workers, open for as many phases as the run has.

        with Collector(seed=0, workers=8) as run:
            materialise(run.table, nets)              # the parent writes σₜ
            rows = run.collect(seat=0, t=1, walks=10_000)

    `table` is the whole-game `PackedTable` the workers walk; write the
    iteration's regrets into `table.cumulative_regret` before each `collect`.
    `workers=0` walks in this process, with no pool and no shared block.
    `deal` reaches the spawned workers by pickle, so it is a module-level
    callable, not a lambda or a function defined inside another.
    """

    def __init__(self, *, seed: int, workers: int, deal: RootSampler = random_deal) -> None:
        _validate_collector(seed, workers, deal)
        self._seed = int(seed)
        self._workers = int(workers)
        self._deal = deal
        self._table: PackedTable | None = None
        self._block: shared_memory.SharedMemory | None = None
        self._pool: ProcessPoolExecutor | None = None

    @property
    def table(self) -> PackedTable:
        """The table the walks read; its `cumulative_regret` is what the workers see."""
        if self._table is None:
            raise RuntimeError("the collector is not open; use it in a `with` block")
        return self._table

    def __enter__(self) -> Collector:
        try:
            self._open()
        except BaseException:
            self._close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._close()

    def collect(self, *, seat: int, t: int, walks: int) -> Collected:
        """Walks 0..walks−1 of `seat` at iteration `t` against the table as it stands now.

        A worker that raises, walking or attaching, re-raises here with its
        message; a worker that dies raises `RuntimeError`. Either way no short
        phase is returned, and the collector stays usable: the chunks still
        queued behind the one that raised are cancelled, so the next phase
        does not wait on walks nobody will read (a chunk already running
        finishes and is dropped).
        """
        table = self.table
        _validate_phase(seat, t, walks)
        if self._pool is None:
            return _walk_range(table, self._deal, self._seed, seat, t, range(walks))
        chunks = _chunks(seat, t, walks, self._workers)
        jobs = [self._pool.submit(_walk_chunk, chunk) for chunk in chunks]
        try:
            parts = [job.result() for job in jobs]
        except BrokenProcessPool as broke:
            raise RuntimeError("a collector worker died") from broke
        finally:
            # After a success every job is done and this is a no-op; after a
            # raise it is what keeps the stale tail from burning the workers.
            for job in jobs:
                job.cancel()
        return joined(parts)

    # -- opening and closing -------------------------------------------------

    def _open(self) -> None:
        table = PackedTable.whole_game(extra_averages=1)
        if self._workers == 0:
            self._table = table
            return
        regret = table.cumulative_regret
        self._block = shared_memory.SharedMemory(create=True, size=regret.nbytes)
        shared = np.ndarray(regret.shape, dtype=regret.dtype, buffer=self._block.buf)
        shared[...] = regret
        self._table = table.over(
            cumulative_regret=shared,
            strategy_sum=table.strategy_sum,
            extra_sums=table.extra_sums,
            stamp=table.stamp,
        )
        self._pool = ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=get_context("spawn"),
            initializer=_attach_worker,
            initargs=(self._block.name, self._seed, self._deal),
        )

    def _close(self) -> None:
        """Undo `_open` as far as it got: stop the pool, privatise the table, unlink the block."""
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
        finally:
            self._pool = None
            table, self._table = self._table, None
            if self._block is not None:
                block, self._block = self._block, None
                if table is not None:
                    # `close` unmaps the block under any array still over it,
                    # and NumPy does not stop it (the array keeps a pointer,
                    # not a buffer export). A caller who kept `run.table` would
                    # read unmapped memory, so the table is moved onto a
                    # private copy first, as `parallel.py` restores its solve's
                    # table. 109 MB, once per run.
                    table.cumulative_regret = np.array(table.cumulative_regret)
                block.close()
                block.unlink()

# ---------------------------------------------------------------------------
# Walking a range of k, in whichever process
# ---------------------------------------------------------------------------

def _walk_range(
    table: PackedTable, deal: RootSampler, seed: int, seat: int, t: int, ks: range
) -> Collected:
    """Walks k in `ks` for `seat` at `t`, as one `Collected` in k order."""
    return joined([rows_of(walk(table, deal, seed, t, seat, k, UNIFORM_COLUMN)) for k in ks])

def _chunks(seat: int, t: int, walks: int, workers: int) -> list[tuple[int, int, int, int]]:
    """The phase's walks as (seat, t, first k, past-the-last k) jobs, ⌈walks / 4W⌉ wide."""
    size = math.ceil(walks / (CHUNKS_PER_WORKER * workers))
    return [(seat, t, start, min(start + size, walks)) for start in range(0, walks, size)]

# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Worker:
    """What a spawned worker keeps for the run: its table over the parent's block, and the block."""

    table: PackedTable
    seed: int
    deal: RootSampler
    # Held so the mapping outlives every walk; the parent unlinks the name.
    block: shared_memory.SharedMemory

# Set once by `_attach_worker` in each spawned process; the pool's jobs read
# it. If attaching failed, `_attach_failure` holds the traceback instead.
_worker: _Worker | None = None
_attach_failure: str = ""

def _attach_worker(block_name: str, seed: int, deal: RootSampler) -> None:
    """Build this process's whole-game index over the parent's regret block, once.

    A failure here is kept, not raised: an initializer that raises breaks the
    pool, and the parent would see only "a worker died" with the reason on
    the worker's stderr. Kept, it is re-raised by the first job, with its
    traceback, in the parent.
    """
    global _worker, _attach_failure
    try:
        _worker = _attached(block_name, seed, deal)
    except BaseException:
        _attach_failure = traceback.format_exc()

def _attached(block_name: str, seed: int, deal: RootSampler) -> _Worker:
    template = PackedTable.whole_game(extra_averages=1)
    # The parent owns the block. From 3.13 the worker can attach without the
    # resource tracker registering it; 3.12 registers regardless, harmlessly,
    # since a spawned worker shares the parent's tracker and the parent's
    # unlink clears the entry.
    if sys.version_info >= (3, 13):
        block = shared_memory.SharedMemory(name=block_name, track=False)
    else:
        block = shared_memory.SharedMemory(name=block_name)
    regret = np.ndarray(
        template.cumulative_regret.shape, dtype=template.cumulative_regret.dtype, buffer=block.buf
    )
    table = template.over(
        cumulative_regret=regret,
        strategy_sum=template.strategy_sum,
        extra_sums=template.extra_sums,
        stamp=template.stamp,
    )
    return _Worker(table=table, seed=int(seed), deal=deal, block=block)

def _walk_chunk(job: tuple[int, int, int, int]) -> Collected:
    """One job of `_chunks`, run in a worker: walks k in [first, past) for its seat and t."""
    seat, t, first, past = job
    worker = _worker
    if worker is None:
        raise RuntimeError(
            "a collector worker failed to start:\n" + (_attach_failure or "it was never attached")
        )
    return _walk_range(worker.table, worker.deal, worker.seed, seat, t, range(first, past))

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_collector(seed: object, workers: object, deal: object) -> None:
    """Refuse a seed, worker count or deal the run could not use, before a table is built."""
    # The walk streams are default_rng([seed, t, seat, k]), which refuses a
    # negative word — but only at the first walk, in a worker.
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    # Zero workers is the in-process reference; a negative count is nothing.
    if not _is_int(workers) or workers < 0:
        raise ValueError(f"workers must be a non-negative int, got {workers!r}")
    # A spawned worker gets `deal` by pickle, which carries a function by its
    # module and name: a lambda or a local function cannot travel, and the
    # pool would only say "a worker died" with the reason on a worker's
    # stderr. Checked at W = 0 as well, so the reference accepts exactly what
    # the workers do.
    try:
        pickle.dumps(deal)
    except (pickle.PicklingError, AttributeError, TypeError) as why:
        raise ValueError(
            f"deal must be picklable to reach a spawned worker, got {deal!r} ({why})"
        ) from why

def _validate_phase(seat: object, t: object, walks: object) -> None:
    """Refuse a phase the walk could not run, before any worker is asked."""
    if seat not in SEATS or not _is_int(seat):
        raise ValueError(f"seat must be one of {SEATS}, got {seat!r}")
    # Iterations start at 1: the memory stamps rows with t and the refit
    # weights them by it.
    if not _is_int(t) or t < 1:
        raise ValueError(f"t must be a positive int, got {t!r}")
    # A phase of no walks would offer the memories nothing and the loop would
    # refit on a stale memory without noticing.
    if not _is_int(walks) or walks < 1:
        raise ValueError(f"walks must be a positive int, got {walks!r}")
