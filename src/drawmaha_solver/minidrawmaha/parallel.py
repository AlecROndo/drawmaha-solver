"""Lockstep MCCFR in W processes over one packed table in shared memory.

`lockstep.train_lockstep` runs a phase's W walks one after another. Here each
walk runs in its own process, and the parent does exactly what the reference
does with their records, in the same order — so a run here equals the
reference with the same W bit for bit (`test_parallel.py` holds them to it).

The shape of one phase (one seat of one iteration):

1. The parent writes (t, seat) into a small control block and posts each
   worker's own `go` semaphore.
2. Each worker walks its hand, reading σₜ from the shared table and writing
   nothing to it, copies its record into its own scratch block, and posts the
   shared `done` semaphore.
3. Once `done` has been taken W times, the parent reads the W records in
   worker order and banks them (`lockstep.apply_arrays`) while every worker
   waits on its `go`.

So the table is only ever written while nobody reads it, and nothing needs a
lock. Semaphores rather than `multiprocessing.Barrier`: the barrier's
condition variable cost ~2 ms a crossing on macOS, more than a walk; a
semaphore post and wait are two system calls, and they order memory, so the
scratch a worker wrote is visible once its `done` is taken.

The table's index (which row and columns each infoset owns) is not shipped to
the workers: each rebuilds it from `keys` — `None` for the whole game, which
is a second of arithmetic — and lays it over the parent's buffers, whose
shapes it is checked against. A worker that raises reports its traceback and
posts `done`; one that dies is noticed by a watchdog thread within a second;
either way the parent raises. Every shared block is created by the parent and
unlinked by it however the run ends.
"""

from __future__ import annotations

import multiprocessing
import queue
import signal
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing import shared_memory
from multiprocessing.process import BaseProcess

import numpy as np

from drawmaha_solver.minidrawmaha.game import InfoSet
from drawmaha_solver.minidrawmaha.lockstep import (
    SEATS,
    Flat,
    LockstepSolve,
    apply_arrays,
    flatten,
    walk,
)
from drawmaha_solver.minidrawmaha.mccfr import RootSampler
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.regret_rules import Average

# A walk records at most ~50 rows (measured: 31 regret rows / 76 values, 48
# strategy rows / 134 values over 1,000 mini-drawmaha walks). Room for fifty
# times that; a walk that overflows raises rather than truncating.
RECORD_CAPACITY = 2_048
VALUE_CAPACITY = 8_192

# A worker reports ready once it has imported, rebuilt the index and
# attached — seconds even on a cold container. One that dies is caught at
# once; this bound is only for one that hangs while starting.
STARTUP_TIMEOUT_S = 300.0

# A worker posts its traceback before `done`, but the queue's feeder thread
# delivers it asynchronously; this long with nothing arriving means none is
# coming, and the parent raises without one.
_REPORT_GRACE_S = 2.0

# The four arrays of a `PackedTable`, each in its own shared block.
_SLOTS = ("cumulative_regret", "strategy_sum", "extra_sums", "stamp")

# The control block: four int64 fields the parent writes before posting `go`
# (and a failing worker sets FAILED in).
_T, _SEAT, _STOP, _FAILED = range(4)

class _Report(StrEnum):
    """What a worker tells the parent on the report queue."""

    READY = "ready"
    FAILED = "failed"

# ---------------------------------------------------------------------------
# What the parent hands a worker
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _SharedArray:
    """An array living in a named shared block: enough to attach to it."""

    block: str
    shape: tuple[int, ...]
    dtype: str

@dataclass(frozen=True, slots=True)
class _WorkerSpec:
    """Everything a worker process needs, picklable."""

    worker: int
    seed: int
    averages: tuple[Average, ...]
    deal: RootSampler
    keys: tuple[InfoSet, ...] | None  # None: the whole game
    table: tuple[_SharedArray, ...]  # in `_SLOTS` order
    control: _SharedArray
    scratch_block: str

@dataclass(frozen=True, slots=True)
class _Scratch:
    """One worker's latest walk record, as arrays over a block it shares with the parent.

    The worker writes it after its walk and before posting `done`; the parent
    reads it only after taking `done`, so the two never touch it at once. The
    capacities are fixed, so one block serves every phase of the run. Every
    field is 8 bytes wide, so the arrays sit end to end in the block, aligned.
    """

    counts: np.ndarray  # regret rows, strategy rows, regret values, strategy values
    regret_rows: np.ndarray
    regret_starts: np.ndarray
    regret_widths: np.ndarray
    strategy_starts: np.ndarray
    strategy_widths: np.ndarray
    regret_values: np.ndarray
    strategy_values: np.ndarray
    extra_values: np.ndarray

    @staticmethod
    def _fields(extra_rows: int) -> tuple[tuple[str, type, tuple[int, ...]], ...]:
        rows, values = (RECORD_CAPACITY,), (VALUE_CAPACITY,)
        return (
            ("counts", np.int64, (4,)),
            ("regret_rows", np.int64, rows),
            ("regret_starts", np.int64, rows),
            ("regret_widths", np.int64, rows),
            ("strategy_starts", np.int64, rows),
            ("strategy_widths", np.int64, rows),
            ("regret_values", np.float64, values),
            ("strategy_values", np.float64, values),
            ("extra_values", np.float64, (extra_rows, VALUE_CAPACITY)),
        )

    @classmethod
    def nbytes(cls, extra_rows: int) -> int:
        """The block size one worker's scratch needs."""
        return sum(8 * int(np.prod(shape)) for _, _, shape in cls._fields(extra_rows))

    @classmethod
    def over(cls, buffer: memoryview, extra_rows: int) -> _Scratch:
        """The scratch arrays laid over `buffer` (at least `nbytes` long)."""
        arrays, offset = {}, 0
        for name, dtype, shape in cls._fields(extra_rows):
            arrays[name] = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
            offset += arrays[name].nbytes
        return cls(**arrays)

    def write(self, flat: Flat) -> None:
        """Copy one walk's record in; a record over capacity raises."""
        records = (len(flat.regret_rows), len(flat.strategy_starts))
        values = (len(flat.regret_values), len(flat.strategy_values))
        if max(records) > RECORD_CAPACITY or max(values) > VALUE_CAPACITY:
            raise RuntimeError(
                f"a walk recorded {records} rows / {values} values, over the scratch "
                f"capacity of {RECORD_CAPACITY} / {VALUE_CAPACITY}"
            )
        self.counts[:] = (*records, *values)
        self.regret_rows[: records[0]] = flat.regret_rows
        self.regret_starts[: records[0]] = flat.regret_starts
        self.regret_widths[: records[0]] = flat.regret_widths
        self.strategy_starts[: records[1]] = flat.strategy_starts
        self.strategy_widths[: records[1]] = flat.strategy_widths
        self.regret_values[: values[0]] = flat.regret_values
        self.strategy_values[: values[1]] = flat.strategy_values
        self.extra_values[:, : values[1]] = flat.extra_values

    def read(self) -> Flat:
        """The record last written, copied out of shared memory."""
        regrets, strategies, regret_values, strategy_values = (int(c) for c in self.counts)
        return Flat(
            regret_rows=self.regret_rows[:regrets].copy(),
            regret_starts=self.regret_starts[:regrets].copy(),
            regret_widths=self.regret_widths[:regrets].copy(),
            regret_values=self.regret_values[:regret_values].copy(),
            strategy_starts=self.strategy_starts[:strategies].copy(),
            strategy_widths=self.strategy_widths[:strategies].copy(),
            strategy_values=self.strategy_values[:strategy_values].copy(),
            extra_values=self.extra_values[:, :strategy_values].copy(),
        )

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

class ParallelLockstep:
    """W worker processes training one lockstep solve over a shared packed table.

        with ParallelLockstep(solve) as run:
            run.train(1_000)

    `keys` must be the keys `solve.table` was listed with (`None` for a
    whole-game table); a list that lays out other shapes is refused when the
    workers start. While open, `solve.table` is the shared table. On exit the
    workers stop, the arrays are copied back into private memory,
    `solve.table` points at the copies, and every shared block is unlinked —
    also when a worker or the caller raised.
    """

    def __init__(self, solve: LockstepSolve, *, keys: Sequence[InfoSet] | None = None):
        if not isinstance(solve.table, PackedTable):
            raise TypeError("a parallel run needs a PackedTable")
        self.solve = solve
        self._keys = None if keys is None else tuple(keys)
        # The caller's table, set aside while the shared one stands in for it.
        self._private: PackedTable | None = None
        self._blocks: list[shared_memory.SharedMemory] = []
        self._processes: list[BaseProcess] = []
        self._shared_table: tuple[_SharedArray, ...] = ()
        self._control: np.ndarray | None = None
        self._shared_control: _SharedArray | None = None
        self._scratch: list[_Scratch] = []
        self._go: list = []
        self._done = None
        self._reports = None
        self._trouble = ""
        self._stopping = threading.Event()
        self._watchdog: threading.Thread | None = None

    def __enter__(self) -> ParallelLockstep:
        try:
            self._start()
        except BaseException:
            self._close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._close()

    def train(self, iterations: int) -> LockstepSolve:
        """Run `iterations` more lockstep iterations, in place; `train_lockstep`'s contract."""
        if iterations < 1:
            raise ValueError(f"iterations must be at least 1, got {iterations}")
        for _ in range(iterations):
            self.step()
        return self.solve

    def step(self) -> None:
        """One lockstep iteration: both seats, W walks each, banked in worker order.

        If it raises, the table may already hold the first seat of the
        iteration it did not finish: the solve is no longer a lockstep run and
        is for discarding, not for saving.
        """
        solve = self.solve
        t = solve.iteration + 1
        for seat in SEATS:
            self._control[[_T, _SEAT]] = (t, seat)
            for go in self._go:
                go.release()
            self._collect()
            flats = [scratch.read() for scratch in self._scratch]
            apply_arrays(solve.table, flats, solve.rule, t)
        solve.iteration = t

    # -- starting ------------------------------------------------------------

    def _start(self) -> None:
        self._share_table()
        self._control, self._shared_control = self._share(np.zeros(4, dtype=np.int64))
        context = multiprocessing.get_context("spawn")
        self._go = [context.Semaphore(0) for _ in range(self.solve.workers)]
        self._done = context.Semaphore(0)
        self._reports = context.Queue()
        for worker in range(self.solve.workers):
            self._spawn(context, worker)
        self._await_ready()
        self._trouble = ""
        self._stopping.clear()
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def _share_table(self) -> None:
        private = self.solve.table
        shared = {slot: self._share(getattr(private, slot)) for slot in _SLOTS}
        self._private = private
        self.solve.table = private.over(**{slot: view for slot, (view, _) in shared.items()})
        self._shared_table = tuple(described for _, described in shared.values())

    def _share(self, array: np.ndarray) -> tuple[np.ndarray, _SharedArray]:
        """A copy of `array` in a new shared block this run owns, and how to attach to it."""
        block = self._new_block(array.nbytes)
        view = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)
        view[...] = array
        return view, _SharedArray(block=block.name, shape=view.shape, dtype=view.dtype.str)

    def _new_block(self, size: int) -> shared_memory.SharedMemory:
        # Recorded before anything else can fail, so `_close` unlinks it.
        # A zero-row table still needs a block: size 0 is refused.
        block = shared_memory.SharedMemory(create=True, size=max(size, 1))
        self._blocks.append(block)
        return block

    def _spawn(self, context, worker: int) -> None:
        extra_rows = len(self.solve.averages)
        scratch_block = self._new_block(_Scratch.nbytes(extra_rows))
        self._scratch.append(_Scratch.over(scratch_block.buf, extra_rows))
        spec = _WorkerSpec(
            worker=worker,
            seed=self.solve.seed,
            averages=self.solve.averages,
            deal=self.solve.deal,
            keys=self._keys,
            table=self._shared_table,
            control=self._shared_control,
            scratch_block=scratch_block.name,
        )
        process = context.Process(
            target=_worker_main,
            args=(spec, self._go[worker], self._done, self._reports),
            daemon=True,
        )
        process.start()
        self._processes.append(process)

    def _await_ready(self) -> None:
        """Wait until every worker has built its index and attached, or fail fast.

        A worker that dies while starting (an import error, a bad spec) never
        reports; without polling for it the parent would sit out the whole
        startup timeout before saying so.
        """
        ready: set[int] = set()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while len(ready) < len(self._processes):
            try:
                report, worker, trace = self._reports.get(timeout=0.5)
            except queue.Empty:
                dead = [i for i, p in enumerate(self._processes) if not p.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"lockstep workers {dead} exited while starting "
                        f"(exit codes {[self._processes[i].exitcode for i in dead]})"
                    ) from None
                if time.monotonic() > deadline:
                    raise RuntimeError("lockstep workers did not start in time") from None
                continue
            if report is _Report.FAILED:
                raise RuntimeError(f"lockstep worker {worker} failed:\n{trace}")
            ready.add(worker)

    # -- one phase -----------------------------------------------------------

    def _collect(self) -> None:
        """Take `done` once per worker, then raise if any worker failed or died.

        A plain blocking wait: macOS has no timed semaphore wait, and Python's
        stand-in polls with sleeps that cost milliseconds a phase. A worker
        that raises posts `done` itself; one that dies is the watchdog's, which
        posts `done` for every worker. A worker that hangs while alive is not
        detected: this wait then lasts as long as the hang.
        """
        for _ in self._processes:
            self._done.acquire()
        if self._trouble:
            self._fail(self._trouble)
        if self._control[_FAILED]:
            self._fail("a worker failed")

    def _watch(self) -> None:
        """Every second: if a worker has died, say so and wake the parent's wait."""
        while not self._stopping.wait(1.0):
            dead = [p.pid for p in self._processes if not p.is_alive()]
            if dead and not self._trouble:
                self._trouble = f"dead workers: {dead}"
                for _ in self._processes:
                    self._done.release()

    def _fail(self, why: str) -> None:
        """Raise with a failed worker's traceback if one arrives, else with `why`."""
        while True:
            try:
                report, worker, trace = self._reports.get(timeout=_REPORT_GRACE_S)
            except queue.Empty:
                break
            if report is _Report.FAILED:
                raise RuntimeError(f"lockstep worker {worker} failed:\n{trace}") from None
        raise RuntimeError(f"the lockstep run broke ({why})") from None

    # -- closing -------------------------------------------------------------

    def _close(self) -> None:
        """Stop everything `_start` began, however far it got."""
        self._stop_watchdog()
        self._stop_workers()
        self._restore_private_table()
        self._unlink_blocks()

    def _stop_watchdog(self) -> None:
        if self._watchdog is not None:
            self._stopping.set()
            self._watchdog.join()
            self._watchdog = None

    def _stop_workers(self) -> None:
        if not self._processes:
            return
        # Live workers are waiting on `go` (or finishing a walk and then will
        # be): the stop flag lets every one of them leave. The control block
        # exists, since it is shared before any worker is spawned.
        self._control[_STOP] = 1
        for go in self._go:
            go.release()
        for process in self._processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
        self._processes = []

    def _restore_private_table(self) -> None:
        if self._private is None:
            return
        shared = self.solve.table
        copies = {slot: np.array(getattr(shared, slot)) for slot in _SLOTS}
        self.solve.table = self._private.over(**copies)
        self._private = None

    def _unlink_blocks(self) -> None:
        self._control = self._shared_control = None
        self._shared_table = ()
        self._scratch = []
        _close_blocks(self._blocks)
        for block in self._blocks:
            block.unlink()
        self._blocks = []

# ---------------------------------------------------------------------------
# The table a run's keys describe
# ---------------------------------------------------------------------------

def packed_table_for(keys: Sequence[InfoSet] | None, *, extra_averages: int) -> PackedTable:
    """A fresh packed table over `keys`, or over the whole game when `keys` is None.

    The one place that convention lives: a run's driver allocates its table
    with it and every worker rebuilds the same index with it.
    """
    if keys is None:
        return PackedTable.whole_game(extra_averages=extra_averages)
    return PackedTable.listed(keys, extra_averages=extra_averages)

# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------

def _worker_main(spec: _WorkerSpec, go, done, reports) -> None:
    # A Ctrl-C or a container's SIGTERM reaches the whole process group; the
    # parent handles it (finish the iteration, save) and stops the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    blocks: list[shared_memory.SharedMemory] = []
    table = control = scratch = None
    try:
        extra_rows = len(spec.averages)
        template = packed_table_for(spec.keys, extra_averages=extra_rows)
        arrays = {
            slot: _attached(shared, blocks) for slot, shared in zip(_SLOTS, spec.table, strict=True)
        }
        # `over` checks each array's shape against the rebuilt index: keys
        # that do not match the parent's table fail here, not in the banking.
        table = template.over(**arrays)
        del template, arrays
        control = _attached(spec.control, blocks)
        scratch = _Scratch.over(_attach(spec.scratch_block, blocks).buf, extra_rows)
        reports.put((_Report.READY, spec.worker, ""))
        while True:
            go.acquire()
            if control[_STOP]:
                return
            t, seat = int(control[_T]), int(control[_SEAT])
            recorded = walk(table, spec.deal, spec.seed, t, seat, spec.worker, spec.averages)
            scratch.write(flatten(recorded, extra_rows))
            done.release()
    except BaseException:
        reports.put((_Report.FAILED, spec.worker, traceback.format_exc()))
        if control is not None:
            control[_FAILED] = 1
        done.release()
    finally:
        # Drop every view before closing, or `close` refuses exported buffers.
        table = control = scratch = None  # noqa: F841
        _close_blocks(blocks)

# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------

def _attached(shared: _SharedArray, opened: list[shared_memory.SharedMemory]) -> np.ndarray:
    block = _attach(shared.block, opened)
    return np.ndarray(shared.shape, dtype=np.dtype(shared.dtype), buffer=block.buf)

def _attach(name: str, opened: list[shared_memory.SharedMemory]) -> shared_memory.SharedMemory:
    """Open a block the parent created, and add it to `opened` for closing."""
    # The parent owns every block. From 3.13 a worker can attach without
    # registering it with the resource tracker; 3.12 always registers, which is
    # harmless here because spawned workers share the parent's tracker, whose
    # registry is a set that the parent's unlink clears.
    if sys.version_info >= (3, 13):
        block = shared_memory.SharedMemory(name=name, track=False)
    else:
        block = shared_memory.SharedMemory(name=name)
    opened.append(block)
    return block

def _close_blocks(blocks: list[shared_memory.SharedMemory]) -> None:
    for block in blocks:
        try:
            block.close()
        except BufferError:
            # Someone still holds an array over the block (a caller keeping a
            # shared table's array, a traceback's frame). `close` refuses
            # rather than unmap memory under it; the mapping goes when the last
            # array does, and the parent's unlink removes the name regardless.
            pass
