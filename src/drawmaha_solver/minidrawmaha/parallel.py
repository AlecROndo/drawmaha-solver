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
scratch a worker wrote is visible once its `done` is taken. A worker that
raises flags it and posts `done`; the parent re-raises its traceback. Shared
memory is unlinked however the run ends.
"""

from __future__ import annotations

import multiprocessing
import queue
import signal
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory

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

STARTUP_TIMEOUT_S = 300.0

_SLOTS = ("cumulative_regret", "strategy_sum", "extra_sums", "stamp")

def _shared(array: np.ndarray) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    """A shared-memory copy of `array`, and the array over it."""
    block = shared_memory.SharedMemory(create=True, size=max(array.nbytes, 1))
    view = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)
    view[...] = array
    return block, view

def _attach(name: str) -> shared_memory.SharedMemory:
    # The parent owns every block; a worker attaching must not register it with
    # its own resource tracker, which would unlink it when the worker exits.
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:  # Python < 3.13 has no `track`
        return shared_memory.SharedMemory(name=name)

@dataclass(frozen=True, slots=True)
class _Layout:
    """Where one worker's scratch arrays sit inside its two blocks."""

    extra_rows: int

    def ints(self, buffer) -> dict[str, np.ndarray]:
        counts = np.ndarray((4,), dtype=np.int64, buffer=buffer)
        arrays = {"counts": counts}
        for index, name in enumerate(
            ("regret_rows", "regret_starts", "regret_widths", "strategy_starts", "strategy_widths")
        ):
            arrays[name] = np.ndarray(
                (RECORD_CAPACITY,),
                dtype=np.int64,
                buffer=buffer,
                offset=8 * (4 + index * RECORD_CAPACITY),
            )
        return arrays

    def floats(self, buffer) -> dict[str, np.ndarray]:
        return {
            "regret_values": np.ndarray((VALUE_CAPACITY,), dtype=np.float64, buffer=buffer),
            "strategy_values": np.ndarray(
                (VALUE_CAPACITY,), dtype=np.float64, buffer=buffer, offset=8 * VALUE_CAPACITY
            ),
            "extra_values": np.ndarray(
                (self.extra_rows, VALUE_CAPACITY),
                dtype=np.float64,
                buffer=buffer,
                offset=16 * VALUE_CAPACITY,
            ),
        }

    @property
    def int_bytes(self) -> int:
        return 8 * (4 + 5 * RECORD_CAPACITY)

    @property
    def float_bytes(self) -> int:
        return 8 * VALUE_CAPACITY * (2 + self.extra_rows)

    def write(self, flat: Flat, ints: dict, floats: dict) -> None:
        records = (len(flat.regret_rows), len(flat.strategy_starts))
        values = (len(flat.regret_values), len(flat.strategy_values))
        if max(records) > RECORD_CAPACITY or max(values) > VALUE_CAPACITY:
            raise RuntimeError(
                f"a walk recorded {records} rows / {values} values, over the scratch "
                f"capacity of {RECORD_CAPACITY} / {VALUE_CAPACITY}"
            )
        ints["counts"][:] = (*records, *values)
        for name in ("regret_rows", "regret_starts", "regret_widths"):
            ints[name][: records[0]] = getattr(flat, name)
        for name in ("strategy_starts", "strategy_widths"):
            ints[name][: records[1]] = getattr(flat, name)
        floats["regret_values"][: values[0]] = flat.regret_values
        floats["strategy_values"][: values[1]] = flat.strategy_values
        floats["extra_values"][:, : values[1]] = flat.extra_values

    def read(self, ints: dict, floats: dict) -> Flat:
        regrets, strategies, regret_values, strategy_values = (int(c) for c in ints["counts"])
        return Flat(
            regret_rows=ints["regret_rows"][:regrets].copy(),
            regret_starts=ints["regret_starts"][:regrets].copy(),
            regret_widths=ints["regret_widths"][:regrets].copy(),
            regret_values=floats["regret_values"][:regret_values].copy(),
            strategy_starts=ints["strategy_starts"][:strategies].copy(),
            strategy_widths=ints["strategy_widths"][:strategies].copy(),
            strategy_values=floats["strategy_values"][:strategy_values].copy(),
            extra_values=floats["extra_values"][:, :strategy_values].copy(),
        )

@dataclass(frozen=True, slots=True)
class _WorkerSpec:
    """Everything a worker process needs, picklable."""

    worker: int
    seed: int
    averages: tuple[Average, ...]
    deal: RootSampler
    keys: tuple[InfoSet, ...] | None  # None: the whole game
    table_blocks: tuple[str, str, str, str]
    control_block: str
    int_block: str
    float_block: str

def _template(keys: Sequence[InfoSet] | None, extra_rows: int) -> PackedTable:
    if keys is None:
        return PackedTable.whole_game(extra_averages=extra_rows)
    return PackedTable.listed(keys, extra_averages=extra_rows)

def _worker_main(spec: _WorkerSpec, go, done, errors) -> None:
    # A Ctrl-C or a container's SIGTERM reaches the whole process group; the
    # parent handles it (finish the iteration, save) and stops the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    blocks: list[shared_memory.SharedMemory] = []
    control = None
    try:
        extra_rows = len(spec.averages)
        template = _template(spec.keys, extra_rows)
        views = {}
        for slot, name in zip(_SLOTS, spec.table_blocks, strict=True):
            block = _attach(name)
            blocks.append(block)
            like = getattr(template, slot)
            views[slot] = np.ndarray(like.shape, dtype=like.dtype, buffer=block.buf)
        table = template.over(**views)
        del template
        control_block = _attach(spec.control_block)
        blocks.append(control_block)
        control = np.ndarray((4,), dtype=np.int64, buffer=control_block.buf)
        layout = _Layout(extra_rows)
        int_block, float_block = _attach(spec.int_block), _attach(spec.float_block)
        blocks += [int_block, float_block]
        ints, floats = layout.ints(int_block.buf), layout.floats(float_block.buf)
        errors.put(("ready", spec.worker, ""))
        while True:
            go.acquire()
            t, seat, stop, _ = (int(value) for value in control)
            if stop:
                return
            recorded = walk(table, spec.deal, spec.seed, t, seat, spec.worker, spec.averages)
            layout.write(flatten(recorded, extra_rows), ints, floats)
            done.release()
    except BaseException:
        errors.put(("failed", spec.worker, traceback.format_exc()))
        if control is not None:
            control[3] = 1
        done.release()
    finally:
        # Drop every view before closing, or `close` refuses exported buffers.
        views = table = control = ints = floats = None  # noqa: F841
        for block in blocks:
            try:
                block.close()
            except BufferError:
                pass

class ParallelLockstep:
    """W worker processes training one lockstep solve over a shared packed table.

        with ParallelLockstep(solve) as run:
            run.train(1_000)

    While open, `solve.table` is the shared table. On exit the workers stop,
    the arrays are copied back into private memory, `solve.table` points at
    the copies, and every shared block is unlinked — also when a worker or
    the caller raised.
    """

    def __init__(self, solve: LockstepSolve, *, keys: Sequence[InfoSet] | None = None):
        if not isinstance(solve.table, PackedTable):
            raise TypeError("a parallel run needs a PackedTable")
        self.solve = solve
        self._keys = None if keys is None else tuple(keys)
        self._blocks: list[shared_memory.SharedMemory] = []
        self._processes: list = []

    def __enter__(self) -> ParallelLockstep:
        try:
            self._start()
        except BaseException:
            self._close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._close()

    def _start(self) -> None:
        solve, workers = self.solve, self.solve.workers
        private = solve.table
        views = {}
        for slot in _SLOTS:
            block, view = _shared(getattr(private, slot))
            self._blocks.append(block)
            views[slot] = view
        self._private = private
        solve.table = private.over(**views)
        # (t, seat, stop, a worker failed)
        control_block, self._control = _shared(np.zeros(4, dtype=np.int64))
        self._blocks.append(control_block)
        self._layout = _Layout(len(solve.averages))
        self._scratch = []
        context = multiprocessing.get_context("spawn")
        self._go = [context.Semaphore(0) for _ in range(workers)]
        self._done = context.Semaphore(0)
        self._errors = context.Queue()
        for worker in range(workers):
            int_block = shared_memory.SharedMemory(create=True, size=self._layout.int_bytes)
            float_block = shared_memory.SharedMemory(create=True, size=self._layout.float_bytes)
            self._blocks += [int_block, float_block]
            self._scratch.append(
                (self._layout.ints(int_block.buf), self._layout.floats(float_block.buf))
            )
            spec = _WorkerSpec(
                worker=worker,
                seed=solve.seed,
                averages=solve.averages,
                deal=solve.deal,
                keys=self._keys,
                table_blocks=tuple(block.name for block in self._blocks[:4]),
                control_block=control_block.name,
                int_block=int_block.name,
                float_block=float_block.name,
            )
            process = context.Process(
                target=_worker_main,
                args=(spec, self._go[worker], self._done, self._errors),
                daemon=True,
            )
            process.start()
            self._processes.append(process)
        self._await_ready()
        self._trouble = ""
        self._stopping = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def _await_ready(self) -> None:
        """Wait until every worker has built its index and attached, or fail fast.

        A worker that dies while starting (an import error, a bad spec) never
        posts `done`; without this the parent would sit out the whole phase
        timeout before saying so.
        """
        ready: set[int] = set()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while len(ready) < len(self._processes):
            try:
                kind, worker, trace = self._errors.get(timeout=0.5)
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
            if kind == "failed":
                raise RuntimeError(f"lockstep worker {worker} failed:\n{trace}")
            ready.add(worker)

    def _collect(self) -> None:
        """Take `done` once per worker.

        A plain blocking wait: macOS has no timed semaphore wait, and Python's
        stand-in polls with sleeps that cost milliseconds a phase. Liveness is
        the watchdog's job — it posts `done` for the missing workers and says
        why, so this wait always ends.
        """
        for _ in self._processes:
            self._done.acquire()
        if self._trouble:
            self._fail(self._trouble)
        if self._control[3]:
            self._fail("a worker failed")

    def _watch(self) -> None:
        """Every second: if a worker has died or a phase has hung, wake the parent."""
        while not self._stopping.wait(1.0):
            dead = [p.pid for p in self._processes if not p.is_alive()]
            if dead and not self._trouble:
                self._trouble = f"dead workers: {dead}"
                for _ in self._processes:
                    self._done.release()

    def _fail(self, why: str) -> None:
        while True:
            try:
                kind, worker, trace = self._errors.get(timeout=2)
            except queue.Empty:
                break
            if kind == "failed":
                raise RuntimeError(f"lockstep worker {worker} failed:\n{trace}") from None
        raise RuntimeError(f"the lockstep run broke ({why})") from None

    def step(self) -> None:
        """One lockstep iteration: both seats, W walks each, banked in worker order."""
        solve = self.solve
        t = solve.iteration + 1
        for seat in SEATS:
            self._control[:3] = (t, seat, 0)
            for go in self._go:
                go.release()
            self._collect()
            flats = [self._layout.read(ints, floats) for ints, floats in self._scratch]
            apply_arrays(solve.table, flats, solve.rule, t)
        solve.iteration = t

    def train(self, iterations: int) -> LockstepSolve:
        if iterations < 1:
            raise ValueError(f"iterations must be at least 1, got {iterations}")
        for _ in range(iterations):
            self.step()
        return self.solve

    def _close(self) -> None:
        if getattr(self, "_watchdog", None) is not None:
            self._stopping.set()
            self._watchdog.join()
            self._watchdog = None
        if self._processes:
            # Live workers are waiting on `go` (or finishing a walk and then
            # will be): the stop flag lets every one of them leave.
            self._control[2] = 1
            for go in self._go:
                go.release()
            for process in self._processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join()
            self._processes = []
        if self._blocks and hasattr(self, "_private"):
            shared = self.solve.table
            copies = {slot: np.array(getattr(shared, slot)) for slot in _SLOTS}
            self.solve.table = self._private.over(**copies)
            del shared
        self._control = self._scratch = None
        for block in self._blocks:
            try:
                block.close()
            except BufferError:
                pass
            block.unlink()
        self._blocks = []
