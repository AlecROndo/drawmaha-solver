"""Drive one lockstep run to its target without losing what it has trained.

    run_to(out, rule="dcfr", target=10_000_000, workers=10)

Everything lives in `out/`:

- `resume.npz` — the latest state, rewritten every `save_every_s` seconds;
- `iter-<t>.npz` — a kept checkpoint at each mark of the grid, never
  overwritten;
- `progress.jsonl` — one line per save: iteration, speed, what was written.

Every save is a temp file renamed into place, then its progress line, then
one `commit()` (a Modal Volume's commit; nothing locally). A checkpoint holds
every iteration up to its t, so a run killed outright loses at most the time
since its last commit. SIGTERM and SIGINT end the run gracefully: the
iteration finishes, the state is saved and committed. Calling `run_to` again
on the same `out` resumes from `resume.npz` and, because the streams are
counter-based, equals never having stopped.
"""

from __future__ import annotations

import json
import signal
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from drawmaha_solver.minidrawmaha.game import InfoSet
from drawmaha_solver.minidrawmaha.lockstep import (
    LockstepSolve,
    load_lockstep,
    new_lockstep,
    save_lockstep,
)
from drawmaha_solver.minidrawmaha.parallel import ParallelLockstep, packed_table_for
from drawmaha_solver.minidrawmaha.regret_rules import Average, RegretRule, validate_averages

# 1-2-5 to a million, then every million: dense where the curve bends and
# through the last 9M, where two close rules are told apart.
GRID = (
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    *range(1_000_000, 10_000_001, 1_000_000),
)

RESUME = "resume.npz"

def kept_name(iteration: int) -> str:
    """The file name of the checkpoint kept at grid mark `iteration`."""
    return f"iter-{iteration:09d}.npz"

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_to(
    out: Path,
    *,
    rule: RegretRule | str,
    target: int,
    workers: int,
    seed: int = 0,
    averages: Sequence[Average | str] = ("uniform", "quadratic"),
    grid: Sequence[int] = GRID,
    save_every_s: float = 600.0,
    deadline_s: float | None = None,
    commit: Callable[[], None] = lambda: None,
    log: Callable[[str], None] = print,
    keys: Sequence[InfoSet] | None = None,
) -> LockstepSolve:
    """Train `out`'s run up to `target` iterations, or until `deadline_s` or a signal.

    Returns the solve as it stands; `solve.iteration == target` means done.
    `keys` allocates a listed slice instead of the whole game (tests). The
    deadline counts from once the run is open, worker startup included.
    Raises whatever the workers raise, without saving: the last commit stands.

    Steps:
    1. Open the run: resume `out/resume.npz`, or start one at iteration 0.
    2. Return at once if it is already at `target`.
    3. With SIGTERM and SIGINT turned into a stop request, train in W
       processes, saving on the grid and every `save_every_s`.
    4. Save the state it stopped at, with why it stopped.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rule = RegretRule(rule)
    averages = validate_averages(averages)

    # Step 1: open the run
    solve = _open(out, rule=rule, averages=averages, workers=workers, seed=seed, keys=keys)

    # Step 2: nothing to do
    if solve.iteration >= target:
        log(f"{rule.value}: already at {solve.iteration:,} of {target:,}")
        return solve

    deadline = None if deadline_s is None else time.monotonic() + deadline_s
    saves = _Saves(out=out, target=target, commit=commit, log=log)
    with _stop_on_signals() as stop, ParallelLockstep(solve, keys=keys) as run:
        log(f"{rule.value}: from {solve.iteration:,} to {target:,} on {workers} workers")

        # Step 3: train, saving as it goes
        why = _train(
            run,
            target=target,
            marks=sorted(m for m in grid if solve.iteration < m <= target),
            save_every_s=save_every_s,
            deadline=deadline,
            stop=stop,
            saves=saves,
        )

        # Step 4: the final save
        saves.save(solve, kept=False, pace=None, why=why)
    return solve

def _open(
    out: Path,
    *,
    rule: RegretRule,
    averages: tuple[Average, ...],
    workers: int,
    seed: int,
    keys: Sequence[InfoSet] | None,
) -> LockstepSolve:
    """The run in `out`: resumed from its `resume.npz`, or new at iteration 0."""
    table = packed_table_for(keys, extra_averages=len(averages))
    resume = out / RESUME
    if not resume.exists():
        return new_lockstep(seed, workers=workers, table=table, rule=rule, averages=averages)
    solve = load_lockstep(resume, table=table, workers=workers, rule=rule, averages=averages)
    # A resume continues the saved run's streams; a different seed would be
    # dropped silently, so refuse it as `load_lockstep` refuses a different W.
    if solve.seed != seed:
        raise ValueError(
            f"{resume} was trained with seed {solve.seed}, and this run asks for {seed}"
        )
    return solve

# ---------------------------------------------------------------------------
# Stopping on a signal
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _StopRequest:
    """Why the run was asked to stop; empty until a signal arrives."""

    why: str = ""

@contextmanager
def _stop_on_signals() -> Iterator[_StopRequest]:
    """Turn SIGTERM and SIGINT into a stop request for the block's duration.

    The run checks the request between iterations, so a signal never cuts an
    iteration (or a save) in half. The previous handlers come back on exit.
    """
    stop = _StopRequest()

    def request_stop(signum, frame) -> None:
        stop.why = signal.Signals(signum).name

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield stop
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

# ---------------------------------------------------------------------------
# Training, saving as it goes
# ---------------------------------------------------------------------------

def _train(
    run: ParallelLockstep,
    *,
    target: int,
    marks: list[int],
    save_every_s: float,
    deadline: float | None,
    stop: _StopRequest,
    saves: _Saves,
) -> str:
    """Step until `target`, the deadline or a signal, and say which ended it.

    Saves after any iteration that lands on a grid mark (the kept checkpoint
    and the resume file) or that comes `save_every_s` after the last save (the
    resume file alone). `deadline` is a `time.monotonic()` reading, or None.
    """
    solve = run.solve
    pace = _Pace(since=time.monotonic(), iteration=solve.iteration)
    while solve.iteration < target and not stop.why:
        if deadline is not None and time.monotonic() > deadline:
            return "deadline"
        run.step()
        kept = bool(marks) and solve.iteration == marks[0]
        if kept:
            marks.pop(0)
        if kept or time.monotonic() - pace.since >= save_every_s:
            saves.save(solve, kept=kept, pace=pace)
    return stop.why or "done"

@dataclass(slots=True)
class _Pace:
    """When the last save finished and at which iteration: the speed's baseline."""

    since: float
    iteration: int

    def lap(self, iteration: int) -> float:
        """Iterations a second since the last lap, which this one replaces."""
        now = time.monotonic()
        # Floored so two laps inside one clock tick cannot divide by zero.
        rate = (iteration - self.iteration) / max(now - self.since, 1e-9)
        self.since, self.iteration = now, iteration
        return rate

@dataclass(frozen=True, slots=True)
class _Saves:
    """Where a run's saves go, and who hears of them: the one writer of `out/`."""

    out: Path
    target: int
    commit: Callable[[], None]
    log: Callable[[str], None]

    def save(
        self, solve: LockstepSolve, *, kept: bool, pace: _Pace | None, why: str = ""
    ) -> None:
        """Write the kept checkpoint (if `kept`) and the resume file, log it, commit once.

        `pace` times the speed since the previous save and is lapped once the
        files are written (None for the final save, which reports no speed);
        `why`, set on the final save only, is why the run stopped.
        """
        wrote = []
        if kept:
            save_lockstep(solve, self.out / kept_name(solve.iteration))
            wrote.append(kept_name(solve.iteration))
        save_lockstep(solve, self.out / RESUME)
        wrote.append(RESUME)
        rate = None if pace is None else pace.lap(solve.iteration)
        line = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rule": solve.rule.value,
            "iteration": solve.iteration,
            "target": self.target,
            "iterations_per_s": None if rate is None else round(rate, 2),
            "eta_h": None if not rate else round((self.target - solve.iteration) / rate / 3600, 1),
            "wrote": wrote,
        }
        if why:
            line["stopped"] = why
        with (self.out / "progress.jsonl").open("a") as file:
            file.write(json.dumps(line) + "\n")
        self.commit()
        self.log(json.dumps(line))
