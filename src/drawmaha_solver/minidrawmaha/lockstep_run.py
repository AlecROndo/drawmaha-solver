"""Drive one lockstep run to its target without losing what it has trained.

    run_to(out, rule="dcfr", target=10_000_000, workers=10)

Everything lives in `out/`:

- `resume.npz` — the latest state, rewritten every `save_every_s` seconds;
- `iter-<t>.npz` — a kept checkpoint at each mark of the grid, never overwritten;
- `progress.jsonl` — one line per save: iteration, speed, what was written.

Every save is a temp file renamed into place, then `commit()` (a Modal Volume's
commit; nothing locally). A checkpoint holds every iteration up to its t, so a
run killed outright loses at most the time since its last commit. SIGTERM and
SIGINT end the run gracefully: the iteration finishes, the state is saved and
committed. Calling `run_to` again on the same `out` resumes from `resume.npz`
and, because the streams are counter-based, equals never having stopped.
"""

from __future__ import annotations

import json
import signal
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from drawmaha_solver.minidrawmaha.lockstep import (
    LockstepSolve,
    load_lockstep,
    new_lockstep,
    save_lockstep,
)
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.parallel import ParallelLockstep
from drawmaha_solver.minidrawmaha.regret_rules import RegretRule, validate_averages

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
    return f"iter-{iteration:09d}.npz"

def run_to(
    out: Path,
    *,
    rule: RegretRule | str,
    target: int,
    workers: int,
    seed: int = 0,
    averages: Sequence[str] = ("uniform", "quadratic"),
    grid: Sequence[int] = GRID,
    save_every_s: float = 600.0,
    deadline_s: float | None = None,
    commit: Callable[[], None] = lambda: None,
    log: Callable[[str], None] = print,
    keys=None,
) -> LockstepSolve:
    """Train `out`'s run up to `target` iterations, or until `deadline_s` or a signal.

    Returns the solve as it stands; `solve.iteration == target` means done.
    `keys` allocates a listed slice instead of the whole game (tests).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rule = RegretRule(rule)
    averages = validate_averages(averages)
    solve = _open(out, rule=rule, averages=averages, workers=workers, seed=seed, keys=keys)
    if solve.iteration >= target:
        log(f"{rule.value}: already at {solve.iteration:,} of {target:,}")
        return solve

    stop = {"why": ""}

    def request_stop(signum, frame):
        stop["why"] = signal.Signals(signum).name

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()
    marks = sorted(m for m in grid if solve.iteration < m <= target)
    try:
        with ParallelLockstep(solve, keys=keys) as run:
            log(f"{rule.value}: from {solve.iteration:,} to {target:,} on {workers} workers")
            last_save = time.monotonic()
            window = (time.monotonic(), solve.iteration)
            while solve.iteration < target and not stop["why"]:
                if deadline_s is not None and time.monotonic() - started > deadline_s:
                    stop["why"] = "deadline"
                    break
                run.step()
                wrote = []
                if marks and solve.iteration == marks[0]:
                    marks.pop(0)
                    save_lockstep(solve, out / kept_name(solve.iteration))
                    wrote.append(kept_name(solve.iteration))
                if wrote or time.monotonic() - last_save >= save_every_s:
                    save_lockstep(solve, out / RESUME)
                    wrote.append(RESUME)
                    commit()
                    now = time.monotonic()
                    rate = (solve.iteration - window[1]) / max(now - window[0], 1e-9)
                    window, last_save = (now, solve.iteration), now
                    _progress(out, solve, rate, target, wrote, commit, log)
            save_lockstep(solve, out / RESUME)
            commit()
            _progress(out, solve, None, target, [RESUME], commit, log, why=stop["why"] or "done")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return solve

def _open(out: Path, *, rule, averages, workers, seed, keys) -> LockstepSolve:
    def table() -> PackedTable:
        if keys is None:
            return PackedTable.whole_game(extra_averages=len(averages))
        return PackedTable.listed(keys, extra_averages=len(averages))

    resume = out / RESUME
    if resume.exists():
        return load_lockstep(resume, table=table(), workers=workers, rule=rule, averages=averages)
    return new_lockstep(seed, workers=workers, table=table(), rule=rule, averages=averages)

def _progress(out, solve, rate, target, wrote, commit, log, *, why: str = "") -> None:
    line = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rule": solve.rule.value,
        "iteration": solve.iteration,
        "target": target,
        "iterations_per_s": None if rate is None else round(rate, 2),
        "eta_h": None if not rate else round((target - solve.iteration) / rate / 3600, 1),
        "wrote": wrote,
    }
    if why:
        line["stopped"] = why
    with (out / "progress.jsonl").open("a") as file:
        file.write(json.dumps(line) + "\n")
    commit()
    log(json.dumps(line))
