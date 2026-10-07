"""Rung 3's race on Modal: four regret rules, 10 lockstep workers each.

`TARGET` (10M) is only the default ceiling; the race itself ran every rule to
2M with `--target 2000000`, and `benchmark`'s `hours_for_default_target` is
measured against that default.

    uv run --with modal modal run scripts/modal_rung3.py::smoke          # ~15 min, one rule: it/s on Modal (`benchmark`)
    uv run --with modal modal deploy scripts/modal_rung3.py              # once, and after any code change
    uv run --with modal modal run scripts/modal_rung3.py::main --target 3000000 --rules vanilla,cfr+,dcfr
    uv run --with modal modal run scripts/modal_rung3.py::status         # where each run stands
    uv run --with modal modal volume get drawmaha-rung3 dcfr/iter-001000000.npz .

Deploy first: `main` then spawns on the DEPLOYED app, whose calls outlive this
terminal and can spawn their own continuations. (Measured 2026-09-30: 30.9
it/s on Modal, 10.5 ms a walk against 2.4 ms on an M5 Pro core.)

One container per rule (11 cores: 10 workers and the parent), writing to the
`drawmaha-rung3` Volume under `<rule>/` (`cfr+` is stored as `cfrplus/`). Every
save is committed at once (see `lockstep_run`), so a container that dies — a
spending limit, a preemption — loses at most ten minutes. A container stops
itself after 23 hours, saves, and spawns its own continuation, so a run crosses
Modal's 24-hour cap unattended. Launching again resumes every unfinished rule
from its `resume.npz`; a rule already at its target returns at once.

Only one container may train a rule at a time. Each writes a heartbeat to
`<rule>/owner.json` at every save, and a newcomer refuses to start while
another's heartbeat is under 15 minutes old, after waiting that long for a dead
one's to lapse — unless that one's last progress line says it stopped and saved
(a preemption), in which case the newcomer takes over and appends a
`claimed_by` line, so the next newcomer is refused. Two newcomers in the same
second are separated by a 30-second settle and a re-read of the owner. Only the
23-hour deadline spawns a continuation; a preempted call is restarted by Modal.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import modal

from drawmaha_solver.minidrawmaha.lockstep import new_lockstep, save_lockstep
from drawmaha_solver.minidrawmaha.lockstep_run import run_to
from drawmaha_solver.minidrawmaha.parallel import ParallelLockstep
from drawmaha_solver.minidrawmaha.regret_rules import RegretRule

# ---------------------------------------------------------------------------
# The app, its Volume and the container shape
# ---------------------------------------------------------------------------

APP = "drawmaha-rung3"  # the deployed app's name, and its Volume's
WORKERS = 10
TARGET = 10_000_000
RULES = tuple(RegretRule)
ROOT = Path("/runs")
# A live trainer beats at every save, at most 10 minutes apart; 15 leaves room
# for a slow save and commit before a heartbeat counts as dead.
STALE_S = 15 * 60
POLL_S = 60
# Long enough for two containers' simultaneous claims to both land on the
# Volume, so each re-reads the same last writer.
CLAIM_SETTLE_S = 30
MODAL_CAP_S = 24 * 3600
DEADLINE_S = MODAL_CAP_S - 3600  # the last hour is room to save and spawn
# Untimed steps before `benchmark`'s clock starts, so starting the workers
# stays out of the steady rate.
WARMUP_ITERATIONS = 50

app = modal.App(APP)
volume = modal.Volume.from_name(APP, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy==2.5.2")
    .add_local_python_source("drawmaha_solver")
)
# `benchmark` runs on the trainer's own shape, so its speed is the race's.
TRAINER = {"image": image, "cpu": WORKERS + 1, "memory": 8192}

def _rule_dir(rule: str) -> Path:
    """A rule's directory on the Volume; `+` is kept out of the path."""
    return ROOT / rule.replace("+", "plus")

# ---------------------------------------------------------------------------
# Launching the race
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(target: int = TARGET, rules: str = ",".join(RULES), seed: int = 0) -> None:
    """Spawn one `train_rule` call per listed rule on the deployed app.

    Every rule name is checked before the first spawn: a typo would otherwise
    start a paid container that claims a directory for a rule that does not
    exist, after the rules ahead of it had already launched.
    """
    chosen = [RegretRule(rule) for rule in rules.split(",")]
    deployed = modal.Function.from_name(APP, "train_rule")
    for rule in chosen:
        call = deployed.spawn(rule.value, target, seed)
        print(f"{rule.value}: launched {call.object_id}")

@app.function(**TRAINER, timeout=MODAL_CAP_S, volumes={str(ROOT): volume})
def train_rule(rule: str, target: int = TARGET, seed: int = 0) -> int:
    """Train one rule toward `target` from its `resume.npz`; return the iteration it stopped at.

    Claims the rule first (`_claim`, then `_settle`) and raises if another
    container is training it. Every save refreshes the heartbeat and commits.
    On return the heartbeat is removed; past the 23-hour deadline the call
    spawns its own continuation. The parameters stay positional: `main` and
    the continuations already running on the deployed app spawn with them.
    """
    out = _rule_dir(rule)
    out.mkdir(parents=True, exist_ok=True)
    volume.reload()
    _claim(out)
    volume.commit()
    _settle(out)
    started = time.monotonic()

    def commit() -> None:
        _beat(out)
        volume.commit()

    solve = run_to(
        out,
        rule=rule,
        target=target,
        workers=WORKERS,
        seed=seed,
        deadline_s=DEADLINE_S,
        commit=commit,
    )
    (out / "owner.json").unlink(missing_ok=True)
    volume.commit()
    if _continues(solve.iteration, target, time.monotonic() - started):
        train_rule.spawn(rule, target, seed)
    return solve.iteration

# ---------------------------------------------------------------------------
# One trainer per rule
# ---------------------------------------------------------------------------

def _claim(out: Path) -> None:
    """Take the rule in `out`, or raise if another live container is training it.

    Waits up to `STALE_S` for a held heartbeat to lapse; a heartbeat already
    stale, or our own, is taken at once. A rule whose last progress line says
    it stopped is free whatever its heartbeat, and taking it over appends a
    `claimed_by` line so a second newcomer sees a live owner.
    """
    me = _task_id()
    owner = out / "owner.json"
    waited = 0
    while owner.exists() and not _stopped(out):
        # A container killed before it could write its `stopped` line leaves a fresh
        # heartbeat behind; wait for it to go stale rather than refuse Modal's restart.
        held = json.loads(owner.read_text())
        if held["task"] == me or time.time() - held["heartbeat"] >= STALE_S:
            break
        if waited >= STALE_S:
            raise RuntimeError(f"{out.name} is being trained by {held['task']}")
        time.sleep(POLL_S)
        waited += POLL_S
        volume.reload()
    if _stopped(out):
        with (out / "progress.jsonl").open("a") as file:
            file.write(json.dumps({"claimed_by": me, "time": time.time()}) + "\n")
    _beat(out)

def _task_id() -> str:
    """This container's Modal task id: the name a claim is held under.

    Never defaulted: two containers sharing a fallback name would each read
    the other's claim as their own.
    """
    return os.environ["MODAL_TASK_ID"]

def _stopped(out: Path) -> bool:
    """Whether the last container on this rule stopped and saved (its last line says so).

    The line says why: a signal (a preemption), the 23-hour `deadline`, or
    `done`. What matters to `_claim` is only that nobody is training: a
    preempted container saves, writes a `stopped` line and commits both, and its
    final commit also refreshes the heartbeat; without this check Modal's
    automatic restart of the same call would be refused as a second trainer. A
    finished run reads as stopped too, which is harmless: a later launch claims
    it, appends its `claimed_by` line, and `run_to` returns at once.
    """
    progress = out / "progress.jsonl"
    if not progress.exists():
        return False
    lines = progress.read_text().splitlines()
    return bool(lines) and "stopped" in json.loads(lines[-1])

def _beat(out: Path) -> None:
    """Write this container's heartbeat: it holds the rule for `STALE_S` from now."""
    (out / "owner.json").write_text(json.dumps({"task": _task_id(), "heartbeat": time.time()}))

def _settle(out: Path) -> None:
    """Refuse to train if another container claimed this rule at the same moment.

    Two containers starting together can both pass `_claim`; commits are
    last-writer-wins, so after a pause both read the same owner and only that
    one trains. The pause is a heuristic: a Volume reload lagging past it could
    leave both alive. They would then write identical files (same resume, same
    counter-based streams) and only waste a container, so the window is narrowed
    rather than closed with a real lock.
    """
    time.sleep(CLAIM_SETTLE_S)
    volume.reload()
    held = json.loads((out / "owner.json").read_text())["task"]
    if held != _task_id():
        raise RuntimeError(f"{out.name} was claimed by {held} at the same time")

def _continues(iteration: int, target: int, elapsed_s: float) -> bool:
    """Whether a container that has stopped should spawn its own continuation.

    Only past the 23-hour deadline. A preempted container stops early, and
    Modal restarts that call itself; spawning here too made a second trainer.
    """
    return iteration < target and elapsed_s >= DEADLINE_S

# ---------------------------------------------------------------------------
# Speed on Modal
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def smoke(rule: str = "dcfr", iterations: int = 20_000) -> None:
    """Print `benchmark`'s speed report for one rule."""
    print(json.dumps(benchmark.remote(rule, iterations), indent=2))

@app.function(**TRAINER, timeout=3600)
def benchmark(rule: str, iterations: int) -> dict:
    """Train a throwaway run (not on the Volume) and report its speed and save time."""
    solve = new_lockstep(0, workers=WORKERS, rule=rule, averages=("uniform", "quadratic"))
    with ParallelLockstep(solve) as run:
        run.train(WARMUP_ITERATIONS)
        started = time.perf_counter()
        run.train(iterations)
        seconds = time.perf_counter() - started
    with tempfile.TemporaryDirectory() as scratch:
        saved = time.perf_counter()
        save_lockstep(solve, Path(scratch) / "checkpoint.npz")
        save_s = time.perf_counter() - saved
    rate = iterations / seconds
    return {
        "rule": rule,
        "iterations_per_s": round(rate, 1),
        "ms_per_iteration": round(1000 / rate, 2),
        "hours_for_default_target": round(TARGET / rate / 3600, 1),
        "checkpoint_save_s": round(save_s, 1),
        "cpu_count": os.cpu_count(),
    }

# ---------------------------------------------------------------------------
# Where each run stands
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def status() -> None:
    """Print each started rule's latest save line."""
    print(json.dumps(read_status.remote(), indent=2))

@app.function(image=image, volumes={str(ROOT): volume})
def read_status() -> dict:
    """Each started rule's last save line from `progress.jsonl` (`claimed_by` lines skipped)."""
    volume.reload()
    status = {}
    for rule in RULES:
        progress = _rule_dir(rule) / "progress.jsonl"
        if progress.exists():
            lines = [json.loads(line) for line in progress.read_text().splitlines()]
            status[rule.value] = [line for line in lines if "iteration" in line][-1]
    return status
