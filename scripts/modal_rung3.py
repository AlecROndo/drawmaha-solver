"""Rung 3's race on Modal: four regret rules, 10 lockstep workers each.

`TARGET` (10M) is only the default ceiling; the race itself ran every rule to
2M with `--target 2000000`, and `benchmark`'s `hours_for_10M` is measured
against that default.

    uv run --with modal modal run scripts/modal_rung3.py::smoke          # ~15 min, one rule: it/s on Modal (`benchmark`)
    uv run --with modal modal deploy scripts/modal_rung3.py              # once, and after any code change
    uv run --with modal modal run scripts/modal_rung3.py::main --target 3000000 --rules vanilla,cfr+,dcfr
    uv run --with modal modal run scripts/modal_rung3.py::status         # where each run stands
    uv run --with modal modal volume get drawmaha-rung3 dcfr/iter-001000000.npz .

Deploy first: `main` then spawns on the DEPLOYED app, whose calls outlive this
terminal and can spawn their own continuations. (Measured 2026-09-30: 30.9
it/s on Modal, 10.5 ms a walk against 2.4 ms on an M5 Pro core.)

One container per rule (11 cores: 10 workers and the parent), writing to the
`drawmaha-rung3` Volume under `<rule>/`. Every save is committed at once (see
`lockstep_run`), so a container that dies — a spending limit, a preemption —
loses at most ten minutes. A container stops itself after 23 hours, saves, and
spawns its own continuation, so a run crosses Modal's 24-hour cap unattended.
Launching again resumes every unfinished rule from its `resume.npz`; a rule
already at its target returns at once.

Only one container may train a rule at a time: each writes a heartbeat to
`<rule>/owner.json` at every save, and a new one refuses to start while
another's heartbeat is under 15 minutes old (it waits that long for a dead one's to lapse) — unless that one's last progress
line says it stopped and saved (a preemption), in which case it takes over and
appends a `claimed_by` line, so the next newcomer is refused. Only the 23-hour
deadline spawns a continuation; a preempted call is restarted by Modal.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

WORKERS = 10
TARGET = 10_000_000
RULES = ("vanilla", "cfr+", "lcfr", "dcfr")
ROOT = Path("/runs")
STALE_S = 15 * 60  # a live trainer beats every 10 minutes
CLAIM_SETTLE_S = 30

app = modal.App("drawmaha-rung3")
volume = modal.Volume.from_name("drawmaha-rung3", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy==2.5.2")
    .add_local_python_source("drawmaha_solver")
)

def _claim(out: Path) -> None:
    """Refuse to train a rule another live container is training."""
    me = os.environ.get("MODAL_TASK_ID", "local")
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
        time.sleep(60)
        waited += 60
        volume.reload()
    if _stopped(out):
        # Taking over a stopped run: mark it, so a second newcomer sees a live owner.
        with (out / "progress.jsonl").open("a") as file:
            file.write(json.dumps({"claimed_by": me, "time": time.time()}) + "\n")
    _beat(out)

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
    owner = out / "owner.json"
    owner.write_text(
        json.dumps({"task": os.environ.get("MODAL_TASK_ID", "local"), "heartbeat": time.time()})
    )

@app.function(
    image=image,
    cpu=WORKERS + 1,
    memory=8192,
    timeout=24 * 3600,
    volumes={str(ROOT): volume},
)
def train_rule(rule: str, target: int = TARGET, seed: int = 0) -> int:
    from drawmaha_solver.minidrawmaha.lockstep_run import run_to

    out = ROOT / rule.replace("+", "plus")
    out.mkdir(parents=True, exist_ok=True)
    volume.reload()
    _claim(out)
    volume.commit()
    # Two containers starting together can both pass `_claim`; commits are last-writer-wins,
    # so after a pause both read the same owner and only that one trains. The pause is a
    # heuristic: a Volume reload lagging past it could leave both alive. They would then
    # write identical files (same resume, same counter-based streams) and only waste a
    # container, so the window is narrowed rather than closed with a real lock.
    time.sleep(CLAIM_SETTLE_S)
    volume.reload()
    held = json.loads((out / "owner.json").read_text())["task"]
    if held != os.environ.get("MODAL_TASK_ID", "local"):
        raise RuntimeError(f"{out.name} was claimed by {held} at the same time")
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
        deadline_s=23 * 3600,
        commit=commit,
    )
    (out / "owner.json").unlink(missing_ok=True)
    volume.commit()
    # Continue only past the 23-hour deadline. A preempted container stops early,
    # and Modal restarts that call itself; spawning here too made a second trainer.
    if solve.iteration < target and time.monotonic() - started >= 23 * 3600:
        train_rule.spawn(rule, target, seed)
    return solve.iteration

@app.function(image=image, cpu=WORKERS + 1, memory=8192, timeout=3600)
def benchmark(rule: str, iterations: int) -> dict:
    """Train a throwaway run (not on the Volume) and report the speed."""
    import tempfile

    from drawmaha_solver.minidrawmaha.lockstep import new_lockstep
    from drawmaha_solver.minidrawmaha.parallel import ParallelLockstep

    solve = new_lockstep(0, workers=WORKERS, rule=rule, averages=("uniform", "quadratic"))
    with ParallelLockstep(solve) as run:
        run.train(50)
        started = time.perf_counter()
        run.train(iterations)
        seconds = time.perf_counter() - started
    with tempfile.TemporaryDirectory() as scratch:
        from drawmaha_solver.minidrawmaha.lockstep import save_lockstep

        saved = time.perf_counter()
        save_lockstep(solve, Path(scratch) / "x.npz")
        save_s = time.perf_counter() - saved
    rate = iterations / seconds
    return {
        "rule": rule,
        "iterations_per_s": round(rate, 1),
        "ms_per_iteration": round(1000 / rate, 2),
        "hours_for_10M": round(TARGET / rate / 3600, 1),
        "checkpoint_save_s": round(save_s, 1),
        "cpu_count": os.cpu_count(),
    }

@app.function(image=image, volumes={str(ROOT): volume})
def read_status() -> dict:
    volume.reload()
    status = {}
    for rule in RULES:
        progress = ROOT / rule.replace("+", "plus") / "progress.jsonl"
        if progress.exists():
            lines = [json.loads(line) for line in progress.read_text().splitlines()]
            status[rule] = [line for line in lines if "iteration" in line][-1]
    return status

@app.local_entrypoint()
def main(target: int = TARGET, rules: str = ",".join(RULES), seed: int = 0) -> None:
    deployed = modal.Function.from_name("drawmaha-rung3", "train_rule")
    for rule in rules.split(","):
        call = deployed.spawn(rule, target, seed)
        print(f"{rule}: launched {call.object_id}")

@app.local_entrypoint()
def smoke(rule: str = "dcfr", iterations: int = 20_000) -> None:
    print(json.dumps(benchmark.remote(rule, iterations), indent=2))

@app.local_entrypoint()
def status() -> None:
    print(json.dumps(read_status.remote(), indent=2))
