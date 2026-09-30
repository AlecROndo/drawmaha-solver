"""Rung 3's race on Modal: four regret rules, 10 lockstep workers each, 10M iterations.

    uv run --with modal modal run scripts/modal_rung3.py::smoke          # ~10 min, one rule: it/s on Modal
    uv run --with modal modal run --detach scripts/modal_rung3.py        # launch (or resume) all four
    uv run --with modal modal run scripts/modal_rung3.py::status         # where each run stands
    uv run --with modal modal volume get drawmaha-rung3 dcfr/iter-001000000.npz .

One container per rule (11 cores: 10 workers and the parent), writing to the
`drawmaha-rung3` Volume under `<rule>/`. Every save is committed at once (see
`lockstep_run`), so a container that dies — a spending limit, a preemption —
loses at most ten minutes. A container stops itself after 23 hours, saves, and
spawns its own continuation, so a run crosses Modal's 24-hour cap unattended.
Launching again resumes every unfinished rule from its `resume.npz`; a rule
already at its target returns at once.

Only one container may train a rule at a time: each writes a heartbeat to
`<rule>/owner.json` at every save, and a new one refuses to start while
another's heartbeat is under 30 minutes old.
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
STALE_S = 30 * 60

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
    if owner.exists():
        held = json.loads(owner.read_text())
        if held["task"] != me and time.time() - held["heartbeat"] < STALE_S:
            raise RuntimeError(f"{out.name} is being trained by {held['task']}")
    _beat(out)

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
    if solve.iteration < target:
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
            status[rule] = json.loads(progress.read_text().splitlines()[-1])
    return status

@app.local_entrypoint()
def main(target: int = TARGET, rules: str = ",".join(RULES), seed: int = 0) -> None:
    for rule in rules.split(","):
        call = train_rule.spawn(rule, target, seed)
        print(f"{rule}: launched {call.object_id}")

@app.local_entrypoint()
def smoke(rule: str = "dcfr", iterations: int = 20_000) -> None:
    print(json.dumps(benchmark.remote(rule, iterations), indent=2))

@app.local_entrypoint()
def status() -> None:
    print(json.dumps(read_status.remote(), indent=2))
