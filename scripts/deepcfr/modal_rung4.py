"""Rung 4 on Modal: one Deep CFR trainer, a grader per grid mark, the smoke and the net-size sweep.

    uv run --with modal modal deploy scripts/deepcfr/modal_rung4.py                 # once, and after any code change
    uv run --with modal modal run scripts/deepcfr/modal_rung4.py::smoke            # 3 iterations at K = 1,000: seconds per step, one mark graded
    uv run --with modal modal run scripts/deepcfr/modal_rung4.py::sweep            # 3 × 64, 4 × 64, 3 × 128 to 100 iterations, three containers
    uv run --with modal modal run scripts/deepcfr/modal_rung4.py::sweep --grade    # once `status` shows all three at 100: figures/rung4/sweep.json
    uv run --with modal modal run scripts/deepcfr/modal_rung4.py::train            # the run: 2,000 iterations at K = 10,000, resumable
    uv run --with modal modal run scripts/deepcfr/modal_rung4.py::status           # where every run stands
    uv run --with modal modal run scripts/deepcfr/modal_rung4.py::grade            # every kept mark, three objects each: figures/rung4/grades.json
    uv run deep-analysis                                                          # the price figure and the answer sheet from grades.json

Deploy first: every launch spawns on the DEPLOYED app, whose calls outlive
this terminal and can spawn their own continuations. Every `modal run` here
starts paid containers; only a person launches one (`scripts/AGENTS.md`).

**What runs where.** `train` is `scripts/deepcfr/run_deep_cfr.py`'s twin on
one container of W + 1 cores (W = 32 workers and the parent; the refits use
`--threads` of them while the workers idle) and 16 GB, writing to the
Volume `drawmaha-rung4` under `<name>/` in exactly the layout `train.run_deep`
writes on a Mac — `nets/` (every iteration's two nets), `marks/t00010/` …
(a full save at each grid mark: the three memories, the nets, the exact
average's sums), `resume/` (the latest state, rewritten every 30 minutes
and at every stop) and `progress.jsonl`. The exact average rides along as
an observer from iteration 1, as the local script passes it: it cannot be
rebuilt later without replaying every kept net. The claim, heartbeat and
23-hour continuation are rung 3's (`scripts/minidrawmaha/modal_rung3.py`),
lifted rather than imported: that script builds its own app and Volume at
import, and these helpers read nothing of Modal's but `volume.reload()`.

`grade` fans out one container per kept mark. Each loads the mark — its
memories, nets and sums — and grades the spec's three objects with the
untouched exact meter: the **exact average** (`ExactAverage.frozen`), **Π**
(a fresh policy net fit on the policy memory as it stands at the mark,
`fit_policy` under the run's own refit config, then `policy_strategy`) and
the **current iterate** (`current_iterate` over the table the mark's nets
materialise). Beside each grade go `hands_per_seat` (t · K, the number that
lines a mark up with the tabular race), `wall_clock_s` (the training clock
through t: the sum of each iteration's seconds off `progress.jsonl`, last
line per t, saves excluded) and the run's config. The three strategies are
kept on the Volume under `frozen/` so any of them can be released as rung
3's was. The local entrypoint writes them into `figures/rung4/grades.json`
keyed `deep/<t>/<exact|pi|current>` (`analysis.deep_key`), regrading the
marks it is asked for — a grade is a pure function of the mark, so a
regrade is idempotent and the fan-out costs the same wall clock for one
mark as for eight.

`smoke` is gate 6: three iterations at K = 1,000 on the trainer's own
shape, in a scratch directory and not on the Volume, then ONE more
iteration resumed from the save (the resume path, exact average included),
then the three objects graded from the kept mark on the same container.
Its report is the seconds per stage, the projection to the full run — only
the walks scale with K, so collect is scaled by 10,000 / K and the rest
taken as measured — and that projection priced at Modal's listed rates,
which replaces the spec's ~26 s and ~$70 budget. The grader's seconds come
with it, so the eight grade containers are budgeted too.

`sweep` is the spec's net-size question: 3 × 64, 4 × 64 and 3 × 128 to 100
iterations each (1M hands per seat) in three trainers at once, under
`sweep/<width>x<depth>/`; `sweep --grade`, once `status` shows all three
at 100, grades each at its mark and writes `figures/rung4/sweep.json` with
the winner. **The winner's rule is fixed before the numbers exist:** the
run uses 3 × 64 unless another size's exact average grades lower by more
than `CLEAR_MARGIN` (10%) of 3 × 64's — one seed, so a smaller gap is
noise until shown otherwise. Train the winner with `train --width --depth`.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

import modal

from drawmaha_solver.deepcfr.average import ExactAverage
from drawmaha_solver.deepcfr.policy import fit_policy, policy_strategy
from drawmaha_solver.deepcfr.train import (
    GRID,
    MARKS,
    PROGRESS,
    STATE,
    DeepConfig,
    NetLearner,
    current_iterate,
    load_run,
    mark_name,
    run_deep,
    saved_config,
)
from drawmaha_solver.minidrawmaha.answer import analysis
from drawmaha_solver.minidrawmaha.answer.analysis import (
    DEEP_COLUMNS,
    DEEP_GRADES,
    RUNG4,
    Grade,
    deep_key,
)
from drawmaha_solver.minidrawmaha.answer.strategy import FrozenStrategy, save_strategy
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable

# ---------------------------------------------------------------------------
# The app, its Volume and the container shapes
# ---------------------------------------------------------------------------

APP = "drawmaha-rung4"  # the deployed app's name, and its Volume's
WORKERS = 32
TARGET = GRID[-1]
RUN = "deep"  # the run's directory on the Volume, and the first word of its keys in grades.json
SWEEP = "sweep"  # the sweep's runs live under `sweep/<width>x<depth>/`
SIZES = ((64, 3), (64, 4), (128, 3))  # (width, depth): the spec's three
DEFAULT_SIZE = (64, 3)
SWEEP_ITERATIONS = 100
# The sweep's pre-registered verdict: a size beats the default only by more
# than this share of the default's exact-average exploitability.
CLEAR_MARGIN = 0.10
SMOKE_WALKS = 1_000
SMOKE_ITERATIONS = 3
ROOT = Path("/runs")
FROZEN = "frozen"  # the graded strategies, beside `marks/` in a run's directory
SWEEP_GRADES = RUNG4 / "sweep.json"
# A live trainer beats at every progress line, under a minute apart; 15
# leaves room for a 3.6 GB resume save and its commit before a heartbeat
# counts as dead.
STALE_S = 15 * 60
POLL_S = 60
# Long enough for two containers' simultaneous claims to both land on the
# Volume, so each re-reads the same last writer.
CLAIM_SETTLE_S = 30
MODAL_CAP_S = 24 * 3600
DEADLINE_S = MODAL_CAP_S - 3600  # the last hour is room to save and spawn
MEMORY_GIB = 16
GRADER_CPU = 8
GRADER_GIB = 8
# Modal's listed on-demand rates, read 2026-10-21 (modal.com/pricing); the
# smoke prices its projection with them.
CPU_USD_PER_CORE_S = 0.0000131
MEMORY_USD_PER_GIB_S = 0.00000222

app = modal.App(APP)
volume = modal.Volume.from_name(APP, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.2", "matplotlib==3.11.1")
    .pip_install("torch==2.14.1", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source("drawmaha_solver")
)
TRAINER = {"image": image, "cpu": WORKERS + 1, "memory": MEMORY_GIB * 1024}
# The grader holds the compiled game (1.5 GB), a table, a policy memory of
# up to 1.2 GB and Π's fit; eight threads for that fit.
GRADER = {"image": image, "cpu": GRADER_CPU, "memory": GRADER_GIB * 1024}

def _run_dir(name: str) -> Path:
    """A run's directory on the Volume."""
    return ROOT / name

def _size_name(width: int, depth: int) -> str:
    """`64x3`: width first, as `Net(width=, depth=)` names them."""
    return f"{width}x{depth}"

def _config_to(config: DeepConfig) -> dict:
    """The config as plain data for a spawn: JSON-clean, the grid a list."""
    return {**asdict(config), "grid": list(config.grid)}

def _config_from(carried: Mapping) -> DeepConfig:
    """The config back from `_config_to`'s data, refusing bad values by name as `DeepConfig` does."""
    return DeepConfig(**{**carried, "grid": tuple(carried["grid"])})

def _run_config(
    *, walks: int, workers: int, width: int, depth: int, threads: int, seed: int
) -> DeepConfig:
    """The spec's run with the flags a launch may set; everything else is `DeepConfig`'s default."""
    return DeepConfig(seed=seed, walks=walks, workers=workers, width=width, depth=depth, threads=threads)

# ---------------------------------------------------------------------------
# Launching the run
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def train(
    target: int = TARGET,
    name: str = RUN,
    walks: int = DeepConfig.walks,
    workers: int = WORKERS,
    width: int = DEFAULT_SIZE[0],
    depth: int = DEFAULT_SIZE[1],
    threads: int = DeepConfig.threads,
    seed: int = DeepConfig.seed,
) -> None:
    """Spawn one `train_run` on the deployed app; re-run with the same flags to resume.

    The config is built here, before the spawn, so a bad flag is refused by
    `DeepConfig` on this machine and never starts a paid container.
    """
    config = _run_config(walks=walks, workers=workers, width=width, depth=depth, threads=threads, seed=seed)
    call = modal.Function.from_name(APP, "train_run").spawn(name, target, _config_to(config))
    print(f"{name}: launched {call.object_id} toward {target:,} iterations ({_size_name(width, depth)}, K = {walks:,}, W = {workers})")

@app.function(**TRAINER, timeout=MODAL_CAP_S, volumes={str(ROOT): volume})
def train_run(name: str, target: int, config: dict) -> int:
    """Train `name` toward `target` from its `resume/`; return the iteration it stopped at.

    Claims the run first (`_claim`, then `_settle`) and raises if another
    container is training it. Every progress line refreshes the heartbeat
    and commits. On return the heartbeat is removed; past the 23-hour
    deadline the call spawns its own continuation, which resumes with the
    exact average's file as `run_deep` requires. The parameters stay
    positional: `train`, `sweep` and the continuations already running on
    the deployed app spawn with them.
    """
    out = _run_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    volume.reload()
    _claim(out)
    volume.commit()
    _settle(out)
    started = time.monotonic()

    def commit() -> None:
        _beat(out)
        volume.commit()

    run = run_deep(
        out,
        _config_from(config),
        target=target,
        deadline_s=DEADLINE_S,
        commit=commit,
        observers=(ExactAverage(),),
    )
    (out / "owner.json").unlink(missing_ok=True)
    volume.commit()
    if _continues(run.t, target, time.monotonic() - started):
        train_run.spawn(name, target, config)
    return run.t

# ---------------------------------------------------------------------------
# The claim, lifted from rung 3's launcher
# ---------------------------------------------------------------------------

def _claim(out: Path) -> None:
    """Take the run in `out`, or raise if another live container is training it.

    Waits up to `STALE_S` for a held heartbeat to lapse; a heartbeat already
    stale, or our own, is taken at once. A run whose last progress line says
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
        with (out / PROGRESS).open("a") as file:
            file.write(json.dumps({"claimed_by": me, "time": time.time()}) + "\n")
    _beat(out)

def _task_id() -> str:
    """This container's Modal task id: the name a claim is held under.

    Never defaulted: two containers sharing a fallback name would each read
    the other's claim as their own.
    """
    return os.environ["MODAL_TASK_ID"]

def _stopped(out: Path) -> bool:
    """Whether the last container on this run stopped and saved (its last line says so).

    `run_deep` writes `stopped` on its last line — a signal, the 23-hour
    `deadline`, or `done` — and commits; a preempted container's final
    commit also refreshes the heartbeat, so without this check Modal's
    automatic restart of the same call would be refused as a second trainer.
    """
    progress = out / PROGRESS
    if not progress.exists():
        return False
    lines = progress.read_text().splitlines()
    return bool(lines) and "stopped" in json.loads(lines[-1])

def _beat(out: Path) -> None:
    """Write this container's heartbeat: it holds the run for `STALE_S` from now."""
    (out / "owner.json").write_text(json.dumps({"task": _task_id(), "heartbeat": time.time()}))

def _settle(out: Path) -> None:
    """Refuse to train if another container claimed this run at the same moment.

    Two containers starting together can both pass `_claim`; commits are
    last-writer-wins, so after a pause both read the same owner and only
    that one trains. The window is narrowed, not closed: two survivors
    would write identical files (the sample is a function of the seed) and
    only waste a container.
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
# Grading: one container per mark, the three objects
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def grade(name: str = RUN, marks: str = "", grades: str = str(DEEP_GRADES), local: str = "") -> None:
    """Grade the run's kept marks — all of them, or the listed ones — and write them into `grades`.

    One `grade_mark` container per mark, all at once; with `--local DIR`
    the marks of a run made on this machine (`scripts/deepcfr/run_deep_cfr.py`)
    are graded here, one after another, by the same `grade_mark_dir`, and
    nothing of Modal's is touched. A mark the run has not kept is refused
    by name before any grade starts. Keys the grader did not write are
    kept; the file is written sorted, so the marks read in training order.
    """
    if local:
        kept = _kept_marks(Path(local))
    else:
        kept = modal.Function.from_name(APP, "read_marks").remote(name)
    asked = sorted(int(m) for m in marks.split(",")) if marks else kept
    missing = [t for t in asked if t not in kept]
    if missing:
        raise ValueError(f"{local or name} has kept no mark at {missing}; its marks are {kept}")
    if local:
        results = [(t, grade_mark_dir(Path(local), t)) for t in asked]
    else:
        deployed = modal.Function.from_name(APP, "grade_mark")
        calls = [(t, deployed.spawn(name, t)) for t in asked]
        results = [(t, call.get()) for t, call in calls]
    path = Path(grades)
    held = json.loads(path.read_text()) if path.exists() else {}
    for t, graded in results:
        for column, entry in graded.items():
            held[deep_key(t, column)] = entry
            print(f"{deep_key(t, column)}: {entry.get('exploitability', float('nan')):.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(held.items())), indent=2) + "\n")
    print(f"wrote {path.resolve()}")

@app.function(**GRADER, timeout=4 * 3600, volumes={str(ROOT): volume})
def grade_mark(name: str, t: int) -> dict[str, dict]:
    """Grade mark t of run `name` on the Volume; the frozen strategies are committed beside it."""
    volume.reload()
    graded = grade_mark_dir(_run_dir(name), t)
    volume.commit()
    return graded

@app.function(image=image, volumes={str(ROOT): volume})
def read_marks(name: str) -> list[int]:
    """The grid marks run `name` has kept on the Volume, in order."""
    volume.reload()
    return _kept_marks(_run_dir(name))

def _kept_marks(run_dir: Path) -> list[int]:
    """The marks under `run_dir/marks/` that reached their `state.json`, in order."""
    marks = Path(run_dir) / MARKS
    if not marks.exists():
        return []
    return sorted(int(mark.name[1:]) for mark in marks.iterdir() if (mark / STATE).exists())

def grade_mark_dir(
    run_dir: Path, t: int, *, grade: Callable[[FrozenStrategy], Grade] = analysis.grade
) -> dict[str, dict]:
    """The three graded objects of mark t in `run_dir`, keyed exact / pi / current, in that order.

    Loads the mark under the run's own config: the nets through
    `NetLearner`, the sums into an `ExactAverage`, the memories as saved.
    Π is fit here, fresh, on the policy memory as the mark left it, under
    the run's refit config. Each strategy is saved under `frozen/` before it
    is graded, so a grade that dies leaves the file. The grade is the exact
    meter, two minutes a call; the tests hand in a stand-in.
    """
    mark = Path(run_dir) / MARKS / mark_name(t)
    config = saved_config(mark)
    learner = NetLearner(config)
    exact = ExactAverage()
    run = load_run(mark, learner=learner, observers=(exact,))
    if run.t != t:
        raise ValueError(f"{mark} holds iteration {run.t}, not {t}")
    beside = {
        "hands_per_seat": t * config.walks,
        "wall_clock_s": _wall_clock_s(Path(run_dir) / PROGRESS, t),
        "config": _config_to(config),
    }
    table = PackedTable.whole_game()
    learner.write(table, tuple(run.nets))
    fit_started = time.perf_counter()
    pi = fit_policy(*run.policy.contents(), seed=config.seed, iteration=t, config=learner.fit_config())
    strategies = {
        "exact": exact.frozen(iteration=t, seed=config.seed),
        "pi": policy_strategy(pi, iteration=t, seed=config.seed),
        "current": current_iterate(table, iteration=t, seed=config.seed),
    }
    pi_fit_s = time.perf_counter() - fit_started
    frozen = Path(run_dir) / FROZEN
    frozen.mkdir(parents=True, exist_ok=True)
    graded = {}
    for column, strategy in strategies.items():
        save_strategy(strategy, frozen / f"{mark_name(t)}-{column}.npz")
        started = time.perf_counter()
        measured = grade(strategy)
        graded[column] = {**measured, **beside, "grade_s": round(time.perf_counter() - started, 1)}
    graded["pi"]["pi_fit_s"] = round(pi_fit_s, 1)
    return graded

def _wall_clock_s(progress: Path, t: int) -> float:
    """The training clock through iteration t: each iteration's seconds, last line per t, summed.

    Saves and the opens are not in it; a crashed-and-resumed stretch counts
    the iterations it re-ran once, as `train` says a reader should. Refuses
    a t the log never reached.
    """
    seconds: dict[int, float] = {}
    for raw in Path(progress).read_text().splitlines():
        line = json.loads(raw)
        if "seconds" in line:
            seconds[int(line["t"])] = float(line["seconds"]["iteration"])
    if t not in seconds:
        raise ValueError(f"{progress} has no iteration {t}; it reached {max(seconds, default=0)}")
    return sum(s for at, s in seconds.items() if at <= t)

# ---------------------------------------------------------------------------
# The smoke: speed on Modal, the resume path, one mark graded
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def smoke(walks: int = SMOKE_WALKS, iterations: int = SMOKE_ITERATIONS, workers: int = WORKERS) -> None:
    """Print `benchmark`'s report: seconds per stage, the projection and its price, the grader's seconds."""
    print(json.dumps(benchmark.remote(walks, iterations, workers), indent=2))

@app.function(**TRAINER, timeout=3 * 3600)
def benchmark(walks: int, iterations: int, workers: int) -> dict:
    """Train a throwaway run (not on the Volume) to `iterations`, resume one more, grade the mark, report.

    The run is the spec's but for K and the grid, so its refits and memories
    are the real run's; the mark at `iterations` is what the grader loads.
    """
    config = DeepConfig(walks=walks, workers=workers, grid=(iterations,))
    lines: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "smoke"
        run_deep(out, config, target=iterations, observers=(ExactAverage(),), log=lines.append)
        run_deep(out, config, target=iterations + 1, observers=(ExactAverage(),), log=lines.append)
        graded = grade_mark_dir(out, iterations)
    progress = [json.loads(line) for line in lines if line.startswith("{")]
    return _speed_report(progress, graded, walks=walks, workers=workers, cpu_count=os.cpu_count())

def _speed_report(
    progress: list[dict], graded: Mapping[str, Mapping], *, walks: int, workers: int, cpu_count: int | None
) -> dict:
    """Seconds per stage over the lines (last per t), the full run's projection, and its price.

    Collect is the one stage that scales with K, so the projection takes
    it × `DeepConfig.walks / walks` and every other stage as measured. The
    price is the trainer's cores and memory at Modal's listed rates for the
    projected hours of `TARGET` iterations.
    """
    last = {int(line["t"]): line for line in progress if "seconds" in line}
    stages = ("materialise", "collect", "merge", "fit")
    seconds = {
        stage: sum(sum(line["seconds"][stage]) for line in last.values()) / len(last) for stage in stages
    }
    per_iteration = sum(line["seconds"]["iteration"] for line in last.values()) / len(last)
    scale = DeepConfig.walks / walks
    projected = per_iteration - seconds["collect"] + seconds["collect"] * scale
    hours = projected * TARGET / 3600
    usd = hours * 3600 * ((workers + 1) * CPU_USD_PER_CORE_S + MEMORY_GIB * MEMORY_USD_PER_GIB_S)
    grade_seconds = {column: float(entry["grade_s"]) for column, entry in graded.items()}
    grade_seconds["pi_fit"] = float(graded["pi"]["pi_fit_s"])
    return {
        "walks": walks,
        "workers": workers,
        "cpu_count": cpu_count,
        "iterations": len(last),
        "seconds": {stage: round(s, 3) for stage, s in seconds.items()},
        "seconds_per_iteration": round(per_iteration, 3),
        "projected_seconds_per_iteration": round(projected, 3),
        "projected_hours_for_full_run": round(hours, 2),
        "projected_usd_for_full_run": round(usd, 2),
        "grade_seconds": grade_seconds,
    }

# ---------------------------------------------------------------------------
# The net-size sweep
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def sweep(
    grade: bool = False,
    target: int = SWEEP_ITERATIONS,
    walks: int = DeepConfig.walks,
    workers: int = WORKERS,
    threads: int = DeepConfig.threads,
    seed: int = DeepConfig.seed,
    out: str = str(SWEEP_GRADES),
) -> None:
    """Launch the three sizes to `target` iterations, or with `--grade` grade them there and name the winner.

    The grade refuses, by name, a size whose run is short of `target`
    before any container starts; a run that has not started is refused the
    same way.
    """
    names = {size: f"{SWEEP}/{_size_name(*size)}" for size in SIZES}
    if not grade:
        deployed = modal.Function.from_name(APP, "train_run")
        for (width, depth), name in names.items():
            config = _run_config(walks=walks, workers=workers, width=width, depth=depth, threads=threads, seed=seed)
            call = deployed.spawn(name, target, _config_to(config))
            print(f"{name}: launched {call.object_id}")
        return
    status = modal.Function.from_name(APP, "read_status").remote()
    short = [f"{name} at {status.get(name, {}).get('t', 0)}" for name in names.values()
             if status.get(name, {}).get("t", 0) < target]
    if short:
        raise RuntimeError(f"not every size has reached {target}: {', '.join(short)}")
    deployed = modal.Function.from_name(APP, "grade_mark")
    calls = {size: deployed.spawn(name, target) for size, name in names.items()}
    by_size = {_size_name(*size): call.get() for size, call in calls.items()}
    winner = sweep_winner(by_size)
    sheet = {
        "iterations": target,
        "hands_per_seat": target * walks,
        "rule": f"{_size_name(*DEFAULT_SIZE)} unless another size's exact average is lower by more "
                f"than {CLEAR_MARGIN:.0%} of its exploitability",
        "winner": winner,
        "sizes": by_size,
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sheet, indent=2) + "\n")
    for size, columns in by_size.items():
        print(f"{size}: " + ", ".join(f"{c} {columns[c]['exploitability']:.4f}" for c in DEEP_COLUMNS))
    print(f"winner by the pre-registered rule: {winner}; wrote {path.resolve()}")

def sweep_winner(by_size: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str:
    """The size the run should use: the default, unless one grades clearly lower on the exact average.

    "Clearly" is `CLEAR_MARGIN`: lower than the default's exact-average
    exploitability by more than that share of it. A tie at the line keeps
    the default.
    """
    default = _size_name(*DEFAULT_SIZE)
    exact = {size: float(columns["exact"]["exploitability"]) for size, columns in by_size.items()}
    best = min(exact, key=exact.__getitem__)
    return best if exact[best] < exact[default] * (1.0 - CLEAR_MARGIN) else default

# ---------------------------------------------------------------------------
# Where each run stands
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def status() -> None:
    """Print each started run's latest iteration line."""
    print(json.dumps(read_status.remote(), indent=2))

@app.function(image=image, volumes={str(ROOT): volume})
def read_status() -> dict:
    """Each started run's last iteration line from `progress.jsonl` (claim lines and bare stops skipped).

    The runs are the main one and the sweep's three, by name; a run with no
    progress file has not started and is left out.
    """
    volume.reload()
    status = {}
    for name in (RUN, *(f"{SWEEP}/{_size_name(*size)}" for size in SIZES)):
        progress = _run_dir(name) / PROGRESS
        if progress.exists():
            lines = [json.loads(line) for line in progress.read_text().splitlines()]
            iterations = [line for line in lines if "seconds" in line]
            if iterations:
                status[name] = iterations[-1]
    return status
