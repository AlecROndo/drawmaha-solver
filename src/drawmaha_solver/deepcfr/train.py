"""The loop: Deep CFR iteration after iteration, and a run that stops and restarts exactly.

PRs 1 to 3 built every part of Deep CFR on mini-drawmaha in isolation: the
encoder (`features`), the net and its from-scratch refit (`net`, `fit`), the
write of a net into the table's regret slots (`materialise`), the memories
(`reservoir`) and the walk collector (`collect`). Nothing yet called them in
order. This module is that order — one iteration is

    for p in (0, 1):                                   # alternating, as lockstep does
        learner.write(table, nets)                     # both seats' current nets into the slots
        for observe in observers: observe(table, t, 1 - p)
        rows = collector.collect(seat=p, t=t, walks=K)
        advantage[p].merge(rows.advantage_rows, rows.advantage_targets, t=t, rng=stream(ADVANTAGE))
        policy.merge(rows.policy_rows, rows.policy_targets, t=t, rng=stream(POLICY))
        nets[p] = learner.fit(*advantage[p].contents(), seed=(seed, t, p)).model

— plus everything a 15-hour run needs around it: the resume state, grid-mark
saves, a progress line per iteration, and a graceful stop on a signal or a
deadline. There is no new math here; the exact average and the policy net Π
are PR 5, which plugs into `observers` without touching the loop.

**Why materialise runs before every phase, not once per iteration.** Seat
1's walks at iteration t sample seat 0's FRESHLY refit net — that is what
alternating lockstep does, and what the spec counts as "iteration t's"
strategy for seat 0. So the write happens twice per iteration, and the
observers, which PR 5 will use to accumulate the exact average, are told
which seat's σ is about to be sampled as the opponent: 1 − p.

**The learner seam.** The loop never names torch. It talks to a `Learner`
with four methods — `fit` a model to the memory, `write` two models into
the table, `save` and `load` a model — and the real one, `NetLearner`,
imports `fit`, `materialise` and `net` inside those methods. That is what
lets the whole-loop test run in the default suite with no torch: the
`OracleLearner` below is a "perfect net" that returns, at every row, the
t′-weighted sum of every regret sample the memory holds there, and with it
Deep CFR collapses into rung 3's tabular LCFR. The one thing the loop asks
of a `write` is that it assigns INTO `table.cumulative_regret` and never
rebinds it: the collector's workers read that array through a shared block,
and a rebound array would leave them walking stale regrets.

**Gate 1, and why it is two pins.** LCFR banks `R += t · Σ r` per column, the
samples summed in walk order (`lockstep.apply_arrays`, then
`regret_rules.bank_rows`). With the oracle and a memory large enough that
nothing is evicted, the loop computes the same sums from the same walks, so
after N iterations its table must equal `train_lockstep(rule="lcfr",
workers=K)`'s. One cast stands between the two: the memory stores a sample
as float32 (`collect.rows_of` pads and casts), and about half of a walk's
regrets are not float32-exact. So the test pins the loop BIT FOR BIT, over
three iterations, against LCFR built from rung 3's own `walk`, `flatten`
and `apply_arrays` with each sample rounded through float32 first — every
join of the loop (a stale table, the wrong t, the wrong seat, a dropped or
reordered row) still breaks that — and to float32 precision against the
untouched `train_lockstep`. The oracle repeats LCFR's float operations in
LCFR's order (per t′ in increasing order, `R[cols] += t′ · totals`) because
a single Σ t′·r over all rows would round differently from t = 2 on.

**The resume state** is three memories, the two latest nets, and a small
`state.json` written LAST and atomically: it is the commit point, so a crash
anywhere in a save leaves the previous `state.json` naming the previous,
complete files. The files carry their iteration in their names for that
reason — overwriting fixed names would let a crash between two memories put
two iterations under one `state.json`. The configuration travels in the
state, and a resume whose seed, K, C or net size differs is refused by the
field's name, as rung 3 refuses a W change: each of them changes the sample.
W itself may change, because the collector's rows are a function of (seed,
t, seat, k) and never of W.

**Determinism.** Everything is a function of (seed, t, seat, k): the walks
draw from `lockstep.stream`, the memories from `reservoir_stream`, the refit
from `(seed, t, p)`. No RNG state is ever checkpointed. The collector's pool
is spawned when the run opens, before torch is imported, and the refit's
threads run while the workers idle, so nothing forks after torch is loaded
and the two never compete for cores.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from drawmaha_solver.deepcfr.reservoir import (
    TARGET_WIDTH,
    Memory,
    Reservoir,
    _is_int,
    reservoir_stream,
)
from drawmaha_solver.minidrawmaha.lockstep import SEATS, _columns
from drawmaha_solver.minidrawmaha.packed_table import PackedTable

# The iterations that line up with the tabular race's hands per seat: at
# K = 10,000 walks per seat per iteration, Deep CFR's iteration 10 is LCFR's
# 10,000 × 10 hands, and so on up to the race's own endpoint.
GRID = (10, 20, 50, 100, 200, 500, 1_000, 2_000)

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepConfig:
    """Everything a run is: the sample, the net and refit, the marks, the save cadence.

    Written into every checkpoint. The net and refit fields are `FitConfig`'s
    seven, carried flat because `fit.py` imports torch at its top and this
    module must not; `NetLearner` builds the `FitConfig` when it is asked to.
    """

    seed: int = 0
    walks: int = 10_000
    workers: int = 8
    capacity: int = 50_000_000
    width: int = 64
    depth: int = 3
    steps: int = 4_000
    batch: int = 10_000
    lr: float = 1e-3
    clip: float = 1.0
    threads: int = 8
    grid: tuple[int, ...] = GRID
    # The resume state carries 3.6 GB of memories at the run's C, so 30
    # minutes rather than rung 3's 10.
    save_every_s: float = 1_800.0

    def __post_init__(self) -> None:
        if not _is_int(self.seed) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {self.seed!r}")
        for name in ("walks", "capacity", "width", "depth", "steps", "batch", "threads"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        # Zero workers walks in this process: the reference the pool must equal.
        if not _is_int(self.workers) or self.workers < 0:
            raise ValueError(f"workers must be a non-negative int, got {self.workers!r}")
        # `not > 0` rather than `<= 0`: a NaN is neither, and would fit in silence.
        for name in ("lr", "clip"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        marks = tuple(self.grid)
        if any(not _is_int(m) or m < 1 for m in marks) or any(
            a >= b for a, b in zip(marks, marks[1:], strict=False)
        ):
            raise ValueError(f"grid must be strictly increasing positive ints, got {self.grid!r}")
        if not self.save_every_s >= 0:
            raise ValueError(f"save_every_s must be non-negative, got {self.save_every_s!r}")

# The fields a resume may not change: each one changes the sample or the
# fit, so a run continued under a different value would be a different run.
# W, the grid and the save cadence change nothing the walks or the nets see.
SAMPLE_FIELDS = (
    "seed",
    "walks",
    "capacity",
    "width",
    "depth",
    "steps",
    "batch",
    "lr",
    "clip",
    "threads",
)

def require_same_run(saved: DeepConfig, asked: DeepConfig) -> None:
    """Refuse a resume that changes any sample field, naming every one that differs."""
    changed = [
        f"{name}: {getattr(saved, name)!r} → {getattr(asked, name)!r}"
        for name in SAMPLE_FIELDS
        if getattr(saved, name) != getattr(asked, name)
    ]
    if changed:
        raise ValueError(
            "the checkpoint's run differs from the one asked for in "
            + ", ".join(changed)
            + "; each of these changes the sample, so the run cannot continue"
        )

@dataclass(slots=True)
class DeepRun:
    """The mutable state of a run: the last completed iteration, two nets, three memories.

    `nets[seat]` is whatever the learner's `fit` returned, or None for a seat
    not yet trained, which `write` must play as uniform.
    """

    config: DeepConfig
    t: int
    nets: list[object | None]
    advantage: tuple[Reservoir, Reservoir]
    policy: Reservoir

def new_run(config: DeepConfig) -> DeepRun:
    """A fresh run at iteration 0: no nets, three empty memories of the run's capacity."""
    return DeepRun(
        config=config,
        t=0,
        nets=[None, None],
        advantage=(Reservoir(config.capacity), Reservoir(config.capacity)),
        policy=Reservoir(config.capacity),
    )

# ---------------------------------------------------------------------------
# The learner seam
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Fitted:
    """What one refit returns: the model, and its loss on a held batch (None if it has none)."""

    model: object
    loss: float | None

class Learner(Protocol):
    """The one seam between the loop and the model.

    `fit` takes a memory's `contents()` and the fit's seed `(seed, t, seat)`;
    `write` plays `models[seat]` for both seats by assigning into
    `table.cumulative_regret`; `save` and `load` round-trip one model
    through one file, for the resume state and the kept nets.
    """

    def fit(
        self, rows: np.ndarray, t: np.ndarray, targets: np.ndarray, *, seed: tuple[int, int, int]
    ) -> Fitted: ...

    def write(self, table: PackedTable, models: Sequence[object | None]) -> None: ...

    def save(self, model: object, path: Path) -> None: ...

    def load(self, path: Path) -> object: ...

# What runs right after each write: `(table, t, seat)`, with `seat` the one
# whose σ is about to be sampled as the opponent, 1 − p.
Observer = Callable[[PackedTable, int, int], None]

# ---------------------------------------------------------------------------
# One iteration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StepStats:
    """What one iteration took: seconds per stage per seat, the refits' losses, the rows offered."""

    t: int
    materialise: tuple[float, float]
    collect: tuple[float, float]
    merge: tuple[float, float]
    fit: tuple[float, float]
    seconds: float
    loss: tuple[float | None, float | None]
    rows: tuple[int, int]

def step(
    run: DeepRun,
    collector,
    learner: Learner,
    observers: Sequence[Observer] = (),
) -> StepStats:
    """One iteration, both seats' phases in the spec's order, timed stage by stage.

    `collector` is a `Collector` (or anything with its `table` and
    `collect`); the iteration is `run.t + 1`, and `run.t` advances once both
    phases are done.
    """
    t = run.t + 1
    config = run.config
    table = collector.table
    materialise, collect, merge, fit, loss, rows = [], [], [], [], [], []
    started = time.perf_counter()
    for p in SEATS:
        with _timed(materialise):
            learner.write(table, tuple(run.nets))
        for observe in observers:
            observe(table, t, 1 - p)
        with _timed(collect):
            offered = collector.collect(seat=p, t=t, walks=config.walks)
        with _timed(merge):
            run.advantage[p].merge(
                offered.advantage_rows,
                offered.advantage_targets,
                t=t,
                rng=reservoir_stream(seed=config.seed, t=t, seat=p, memory=Memory.ADVANTAGE),
            )
            run.policy.merge(
                offered.policy_rows,
                offered.policy_targets,
                t=t,
                rng=reservoir_stream(seed=config.seed, t=t, seat=p, memory=Memory.POLICY),
            )
        with _timed(fit):
            fitted = learner.fit(*run.advantage[p].contents(), seed=(config.seed, t, p))
        run.nets[p] = fitted.model
        loss.append(fitted.loss)
        rows.append(int(offered.advantage_rows.shape[0]))
    run.t = t
    return StepStats(
        t=t,
        materialise=tuple(materialise),
        collect=tuple(collect),
        merge=tuple(merge),
        fit=tuple(fit),
        seconds=time.perf_counter() - started,
        loss=tuple(loss),
        rows=tuple(rows),
    )

@contextmanager
def _timed(into: list[float]):
    """Append the block's wall seconds to `into` when it ends, also on an exception."""
    started = time.perf_counter()
    try:
        yield
    finally:
        into.append(time.perf_counter() - started)

# ---------------------------------------------------------------------------
# The resume state
# ---------------------------------------------------------------------------

# The commit point of a save: written last, and the only file a load trusts.
STATE = "state.json"

def save_run(run: DeepRun, directory: Path, *, learner: Learner) -> None:
    """Write the run's three memories, its nets and `state.json` into `directory`; the state last.

    Every file carries the iteration in its name, so the files of the
    previous save stand untouched until the new `state.json` is in place;
    then whatever the state does not name is removed. A crash anywhere
    before the state's rename leaves the previous state loadable.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    t = run.t
    advantage = [_memory_name("advantage", t, seat=seat) for seat in SEATS]
    policy = _memory_name("policy", t)
    nets = [None if net is None else _net_name(seat, t) for seat, net in enumerate(run.nets)]
    for seat in SEATS:
        run.advantage[seat].save(directory / advantage[seat])
    run.policy.save(directory / policy)
    for seat, name in enumerate(nets):
        if name is not None:
            _atomically(learner.save, run.nets[seat], directory / name)
    state = {
        "t": t,
        "config": asdict(run.config),
        "files": {"advantage": advantage, "policy": policy, "nets": nets},
    }
    _write_state(directory, state)
    keep = {STATE, *advantage, policy, *(name for name in nets if name is not None)}
    for stale in directory.iterdir():
        if stale.is_file() and stale.name not in keep:
            stale.unlink()

def load_run(directory: Path, *, learner: Learner) -> DeepRun:
    """The run as `save_run` left it in `directory`, ready for the next `step`.

    Refuses, by name, a directory with no `state.json` and a state naming a
    file that is not there, rather than continuing from a memory it does
    not have.
    """
    directory = Path(directory)
    state_path = directory / STATE
    if not state_path.exists():
        raise ValueError(f"{directory} holds no {STATE}; there is no run to resume")
    state = json.loads(state_path.read_text())
    files = state["files"]
    named = [*files["advantage"], files["policy"], *(n for n in files["nets"] if n is not None)]
    for name in named:
        if not (directory / name).exists():
            raise ValueError(f"{state_path} names {name}, which is not in {directory}")
    saved = dict(state["config"])
    saved["grid"] = tuple(saved["grid"])
    config = DeepConfig(**saved)
    return DeepRun(
        config=config,
        t=int(state["t"]),
        nets=[None if name is None else learner.load(directory / name) for name in files["nets"]],
        advantage=tuple(Reservoir.load(directory / name) for name in files["advantage"]),
        policy=Reservoir.load(directory / files["policy"]),
    )

def _memory_name(kind: str, t: int, *, seat: int | None = None) -> str:
    who = "" if seat is None else f"-p{seat}"
    return f"{kind}{who}-t{t:05d}.npz"

def _net_name(seat: int, t: int) -> str:
    return f"p{seat}-t{t:05d}.pt"

def _write_state(directory: Path, state: dict) -> None:
    """`state.json`, written as a temp file, fsynced and renamed into place."""
    _atomically(lambda text, path: path.write_text(text), json.dumps(state, indent=1), directory / STATE)

def _atomically(write: Callable[[object, Path], None], value: object, path: Path) -> None:
    """Run `write(value, temp)` then rename the temp onto `path`, so a crash leaves `path` whole."""
    temporary = path.with_name(path.name + ".tmp")
    write(value, temporary)
    with temporary.open("rb") as file:
        os.fsync(file.fileno())
    os.replace(temporary, path)

# ---------------------------------------------------------------------------
# The oracle: gate 1's perfect net
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OracleModel:
    """A copy of one advantage memory at fit time: the "net" that remembers every sample."""

    rows: np.ndarray  # int32 (n,)
    t: np.ndarray  # int32 (n,)
    targets: np.ndarray  # float32 (n, TARGET_WIDTH)

class OracleLearner:
    """The learner that makes the loop LCFR: its model is the memory, its write is LCFR's bank.

    `fit` copies the memory. `write` puts at every row of each seat the
    t′-weighted sum of the samples its model holds there, computed as LCFR
    computes it — per t′ in increasing order, the samples of that t′ summed
    per column in memory order with `np.add.at`, then `R[cols] += t′ ·
    totals` — so the bits match `bank_rows` through any number of
    iterations, not only the first. A row no sample names stays exactly
    0.0. NumPy only, so gate 1 runs without torch.
    """

    def __init__(self) -> None:
        # Row → first column, from the table's widths; built once per table.
        self._starts: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(
        self, rows: np.ndarray, t: np.ndarray, targets: np.ndarray, *, seed: tuple[int, int, int]
    ) -> Fitted:
        return Fitted(OracleModel(np.array(rows), np.array(t), np.array(targets)), None)

    def write(self, table: PackedTable, models: Sequence[OracleModel | None]) -> None:
        starts, widths = self._layout(table)
        summed = np.zeros(table.cumulative_regret.size)
        for model in models:
            if model is None:
                continue
            for t in np.unique(model.t):
                at = model.t == t
                rows = model.rows[at]
                width = widths[rows]
                columns = _columns(starts[rows], width)
                legal = np.arange(TARGET_WIDTH)[None, :] < width[:, None]
                values = model.targets[at][legal].astype(np.float64)
                touched, _, inverse = np.unique(columns, return_index=True, return_inverse=True)
                totals = np.zeros(touched.size)
                np.add.at(totals, inverse, values)
                summed[touched] += int(t) * totals
        # Assigned into the array, never rebound: the workers read this block.
        table.cumulative_regret[...] = summed

    def save(self, model: OracleModel, path: Path) -> None:
        # Through an open file, so NumPy does not append `.npz` to the name.
        with Path(path).open("wb") as file:
            np.savez(file, rows=model.rows, t=model.t, targets=model.targets)

    def load(self, path: Path) -> OracleModel:
        with np.load(Path(path), allow_pickle=False) as saved:
            return OracleModel(saved["rows"], saved["t"], saved["targets"])

    def _layout(self, table: PackedTable) -> tuple[np.ndarray, np.ndarray]:
        if id(table) not in self._starts:
            widths = table.widths()
            self._starts[id(table)] = (np.cumsum(widths) - widths, widths)
        return self._starts[id(table)]
