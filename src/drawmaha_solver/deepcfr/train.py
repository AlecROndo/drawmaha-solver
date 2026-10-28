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
are PR 5 (`average`, `policy`), which plug into `observers` without
touching the loop.

**Why materialise runs before every phase, not once per iteration.** Seat
1's walks at iteration t sample seat 0's FRESHLY refit net — that is what
alternating lockstep does, and what the spec counts as "iteration t's"
strategy for seat 0. So the write happens twice per iteration, and the
observers — PR 5's `ExactAverage` accumulates the exact average there — are
told which seat's σ is about to be sampled as the opponent: 1 − p.

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

**The resume state** is three memories, the two latest nets, the file of
every observer that carries state (PR 5's `ExactAverage`: an observer with
`name`, `save` and `load` — see `StatefulObserver`), and a small
`state.json` written LAST and atomically: it is the commit point, so a crash
anywhere in a save leaves the previous `state.json` naming the previous,
complete files. A resume refuses to drop an observer the state names, and
an observer whose file the state lacks, because either would make the
average silently wrong. The files carry their iteration in their names for that
reason — overwriting fixed names would let a crash between two memories put
two iterations under one `state.json`. The configuration travels in the
state, and a resume whose seed, K, C, net size or any refit field but the
device differs is refused by the field's name, as rung 3 refuses a W
change: each changes the sample or the fit — the thread count too, since a
matmul's summation order can follow it, so a run begun at 8 threads resumes
at 8. W itself may change, because the collector's rows are a function of
(seed, t, seat, k) and never of W. The device may change too, but not for
W's reason: it is let through on purpose, so a run can move to the GPU,
and a run that did is no longer reproducible from its seed and config.

**Determinism.** Everything is a function of (seed, t, seat, k): the walks
draw from `lockstep.stream`, the memories from `reservoir_stream`, the refit
from `(seed, t, p)`. No RNG state is ever checkpointed. The collector's
workers are spawned, not forked, so a torch import before or after them
changes nothing; the refit's threads run while the workers idle, so the two
never compete for cores.

**Stopping.** SIGTERM and SIGINT are turned into a stop request the loop
reads after each iteration and before the next, so a signal never cuts an
iteration or a save in half and never costs an extra iteration either. A
terminal's Ctrl-C reaches the whole process group, workers included: a
worker that died on it would break the pool and end the run without its
save. So the workers are spawned while the parent ignores both signals —
a spawned child inherits an ignored disposition, and Python installs its
own SIGINT handler only over the default — and only then are the parent's
handlers put in. Rung 3's `parallel.py` has its workers ignore the signals
themselves; PR 3's collector does not, and this module does not edit it.

**The log.** `progress.jsonl` gets one line per iteration, and one more
when a stop is noticed before an iteration starts. A run that crashed and
resumed re-appends the iterations since its last resume save, so a reader
takes the LAST line per t.

Measured on this Mac (2026-10-20, W = 8, the default 3 × 64 net and
4,000 × 10,000 refit, iteration 2 of a run): at K = 1,000, 25.9 s per
iteration — materialise 0.46 + 0.47, collect 0.33 + 0.26, merge 0.001 ×2,
fit 12.4 + 12.0; at K = 10,000, 35.0 s — materialise 0.46 + 0.46, collect
3.95 + 2.97, merge 0.007 + 0.006, fit 13.7 + 13.4. The refits are the cost
at either K; the spec budgeted ~26 s per iteration on 33 Modal cores.
"""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from drawmaha_solver.deepcfr.collect import Collector
from drawmaha_solver.deepcfr.reservoir import (
    TARGET_WIDTH,
    Memory,
    Reservoir,
    _is_int,
    reservoir_stream,
)
from drawmaha_solver.minidrawmaha.rules.features import layout
from drawmaha_solver.minidrawmaha.rules.game import chip_state
from drawmaha_solver.minidrawmaha.solver.lockstep import SEATS, _columns
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.answer.strategy import FrozenStrategy, StrategyInfo, average_rows

# The iterations that line up with the tabular race's hands per seat: at
# K = 10,000 walks per seat per iteration, Deep CFR's iteration 10 is LCFR's
# 10,000 × 10 hands, and so on up to the race's own endpoint.
GRID = (10, 20, 50, 100, 200, 500, 1_000, 2_000)

# Where a refit may run. `fit.FitConfig` spells the same two out, since it
# must not import this module's loop; a test keeps them equal.
DEVICES = ("cpu", "mps")
# The regret rules a refit can fit (`rule_targets`): LCFR, the spec's, and two
# screens of the discounting rules rung 3 raced, by their per-sample form.
RULES = ("linear", "dcfr", "cfr+")

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepConfig:
    """Everything a run is: the sample, the net and refit, the marks, the save cadence.

    Written into every checkpoint. The net and refit fields are `FitConfig`'s
    eight, the device among them, carried flat because `fit.py` imports
    torch at its top and this module must not; `NetLearner` builds the
    `FitConfig` when it is asked to. A `state.json` written before a field
    existed loads with the value that run had, not today's default
    (`ADDED_LATER`): "cpu" for the device, 0 for the pot power.
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
    # Read by the cpu refit only; an mps refit ignores it. It stays a sample
    # field all the same, so a run begun at 8 threads that moves to mps and
    # back resumes on the cpu at 8, and a move must repeat the run's count.
    threads: int = 8
    # Where the refit runs: "cpu", or "mps" for Apple's GPU. A resume may
    # change it, but not as it may change W, which changes nothing: the
    # device orders the refit's arithmetic, so a moved run is the same rule
    # with different nets from the move on, and its seed and config no
    # longer reproduce it bit for bit. The device is let through anyway so
    # a long run can move to the GPU mid-way. Only the name is checked here,
    # since this module must not import torch: whether the machine has the
    # GPU is checked by `run_deep_cfr.py` at parse time, and otherwise by the
    # first refit, after that iteration's walks.
    device: str = "cpu"
    # Which regret rule the refit's targets follow; see `rule_targets`.
    rule: str = "linear"
    # Each regret sample is divided by its spot's pot to this power before the
    # refit; see `pot_scaled`. 0 is the spec's raw chips; 1/2, the default
    # since 2026-10-27, weighs each spot by what a σ mistake there costs, and
    # cut the 4 × 256, 16,000-step, lr 5e-4 arm's exact average by a quarter
    # at 2M hands per seat, one seed (`docs/notes/rung4-run-log.md`, finding
    # 7). That arm ran at half this config's `lr`: √pot with lr 1e-3 has not
    # been run, so the quarter is the 5e-4 pairing's, not this default's.
    pot_power: float = 0.5
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
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if not self.save_every_s >= 0:
            raise ValueError(f"save_every_s must be non-negative, got {self.save_every_s!r}")
        # Written as `not (0 <= x <= 1)` so a NaN, which is in no interval, is refused too.
        if not 0 <= self.pot_power <= 1:
            raise ValueError(f"pot_power must be in [0, 1], got {self.pot_power!r}")

# Fields added after runs had been saved, each with the value those runs had.
# A saved config that lacks one is from before the field existed, and must
# read as what that run was: a run saved before the pot power existed fitted
# raw chips, and resuming it under today's default would change its weighting
# half-way with nothing in the log. `config_from_saved` applies these; a
# resume then refuses the mismatch by name, as it refuses any sample field.
ADDED_LATER: Mapping[str, object] = {"device": "cpu", "pot_power": 0.0}

# The fields a resume may not change: each one changes the sample or the
# fit, so a run continued under a different value would be a different run.
# W, the grid and the save cadence change nothing the walks or the nets see.
# The device does change the nets, and is let through on purpose, so a run
# can move to the GPU mid-way (see `DeepConfig.device`).
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
    "rule",
    "pot_power",
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

@runtime_checkable
class StatefulObserver(Protocol):
    """An observer that carries state across iterations: PR 5's exact average.

    Its `name` names its file, `{name}-t{t:05d}.npz`, which `save_run`
    writes beside the memories (atomically, as `Reservoir.save` is: the
    observer's own `save` is trusted to be) and `state.json` names; a
    resumed run `load`s it before its first step. An observer without the
    three members is stateless and is left alone — `isinstance` on this
    protocol is that test.
    """

    name: str

    def __call__(self, table: PackedTable, t: int, seat: int) -> None: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

def _stateful(observers: Sequence[Observer]) -> list[StatefulObserver]:
    """The observers that carry state, in the order given; two with one name are refused."""
    found = [o for o in observers if isinstance(o, StatefulObserver)]
    names = [o.name for o in found]
    for name in names:
        if names.count(name) > 1:
            raise ValueError(f"two stateful observers are named {name!r}; their files would collide")
    return found

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
    phases are done. A raise mid-way leaves `run` half-advanced (seat 0's
    net and memories updated, `t` not), which `run_deep` never saves.
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

def save_run(
    run: DeepRun, directory: Path, *, learner: Learner, observers: Sequence[Observer] = ()
) -> None:
    """Write the run's three memories, its nets, its stateful observers and `state.json` into `directory`; the state last.

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
    stateful = _stateful(observers)
    extras = {observer.name: _observer_name(observer.name, t) for observer in stateful}
    for seat in SEATS:
        run.advantage[seat].save(directory / advantage[seat])
    run.policy.save(directory / policy)
    for seat, name in enumerate(nets):
        if name is not None:
            _atomically(learner.save, run.nets[seat], directory / name)
    for observer in stateful:
        observer.save(directory / extras[observer.name])
    state = {
        "t": t,
        "config": asdict(run.config),
        "files": {"advantage": advantage, "policy": policy, "nets": nets, "observers": extras},
    }
    _write_state(directory, state)
    keep = {STATE, *advantage, policy, *(name for name in nets if name is not None), *extras.values()}
    for stale in directory.iterdir():
        if stale.is_file() and stale.name not in keep:
            stale.unlink()

def load_run(
    directory: Path, *, learner: Learner, observers: Sequence[Observer] = ()
) -> DeepRun:
    """The run as `save_run` left it in `directory`, ready for the next `step`.

    Each stateful observer given is loaded from the file the state names
    for it. Refuses, by name, a directory with no `state.json`, a state
    naming a file that is not there, and a stateful observer the state has
    no file for (a save made without it), rather than continuing from a
    memory or a sum it does not have. A state that names an observer NOT
    given here loads fine — a grader opening a mark for its memories and
    nets needs no observer; it is `run_deep`'s resume that refuses to drop
    one.
    """
    directory = Path(directory)
    state = _read_state(directory)
    files = state["files"]
    named = [*files["advantage"], files["policy"], *(n for n in files["nets"] if n is not None)]
    saved = files.get("observers", {})
    stateful = _stateful(observers)
    for observer in stateful:
        if observer.name not in saved:
            raise ValueError(
                f"{directory / STATE} holds no file for the observer {observer.name!r}: "
                "that save was made without it"
            )
        named.append(saved[observer.name])
    for name in named:
        if not (directory / name).exists():
            raise ValueError(f"{directory / STATE} names {name}, which is not in {directory}")
    run = DeepRun(
        config=_config_of(state),
        t=int(state["t"]),
        nets=[None if name is None else learner.load(directory / name) for name in files["nets"]],
        advantage=tuple(Reservoir.load(directory / name) for name in files["advantage"]),
        policy=Reservoir.load(directory / files["policy"]),
    )
    # Last, so a refused config or memory leaves the caller's observers as
    # they were.
    for observer in stateful:
        observer.load(directory / saved[observer.name])
    return run

def saved_config(directory: Path) -> DeepConfig:
    """The config a save was made under, read from its `state.json` alone.

    What a resume checks BEFORE loading: a refusal costs one small file,
    not the gigabytes of memories behind it.
    """
    return _config_of(_read_state(Path(directory)))

def _read_state(directory: Path) -> dict:
    state_path = directory / STATE
    if not state_path.exists():
        raise ValueError(f"{directory} holds no {STATE}; there is no run to resume")
    return json.loads(state_path.read_text())

def config_from_saved(record: Mapping) -> DeepConfig:
    """The config a record was written under, by code that may have had fewer fields.

    `record` is a config as JSON left it: the grid a list, and any field in
    `ADDED_LATER` possibly absent, in which case it takes the value that
    run had. Bad values are refused by name, as `DeepConfig` refuses them,
    and so is a field this code does not know: that record was written by
    newer code, and this code cannot continue the run it describes.
    """
    unknown = sorted(set(record) - set(DeepConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(
            f"the saved config has fields this code does not know: {', '.join(unknown)}; "
            "it was written by newer code, and only that code can continue the run"
        )
    saved = {**ADDED_LATER, **record}
    saved["grid"] = tuple(saved["grid"])
    return DeepConfig(**saved)

def _config_of(state: dict) -> DeepConfig:
    return config_from_saved(state["config"])

def _memory_name(kind: str, t: int, *, seat: int | None = None) -> str:
    who = "" if seat is None else f"-p{seat}"
    return f"{kind}{who}-t{t:05d}.npz"

def _net_name(seat: int, t: int) -> str:
    return f"p{seat}-t{t:05d}.pt"

def _observer_name(name: str, t: int) -> str:
    return f"{name}-t{t:05d}.npz"

def _write_state(directory: Path, state: dict) -> None:
    """`state.json`, written as a temp file, fsynced and renamed into place."""
    _atomically(
        lambda text, path: path.write_text(text), json.dumps(state, indent=1), directory / STATE
    )

def _atomically(write: Callable[[object, Path], None], value: object, path: Path) -> None:
    """Run `write(value, temp)` then rename the temp onto `path`, so a crash leaves `path` whole."""
    temporary = path.with_name(path.name + ".tmp")
    write(value, temporary)
    with temporary.open("rb") as file:
        os.fsync(file.fileno())
    os.replace(temporary, path)

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

    `lockstep_run._stop_on_signals`, copied rather than imported from a
    private name. The run checks the request after each iteration, so a
    signal never cuts an iteration or a save in half; the previous handlers
    come back on exit.
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
# The driver
# ---------------------------------------------------------------------------

# The layout of a run's directory.
NETS = "nets"  # every iteration's two nets, never overwritten
MARKS = "marks"  # a full save at each grid mark
RESUME = "resume"  # the latest state, rewritten on the clock
PROGRESS = "progress.jsonl"  # one line per iteration

def net_name(seat: int, t: int) -> str:
    """The kept net's file name under `nets/`: `p{seat}-t{t:05d}.pt`."""
    return _net_name(seat, t)

def mark_name(t: int) -> str:
    """The directory name of the save kept at grid mark t: `t{t:05d}`."""
    return f"t{t:05d}"

def run_deep(
    out: Path,
    config: DeepConfig,
    *,
    target: int,
    learner: Learner | None = None,
    deadline_s: float | None = None,
    commit: Callable[[], None] = lambda: None,
    log: Callable[[str], None] = print,
    observers: Sequence[Observer] = (),
) -> DeepRun:
    """Train `out`'s run up to `target` iterations, or until `deadline_s` or a signal.

    Returns the run as it stands; `run.t == target` means done. `learner`
    defaults to `NetLearner(config)`, the real one. The deadline counts from
    once the run is open, worker startup included, and is read before each
    iteration and after it, as a signal is. Raises whatever the collector or
    the learner raises, without saving: the last commit stands.

    Steps:
    1. Open the run: resume `out/resume`, or start one at iteration 0.
    2. Return at once if it is already at `target`.
    3. Open one collector for the whole run, its workers started under
       ignored signals; then, with SIGTERM and SIGINT turned into a stop
       request, step until `target`, the deadline or a signal; after every
       iteration keep both nets, save on the grid and on the clock, append
       a progress line, commit once.
    4. The last save includes the resume state, and the last line says why
       the run stopped.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if learner is None:
        learner = NetLearner(config)
    # Two stateful observers with one name are refused here, before an
    # iteration is paid for, not at the first save.
    _stateful(observers)

    # Step 1: open the run
    resumed_on = saved_config(out / RESUME).device if (out / RESUME / STATE).exists() else None
    run = _open(out, config, learner=learner, observers=observers)
    # A resume may move the refit (see `DeepConfig.device`), and the CLI's
    # default is the cpu, so a restart that forgot `--device mps` would move
    # the run back without a word; the move is always said.
    if resumed_on is not None and resumed_on != config.device:
        log(f"deep: the refit moves from {resumed_on} to {config.device} at iteration {run.t:,}")

    # Step 2: nothing to do
    if run.t >= target:
        log(f"deep: already at {run.t:,} of {target:,}")
        return run

    deadline = None if deadline_s is None else time.monotonic() + deadline_s
    with Collector(seed=config.seed, workers=config.workers) as collector:
        # The workers come up under ignored signals, before the parent's
        # handlers go in (see the module docstring on stopping).
        _spawn_ignoring_signals(collector, workers=config.workers)
        log(
            f"deep: from {run.t:,} to {target:,}, K = {config.walks:,} on {config.workers} workers,"
            f" refit on {config.device}"
        )
        # The pace and the save clock start once the workers are up, as the
        # resume state on disk is as fresh as the open.
        saves = _Saves(
            out=out,
            target=target,
            learner=learner,
            commit=commit,
            log=log,
            observers=observers,
            opened=time.monotonic(),
            opened_t=run.t,
            resumed=time.monotonic(),
            resume_t=run.t if (out / RESUME / STATE).exists() else -1,
        )
        with _stop_on_signals() as stop:
            _train(
                run,
                collector,
                learner,
                observers,
                target=target,
                marks=sorted(m for m in config.grid if run.t < m <= target),
                deadline=deadline,
                stop=stop,
                saves=saves,
            )
    return run

def _spawn_ignoring_signals(collector: Collector, *, workers: int) -> None:
    """Start every worker of `collector` while SIGINT and SIGTERM are ignored here.

    A spawned child inherits an ignored disposition and Python installs its
    SIGINT handler only over the default, so the workers ignore both
    signals for the run, as rung 3's do. The pool starts a worker per
    submitted chunk while none is idle, so one phase of 4W one-walk chunks —
    submitted in a tight loop, a second before the first worker is even up —
    starts all W in practice (pinned at W = 3). Its rows are thrown away.
    Nothing to do at W = 0.
    """
    if workers == 0:
        return
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        collector.collect(seat=0, t=1, walks=4 * workers)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

def _open(
    out: Path, config: DeepConfig, *, learner: Learner, observers: Sequence[Observer]
) -> DeepRun:
    """The run in `out`: resumed from `out/resume` if it has a state, else new at iteration 0.

    A resume state naming a stateful observer the run was not given is
    refused by name: continuing without it would drop its state at the next
    save, and the average it held would be gone with nothing in the log.
    """
    resume = out / RESUME
    if not (resume / STATE).exists():
        return new_run(config)
    # The saved config is the run's; the asked one may differ only in what
    # the walks and nets never see. Checked before the memories are read.
    require_same_run(saved_config(resume), config)
    given = {observer.name for observer in _stateful(observers)}
    dropped = sorted(set(_read_state(resume)["files"].get("observers", {})) - given)
    if dropped:
        raise ValueError(
            f"{resume / STATE} carries state for the observer(s) {dropped}, which this "
            "run was not given; resuming without them would drop it"
        )
    run = load_run(resume, learner=learner, observers=observers)
    run.config = config
    return run

def _train(
    run: DeepRun,
    collector: Collector,
    learner: Learner,
    observers: Sequence[Observer],
    *,
    target: int,
    marks: list[int],
    deadline: float | None,
    stop: _StopRequest,
    saves: _Saves,
) -> str:
    """Step until `target`, the deadline or a signal; return why it stopped.

    Every iteration keeps its two nets. A grid mark keeps a full save under
    `marks/` and rewrites the resume state; so does the clock
    (`save_every_s` since the last resume save) and the stop itself. The
    stop is read after each iteration and again before the next.
    """
    while True:
        # A signal or the deadline that landed during the last save stops
        # the run here, before another iteration is paid for.
        why = _why_stop(stop, deadline)
        if why:
            saves.stopped(run, why=why)
            return why
        stats = step(run, collector, learner, observers)
        why = "done" if run.t >= target else _why_stop(stop, deadline)
        kept = bool(marks) and run.t == marks[0]
        if kept:
            marks.pop(0)
        resume = kept or bool(why) or saves.due(run.config.save_every_s)
        saves.after_iteration(run, stats, kept=kept, resume=resume, why=why)
        if why:
            return why

def _why_stop(stop: _StopRequest, deadline: float | None) -> str:
    """The signal's name, "deadline", or "" to carry on."""
    if stop.why:
        return stop.why
    if deadline is not None and time.monotonic() > deadline:
        return "deadline"
    return ""

@dataclass(slots=True)
class _Saves:
    """Where a run's files go and who hears of them: the one writer of `out/` after the open."""

    out: Path
    target: int
    learner: Learner
    commit: Callable[[], None]
    log: Callable[[str], None]
    observers: Sequence[Observer]
    opened: float  # monotonic seconds when the run opened, the pace's baseline
    opened_t: int  # the iteration it opened at
    resumed: float  # monotonic seconds of the last resume save
    resume_t: int  # the iteration the resume state on disk holds; -1 for none

    def due(self, every_s: float) -> bool:
        """Whether the clock asks for a resume save."""
        return time.monotonic() - self.resumed >= every_s

    def after_iteration(
        self, run: DeepRun, stats: StepStats, *, kept: bool, resume: bool, why: str
    ) -> None:
        """Keep the nets, save what is due, append the progress line, commit once."""
        wrote = []
        nets = self.out / NETS
        nets.mkdir(exist_ok=True)
        for seat, net in enumerate(run.nets):
            _atomically(self.learner.save, net, nets / net_name(seat, run.t))
            wrote.append(f"{NETS}/{net_name(seat, run.t)}")
        if kept:
            save_run(
                run, self.out / MARKS / mark_name(run.t), learner=self.learner, observers=self.observers
            )
            wrote.append(f"{MARKS}/{mark_name(run.t)}")
        if resume:
            self._save_resume(run, wrote)
        rate = (run.t - self.opened_t) / max(time.monotonic() - self.opened, 1e-9)
        line = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "t": run.t,
            "target": self.target,
            "seconds": {
                "materialise": [round(s, 3) for s in stats.materialise],
                "collect": [round(s, 3) for s in stats.collect],
                "merge": [round(s, 3) for s in stats.merge],
                "fit": [round(s, 3) for s in stats.fit],
                "iteration": round(stats.seconds, 3),
            },
            "rows": {
                "advantage": [memory.offered for memory in run.advantage],
                "policy": run.policy.offered,
                "resident": {
                    "advantage": [memory.size for memory in run.advantage],
                    "policy": run.policy.size,
                },
            },
            "loss": list(stats.loss),
            "iterations_per_s": round(rate, 4),
            "eta_h": round((self.target - run.t) / rate / 3600, 2),
            "wrote": wrote,
        }
        if why:
            line["stopped"] = why
        self._record(line)

    def stopped(self, run: DeepRun, *, why: str) -> None:
        """A stop noticed before an iteration: the resume state if it is behind, and one line."""
        wrote: list[str] = []
        if self.resume_t != run.t:
            self._save_resume(run, wrote)
        self._record(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "t": run.t,
                "target": self.target,
                "wrote": wrote,
                "stopped": why,
            }
        )

    def _save_resume(self, run: DeepRun, wrote: list[str]) -> None:
        save_run(run, self.out / RESUME, learner=self.learner, observers=self.observers)
        wrote.append(RESUME)
        self.resumed = time.monotonic()
        self.resume_t = run.t

    def _record(self, line: dict) -> None:
        """Append the line to `progress.jsonl`, commit once, and say it."""
        with (self.out / PROGRESS).open("a") as file:
            file.write(json.dumps(line) + "\n")
        self.commit()
        self.log(json.dumps(line))

# ---------------------------------------------------------------------------
# The real learner
# ---------------------------------------------------------------------------

def rule_targets(targets: np.ndarray, t: np.ndarray, *, now: int, rule: str) -> np.ndarray:
    """The memory's regret samples rescaled to what `rule` would weigh them at iteration `now`.

    Deep CFR keeps no running regret: each refit learns the weighted MEAN of
    every sample the memory holds, and regret matching reads only that mean's
    sign pattern and ratios, so any rule that is a weight per iteration is a
    rescaling of each sample. LCFR's weight t is the spec's and rides in the
    loss (`fit.batch_loss`); the rules here rescale the target instead and
    leave the loss unweighted, which `NetLearner.fit` does by passing t = 1.

    DCFR (Brown & Sandholm 2019, α = 1.5, β = 0) multiplies the running sum
    after iteration k by k^1.5/(k^1.5 + 1) where it is positive and by 1/2
    where it is negative, so a sample from iteration s carries, at `now`,

        positive:  ∏ₖ₌ₛ^(now−1) k^1.5 / (k^1.5 + 1)
        negative:  (1/2)^(now − s)

    CFR+ is DCFR(∞, −∞) — the paper's own identity: positive samples keep
    weight 1 forever, negative ones count only on the iteration they were
    drawn. (DCFR(1, 1) gives s/now, LCFR's weight up to a common factor.)

    The approximation, and the reason this is a screen and not the rules:
    both act on the sign of the CUMULATIVE regret, which Deep CFR never
    stores, so here the sign is each sample's own. Padding columns are
    rescaled too and stay ignored by the loss.
    """
    t = np.asarray(t, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float32)
    if rule == "cfr+":
        positive = np.ones(len(t))
        negative = (t == now).astype(np.float64)
    elif rule == "dcfr":
        k = np.arange(1, now + 1, dtype=np.float64)
        # tail[s] = Σ_{k=s}^{now−1} log(k^1.5/(k^1.5 + 1)); tail[now] = 0.
        logs = np.log(k**1.5 / (k**1.5 + 1))
        tail = np.zeros(now + 2)
        tail[1:now] = np.cumsum(logs[: now - 1][::-1])[::-1]
        positive = np.exp(tail[t])
        negative = 0.5 ** (now - t).astype(np.float64)
    else:
        raise ValueError(f"no per-sample form for rule {rule!r}")
    scale = np.where(targets > 0, positive[:, None], negative[:, None])
    return (targets * scale).astype(np.float32)

@cache
def spot_pots() -> np.ndarray:
    """The pot at each of the 141 public decision points, in the table's point order; (141,) float64.

    `ChipState.pot` after replaying the betting line alone (`chip_state`):
    every chip in the middle as the actor decides, the bet they face
    included, so it is a property of the spot and not of the hand. The
    seven pots the game reaches are 2, 4, 6, 12, 18, 35 and 52: 2 at the
    ante, 12 for the opener facing the raise, 35 facing the all-in third
    bet, and 52, the largest, once that bet is called and the hand goes to
    the draw. Built once a process, as `layout` is.
    """
    pots = np.array([chip_state(point.betting).pot for point in layout().points], dtype=np.float64)
    pots.setflags(write=False)
    return pots

def pot_scaled(rows: np.ndarray, targets: np.ndarray, *, power: float) -> np.ndarray:
    """Each regret sample divided by its spot's pot^`power`: where the refit spends its accuracy, not what it learns.

    The problem it solves: one net fits every spot's mean regret with one
    accuracy budget, and the loss decides where that budget goes. A regret
    is in chips, so a spot's regrets are about as large as its pot, and a
    relative error ε in the net's output there costs (ε · pot)² per sample.
    Squared error in raw chips therefore weighs a spot by how often it is
    reached × pot²: a 52-chip spot counts 676 times a 2-chip spot reached
    as often, and the ante spots, where every hand begins, are fitted last.

    Why the divisor is free: it is the pot at the decision, which every
    sample at that spot shares, so the spot's mean regret is scaled by one
    constant across its actions, and regret matching, which reads only the
    sign pattern and the ratios of a row (`RegretMatcher.strategy`, and the
    all-negative one-hot `materialise` writes), plays the same σ from the
    scaled regrets as from the raw ones. That is exact for the regrets
    themselves, a spot's true mean or any sample of it; what the fitted net
    gets wrong, and so the σ it actually plays, is what the weighting moves,
    and is the point. A divisor drawn from the sampled hand's outcome would
    reweight samples within a spot and move its mean; this one cannot.

    What k does: dividing by pot^k leaves a weight of how often × pot^(2 − 2k).
    k = 0 is raw chips, pot². k = 1/2 leaves how often × pot, the chips a
    σ mistake at that spot actually costs the player. k = 1 weighs every
    spot alike, however small its pot. `targets` is returned itself at
    k = 0; otherwise a float32 copy, the padding columns scaled and still
    ignored by the loss. `rows` are whole-game table rows; a row outside the
    table is refused as `Layout.locate_rows` refuses it.
    """
    if power == 0:
        return targets
    point, _ = layout().locate_rows(rows)
    return (targets / (spot_pots()[point] ** power)[:, None]).astype(np.float32)

# The word added to a fit's seed to draw its held batch: "HELD" in ASCII.
_HELD_TAG = 0x48454C44

class NetLearner:
    """The learner the run uses: PR 2's refit and materialise, torch imported when first used.

    `fit` is `fit_from_scratch` on the memory, then the loss of the fresh
    net on one held batch — `config.batch` rows drawn with replacement from
    the memory it was fitted to, by a generator seeded from the fit's seed
    plus a tag, so the draw is the same rule every iteration and never the
    fit's own first minibatch. It is a training loss on rows the net may
    have seen, not a held-out one; it says how well the refit fits, and it
    is comparable across iterations of one run, in the run's own target
    units: the targets are `rule_targets`' under a rule other than linear,
    and `pot_scaled`'s under a pot power other than 0, so a loss is not
    comparable across runs that differ in either. `write` is `materialise`. The thread
    count comes from the config, and the fit's own `set_num_threads` lasts
    the fit only. `fit_config` is the `FitConfig` the run's fields describe,
    which Π's fit at a grid mark shares (`policy.fit_policy`).
    """

    def __init__(self, config: DeepConfig) -> None:
        self._config = config
        self._fit_config = None

    def fit(
        self, rows: np.ndarray, t: np.ndarray, targets: np.ndarray, *, seed: tuple[int, int, int]
    ) -> Fitted:
        import torch

        from drawmaha_solver.deepcfr.fit import _fold_seed, batch_loss, fit_from_scratch

        config = self.fit_config()
        if self._config.rule != "linear":
            targets = rule_targets(targets, t, now=seed[1], rule=self._config.rule)
            t = np.ones(len(t), dtype=np.float32)
        targets = pot_scaled(rows, targets, power=self._config.pot_power)
        net = fit_from_scratch(rows, t, targets, seed=seed, config=config)
        draw = torch.Generator().manual_seed(_fold_seed((*seed, _HELD_TAG)))
        held = torch.randint(len(rows), (config.batch,), generator=draw).numpy()
        with torch.inference_mode():
            loss = float(batch_loss(net, rows[held], t[held], targets[held]).item())
        return Fitted(net, loss)

    def write(self, table: PackedTable, models: Sequence[object | None]) -> None:
        from drawmaha_solver.deepcfr.materialise import materialise

        materialise(table, models)

    def save(self, model: object, path: Path) -> None:
        from drawmaha_solver.deepcfr.net import save_net

        save_net(model, path)

    def load(self, path: Path) -> object:
        from drawmaha_solver.deepcfr.net import load_net

        return load_net(path)

    def fit_config(self):
        """The `FitConfig` of this run's refits, built once torch is first needed."""
        if self._fit_config is None:
            from drawmaha_solver.deepcfr.fit import FitConfig

            c = self._config
            self._fit_config = FitConfig(
                width=c.width,
                depth=c.depth,
                steps=c.steps,
                batch=c.batch,
                lr=c.lr,
                clip=c.clip,
                threads=c.threads,
                device=c.device,
            )
        return self._fit_config

# ---------------------------------------------------------------------------
# The current iterate
# ---------------------------------------------------------------------------

def current_iterate(table: PackedTable, *, iteration: int = 0, seed: int = 0) -> FrozenStrategy:
    """Regret matching over the materialised table, frozen for the exact grader.

    Deep CFR's eq. (3) exactly, because `materialise` already wrote the
    all-negative case as a one-hot; an all-zero row plays uniformly, as
    `RegretMatcher.strategy` reads it. `iteration` and `seed` only label the
    file.
    """
    widths = table.widths()
    info = StrategyInfo(rule="deep", column="current", iteration=iteration, seed=seed, workers=0)
    return FrozenStrategy(
        average_rows(np.maximum(table.cumulative_regret, 0.0), widths), widths, info
    )

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
    iterations, not only the first. The two models are summed one after
    the other, which is only LCFR's order because a row belongs to one
    acting seat and so appears in one model. A row no sample names stays
    exactly 0.0. NumPy only, so gate 1 runs without torch.
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
