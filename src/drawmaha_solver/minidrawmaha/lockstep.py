"""Lockstep MCCFR: W sampled hands per seat per iteration, all against one frozen σₜ.

`mccfr.run_iteration` walks one hand per seat and writes as it goes. A lockstep
iteration walks W hands per seat — one per worker — and writes nothing while it
walks. Each walk records what it WOULD have written; once all W are in, every
touched row is banked once with the sum of its regrets, and the strategy mass
is added. Then the other seat does the same against the updated regrets
(alternating updates, as before). The W walks of one seat are a PHASE.

That is MCCFR with W samples per iteration, and it is what lets W processes
share one table: during a phase the table is read-only, so the walks cannot
race, and the one write per row happens in a fixed order, so the float sums
are the same however the walks were scheduled. `parallel.py` runs the W walks
in W processes; this module runs them one after another and is the reference
the processes must equal bit for bit.

Two consequences worth knowing:

- Every rule is exact for W-sample lockstep iterations, including CFR+: a row
  is floored once per iteration, on the total of its W sampled regrets, which
  is the paper's rule with that total as the iteration's regret. It is not W
  separately floored one-hand banks. (`mccfr`'s one-hand walk
  floors after each bank, which differs on the rare row one walk reaches twice.)
- A walk that reaches the same row twice (suit relabelling) reads σₜ both
  times, instead of seeing its own first bank the second time. Every sample of
  iteration t sees the same strategy, which is the textbook semantics.

A walk's record has two forms. `Recorded` is what the walk itself produces:
per visit the infoset key, its packed SLOT (row, first column, width — None on
a table that is not packed) and the values it would have written. `Flat` is
the same record as eight arrays, which is what a worker process can hand its
parent through shared memory; `apply_arrays` banks a phase from those arrays
alone, vectorised, and `apply` is the per-ledger reference it must equal.

The randomness is counter-based: worker w's walk for seat s at iteration t
draws from `default_rng([seed, t, s, w])`. A run therefore carries no RNG
state, and resuming from any checkpoint equals never having stopped. A
checkpoint holds `mccfr.save_solve`'s arrays plus `workers`, and it is the
`workers` key that marks it as lockstep.
"""

from __future__ import annotations

import os
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drawmaha_solver.minidrawmaha.game import random_deal
from drawmaha_solver.minidrawmaha.mccfr import (
    RootSampler,
    Table,
    _flat_slots,
    _pour_slots,
    _require_extra_rows,
    _require_same_run,
    _saved_run,
    _validate_columns,
    _validate_shape,
    _widths,
    traverse,
)
from drawmaha_solver.minidrawmaha.packed_table import PackedTable, Slot
from drawmaha_solver.minidrawmaha.regret_rules import (
    RULES,
    Average,
    RegretRule,
    bank_rows,
    extra_weights,
    validate_averages,
)
from drawmaha_solver.regret_matching import RegretMatcher

SEATS = (0, 1)

# Present in every lockstep checkpoint and in no serial one: W hands per
# iteration is part of what a lockstep run is, and a serial run has no W.
_WORKERS_KEY = "workers"

# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LockstepSolve:
    """A lockstep run: the table, the rule, the columns, W, and the last completed t.

    No random state: the streams are a function of (seed, t, seat, worker).
    Mutable on purpose, as `mccfr.Solve` is: training advances `iteration` in
    place, and `parallel.py` swaps `table` for one over shared memory.
    """

    table: Table
    seed: int
    workers: int
    deal: RootSampler = random_deal
    iteration: int = 0
    rule: RegretRule = RegretRule.VANILLA
    averages: tuple[Average, ...] = ()

def new_lockstep(
    seed: int,
    *,
    workers: int,
    table: Table | None = None,
    deal: RootSampler = random_deal,
    rule: RegretRule | str = RegretRule.VANILLA,
    averages: Sequence[Average | str] = (),
) -> LockstepSolve:
    """A fresh lockstep run at iteration 0. `table=None` is the whole packed game.

    A table the caller hands in must already carry one `extra_sums` row per
    averaging column; it is trained in place.
    """
    _validate_run(seed, workers)
    rule = RegretRule(rule)
    averages = validate_averages(averages)
    if table is None:
        table = PackedTable.whole_game(extra_averages=len(averages))
    else:
        _require_extra_rows(table, averages)
    return LockstepSolve(
        table=table, seed=seed, workers=workers, deal=deal, rule=rule, averages=averages
    )

def train_lockstep(solve: LockstepSolve, iterations: int) -> LockstepSolve:
    """Run `iterations` more lockstep iterations in this process, in place, and return the run."""
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    for _ in range(iterations):
        t = solve.iteration + 1
        for seat in SEATS:
            recorded = [
                walk(solve.table, solve.deal, solve.seed, t, seat, worker, solve.averages)
                for worker in range(solve.workers)
            ]
            apply(solve.table, recorded, solve.rule, t)
        solve.iteration = t
    return solve

# ---------------------------------------------------------------------------
# One walk, recorded instead of written
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Recorded:
    """What one walk would have written, in the order it would have written it.

    `regret`: (key, slot, r) per traverser visit. `strategy`: (key, slot, t·σ,
    column-weighted σ) per opponent visit. `slot` is None on a table that is
    not packed.
    """

    regret: list[tuple[Hashable, Slot | None, np.ndarray]]
    strategy: list[tuple[Hashable, Slot | None, np.ndarray, np.ndarray]]

def walk(
    table: Table,
    deal: RootSampler,
    seed: int,
    t: int,
    seat: int,
    worker: int,
    averages: tuple[Average, ...],
) -> Recorded:
    """Worker `worker`'s walk for `seat` at iteration t, reading σₜ and writing nothing."""
    rng = stream(seed, t, seat, worker)
    recorder = _Recorder(table, len(averages))
    traverse(
        deal(rng),
        recorder,
        seat,
        rng,
        t,
        bank=recorder.bank,
        column_weights=extra_weights(averages, t),
    )
    return recorder.recorded()

def stream(seed: int, t: int, seat: int, worker: int) -> np.random.Generator:
    """The random stream of worker `worker`'s walk for `seat` at iteration `t`."""
    return np.random.default_rng([seed, t, seat, worker])

class _Recorder(Mapping[Hashable, RegretMatcher]):
    """What `traverse` is handed as its table during a lockstep walk.

    `table[key]` returns a ledger that reads the real regrets (so σ is σₜ) but
    whose strategy slots are fresh zeros, so the walk's `+=` lands in scratch;
    `bank` records the regret instead of banking it.
    """

    def __init__(self, table: Table, extra_rows: int) -> None:
        self._table = table
        self._packed = isinstance(table, PackedTable)
        self._extra_rows = extra_rows
        self._visits: list[tuple[Hashable, Slot | None, RegretMatcher]] = []
        self._banked: dict[int, np.ndarray] = {}

    def __getitem__(self, key: Hashable) -> RegretMatcher:
        if self._packed:
            slot = self._table.slot_of(key)
            _, start, width = slot
            regret = self._table.cumulative_regret[start : start + width]
        else:
            slot = None
            regret = self._table[key].cumulative_regret
            width = len(regret)
        scratch = RegretMatcher.over(
            cumulative_regret=regret,
            strategy_sum=np.zeros(width),
            extra_sums=np.zeros((self._extra_rows, width)),
            stamp=np.zeros(1, dtype=np.int64),
        )
        self._visits.append((key, slot, scratch))
        return scratch

    def __contains__(self, key: object) -> bool:
        # Mapping's default would call __getitem__ and record a phantom visit.
        return key in self._table

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)

    def bank(self, ledger: RegretMatcher, regret: np.ndarray, t: int) -> None:
        # Keyed by identity: the scratch ledger is this visit's own object,
        # and a row reached twice gets two of them.
        self._banked[id(ledger)] = regret

    def recorded(self) -> Recorded:
        """The visits so far: banked ones are traverser visits, the rest opponent ones."""
        regret, strategy = [], []
        for key, slot, ledger in self._visits:
            banked = self._banked.get(id(ledger))
            if banked is not None:
                regret.append((key, slot, banked))
            else:
                strategy.append((key, slot, ledger.strategy_sum, ledger.extra_sums))
        return Recorded(regret=regret, strategy=strategy)

# ---------------------------------------------------------------------------
# Applying a phase
# ---------------------------------------------------------------------------

def apply(table: Table, recorded: Sequence[Recorded], rule: RegretRule, t: int) -> None:
    """Bank one phase's walks into `table`: each touched row once, walks in worker order.

    A packed table goes through `apply_arrays`; any other is banked ledger by
    ledger, which is the reference the vectorised path is tested against.
    """
    if isinstance(table, PackedTable):
        k = table.extra_sums.shape[0]
        apply_arrays(table, [flatten(walked, k) for walked in recorded], rule, t)
        return
    totals: dict[Hashable, np.ndarray] = {}
    for walked in recorded:
        for key, _, regret in walked.regret:
            # From 0.0, as `apply_arrays` sums from np.zeros, so a single
            # sample comes out of both paths as the same bits (0.0 + −0.0 is 0.0).
            totals[key] = totals.get(key, 0.0) + regret
    bank = RULES[rule]
    for key, total in totals.items():
        bank(table[key], total, t)
    for walked in recorded:
        for key, _, primary, extra in walked.strategy:
            ledger = table[key]
            ledger.strategy_sum += primary
            ledger.extra_sums += extra

@dataclass(frozen=True, slots=True)
class Flat:
    """One walk's record as arrays — what a worker process hands the parent.

    Regret: per traverser visit its row, first column and width, and the
    values end to end. Strategy: per opponent visit its first column and
    width, the primary values end to end, and the column-weighted values as
    (k, n).
    """

    regret_rows: np.ndarray
    regret_starts: np.ndarray
    regret_widths: np.ndarray
    regret_values: np.ndarray
    strategy_starts: np.ndarray
    strategy_widths: np.ndarray
    strategy_values: np.ndarray
    extra_values: np.ndarray

def flatten(walked: Recorded, k: int) -> Flat:
    """A packed-table walk's record as arrays; `k` is the run's number of extra columns.

    Only a walk over a packed table has slots to flatten; the caller has
    checked that.
    """

    def ints(values: list[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    extras = [extra for _, _, _, extra in walked.strategy]
    return Flat(
        regret_rows=ints([slot[0] for _, slot, _ in walked.regret]),
        regret_starts=ints([slot[1] for _, slot, _ in walked.regret]),
        regret_widths=ints([slot[2] for _, slot, _ in walked.regret]),
        regret_values=np.concatenate([r for _, _, r in walked.regret] or [np.zeros(0)]),
        strategy_starts=ints([slot[1] for _, slot, _, _ in walked.strategy]),
        strategy_widths=ints([slot[2] for _, slot, _, _ in walked.strategy]),
        strategy_values=np.concatenate(
            [p for _, _, p, _ in walked.strategy] or [np.zeros(0)]
        ),
        extra_values=np.concatenate(extras, axis=1) if extras else np.zeros((k, 0)),
    )

def apply_arrays(table: PackedTable, flats: Sequence[Flat], rule: RegretRule, t: int) -> None:
    """`apply` on a packed table, from the walks' arrays, vectorised.

    Regret: every column's values are summed in walk order (`np.add.at` adds
    in the order given), then `bank_rows` banks each touched row once.
    Strategy: added in walk order, column by column, exactly as the one-hand
    walk's `+=` would have.
    """
    columns = np.concatenate([_columns(f.regret_starts, f.regret_widths) for f in flats])
    if columns.size:
        rows = np.concatenate([np.repeat(f.regret_rows, f.regret_widths) for f in flats])
        values = np.concatenate([f.regret_values for f in flats])
        touched, first, inverse = np.unique(columns, return_index=True, return_inverse=True)
        totals = np.zeros(touched.size)
        np.add.at(totals, inverse, values)
        bank_rows(
            rule,
            regret=table.cumulative_regret,
            stamp=table.stamp,
            columns=touched,
            rows=rows[first],
            totals=totals,
            t=t,
        )
    columns = np.concatenate([_columns(f.strategy_starts, f.strategy_widths) for f in flats])
    if columns.size:
        np.add.at(table.strategy_sum, columns, np.concatenate([f.strategy_values for f in flats]))
        extra = np.concatenate([f.extra_values for f in flats], axis=1)
        for row in range(table.extra_sums.shape[0]):
            np.add.at(table.extra_sums[row], columns, extra[row])

def _columns(starts: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """start, start+1, …, start+width−1 for every (start, width), end to end."""
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.arange(int(widths.sum())) - np.repeat(np.cumsum(widths) - widths, widths)
    return np.repeat(starts, widths) + offsets

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_lockstep(solve: LockstepSolve, path: Path) -> None:
    """Write a lockstep run to one `.npz`, atomically: a temp file renamed into place.

    The arrays are `mccfr.save_solve`'s, so the grader reads either; there is
    no RNG state (the streams are counter-based) and `workers` is recorded,
    because W hands per iteration is part of what the run is. Raises on an
    empty table, which has nothing to restore.
    """
    if not solve.table:
        raise ValueError("the table is empty; there is nothing to checkpoint")
    path = Path(path)
    regret, strategy_sum, extra_sums, stamps = _flat_slots(solve.table)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez(
            file,
            regret=regret,
            strategy_sum=strategy_sum,
            extra_sums=extra_sums,
            stamps=stamps,
            widths=_widths(solve.table),
            iteration=np.int64(solve.iteration),
            seed=np.int64(solve.seed),
            workers=np.int64(solve.workers),
            rule=np.str_(RegretRule(solve.rule).value),
            averages=np.array([Average(a).value for a in solve.averages], dtype=np.str_),
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)

def load_lockstep(
    path: Path,
    *,
    table: Table,
    workers: int,
    rule: RegretRule | str,
    averages: Sequence[Average | str],
    deal: RootSampler = random_deal,
) -> LockstepSolve:
    """Pour a lockstep checkpoint into `table` and return the run, ready to continue.

    Refuses a serial checkpoint, another worker count, rule or column set, and
    a table the widths do not fit — all before writing anything.
    """
    rule = RegretRule(rule)
    averages = validate_averages(averages)
    with np.load(Path(path), allow_pickle=False) as saved:
        _require_lockstep(saved, path)
        saved_workers = int(saved[_WORKERS_KEY])
        if saved_workers != workers:
            raise ValueError(
                f"the checkpoint ran {saved_workers} workers, and this run asks for {workers}"
            )
        _require_same_run(_saved_run(saved), (rule, averages))
        widths = saved["widths"]
        _validate_shape(widths, table)
        _require_extra_rows(table, averages)
        extra_sums, stamps = saved["extra_sums"], saved["stamps"]
        _validate_columns(extra_sums, stamps, widths, len(averages))
        _pour_slots(table, widths, saved["regret"], saved["strategy_sum"], extra_sums, stamps)
        return LockstepSolve(
            table=table,
            seed=int(saved["seed"]),
            workers=workers,
            deal=deal,
            iteration=int(saved["iteration"]),
            rule=rule,
            averages=averages,
        )

def is_lockstep(path: Path) -> bool:
    """Whether a checkpoint was written by a lockstep run (it records `workers`)."""
    with np.load(Path(path), allow_pickle=False) as saved:
        return _WORKERS_KEY in saved.files

def read_lockstep(path: Path) -> dict[str, object]:
    """A lockstep checkpoint's run description, read without loading its arrays.

    Raises on a serial checkpoint, which has no worker count to describe.
    """
    with np.load(Path(path), allow_pickle=False) as saved:
        _require_lockstep(saved, path)
        return {
            "iteration": int(saved["iteration"]),
            "seed": int(saved["seed"]),
            "workers": int(saved[_WORKERS_KEY]),
            "rule": str(saved["rule"]),
            "averages": [str(a) for a in saved["averages"]],
        }

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_run(seed: object, workers: object) -> None:
    """Refuse a seed or worker count the run could not use, before anything is built."""
    # The streams are default_rng([seed, t, seat, worker]), which refuses a
    # negative entry — but only at the first walk, after a table is built.
    # A bool is an int to Python and never what a caller meant.
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    # W is the number of hands per seat per iteration; zero would train nothing.
    if (
        not isinstance(workers, (int, np.integer))
        or isinstance(workers, bool)
        or workers < 1
    ):
        raise ValueError(f"workers must be a positive int, got {workers!r}")

def _require_lockstep(saved: np.lib.npyio.NpzFile, path: Path) -> None:
    """Refuse a serial checkpoint where a lockstep one is expected.

    Only a lockstep run records `workers`; without it there is no W to resume
    with or to describe, so loading or reading on would have to invent one.
    """
    if _WORKERS_KEY not in saved.files:
        raise ValueError(f"{path} is not a lockstep checkpoint")
