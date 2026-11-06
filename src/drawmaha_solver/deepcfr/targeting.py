"""Targeting: which table rows the boost stage computes exactly, and how they are drawn.

The boost stage (the targeted-sampling plan's PR 4) adds, at a chosen infoset,
one row holding the EXACT regret under the current strategy, at the weight the
walk's own noisy samples would have carried there in expectation. It can
afford that at B infosets per seat per iteration. This module decides which B:
it turns "where does the strategy leak, and how noisy are the walk's samples
there" into a frozen list, refreshed every R iterations, and draws from it.

**Four ways to aim the same budget.** Full Drawmaha cannot name exact infosets;
it can name betting lines and hand classes. So the experiment races four arms
that differ only in the GROUP they rank:

- `infoset` — a group is one table row. The B best rows, fixed for R
  iterations. The most concentration possible; it cannot carry to the full game.
- `line` — a group is a public point, a betting line with the draw counts
  (141 in the whole game, both seats).
- `line_class` — a public point × the `InnerCategory` of the three private cards
  × the `OuterCategory` they make with both board cards (3,870 non-empty
  groups). Where only one board card is out (round 1 and the draw) there is no
  outer hand, and the class is the inner category alone.
- `random` — one group per seat: B rows drawn from the whole seat by how often
  the walk arrives there. Same exact rows, no aim; without it a win could just
  mean "exact rows help anywhere".

"Off" is not a fifth granularity: a run without targeting has no `TargetSpec`.

**The score of a group** is leak × noise per reached infoset,

    score(g) = leak(g) · min(noise(g), cap) / reached(g)

where leak(g) is the summed one-step best-response gain over g's rows (chips
per hand; the grader computes it, never this module), reached(g) counts g's
rows with positive play frequency, and noise(g) is how badly the walk's
samples at g estimate their own mean (below). Dividing by reached(g) makes it
a density — the leak a budget of one infoset buys inside g — so a line that
leaks a lot only because it holds a hundred thousand infosets does not outrank
a small one that leaks more per row. Multiplying by noise sends the budget
where an exact row removes the most error: a leak the samples already see
clearly will be learned without help.

**The noise of a group** is the relative standard error of what one walk banks
there. A walk visits g with probability p (visits per walk) and banks a regret
sample r each time; its per-walk total has mean p·μ and, for small p, variance
p·E[r²] = p·(σ² + μ²). Their ratio,

    noise(g) = √(E[r²] / p) / |E[r]|,

grows as the samples scatter (E[r²] ≫ μ²), as the mean they estimate shrinks,
and as visits get rarer (1/√p). It is read at a* = the column of largest |mean|,
the action whose regret decides the strategy. Everything comes from the seat's
advantage memory: E[r] and E[r²] are means over g's resident rows, and p is
g's share of the memory times the rows offered per walk,
p = (n_g / size) · offered / (K · t_last), K the walks per iteration and
t_last the latest iteration in the memory. Only the walk's OWN rows count —
weight 1, or α where the boost turned them down — because an exact row is a
mean, not a sample, and would read as no scatter at all. A group with fewer
than two such rows, or a mean of exactly zero, gets the cap, so a rare
infoset the walk has hardly seen ranks as noisy rather than silent.

**The pool and the draws.** `infoset` takes the B best rows outright. `line`
and `line_class` take groups best first until their reached rows number at
least `pool × B` (20 × 2,000 = 40,000 by default; the group that crosses the
line is in), so a refresh spreads the budget over the best groups without
collapsing onto the top few infosets; `random` pools the whole seat. Each
iteration then draws B rows of the pool WITHOUT replacement, with probability
∝ each row's weight — the walk's arrival rate, `BoostTargets.arrivals`. A
repeated row would get two exact rows and double its expected weight, which
is the one bias the boost exists to avoid. The draw uses exponential keys
(Efraimidis and Spirakis, 2006): each row gets E / w with E ~ Exp(1), and the B
smallest win, which is the same law as drawing one row at a time ∝ w and
removing it. Its stream is a function of (seed, t, seat, stream) alone, with
a fifth word no walk, reservoir or boost stream carries, so draws never
borrow randomness from the walk and a resumed run draws the same rows.

**What this module never sees: the grader.** Leak and frequency arrive as
arrays and the draw weight as a callable. The script
`scripts/deepcfr/targeting_refresh.py` is where the exact meter is run and its
output handed in; a test walks every import block under `deepcfr/` to keep
it so, because a trainer that shared code with its grader could be certified
by its own bug.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from drawmaha_solver.deepcfr.reservoir import TARGET_WIDTH, Reservoir
from drawmaha_solver.minidrawmaha.rules.enumeration import private_keys, public_decision_points
from drawmaha_solver.minidrawmaha.rules.hands import inner_score, outer_score

# The fifth word of every draw's stream: "TARG" in ASCII. A walk's stream has
# four words, a reservoir's ends in "RESV", a sampled boost row's in "BOOST".
_STREAM_TAG = int.from_bytes(b"TARG", "big")

Granularity = Literal["random", "infoset", "line", "line_class"]
GRANULARITIES: tuple[str, ...] = get_args(Granularity)
SEATS = (0, 1)

# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class TargetSpec:
    """How one arm aims: the granularity, the budget, the refresh, and the draw's stream.

    `budget` (B) and `refresh` (R) default to the plan's proposals, 2,000
    rows per seat per iteration (about 1.5% of an iteration's time) and 25
    iterations between refreshes. `alpha` is the weight the boost leaves on
    the walk's own rows at a targeted infoset; it has no default because PR 4
    settles it, and this module reads it only to tell the walk's rows from
    exact ones in the noise score.
    """

    granularity: Granularity
    alpha: float
    budget: int = 2_000
    refresh: int = 25
    noise_cap: float = 20.0
    pool: int = 20
    stream: int = 0x7A52

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {GRANULARITIES}, got {self.granularity!r} "
                "(a run without targeting has no TargetSpec)"
            )
        for name in ("budget", "refresh", "pool"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if not _is_int(self.stream) or self.stream < 0:
            raise ValueError(f"stream must be a non-negative int, got {self.stream!r}")
        if not (isinstance(self.alpha, (int, float)) and 0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be a number in [0, 1], got {self.alpha!r}")
        if not (isinstance(self.noise_cap, (int, float)) and 0.0 < self.noise_cap < math.inf):
            raise ValueError(f"noise_cap must be positive and finite, got {self.noise_cap!r}")

def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

# ---------------------------------------------------------------------------
# The groups
# ---------------------------------------------------------------------------

@cache
def _layout() -> tuple[np.ndarray, np.ndarray]:
    """Where each public point's rows start (142 entries, the end last), and each point's acting seat.

    The whole-game table lays its rows out point by point in walk order, each
    point's keys in its shape's sorted `private_keys` order — the packed
    table's own arithmetic, restated from the enumerator.
    """
    points = public_decision_points()
    sizes = [len(private_keys(board_cards=p.board_cards, discards=p.discards)) for p in points]
    offset = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    player = np.array([p.player for p in points], dtype=np.int64)
    return offset, player

def n_rows() -> int:
    """How many rows the whole-game table has: 6,220,050."""
    return int(_layout()[0][-1])

@cache
def _classes(board_cards: int, discards: int) -> tuple[np.ndarray, np.ndarray]:
    """Inner and outer category of every key of one shape, in key order; outer 0 with one board card out."""
    keys = private_keys(board_cards=board_cards, discards=discards)
    inner = np.array([int(inner_score(hole).category) for hole, _, _ in keys], dtype=np.int64)
    if board_cards < 2:
        return inner, np.zeros_like(inner)
    outer = np.array([int(outer_score(hole, board).category) for hole, _, board in keys], dtype=np.int64)
    return inner, outer

# Three bits each: InnerCategory runs 1–6 and OuterCategory 1–7, with 0 for "no outer hand".
_CLASS_BITS = 3

def line_class_id(point: int, inner: int, outer: int) -> int:
    """The group id of (point, inner, outer): point × 64 + inner × 8 + outer."""
    return (point << (2 * _CLASS_BITS)) | (inner << _CLASS_BITS) | outer

def decode_line_class(group: int) -> tuple[int, int, int]:
    """(point, inner, outer) of a `line_class` group id."""
    mask = (1 << _CLASS_BITS) - 1
    return group >> (2 * _CLASS_BITS), (group >> _CLASS_BITS) & mask, group & mask

@cache
def groups(granularity: Granularity, seat: int) -> np.ndarray:
    """The group id of every table row of `seat` at `granularity`, −1 on the other seat's rows.

    Read-only, int64, one entry per row of the whole game. `infoset` ids are
    the rows themselves, `line` ids the public point, `line_class` ids
    `line_class_id(point, inner, outer)`, `random` 0.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
    if seat not in SEATS:
        raise ValueError(f"seat must be one of {SEATS}, got {seat!r}")
    offset, player = _layout()
    points = public_decision_points()
    out = np.full(n_rows(), -1, dtype=np.int64)
    for p in np.flatnonzero(player == seat).tolist():
        span = slice(offset[p], offset[p + 1])
        if granularity == "infoset":
            out[span] = np.arange(offset[p], offset[p + 1])
        elif granularity == "line":
            out[span] = p
        elif granularity == "random":
            out[span] = 0
        else:
            inner, outer = _classes(points[p].board_cards, points[p].discards)
            out[span] = line_class_id(p, 0, 0) | (inner << _CLASS_BITS) | outer
    out.setflags(write=False)
    return out

# ---------------------------------------------------------------------------
# The noise score
# ---------------------------------------------------------------------------

def noise(
    memory: Reservoir, group_of_row: np.ndarray, *, walks: int, alpha: float, cap: float
) -> np.ndarray:
    """min(√(E[r²] / p) / |E[r]|, cap) per group id, from one seat's advantage memory; the cap where unread.

    Indexed by group id, one entry per id up to the largest in
    `group_of_row` (a `groups(...)` array). Only the walk's own rows are read,
    weight 1 or `alpha`. E[r] and E[r²] are at the column of largest |E[r]|;
    p = (n_g / size) · offered / (walks · t_last) is visits per walk, with
    `walks` the walks per iteration (K) and t_last the memory's latest t. A
    group with fewer than two of the walk's rows, or a mean of exactly zero,
    reads as `cap`, as does every group of an empty memory.
    """
    if not _is_int(walks) or walks < 1:
        raise ValueError(f"walks must be a positive int (walks per iteration), got {walks!r}")
    n_groups = int(group_of_row.max()) + 1
    out = np.full(n_groups, float(cap))
    rows, t, targets, weight = memory.contents()
    if rows.size == 0:
        return out
    group = group_of_row[rows]
    if (group < 0).any():
        raise ValueError(
            f"the memory holds row {int(rows[np.argmax(group < 0)])}, which is not this seat's: "
            "a seat's noise reads that seat's advantage memory"
        )
    # Indices, not a boolean copy of the targets: a 50M-row memory's targets
    # are 800 MB, and every column is gathered one at a time below.
    natural = np.flatnonzero((weight == np.float32(1.0)) | (weight == np.float32(alpha)))
    group = group[natural]
    count = np.bincount(group, minlength=n_groups).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.stack(
            [
                np.bincount(group, weights=targets[natural, a], minlength=n_groups)
                for a in range(TARGET_WIDTH)
            ],
            axis=1,
        ) / count[:, None]
        best = np.argmax(np.abs(np.nan_to_num(mean)), axis=1)
        at_best = targets[natural, best[group]].astype(np.float64)
        square = np.bincount(group, weights=at_best * at_best, minlength=n_groups)
        mu = mean[np.arange(n_groups), best]
        visits = count / memory.size * memory.offered / (walks * int(t.max()))
        ratio = np.sqrt(square / count / visits) / np.abs(mu)
    readable = (count >= 2) & (mu != 0.0)
    out[readable] = np.minimum(ratio[readable], cap)
    return out

# ---------------------------------------------------------------------------
# Scores and selection
# ---------------------------------------------------------------------------

def choose(
    spec: TargetSpec,
    *,
    seat: int,
    leak: np.ndarray | None,
    frequency: np.ndarray | None,
    noise_of_group: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """One seat's pool of rows and the groups it came from, best first.

    `leak` and `frequency` are per table row (the seat's one-step gains and
    play frequencies, from whatever meter the refresher runs) and
    `noise_of_group` is `noise(...)` at the same granularity. A group is
    eligible when one of its rows is reached (frequency > 0), and ranked by
    leak · noise / reached, ties to the lower id. `infoset` returns the B best
    rows as the pool; `line` and `line_class` take groups until their reached
    rows number `pool × B`, the crossing group included, and pool those
    reached rows; `random` pools every row of the seat and reads nothing.
    """
    group = groups(spec.granularity, seat)
    mine = np.flatnonzero(group >= 0)
    if spec.granularity == "random":
        return mine, np.zeros(1, dtype=np.int64)
    leak = _per_row("leak", leak)
    frequency = _per_row("frequency", frequency)
    owned = group[mine]
    n_groups = int(owned.max()) + 1
    if noise_of_group is None or np.shape(noise_of_group) != (n_groups,):
        raise ValueError(
            f"noise_of_group must hold one value per {spec.granularity} group id of seat {seat} "
            f"({n_groups:,}), got shape {np.shape(noise_of_group)}"
        )
    if not (np.isfinite(noise_of_group).all() and (noise_of_group >= 0).all()):
        raise ValueError("noise_of_group must be finite and non-negative")
    reached_row = frequency[mine] > 0.0
    reached = np.bincount(owned, weights=reached_row, minlength=n_groups)
    total = np.bincount(owned, weights=leak[mine], minlength=n_groups)
    eligible = np.flatnonzero(reached > 0)
    score = total[eligible] * noise_of_group[eligible] / reached[eligible]
    order = eligible[np.lexsort((eligible, -score))]
    if spec.granularity == "infoset":
        chosen = order[: spec.budget]
        return np.sort(chosen), chosen
    filled = np.cumsum(reached[order])
    chosen = order[: int(np.searchsorted(filled, spec.pool * spec.budget)) + 1]
    picked = np.zeros(n_groups, dtype=bool)
    picked[chosen] = True
    return mine[picked[owned] & reached_row], chosen

def _per_row(name: str, values: np.ndarray | None) -> np.ndarray:
    """A whole-game per-row array, finite and non-negative, as float64; refused by name otherwise."""
    if values is None:
        raise ValueError(f"{name} is required to aim: one value per table row")
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n_rows(),):
        raise ValueError(f"{name} must hold one value per table row ({n_rows():,}), got shape {values.shape}")
    if not (np.isfinite(values).all() and (values >= 0.0).all()):
        raise ValueError(f"{name} must be finite and non-negative at every row")
    return values

# ---------------------------------------------------------------------------
# The frozen set and its draws
# ---------------------------------------------------------------------------

def target_stream(*, seed: int, t: int, seat: int, stream: int) -> np.random.Generator:
    """The random stream of seat `seat`'s draw at iteration t: five words, the last "TARG"."""
    return np.random.default_rng([int(seed), int(t), int(seat), int(stream), _STREAM_TAG])

# What a saved set holds; `built_at` beside `rows0` marks a file as a target set.
_SPEC_KEYS = ("granularity", "alpha", "budget", "refresh", "noise_cap", "pool", "stream")
_FILE_KEYS = ("built_at", *_SPEC_KEYS, "rows0", "rows1", "weight0", "weight1", "groups0", "groups1")

# eq=False: the fields are arrays.
@dataclass(frozen=True, eq=False)
class TargetSet:
    """One refresh's choice, frozen for the `spec.refresh` iterations from `built_at` on.

    Per seat: `rows`, the pool (sorted table rows of that seat; for `infoset`,
    the B rows themselves); `weight`, each pool row's draw weight — the walk's
    arrival rate, so a row the walk cannot reach is never drawn; `groups`, the
    group ids the pool came from, best first, kept for the record. Every array
    is copied and made read-only.

        rows = targets.rows_for(t, seat, seed)   # ≤ B distinct rows, sorted

    `save` and `load` round-trip it through one `.npz`, so a resumed run
    draws from the set the run drew from.
    """

    spec: TargetSpec
    built_at: int
    rows: tuple[np.ndarray, np.ndarray]
    weight: tuple[np.ndarray, np.ndarray]
    groups: tuple[np.ndarray, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TargetSpec):
            raise ValueError(f"spec must be a TargetSpec, got {type(self.spec).__name__}")
        if not _is_int(self.built_at) or self.built_at < 1:
            raise ValueError(f"built_at must be the first iteration served, a positive int, got {self.built_at!r}")
        offset, player = _layout()
        rows, weight, chosen = [], [], []
        for seat in SEATS:
            r = np.array(self.rows[seat], dtype=np.int64)
            w = np.array(self.weight[seat], dtype=np.float64)
            g = np.array(self.groups[seat], dtype=np.int64)
            if r.ndim != 1 or (r.size and np.any(np.diff(r) <= 0)):
                raise ValueError(f"seat {seat}'s rows must be sorted, distinct and 1-D")
            if r.size and (r[0] < 0 or r[-1] >= offset[-1]):
                raise ValueError(f"seat {seat} has a row outside the table's {int(offset[-1]):,} rows")
            if r.size and np.any(player[np.searchsorted(offset, r, side="right") - 1] != seat):
                raise ValueError(f"seat {seat}'s pool holds a row of the other seat")
            if w.shape != r.shape or not (np.isfinite(w).all() and (w >= 0.0).all()):
                raise ValueError(f"seat {seat}'s weight must be finite, non-negative, one per row")
            if self.spec.granularity == "infoset" and r.size > self.spec.budget:
                raise ValueError(
                    f"an infoset set holds at most the budget, {self.spec.budget}, rows a seat; got {r.size}"
                )
            for array in (r, w, g):
                array.setflags(write=False)
            rows.append(r)
            weight.append(w)
            chosen.append(g)
        object.__setattr__(self, "built_at", int(self.built_at))
        object.__setattr__(self, "rows", tuple(rows))
        object.__setattr__(self, "weight", tuple(weight))
        object.__setattr__(self, "groups", tuple(chosen))

    def rows_for(self, t: int, seat: int, seed: int) -> np.ndarray:
        """The rows seat `seat` boosts at iteration t: B of the pool without replacement ∝ weight, sorted.

        Fewer when the pool has fewer than B rows of positive weight (then
        all of them). `infoset` serves its fixed rows. Deterministic in
        (seed, t, seat, `spec.stream`). Refuses an iteration outside
        [built_at, built_at + refresh): a stale set is a missed refresh.
        """
        if seat not in SEATS:
            raise ValueError(f"seat must be one of {SEATS}, got {seat!r}")
        if not (self.built_at <= t < self.built_at + self.spec.refresh):
            raise ValueError(
                f"this set serves iterations {self.built_at} to {self.built_at + self.spec.refresh - 1}, "
                f"not iteration {t}: refresh it"
            )
        rows, weight = self.rows[seat], self.weight[seat]
        if self.spec.granularity == "infoset":
            return rows.copy()
        positive = weight > 0.0
        if np.count_nonzero(positive) <= self.spec.budget:
            return rows[positive]
        # Exponential keys: E / w with E ~ Exp(1), the B smallest. A weight of 0
        # gives an infinite key, never among them. One key per pool row, so a
        # row's key does not depend on which other rows have weight.
        stream = target_stream(seed=seed, t=t, seat=seat, stream=self.spec.stream)
        with np.errstate(divide="ignore"):
            keys = stream.standard_exponential(rows.size) / weight
        return np.sort(rows[np.argpartition(keys, self.spec.budget - 1)[: self.spec.budget]])

    # -- the file ------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the set to one `.npz`, atomically: a temp file renamed into place."""
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        arrays = {key: np.asarray(value) for key, value in asdict(self.spec).items()}
        for seat in SEATS:
            arrays[f"rows{seat}"] = self.rows[seat]
            arrays[f"weight{seat}"] = self.weight[seat]
            arrays[f"groups{seat}"] = self.groups[seat]
        # Through an open file, so NumPy does not append `.npz` to the name.
        with temporary.open("wb") as file:
            np.savez(file, built_at=np.asarray(self.built_at), **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> TargetSet:
        """Read a saved set; a file without every key is refused by name."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as saved:
            missing = [key for key in _FILE_KEYS if key not in saved.files]
            if missing:
                raise ValueError(f"{path} is not a target set (no {missing[0]!r})")
            data = {key: saved[key] for key in _FILE_KEYS}
        spec = TargetSpec(
            granularity=str(data["granularity"]),
            alpha=float(data["alpha"]),
            budget=int(data["budget"]),
            refresh=int(data["refresh"]),
            noise_cap=float(data["noise_cap"]),
            pool=int(data["pool"]),
            stream=int(data["stream"]),
        )
        return cls(
            spec=spec,
            built_at=int(data["built_at"]),
            rows=(data["rows0"], data["rows1"]),
            weight=(data["weight0"], data["weight1"]),
            groups=(data["groups0"], data["groups1"]),
        )
