"""Π: the average strategy learned by regression from the policy memory, and frozen for the grader.

`average.py` keeps the average exactly, one float per table column, which
is fine at mini-drawmaha's 6.2 million rows and impossible at a game whose
table cannot exist. Π is the way the original Deep CFR (Brown, Lerer, Gross
& Sandholm 2019) keeps it anyway: during every phase the opponent's walks
copy, at each spot they pass, the strategy they sampled from — one
`(row, t, σ)` row into the policy memory, a fixed-size reservoir — and a
fresh net is fit to those rows with the same t′-weighted squared error the
advantage nets use. That fit IS the average. At a spot with memory rows
σ₁, σ₂, σ₃ at weights t₁, t₂, t₃, the output y that minimises
`Σ tᵢ · (y − σᵢ)²` is `Σ tᵢ σᵢ / Σ tᵢ`, the t-weighted mean, and no row
ever held it. The reach weight the exact average multiplies in rides in
the SAMPLE COUNTS instead: the walks visit a spot in proportion to how
often play arrives there, so a spot reached ten times as often at t = 5
as at t = 2 contributes ten times as many rows from t = 5.

So Π is a lookup table for the average strategy, compressed into weights
and smoothed across neighbouring spots, carrying two errors the exact
average does not have — the reservoir's sampling and the fit's — which is
exactly what grading both on mini-drawmaha measures (spec: "Three graded
objects per grid mark").

**The fit.** `fit_from_scratch` on the memory's `contents()`; the targets
are σ rows zero-padded to `TARGET_WIDTH`, and `batch_loss` masks the
padding before the subtraction, as it does for advantage rows. Nothing in
the fit knows the target is a probability, so a regret memory handed by
mistake would fit in silence; `fit_policy` refuses one by checking that
every row is a distribution at its legal columns. A fresh net every time,
at every grid mark and not only at the end, so the curve has Π at each
point. The seed is `(seed, t, POLICY_TAG)`: `NetLearner` fits the two
advantage nets at `(seed, t, 0)` and `(seed, t, 1)` and draws its held
batch at `(seed, t, p, HELD)`, so Π's third word is "PI" in ASCII and no
two fits of one iteration share an initialisation.

**The freeze.** The net gives seven signed numbers per spot. Gathered at
the spot's legal columns, clipped at zero and divided by their sum they
are a strategy; where every output is at or below zero the spot plays
uniform. That is `strategy.average_rows` on the clipped outputs, the same
read `current_iterate` makes of the table, and the same arithmetic the
ledger's regret matching does — but it means something else here: the
targets were already distributions, so a perfect net would output rows
that need no clipping, and the clip only repairs the fit's error. The run
over every key is `materialise`'s per-point batching, restated: one
forward pass per public point over the layout's contiguous rows.

torch is imported inside the functions, so the package's default import
and the default test suite stay torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drawmaha_solver.minidrawmaha.rules.features import NO_ACTION, layout, legal_gather
from drawmaha_solver.minidrawmaha.answer.strategy import (
    FrozenStrategy,
    StrategyInfo,
    average_rows,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from drawmaha_solver.deepcfr.fit import FitConfig
    from drawmaha_solver.deepcfr.net import Net

# The third word of Π's fit seed: "PI" in ASCII. The advantage fits use the
# seat, 0 or 1, there.
POLICY_TAG = 0x5049

# A σ row sums to one; float32 rows land within this of it.
_ROW_SUM_TOLERANCE = 1e-5

def policy_seed(seed: int, t: int) -> tuple[int, int, int]:
    """The seed of Π's fit at iteration t: `(seed, t, POLICY_TAG)`."""
    return (int(seed), int(t), POLICY_TAG)

# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit_policy(
    rows: np.ndarray,
    t: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    iteration: int,
    config: FitConfig | None = None,
) -> Net:
    """A fresh net fit to the policy memory's `(rows, t, targets)` at `iteration`, under `policy_seed`.

    `config` defaults to `FitConfig()`, the spec's refit. Refuses, before
    anything is built, a targets row that is not a distribution at its
    legal columns — a regret memory handed in place of the policy memory.
    """
    from drawmaha_solver.deepcfr.fit import FitConfig, fit_from_scratch

    targets = np.asarray(targets, dtype=np.float32)
    _require_distributions(np.asarray(rows), targets)
    return fit_from_scratch(
        rows, t, targets, seed=policy_seed(seed, iteration), config=config or FitConfig()
    )

def _require_distributions(rows: np.ndarray, targets: np.ndarray) -> None:
    """Every row's legal columns non-negative and summing to one; the padding is not looked at."""
    if targets.ndim != 2 or rows.ndim != 1 or len(rows) != len(targets):
        return  # `fit_from_scratch` names the shape fault
    point, _ = layout().locate_rows(rows.astype(np.int64))
    legal = legal_gather()[point] != NO_ACTION
    values = np.where(legal, targets.astype(np.float64), 0.0)
    fine = (values >= 0.0).all(axis=1) & (np.abs(values.sum(axis=1) - 1.0) <= _ROW_SUM_TOLERANCE)
    bad = np.flatnonzero(~fine)
    if bad.size:
        raise ValueError(
            f"policy target {int(bad[0])} is not a distribution over its legal columns: "
            f"{targets[bad[0]].tolist()}; Π is fit on σ rows, not regret samples"
        )

# ---------------------------------------------------------------------------
# The freeze
# ---------------------------------------------------------------------------

def policy_strategy(net: Net, *, iteration: int, seed: int) -> FrozenStrategy:
    """Π over every key of the whole game: outputs at the legal columns, clipped, normalised, uniform if all zero.

    `iteration` and `seed` only label the file.
    """
    from drawmaha_solver.deepcfr.net import predict

    built = layout()
    gather = legal_gather()
    clipped = np.empty(int(built.column_offset[-1]))
    for p in range(len(built.points)):
        rows = built.row_offset[p + 1] - built.row_offset[p]
        width = built.width[p]
        private = built.private[built.private_offset[p] : built.private_offset[p] + rows]
        public = np.broadcast_to(built.public[p], (rows, built.public.shape[1]))
        outputs = predict(net, np.concatenate([private, public], axis=1))
        columns = slice(built.column_offset[p], built.column_offset[p + 1])
        clipped[columns] = np.maximum(outputs[:, gather[p, :width]], 0.0).ravel()
    widths = np.repeat(built.width, np.diff(built.row_offset))
    info = StrategyInfo(rule="deep", column="pi", iteration=iteration, seed=seed, workers=0)
    return FrozenStrategy(average_rows(clipped, widths), widths, info)
