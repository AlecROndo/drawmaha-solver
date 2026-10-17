"""Materialise: two seats' nets become the table's regret slots, eq. (3) by the write.

Deep CFR plays, at iteration t, the strategy its regret net implies: regret
matching over the net's outputs when any is positive, and — eq. (3) of
Brown et al. — the single best action when none is. Rung 3's walk already
plays regret matching over `cumulative_regret`: `RegretMatcher.strategy()`
clips the slots at zero and normalises, falling back to uniform when the
clipped sum is zero. So the walk needs no net at all. It needs the net's
verdict WRITTEN into the slots in a form the ledger's rule reads as eq. (3):

- some output positive: the outputs as they are. The ledger clips the
  negatives and normalises the rest, which is regret matching.
- every output <= 0, not all equal: a one-hot at the argmax, shared by tied
  maxima. The ledger normalises it to "play the best action(s)", the clause
  regret matching alone cannot express because it would read all-negative
  as uniform.
- all equal: zeros, which the ledger reads as uniform. An untrained net
  (iteration 1) and an exact tie — bit-equal float32 outputs, which a
  trained net all but never produces — land here, and so does a seat with
  no net.

**One pass per public point.** The table's rows are laid out point by point,
each point's rows in its shape's `private_keys` order, and `features.layout()`
holds the same arithmetic: a point's rows are a contiguous slice of the
private table beside one public row, and its columns are one contiguous
slice of `cumulative_regret`, `width[point]` per row. So materialise is 141
batched forward passes, each gathered through the point's legal actions,
each written with one slice assignment. No key is ever built.

**Whose net.** Every row on a point is a decision of `point.player`, so the
seat's net scores the whole point; the other seat's net never sees it.

**Only the whole game.** A `listed` table's rows are in whatever order the
caller gave, so the layout's offsets mean nothing in it. The table does not
say which index built it, so it is refused by its fingerprint: the row
count, the column count and every row's width (`PackedTable.widths()`)
must be the whole game's. A listed table over all 6.2 million keys in an
order that happens to preserve every width would pass, and nothing builds
one: its index would be the gigabytes of dict the packed table exists to
avoid. Anything but exactly two nets is refused too.

Measured on this Mac (6 threads): one whole-game materialise with two
3 x 64 nets, 6.2M rows, in 0.47 s (the spec measured the bare forward pass at 0.32 s).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drawmaha_solver.deepcfr.net import Net, predict
from drawmaha_solver.minidrawmaha.features import layout, legal_gather
from drawmaha_solver.minidrawmaha.packed_table import PackedTable

# ---------------------------------------------------------------------------
# Eq. (3) as a write
# ---------------------------------------------------------------------------

def regret_slots_from_outputs(outputs: np.ndarray) -> np.ndarray:
    """What to write into a ledger's slots so `strategy()` plays the net's eq. (3).

    `outputs` is (n, w): each row a spot's outputs at its w legal columns.
    Returns (n, w) float64 — the table's dtype — by the three clauses in the
    module docstring, classified row by row with no Python loop. A tie is
    exact equality of the outputs (float32 from the net, widened without
    rounding), not a tolerance. A non-finite output is refused: no clause
    is right for it.
    """
    outputs = np.asarray(outputs, dtype=np.float64)
    if outputs.ndim != 2:
        raise ValueError(f"outputs must be a 2-D (rows, width) batch, got shape {outputs.shape}")
    # A NaN row would fall through every clause and write zeros: a diverged
    # net playing uniform with nothing in the log.
    if not np.isfinite(outputs).all():
        raise ValueError("every output must be finite: the net has diverged")
    top = outputs.max(axis=1, keepdims=True)
    some_positive = top > 0.0
    all_equal = (outputs == top).all(axis=1, keepdims=True)
    one_hot = (outputs == top).astype(np.float64)
    return np.where(some_positive, outputs, np.where(all_equal, 0.0, one_hot))

# ---------------------------------------------------------------------------
# The write
# ---------------------------------------------------------------------------

def materialise(table: PackedTable, nets: Sequence[Net | None]) -> None:
    """Write `nets[seat]`'s eq.-(3) slots for every row of the whole game into `table`.

    `cumulative_regret` is overwritten in place, every column of it; no
    other array of the table is read or written. A seat whose net is `None`
    writes zeros over its points — uniform — without building a feature.
    """
    if len(nets) != 2:
        raise ValueError(f"materialise takes two nets, one per seat, got {len(nets)}")
    built = layout()
    if (
        len(table) != built.row_offset[-1]
        or table.cumulative_regret.size != built.column_offset[-1]
    ):
        raise ValueError(
            "materialise writes the whole game's layout and was handed a table of "
            f"{len(table):,} rows and {table.cumulative_regret.size:,} columns, not the "
            f"whole game's {built.row_offset[-1]:,} and {built.column_offset[-1]:,}"
        )
    # Same size is not the same order: the widths are the layout's row by row
    # only when the rows are. A 6.2M-element compare, about 3 ms.
    if not np.array_equal(table.widths(), np.repeat(built.width, np.diff(built.row_offset))):
        raise ValueError(
            "materialise writes the whole game's layout and was handed a table whose "
            "rows are not in the whole game's order"
        )
    gather = legal_gather()
    slots = table.cumulative_regret
    for p, point in enumerate(built.points):
        columns = slice(built.column_offset[p], built.column_offset[p + 1])
        net = nets[point.player]
        if net is None:
            slots[columns] = 0.0
            continue
        rows = built.row_offset[p + 1] - built.row_offset[p]
        width = built.width[p]
        private = built.private[built.private_offset[p] : built.private_offset[p] + rows]
        public = np.broadcast_to(built.public[p], (rows, built.public.shape[1]))
        outputs = predict(net, np.concatenate([private, public], axis=1))
        slots[columns] = regret_slots_from_outputs(outputs[:, gather[p, :width]]).ravel()
