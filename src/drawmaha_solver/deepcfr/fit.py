"""The from-scratch refit: a fresh net fitted to (row, t, target) rows.

Deep CFR does not update a net between iterations; it throws the net away
and fits a new one to everything the memory holds. The memory is three
arrays the same length: `rows`, the table row each sample was taken at
(the net never sees the row number — `Layout.features` turns it into the
136 floats the player could see); `t`, the iteration the sample was taken
on; and `targets`, what the net should say at that row's legal columns,
`MAX_WIDTH` wide with padding past the row's width. Nothing here knows
what a target IS — a sampled regret, an action probability — so the same
fit serves the regret nets and the policy net.

**The loss.** Per row, the squared error summed over its legal columns,
read off the head through `features.legal_gather()`; the padding is masked
out BEFORE the subtraction, so a padded column holding NaN is never looked
at and never leaks into a gradient. Each row is weighted by its iteration
t, divided by the batch's mean t so the loss keeps one scale as the run
lengthens, and the batch loss is the mean over rows: a 2-wide and a 4-wide
spot each count as one row. The weight is Deep CFR's (Brown, Lerer, Gross &
Sandholm 2019), borrowed from Linear CFR (Brown & Sandholm 2019): the
early iterations' strategies are far from equilibrium, so the regrets
sampled against them say the least about the game, and a weight that
grows with t lets the later samples outvote them.

**The optimiser.** Adam at the spec's learning rate, the gradient's norm
clipped to the spec's bound before each step so one wild sample cannot
throw the weights, `steps` steps of `batch` rows each, the rows drawn
uniformly WITH replacement from the memory so a memory smaller than a
batch still fills one.

**Determinism.** The seed is a tuple of ints — PR 4 passes (seed, t, seat)
— folded by `np.random.SeedSequence` into one 63-bit integer so that two
tuples with the same sum cannot collide. That one integer seeds
`torch.manual_seed` for the net's initialisation and a `torch.Generator`
for the minibatch indices. The thread count is set for the fit's duration
and put back afterwards, because it is process-global and a different
count CAN change the summation order of a matmul; nothing promises two
counts the same weights (1 and 4 threads happened to agree bit for bit on
this Mac), so a run that must reproduce another keeps the count it used.

Measured on this Mac (6 cores): one default refit — 4,000 steps of
10,000 rows drawn from 1M real rows, features gathered on the fly — in
15.1 s at `threads=8` and 14.2 s at 6, against the spec's 10.8 s for the
bare net on ready-made features. The gather alone is 3.9 s of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from drawmaha_solver.deepcfr.net import Net
from drawmaha_solver.minidrawmaha.features import MAX_WIDTH, NO_ACTION, layout, legal_gather

# ---------------------------------------------------------------------------
# The run parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FitConfig:
    """What one refit is: the net's size, the steps, the batch, Adam's rate, the clip, the threads.

    The defaults are the spec's: 3 x 64, 4,000 steps of 10,000 rows, lr
    1e-3, norm clipped to 1. `threads` is handed to `torch.set_num_threads`
    for the fit's duration.
    """

    width: int = 64
    depth: int = 3
    steps: int = 4_000
    batch: int = 10_000
    lr: float = 1e-3
    clip: float = 1.0
    threads: int = 8

    def __post_init__(self) -> None:
        for name in ("steps", "batch", "threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        # `not > 0` rather than `<= 0`: a NaN is neither, and a clip of 0
        # would scale every gradient to nothing and fit in silence.
        for name in ("lr", "clip"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

def batch_loss(net: Net, rows: np.ndarray, t: np.ndarray, targets: np.ndarray) -> torch.Tensor:
    """One batch's loss, a scalar tensor: what every step minimises.

    Public so a test and PR 4's logging read the same number the optimiser
    does. Builds the batch's features on the fly from `layout()`. Assumes
    what `fit_from_scratch` checks: every t is >= 1, so the batch's mean t
    the weights are divided by is positive.
    """
    built = layout()
    point, _ = built.locate_rows(rows)
    gather = torch.from_numpy(legal_gather()[point].astype(np.int64))
    legal = gather != NO_ACTION
    outputs = net(torch.from_numpy(built.features(rows)))
    predicted = outputs.gather(1, gather.clamp(min=0))
    # The padding is replaced before the subtraction: a NaN there would
    # survive `where` in the backward pass, where 0 * NaN is NaN.
    target = torch.where(legal, torch.from_numpy(np.asarray(targets, dtype=np.float32)), 0.0)
    error = torch.where(legal, (predicted - target) ** 2, 0.0).sum(dim=1)
    weight = torch.from_numpy(np.asarray(t, dtype=np.float32))
    weight = weight / weight.mean()
    return (weight * error).mean()

# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit_from_scratch(
    rows: np.ndarray,
    t: np.ndarray,
    targets: np.ndarray,
    *,
    seed: tuple[int, ...],
    config: FitConfig = FitConfig(),
) -> Net:
    """A fresh `Net(config.width, config.depth)` fitted to the memory; returned in `eval()` mode.

    `rows` (n,) int table rows, `t` (n,) iteration numbers >= 1 (an
    integer-valued float is accepted: t enters only as a float32 weight),
    `targets` (n, MAX_WIDTH) float in ledger-column order with padding
    ignored.

    Refused by name before anything is built:

    - a `rows` or `t` that is not 1-D, or non-integer rows;
    - mismatched lengths, or an empty memory;
    - a non-finite t, or one below 1;
    - targets not `MAX_WIDTH` wide, or non-finite at a legal column;
    - a row outside the table.

    Refused afterwards: a fit whose weights went non-finite.
    """
    rows = np.asarray(rows)
    t = np.asarray(t)
    targets = np.asarray(targets, dtype=np.float32)
    # A (n, 1) t would broadcast against the (n,) error into an (n, n) loss
    # and train on the wrong number without a word, so the shapes are
    # refused by name, not left to broadcasting.
    if rows.ndim != 1 or t.ndim != 1:
        raise ValueError(f"rows and t must be 1-D, got shapes {rows.shape} and {t.shape}")
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(f"rows must be integer row numbers, got dtype {rows.dtype}")
    if not len(rows) == len(t) == len(targets):
        raise ValueError(
            f"rows, t and targets must have the same length, got {len(rows)}, {len(t)}, {len(targets)}"
        )
    if len(rows) == 0:
        raise ValueError("the memory is empty: nothing to fit")
    if targets.ndim != 2 or targets.shape[1] != MAX_WIDTH:
        raise ValueError(f"targets must be (n, {MAX_WIDTH}), got shape {targets.shape}")
    if not np.isfinite(t).all():
        raise ValueError("every t must be finite")
    if t.min() < 1:
        raise ValueError(f"every t must be an iteration number >= 1, got a minimum of {t.min()}")
    rows = rows.astype(np.int64)
    point, _ = layout().locate_rows(rows)  # raises IndexError for a row outside the table
    # Only the legal columns are checked: the padding may hold anything.
    if not np.isfinite(targets[legal_gather()[point] != NO_ACTION]).all():
        raise ValueError("every target at a legal column must be finite")
    folded = _fold_seed(seed)
    threads_before = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        torch.manual_seed(folded)
        net = Net(width=config.width, depth=config.depth)
        draw = torch.Generator().manual_seed(folded)
        optimiser = torch.optim.Adam(net.parameters(), lr=config.lr)
        net.train()
        for _ in range(config.steps):
            picked = torch.randint(len(rows), (config.batch,), generator=draw).numpy()
            optimiser.zero_grad()
            batch_loss(net, rows[picked], t[picked], targets[picked]).backward()
            clip_grad_norm_(net.parameters(), config.clip)
            optimiser.step()
    finally:
        torch.set_num_threads(threads_before)
    # A diverged fit would materialise to uniform play with nothing in the
    # log; it is an error here, where the run can see it.
    if not all(torch.isfinite(parameter).all() for parameter in net.parameters()):
        raise ValueError("the fit diverged: a weight is not finite")
    net.eval()
    return net

def _fold_seed(seed: tuple[int, ...]) -> int:
    """One 63-bit integer from a tuple of ints, so (1, 2) and (2, 1) seed differently."""
    high, low = (int(word) for word in np.random.SeedSequence(seed).generate_state(2, np.uint32))
    return ((high & 0x7FFF_FFFF) << 32) | low
