"""The from-scratch refit: a fresh net fitted to (row, t, target, weight) rows.

Deep CFR does not update a net between iterations; it throws the net away
and fits a new one to everything the memory holds. The memory is four
arrays the same length: `rows`, the table row each sample was taken at
(the net never sees the row number — `Layout.features` turns it into the
136 floats the player could see); `t`, the iteration the sample was taken
on; `targets`, what the net should say at that row's legal columns,
`MAX_WIDTH` wide with padding past the row's width; and `weight`, how many
sampled rows the row stands for, 1 for a walk's own sample (see
`reservoir.py` for the rows that are not one). Nothing here knows what a
target IS — a sampled regret, an action probability — so the same fit
serves the regret nets and the policy net.

**The loss.** Per row, the squared error summed over its legal columns,
read off the head through `features.legal_gather()`; the padding is masked
out BEFORE the subtraction, so a padded column holding NaN is never looked
at and never leaks into a gradient. Each row is weighted by t × weight —
its iteration times how many samples it stands for — divided by the
batch's mean of t × weight so the loss keeps one scale as the run
lengthens, and the batch loss is the mean over rows: a 2-wide and a 4-wide
spot each count as one row. The t is Deep CFR's (Brown, Lerer, Gross &
Sandholm 2019), borrowed from Linear CFR (Brown & Sandholm 2019): the
early iterations' strategies are far from equilibrium, so the regrets
sampled against them say the least about the game, and a weight that
grows with t lets the later samples outvote them. The weight multiplies t
rather than the target, so it changes how much a row counts and never
what the row says; at weight 1 throughout it is t / mean(t) bit for bit
(t · 1.0 is t in float32), and a row of weight 0 contributes nothing. A
batch whose every row weighs 0 has a mean of 0 to divide by; its loss is
0 rather than 0/0, so a tiny memory under the boost cannot end a fit as
"diverged" on one unlucky draw.

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

**The GPU.** `device="mps"` is the same fit on Apple's GPU: the same init,
the same minibatch draws (still from a cpu generator), the same loss, one
`_weighted_loss` shared by both paths. The memory, the layout's two
feature halves and the legal gather go to the device once per fit and each
batch's features are gathered there, so no step goes back through numpy.
That copy holds the whole memory a second time for the fit's duration
(rows, t, weight, and `MAX_WIDTH` float32 targets per row), so peak memory is
about twice the memory's; unified memory makes the copy cheap, not free.
Only the order of the arithmetic changes, so it is the same rule but not
the same net. Adam divides by √v, so a near-zero gradient component turns
rounding into a full-size step and the two fits can part: over 300 steps on
2,000 real rows the predictions agreed to 1e-6 on some memories and parted
by 0.04 on another, while the loss on the memory stayed within 2% of the
cpu fit's on all twelve tried. The test pins the loss, not the weights, at
5% for headroom over that 2%.
Measured on this Mac (M5 Pro) on a real 17M-row memory, per step: 4 x 256
at 5.0 ms against 23.9 ms on the cpu, 3 x 128 at 2.5 ms against 9.5 ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from drawmaha_solver.deepcfr.net import Net
from drawmaha_solver.minidrawmaha.rules.features import MAX_WIDTH, NO_ACTION, layout, legal_gather

# ---------------------------------------------------------------------------
# The run parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FitConfig:
    """What one refit is: the net's size, steps, batch, Adam's rate, clip, threads and device.

    The defaults are the spec's: 3 x 64, 4,000 steps of 10,000 rows, lr
    1e-3, norm clipped to 1. `threads` is handed to `torch.set_num_threads`
    for the fit's duration. `device` "mps" runs the steps on Apple's GPU
    (`_fit_on_device`), and `threads` is then not set.
    """

    width: int = 64
    depth: int = 3
    steps: int = 4_000
    batch: int = 10_000
    lr: float = 1e-3
    clip: float = 1.0
    threads: int = 8
    # "cpu", or "mps" for Apple's GPU: the device only orders the arithmetic,
    # so an mps fit is the same rule but not the same weights bit for bit.
    # The two names are `train.DEVICES`, spelled out here because this
    # module must not import the loop; a test keeps the two equal.
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("steps", "batch", "threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.device not in ("cpu", "mps"):
            raise ValueError(f"device must be 'cpu' or 'mps', got {self.device!r}")
        # `not > 0` rather than `<= 0`: a NaN is neither, and a clip of 0
        # would scale every gradient to nothing and fit in silence.
        for name in ("lr", "clip"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

def batch_loss(
    net: Net,
    rows: np.ndarray,
    t: np.ndarray,
    targets: np.ndarray,
    weight: np.ndarray | None = None,
) -> torch.Tensor:
    """One batch's loss, a scalar tensor: what every step minimises.

    Public so a test and PR 4's logging read the same number the optimiser
    does. Builds the batch's features on the fly from `layout()`. `weight`
    left out is a batch of plain samples, every row weighing 1: a default
    here, where a caller hands arrays it built itself, while `Learner.fit`
    in `train.py` has none, because that is where a memory's column could
    be dropped by accident. Assumes what `fit_from_scratch` checks: every t
    is >= 1, every weight finite and non-negative, and t × weight within
    float32's range.
    """
    built = layout()
    point, _ = built.locate_rows(rows)
    gather = torch.from_numpy(legal_gather()[point].astype(np.int64))
    if weight is None:
        weight = np.ones(len(rows), dtype=np.float32)
    return _weighted_loss(
        net(torch.from_numpy(built.features(rows))),
        gather,
        torch.from_numpy(np.asarray(targets, dtype=np.float32)),
        torch.from_numpy(np.asarray(t, dtype=np.float32)),
        torch.from_numpy(np.asarray(weight, dtype=np.float32)),
    )

def _weighted_loss(
    outputs: torch.Tensor,
    gather: torch.Tensor,
    targets: torch.Tensor,
    t: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """The loss from the net's outputs on: the one copy both the cpu and the device fit call.

    `outputs` (b, head) the net's raw head, `gather` (b, MAX_WIDTH) each
    row's ledger columns with `NO_ACTION` padding, `targets` (b, MAX_WIDTH),
    `t` (b,) and `weight` (b,) float32, all on one device. Everything
    before this — the features, the gather's rows — differs between the
    paths only in where it is looked up, never in what it is.
    """
    legal = gather != NO_ACTION
    predicted = outputs.gather(1, gather.clamp(min=0))
    # The padding is replaced before the subtraction: a NaN there would
    # survive `where` in the backward pass, where 0 * NaN is NaN.
    target = torch.where(legal, targets, 0.0)
    error = torch.where(legal, (predicted - target) ** 2, 0.0).sum(dim=1)
    scale = t * weight
    # A batch of weight-0 rows alone has a mean of 0; clamped, its loss is 0
    # and not NaN. Any positive mean is far above `tiny`, so the clamp
    # returns it unchanged and the division is the plain one.
    weight = scale / scale.mean().clamp(min=torch.finfo(scale.dtype).tiny)
    return (weight * error).mean()

# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit_from_scratch(
    rows: np.ndarray,
    t: np.ndarray,
    targets: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    seed: tuple[int, ...],
    config: FitConfig = FitConfig(),
) -> Net:
    """A fresh `Net(config.width, config.depth)` fitted to the memory; returned in `eval()` mode.

    `rows` (n,) int table rows, `t` (n,) iteration numbers >= 1 (an
    integer-valued float is accepted: t enters only as a float32 weight),
    `targets` (n, MAX_WIDTH) float in ledger-column order with padding
    ignored, `weight` (n,) finite and >= 0, or None for a memory of plain
    samples, every row weighing 1.

    Refused by name before anything is built:

    - a `rows`, `t` or `weight` that is not 1-D, or non-integer rows;
    - mismatched lengths, or an empty memory;
    - a non-finite t, or one below 1;
    - a non-finite or negative weight, one whose t × weight overflows
      float32 (the loss's scale), or weights with no positive entry
      (nothing to fit, as an empty memory is);
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
    with np.errstate(over="ignore"):  # a weight past float32's range is refused below as the inf it became
        weight = np.ones(len(rows), dtype=np.float32) if weight is None else np.asarray(weight, dtype=np.float32)
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D, got shape {weight.shape}")
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(f"rows must be integer row numbers, got dtype {rows.dtype}")
    if not len(rows) == len(t) == len(targets) == len(weight):
        raise ValueError(
            "rows, t, targets and weight must have the same length, "
            f"got {len(rows)}, {len(t)}, {len(targets)}, {len(weight)}"
        )
    if len(rows) == 0:
        raise ValueError("the memory is empty: nothing to fit")
    if targets.ndim != 2 or targets.shape[1] != MAX_WIDTH:
        raise ValueError(f"targets must be (n, {MAX_WIDTH}), got shape {targets.shape}")
    if not np.isfinite(t).all():
        raise ValueError("every t must be finite")
    if t.min() < 1:
        raise ValueError(f"every t must be an iteration number >= 1, got a minimum of {t.min()}")
    # Checked after the cast to float32, so a float64 weight past float32's
    # range is caught as the inf it became, not passed as the finite number
    # it was given as.
    if not np.isfinite(weight).all() or weight.min() < 0:
        raise ValueError("every weight must be finite and non-negative in float32")
    # The loss scales a row by t × weight in float32; a product past its
    # range would reach the batch as inf and the loss as NaN.
    with np.errstate(over="ignore"):
        if not np.isfinite(t.astype(np.float32) * weight).all():
            raise ValueError("t × weight must fit float32")
    # Every row inert: the batches would all be the weightless case the
    # loss guards, and 4,000 steps of zero gradient would "fit" nothing.
    if not (weight > 0).any():
        raise ValueError("no row carries weight: nothing to fit")
    rows = rows.astype(np.int64)
    point, _ = layout().locate_rows(rows)  # raises IndexError for a row outside the table
    # Only the legal columns are checked: the padding may hold anything.
    if not np.isfinite(targets[legal_gather()[point] != NO_ACTION]).all():
        raise ValueError("every target at a legal column must be finite")
    folded = _fold_seed(seed)
    # Any device but the cpu takes the device loop, so a name added to the
    # config can never quietly run the cpu loop; the device loop checks only
    # mps's availability, so a new name must add its own check there.
    if config.device != "cpu":
        return _fit_on_device(rows, t, targets, weight, folded, config, device=config.device)
    threads_before = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        torch.manual_seed(folded)
        net = Net(width=config.width, depth=config.depth)
        draw = torch.Generator().manual_seed(folded)
        optimiser = torch.optim.Adam(net.parameters(), lr=config.lr)
        net.train()
        for _ in range(config.steps):
            # The same draw as `_fit_on_device`'s; change both or neither
            # (the cpu-parity test in test_fit.py is the guard).
            picked = torch.randint(len(rows), (config.batch,), generator=draw).numpy()
            optimiser.zero_grad()
            batch_loss(net, rows[picked], t[picked], targets[picked], weight[picked]).backward()
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

def _fit_on_device(
    rows: np.ndarray,
    t: np.ndarray,
    targets: np.ndarray,
    weight: np.ndarray,
    folded: int,
    config: FitConfig,
    *,
    device: str,
) -> Net:
    """`fit_from_scratch`'s loop on `device`, the net returned on the cpu in `eval()` mode.

    The same init, the same minibatch draws (from a cpu generator, so they
    match the cpu fit's), the same `_weighted_loss` as `batch_loss`. What
    changes is where the work happens: the memory, the layout's two halves
    and the legal gather go to the device once, and each step's features
    are gathered there, so no step goes back to numpy. `device` is a
    parameter, not read off `config`, so a test can run this loop on the
    cpu, where it must match `batch_loss`'s fit, on a machine without a GPU;
    `config.device` and `config.threads` are not read here.
    An "mps" asked for on a machine without one is refused here, before
    anything is copied, rather than failing inside the first `.to`.
    """
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' was asked for, but this machine has no usable Apple GPU")
    device = torch.device(device)
    built = layout()
    row_offset = torch.from_numpy(built.row_offset).to(device)
    private_offset = torch.from_numpy(built.private_offset).to(device)
    private = torch.from_numpy(np.array(built.private)).to(device)
    public = torch.from_numpy(np.array(built.public)).to(device)
    gather_all = torch.from_numpy(legal_gather().astype(np.int64)).to(device)
    memory_rows = torch.from_numpy(rows).to(device)
    memory_t = torch.from_numpy(np.asarray(t, dtype=np.float32)).to(device)
    memory_weight = torch.from_numpy(np.asarray(weight, dtype=np.float32)).to(device)
    # `from_numpy` warns on a read-only array (a memory loaded from a file
    # can be one); `require` copies only in that case.
    memory_targets = torch.from_numpy(np.require(targets, requirements="W")).to(device)

    torch.manual_seed(folded)
    net = Net(width=config.width, depth=config.depth).to(device)
    draw = torch.Generator().manual_seed(folded)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.lr)
    net.train()
    for _ in range(config.steps):
        # The same draw as the cpu loop's in `fit_from_scratch`; change both
        # or neither (the cpu-parity test in test_fit.py is the guard).
        picked = torch.randint(len(rows), (config.batch,), generator=draw).to(device)
        batch_rows = memory_rows[picked]
        point = torch.searchsorted(row_offset, batch_rows, right=True) - 1
        position = batch_rows - row_offset[point]
        features = torch.cat([private[private_offset[point] + position], public[point]], dim=1)
        loss = _weighted_loss(
            net(features),
            gather_all[point],
            memory_targets[picked],
            memory_t[picked],
            memory_weight[picked],
        )
        optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(net.parameters(), config.clip)
        optimiser.step()
    net = net.to("cpu")
    if not all(torch.isfinite(parameter).all() for parameter in net.parameters()):
        raise ValueError("the fit diverged: a weight is not finite")
    net.eval()
    return net

def _fold_seed(seed: tuple[int, ...]) -> int:
    """One 63-bit integer from a tuple of ints, so (1, 2) and (2, 1) seed differently."""
    high, low = (int(word) for word in np.random.SeedSequence(seed).generate_state(2, np.uint32))
    return ((high & 0x7FFF_FFFF) << 32) | low
