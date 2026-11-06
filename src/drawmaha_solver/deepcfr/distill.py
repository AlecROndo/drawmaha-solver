"""Distillation: the rung-4 net fitted straight to a known strategy, to measure what the net can hold.

Deep CFR's ÷ pot arm, a 4 × 256 net, flattens at an exact-average
exploitability of 0.068 while the tabular LCFR run keeps falling (0.0157
at iteration 10,000,000). Two things could hold the net there: the net
cannot REPRESENT a better strategy, or Deep CFR's way of training it —
sampled regrets, reservoir memories, from-scratch refits — never finds
one. This module takes Deep CFR out of the picture. It fits the same net
architecture directly to the table's average strategy, by supervised
learning over every one of the 6,220,050 infosets, and the copy is frozen
and graded by the exact meter like any other strategy. Three gaps then
read off one axis: table vs copy is representation, copy vs Deep CFR is
training, table vs Deep CFR is both.

**The target is the average strategy, not the regrets.** The regrets would
copy the current iterate, which CFR never promises anything about (0.191
at 20M hands against the average's 0.068); the average is what is graded.
A target row is the teacher's distribution over the row's legal actions,
read from a `FrozenStrategy` and laid out `MAX_WIDTH` wide with zeros past
the row's width, the same padding the policy memory uses. A row the
teacher never banked plays uniform in the file, so every target is a
distribution and none is NaN.

**The loss is cross-entropy on a softmax over the legal columns.** The Π
loss sweep (`scripts/deepcfr/pi_loss_sweep.py`) showed cross-entropy on a
softmax head copying a distribution better than squared error on a linear
head, 0.136 → 0.109 at one mark. The net's head stays the seven signed
outputs of `net.Net`; the softmax is taken at the row's legal columns with
the others held at −1e9, so an illegal column gets no probability and no
gradient. The loss at a row is smallest where the softmax IS the teacher's
row, and that minimum is the row's entropy; a net reaches it only where it
can represent the row, and a teacher entry of exactly zero is approached,
never reached, as its logit runs toward −∞.

**Each row weighs what arrives at it.** Exploitability is measured on play
that happens: a best response exploits a spot in proportion to how often
the strategy's own choices and the deck bring the player there, and a spot
the player's own play never reaches cannot be exploited at all. So a row's
weight is its seat's own reach under the teacher (`average.reach_of_strategy`,
the textbook π^q(I) through the predecessor table) times the number of
physical deals its canonical key stands for (`deal_multiplicity`: a key with
a suit symmetry — a pair whose suits the board does not tell apart — is
fewer deals than one without, 6 over the size of its stabiliser). That is
the probability mass of the player's own play at the row, up to one
constant. `pi_reach_weighted.py` found that a reach-weighted error tracks
exploitability where an unweighted one does not; the weight is the same
one, carried into the loss. Rows of weight zero are never trained on, and
the net says whatever it generalises there.

**The weight is applied by sampling.** Minibatch rows are drawn in
proportion to their weight (`ReachSampler`: a float64 cumulative sum and a
`searchsorted`, on the cpu where float64 exists) and the batch loss is a
plain mean. That estimates the weighted objective Σ wᵢ CEᵢ / Σ wᵢ without
bias and at far lower variance than drawing rows uniformly and multiplying
by weights that span many orders of magnitude.

**No held-out split for the headline.** The exact grader scores every
infoset, so the grade over the whole game is the test and memorising is
the goal. `train_fraction` below one is the second experiment: train on a
random share of the rows, grade on the whole game, and report the error
on the rows the fit never saw — what inference at unseen spots would be
worth in a game that cannot be enumerated.

**The fit is `fit._fit_on_device`'s pattern** with a different loss and a
different draw: the layout's two feature halves go to the device once, each
batch's features are gathered there, Adam under a cosine schedule from
`lr`, the gradient's norm clipped. Every `eval_every` steps the student's
distribution at every row is computed on the device and its reach-weighted
mean absolute error against the teacher recorded, so the curve is of the
quantity that tracks the grade, not only of the loss. The curve's error is
summed in float32 on the device (mps has no float64), the recorded
`reach_weighted_error` of the frozen file in float64 on the cpu; in every
committed arm the curve's last entry and the file's error agree to about
1e-8.

**The freeze** is `net.predict_by_point` with a softmax per row, into a
`FrozenStrategy` whose info names the teacher's rule, column, iteration
and seed under `rule="distill"`, so a reader of the file knows what was
copied. torch is imported inside the functions that need it; the dataset
half is NumPy and runs in the default suite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from drawmaha_solver.deepcfr.average import reach_of_strategy
from drawmaha_solver.minidrawmaha.rules.cards import SUIT_RELABELLINGS, relabel
from drawmaha_solver.minidrawmaha.rules.enumeration import private_keys
from drawmaha_solver.minidrawmaha.rules.features import MAX_WIDTH, layout, legal_gather
from drawmaha_solver.minidrawmaha.solver.lockstep import SEATS
from drawmaha_solver.minidrawmaha.answer.strategy import FrozenStrategy, StrategyInfo

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from drawmaha_solver.deepcfr.net import Net

# ---------------------------------------------------------------------------
# The dataset: targets, multiplicity, weights
# ---------------------------------------------------------------------------

def teacher_targets(teacher: FrozenStrategy) -> np.ndarray:
    """Every row's distribution, (rows, MAX_WIDTH) float32, zeros past the row's width."""
    built = layout()
    targets = np.zeros((int(built.row_offset[-1]), MAX_WIDTH), dtype=np.float32)
    probabilities = teacher.probabilities
    for p in range(len(built.points)):
        rows = slice(built.row_offset[p], built.row_offset[p + 1])
        columns = slice(built.column_offset[p], built.column_offset[p + 1])
        width = int(built.width[p])
        targets[rows, :width] = probabilities[columns].reshape(-1, width)
    return targets

@cache
def deal_multiplicity() -> np.ndarray:
    """Per row, how many physical (hole, discard, ordered board) deals its canonical key stands for; int64, read-only.

    The orbit of the key under the six suit relabellings: a key every
    relabelling moves is 6 deals, one with a suit symmetry fewer. The hole
    and the discard are sets of cards, the board is ordered, as the game
    deals them.
    """
    built = layout()
    shapes = sorted({(point.board_cards, point.discards) for point in built.points})
    per_shape = {}
    for shape in shapes:
        keys = private_keys(board_cards=shape[0], discards=shape[1])
        counts = np.empty(len(keys), dtype=np.int64)
        for position, (hole, discarded, board) in enumerate(keys):
            images = {
                (
                    frozenset(relabel(relabelling, hole)),
                    frozenset(relabel(relabelling, discarded)),
                    relabel(relabelling, board),
                )
                for relabelling in SUIT_RELABELLINGS
            }
            counts[position] = len(images)
        per_shape[shape] = counts
    multiplicity = np.concatenate(
        [per_shape[(point.board_cards, point.discards)] for point in built.points]
    )
    multiplicity.setflags(write=False)
    return multiplicity

def row_weights(teacher: FrozenStrategy) -> np.ndarray:
    """Each row's share of the teacher's own play: own reach × deal multiplicity, summing to one; float64."""
    sigma = teacher.probabilities.astype(np.float64)
    reach = sum(reach_of_strategy(sigma, seat=seat) for seat in SEATS)
    weights = reach * deal_multiplicity()
    return weights / weights.sum()

# ---------------------------------------------------------------------------
# The run parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistillConfig:
    """What one distillation is: the net's size, the steps, the batch, the schedule, the device, the data share.

    `lr` is the cosine schedule's peak, at step 0; `clip` bounds the
    gradient's norm; `train_fraction` is the share of rows the fit may
    draw from (1.0 is every row); `weighted=False` draws rows uniformly
    instead of by reach, an ablation. `eval_every` is the curve's grain.
    `device` defaults to the cpu, which every machine has, as `FitConfig`'s
    does; `scripts/deepcfr/distill_teacher.py` asks for `mps`.
    """

    width: int = 256
    depth: int = 4
    steps: int = 100_000
    batch: int = 16_384
    lr: float = 1e-3
    clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    train_fraction: float = 1.0
    weighted: bool = True
    eval_every: int = 5_000

    def __post_init__(self) -> None:
        for name in ("steps", "batch", "eval_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        for name in ("lr", "clip"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 < self.train_fraction <= 1.0:
            raise ValueError(f"train_fraction must be in (0, 1], got {self.train_fraction}")
        if self.device not in ("cpu", "mps"):
            raise ValueError(f"device must be 'cpu' or 'mps', got {self.device!r}")

# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

class ReachSampler:
    """Rows drawn in proportion to their weight: a float64 cumulative sum and a `searchsorted`.

    A row of weight zero sits on a step of zero height in the cumulative
    sum and no uniform draw lands on it. Refuses weights that are negative,
    non-finite, or all zero, by name: an all-zero vector would otherwise
    sample the last row forever.
    """

    def __init__(self, weights: np.ndarray, *, seed: int) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {weights.shape}")
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("every weight must be finite and non-negative")
        total = weights.sum()
        if not total > 0.0:
            raise ValueError("every weight is zero: nothing to sample")
        self.cumulative = np.cumsum(weights) / total
        self.cumulative[-1] = 1.0
        self._rng = np.random.default_rng(seed)

    def draw(self, n: int) -> np.ndarray:
        """`n` row numbers, int64, with replacement."""
        return np.searchsorted(self.cumulative, self._rng.random(n), side="right").astype(np.int64)

def training_rows(config: DistillConfig, rows: int) -> np.ndarray | None:
    """The rows a fraction arm may draw from, a seeded random subset; None when every row may be."""
    if config.train_fraction >= 1.0:
        return None
    rng = np.random.default_rng((config.seed, 0x5452))  # "TR": the subset's own stream
    picked = rng.random(rows) < config.train_fraction
    return np.flatnonzero(picked).astype(np.int64)

def sampling_weights(weights: np.ndarray, config: DistillConfig) -> np.ndarray:
    """What the sampler draws by: the reach weights, or ones for the unweighted ablation, zero off the training rows."""
    weights = np.asarray(weights, dtype=np.float64)
    drawn = weights.copy() if config.weighted else np.ones_like(weights)
    rows = training_rows(config, len(weights))
    if rows is not None:
        kept = np.zeros_like(drawn)
        kept[rows] = drawn[rows]
        drawn = kept
    return drawn

# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    outputs: torch.Tensor, gather: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Mean over rows of −Σ_legal target · log softmax(outputs at the legal columns); a scalar tensor.

    `outputs` (b, head) the net's raw head, `gather` (b, MAX_WIDTH) each
    row's head columns with `NO_ACTION` padding, `targets` (b, MAX_WIDTH)
    distributions with zeros in the padding. An illegal column's logit is
    replaced by −1e9 before the softmax, so it holds no probability and
    receives no gradient; its target is zero, so it adds nothing to the
    sum either way.
    """
    import torch

    from drawmaha_solver.minidrawmaha.rules.features import NO_ACTION

    legal = gather != NO_ACTION
    picked = outputs.gather(1, gather.clamp(min=0))
    log_probabilities = torch.log_softmax(torch.where(legal, picked, -1e9), dim=1)
    per_row = -torch.where(legal, targets * log_probabilities, 0.0).sum(dim=1)
    return per_row.mean()

# ---------------------------------------------------------------------------
# The layout on the device
# ---------------------------------------------------------------------------

# Rows per forward pass when the student is read over every row of the game.
EVAL_CHUNK = 262_144

class _DeviceLayout:
    """The layout's two feature halves and the gather, on one device, and the student read over rows there."""

    def __init__(self, device: torch.device) -> None:
        import torch

        from drawmaha_solver.minidrawmaha.rules.features import NO_ACTION

        built = layout()
        self.device = device
        self.rows = int(built.row_offset[-1])
        self.row_offset = torch.from_numpy(built.row_offset).to(device)
        self.private_offset = torch.from_numpy(built.private_offset).to(device)
        self.private = torch.from_numpy(np.array(built.private)).to(device)
        self.public = torch.from_numpy(np.array(built.public)).to(device)
        self.gather = torch.from_numpy(legal_gather().astype(np.int64)).to(device)
        self.width = torch.from_numpy(built.width).to(device)
        self.no_action = NO_ACTION

    def point_of(self, rows: torch.Tensor) -> torch.Tensor:
        import torch

        return torch.searchsorted(self.row_offset, rows, right=True) - 1

    def features_and_gather(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(n, 136) features and (n, MAX_WIDTH) gather for row numbers on the device."""
        import torch

        point = self.point_of(rows)
        position = rows - self.row_offset[point]
        features = torch.cat([self.private[self.private_offset[point] + position], self.public[point]], dim=1)
        return features, self.gather[point]

    def probabilities(self, net: Net, rows: torch.Tensor) -> torch.Tensor:
        """The student's softmax at the legal columns of `rows`, (n, MAX_WIDTH), zeros in the padding."""
        import torch

        features, gather = self.features_and_gather(rows)
        legal = gather != self.no_action
        picked = net(features).gather(1, gather.clamp(min=0))
        return torch.where(legal, torch.softmax(torch.where(legal, picked, -1e9), dim=1), 0.0)

# ---------------------------------------------------------------------------
# The reach-weighted error
# ---------------------------------------------------------------------------

def reach_weighted_error(
    student: FrozenStrategy, targets: np.ndarray, weights: np.ndarray
) -> float:
    """Σ over columns of weight · |student − teacher| over Σ weight, each column at its row's weight.

    The `rw_mad` of `scripts/deepcfr/pi_reach_weighted.py` restated over
    the padded layout: the diagnostic that tracked exploitability across
    the Π refits. `targets` and `weights` are `teacher_targets` and
    `row_weights` (or any (rows,) weights); a weight of zero drops the row.
    """
    weights = np.asarray(weights, dtype=np.float64)
    difference = np.abs(teacher_targets(student).astype(np.float64) - np.asarray(targets, dtype=np.float64))
    widths = np.repeat(layout().width, np.diff(layout().row_offset))
    return float((weights * difference.sum(axis=1)).sum() / (weights * widths).sum())

def _device_error(
    net: Net, on_device: _DeviceLayout, targets: torch.Tensor, weights: torch.Tensor
) -> float:
    """`reach_weighted_error` computed on the device in chunks, without freezing the student."""
    import torch

    numerator = torch.zeros((), dtype=torch.float32, device=on_device.device)
    denominator = torch.zeros((), dtype=torch.float32, device=on_device.device)
    with torch.inference_mode():
        for start in range(0, on_device.rows, EVAL_CHUNK):
            rows = torch.arange(start, min(start + EVAL_CHUNK, on_device.rows), device=on_device.device)
            difference = (on_device.probabilities(net, rows) - targets[rows]).abs().sum(dim=1)
            widths = on_device.width[on_device.point_of(rows)].to(torch.float32)
            numerator += (weights[rows] * difference).sum()
            denominator += (weights[rows] * widths).sum()
    return float(numerator / denominator)

def student_probabilities(net: Net, *, device: str) -> np.ndarray:
    """The student's distribution at every row, (rows, MAX_WIDTH) float32, read on `device`: what the curve's error is taken from."""
    import torch

    on_device = _DeviceLayout(torch.device(device))
    net = net.to(on_device.device).eval()
    out = np.empty((on_device.rows, MAX_WIDTH), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, on_device.rows, EVAL_CHUNK):
            rows = torch.arange(start, min(start + EVAL_CHUNK, on_device.rows), device=on_device.device)
            out[start : start + len(rows)] = on_device.probabilities(net, rows).cpu().numpy()
    net.to("cpu")
    return out

# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

# The second word of the fit's seed, "DS" in ASCII, beside the config's seed;
# the Π fit's third word is "PI" and the advantage fits' is the seat.
DISTILL_TAG = 0x4453

Curve = list[dict[str, float]]

def distill(
    targets: np.ndarray,
    weights: np.ndarray,
    config: DistillConfig,
    *,
    log: Callable[[str], None] = print,
) -> tuple[Net, Curve]:
    """A fresh `Net(config.width, config.depth)` fitted to `targets` under `weights`; the net on the cpu in `eval()` mode, and its curve.

    `targets` is `teacher_targets`'s (rows, MAX_WIDTH); `weights` is
    `row_weights`'s (rows,), the measure the error is taken under, and —
    unless the config says otherwise — the one rows are drawn by. The curve
    has one entry per `eval_every` steps and at the last step: the step,
    the mean loss since the previous entry, the learning rate, the seconds
    so far, and `rw_error` over every row; a fraction arm adds
    `rw_error_seen` and `rw_error_unseen`. Refuses, by name, targets of the
    wrong shape, weights of the wrong length, and an `mps` device on a
    machine without one.
    """
    import math
    import time

    import torch
    from torch.nn.utils import clip_grad_norm_

    from drawmaha_solver.deepcfr.fit import _fold_seed
    from drawmaha_solver.deepcfr.net import Net

    rows = int(layout().row_offset[-1])
    targets = np.asarray(targets, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    if targets.shape != (rows, MAX_WIDTH):
        raise ValueError(f"targets must be ({rows:,}, {MAX_WIDTH}), got shape {targets.shape}")
    if weights.shape != (rows,):
        raise ValueError(f"weights must be ({rows:,},), got shape {weights.shape}")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' was asked for, but this machine has no usable Apple GPU")
    device = torch.device(config.device)
    on_device = _DeviceLayout(device)
    device_targets = torch.from_numpy(targets).to(device)
    device_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    sampler = ReachSampler(sampling_weights(weights, config), seed=config.seed)
    seen = training_rows(config, rows)
    seen_mask = None
    if seen is not None:
        seen_mask = np.zeros(rows, dtype=bool)
        seen_mask[seen] = True

    folded = _fold_seed((config.seed, DISTILL_TAG))
    torch.manual_seed(folded)
    net = Net(width=config.width, depth=config.depth).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.lr)
    curve: Curve = []
    started = time.perf_counter()
    running, since = 0.0, 0
    for step in range(1, config.steps + 1):
        lr = 0.5 * config.lr * (1.0 + math.cos(math.pi * (step - 1) / config.steps))
        for group in optimiser.param_groups:
            group["lr"] = lr
        picked = torch.from_numpy(sampler.draw(config.batch)).to(device)
        features, gather = on_device.features_and_gather(picked)
        net.train()
        loss = masked_cross_entropy(net(features), gather, device_targets[picked])
        optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(net.parameters(), config.clip)
        optimiser.step()
        running += loss.item()
        since += 1
        if step % config.eval_every == 0 or step == config.steps:
            net.eval()
            entry = {
                "step": step,
                "loss": running / since,
                "lr": lr,
                "seconds": time.perf_counter() - started,
                "rw_error": _device_error(net, on_device, device_targets, device_weights),
            }
            if seen_mask is not None:
                mask = torch.from_numpy(seen_mask.astype(np.float32)).to(device)
                entry["rw_error_seen"] = _device_error(net, on_device, device_targets, device_weights * mask)
                entry["rw_error_unseen"] = _device_error(
                    net, on_device, device_targets, device_weights * (1.0 - mask)
                )
            curve.append(entry)
            running, since = 0.0, 0
            unseen = f"  unseen {entry['rw_error_unseen']:.5f}" if "rw_error_unseen" in entry else ""
            log(
                f"step {step:>7,}  loss {entry['loss']:.5f}  rw_error {entry['rw_error']:.5f}{unseen}"
                f"  lr {lr:.2e}  {entry['seconds']:.0f} s"
            )
    net = net.to("cpu")
    if not all(torch.isfinite(parameter).all() for parameter in net.parameters()):
        raise ValueError("the fit diverged: a weight is not finite")
    net.eval()
    return net, curve

# ---------------------------------------------------------------------------
# The freeze
# ---------------------------------------------------------------------------

def student_strategy(net: Net, *, teacher: StrategyInfo) -> FrozenStrategy:
    """The student over every key: the softmax of its outputs at the legal columns, as a strategy the exact grader reads.

    The info is the teacher's with `rule="distill"`: what was copied, from
    which run, column, iteration and seed, so the file says what it is.
    """
    from drawmaha_solver.deepcfr.net import predict_by_point

    built = layout()
    flat = np.empty(int(built.column_offset[-1]), dtype=np.float64)
    for p, outputs in predict_by_point(net):
        shifted = np.exp(outputs.astype(np.float64) - outputs.max(axis=1, keepdims=True))
        flat[built.column_offset[p] : built.column_offset[p + 1]] = (
            shifted / shifted.sum(axis=1, keepdims=True)
        ).ravel()
    widths = np.repeat(built.width, np.diff(built.row_offset))
    info = StrategyInfo(
        rule="distill", column=teacher.column, iteration=teacher.iteration,
        seed=teacher.seed, workers=teacher.workers,
    )
    return FrozenStrategy(flat.astype(np.float32), widths, info)
