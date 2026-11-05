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
gradient. At the minimum the softmax IS the teacher's row and the loss is
the row's entropy.

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
quantity that tracks the grade, not only of the loss.

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
    """

    width: int = 256
    depth: int = 4
    steps: int = 100_000
    batch: int = 16_384
    lr: float = 1e-3
    clip: float = 1.0
    seed: int = 0
    device: str = "mps"
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
