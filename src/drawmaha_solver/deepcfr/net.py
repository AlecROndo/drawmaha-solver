"""The net: `136 -> 64 -> 64 -> 64 -> 7`, chunked prediction, and its save file.

A plain MLP. Its input is one infoset's 136 features (`features.encode`,
or a batch of rows from `Layout.features`); its output is 7 signed numbers,
one per `Action` in `int(Action)` order, so an output column means the
same thing at every spot in the game. A spot's ledger reads 2, 3 or 4 of
them through `features.legal_gather()`; the other columns are trained on
nothing and read by nobody.

**Why the head is linear.** The outputs stand in for cumulative regrets,
which are signed — a negative one says "this action has done worse than
the mix" and `materialise` needs the sign to tell eq. (3)'s clauses apart.
A ReLU or a softmax on the head would throw that away.

**Width and depth are parameters**, not constants, because the spec's size
sweep (3 x 64, 4 x 64, 3 x 128) must run through one trainer. The default
is the spec's 3 x 64: 17,543 parameters, which the test pins.

**Prediction is chunked** under `torch.inference_mode()`. `materialise`
runs the net over 100,400-row shapes; a chunk of 262,144 keeps one forward
pass to a few tens of megabytes and builds no autograd graph. The result is
float32 NumPy, which is what the table's writer gathers from.

**The whole game is 141 forward passes.** The table's rows are laid out
point by point, each point's rows a contiguous slice of the layout's
private half beside one public row, and each point's legal actions are a
fixed set of head columns (`features.legal_gather()`). So reading the net
over every key is one batched pass per public point, gathered at that
point's columns — `predict_by_point`. `materialise` writes those outputs
into the regret slots, Π's freeze clips and normalises them, and the
distillation freeze takes their softmax; the loop is here so the three
cannot drift apart. No key is ever built.

**The save file** is one `torch.save` of the size and the weights, read
back with `weights_only=True` so a checkpoint cannot execute code on load.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn

from drawmaha_solver.minidrawmaha.rules.features import (
    FEATURE_WIDTH,
    HEAD_WIDTH,
    layout,
    legal_gather,
)

# Rows per forward pass in `predict`; a run parameter only in the sense
# that a test shrinks it to force several chunks.
PREDICT_CHUNK = 262_144

# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """`depth` hidden layers of `width` units, ReLU after each, then a linear head of 7.

    Keyword-only sizes: `Net(64, 3)` and `Net(3, 64)` would both build, and
    only one of them is the net the spec describes.
    """

    def __init__(self, *, width: int = 64, depth: int = 3) -> None:
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        super().__init__()
        self.width = width
        self.depth = depth
        layers: list[nn.Module] = []
        fan_in = FEATURE_WIDTH
        for _ in range(depth):
            layers += [nn.Linear(fan_in, width), nn.ReLU()]
            fan_in = width
        layers.append(nn.Linear(fan_in, HEAD_WIDTH))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(n, 136) float32 -> (n, 7) float32, signed."""
        return self.layers(x)

# ---------------------------------------------------------------------------
# Prediction into NumPy
# ---------------------------------------------------------------------------

def predict(net: Net, features: np.ndarray) -> np.ndarray:
    """The net's 7 outputs for every row of `features`; (n, 7) float32.

    Refuses anything but an (n, 136) input by name: a transposed or
    half-built batch would otherwise surface as a matmul shape error deep
    in torch.
    """
    if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
        raise ValueError(
            f"predict takes an (n, {FEATURE_WIDTH}) feature batch, got shape {features.shape}"
        )
    net.eval()
    out = np.empty((len(features), HEAD_WIDTH), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(features), PREDICT_CHUNK):
            chunk = np.ascontiguousarray(features[start : start + PREDICT_CHUNK], dtype=np.float32)
            out[start : start + len(chunk)] = net(torch.from_numpy(chunk)).numpy()
    return out

# ---------------------------------------------------------------------------
# The whole game, one public point at a time
# ---------------------------------------------------------------------------

def predict_by_point(
    net: Net, points: Iterable[int] | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Per public point in walk order: `(p, outputs)`, outputs (rows_p, width_p) float32 at the point's legal columns.

    Row k of a point's outputs is the key at position k of the point's
    shape of `private_keys`, the order the whole-game table keeps, so a
    caller writes or reads `layout().column_offset[p]` onward with one
    `ravel()`. Column j is the net's head at `legal_gather()[p, j]`: the
    j-th legal action of every key on the point, in ledger-column order.
    `points` restricts the pass to those point indices, in the order given
    (`materialise` passes one seat's points, so a net is never run over
    the other seat's keys); the default is every point.
    """
    built = layout()
    gather = legal_gather()
    for p in range(len(built.points)) if points is None else points:
        rows = built.row_offset[p + 1] - built.row_offset[p]
        width = built.width[p]
        private = built.private[built.private_offset[p] : built.private_offset[p] + rows]
        public = np.broadcast_to(built.public[p], (rows, built.public.shape[1]))
        outputs = predict(net, np.concatenate([private, public], axis=1))
        yield p, outputs[:, gather[p, :width]]

# ---------------------------------------------------------------------------
# The save file
# ---------------------------------------------------------------------------

def save_net(net: Net, path: Path) -> None:
    """One file holding the size and the weights, enough to rebuild the net alone."""
    torch.save({"width": net.width, "depth": net.depth, "state_dict": net.state_dict()}, path)

def load_net(path: Path) -> Net:
    """The net `save_net` wrote, in `eval()` mode; weights only, so the file runs no code."""
    saved = torch.load(path, weights_only=True)
    net = Net(width=saved["width"], depth=saved["depth"])
    net.load_state_dict(saved["state_dict"])
    net.eval()
    return net
