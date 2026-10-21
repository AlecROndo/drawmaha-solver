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

**The save file** is one `torch.save` of the size and the weights, read
back with `weights_only=True` so a checkpoint cannot execute code on load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from drawmaha_solver.minidrawmaha.rules.features import FEATURE_WIDTH, HEAD_WIDTH

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
