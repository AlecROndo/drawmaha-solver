"""Shared figure chrome: the palette and the axis styling every rung's plots use.

Extracted when rung 1 needed the same look as rung 0 — one house style for the
whole ladder, so a reader can compare an RPS convergence curve against a Kuhn
one without re-reading the axes.

The three categorical slots are validated all-pairs for up to three series in
light mode; a plot needing a fourth series wants a different encoding, not a
fourth colour. Everything else here is neutral ink, chosen so the data is the
only saturated thing on the page.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# ---------------------------------------------------------------------------
# Axis scaffolding
# ---------------------------------------------------------------------------

def new_axes(title: str, *, figsize: tuple[float, float] = (8, 4.5)) -> tuple[plt.Figure, plt.Axes]:
    """A styled figure and axes: left-aligned title, y-grid, two spines."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)
    return fig, ax

def legend(ax: plt.Axes, **kwargs) -> None:
    """Frameless legend in the secondary ink."""
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in leg.get_texts():
        text.set_color(SECONDARY)

def save(fig: plt.Figure, path: Path) -> None:
    """Write the figure and print its absolute path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.resolve()}")

def log_indices(n: int, k: int = 500) -> np.ndarray:
    """Up to k log-spaced 0-based indices into a length-n trajectory."""
    return np.unique(np.geomspace(1, n, k).astype(int)) - 1
