"""Validation grid for the rung0-rps writeup — house style (see skill assets)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

os.makedirs("figures", exist_ok=True)

BLUE  = "#134A6B"
INK   = "#1A1A1A"
GRAY  = "#667079"

_sans = next((f for f in ["Helvetica", "Arial", "DejaVu Sans"]
              if any(fn.name == f for fn in font_manager.fontManager.ttflist)),
             "DejaVu Sans")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "sans-serif", "font.sans-serif": [_sans],
    "text.color": INK, "axes.titlecolor": BLUE,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

PASS_FILL, PASS_EDGE, PASS_TXT = "#DBEDE7", "#2C6E63", "#245A50"
FAIL_FILL, FAIL_EDGE, FAIL_TXT = "#F6E4D9", "#B4623C", "#8F4B2E"
SKIP_FILL, SKIP_EDGE, SKIP_TXT = "#EEF1F4", "#C9CED3", "#667079"
_STATUS = {
    "pass": (PASS_FILL, PASS_EDGE, PASS_TXT),
    "fail": (FAIL_FILL, FAIL_EDGE, FAIL_TXT),
    "skip": (SKIP_FILL, SKIP_EDGE, SKIP_TXT),
}

def save(fig, name):
    fig.savefig(f"figures/{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote figures/{name}.png")

def validation_grid(tiles, ncols=3, title="Validation: all 38 tests pass (run locally; the repo has no CI)"):
    n = len(tiles)
    nrows = (n + ncols - 1) // ncols
    fig, ax = plt.subplots(figsize=(2.7 * ncols, 1.5 * nrows))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.axis("off")
    pad = 0.05
    for i, t in enumerate(tiles):
        r, c = divmod(i, ncols)
        fill, edge, txt = _STATUS.get(t.get("status", "skip"), _STATUS["skip"])
        ax.add_patch(plt.Rectangle(
            (c + pad, r + pad), 1 - 2 * pad, 1 - 2 * pad,
            facecolor=fill, edgecolor=edge, linewidth=1.6,
            joinstyle="round", zorder=1))
        cx = c + 0.5
        ax.text(cx, r + 0.36, t["name"], ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=txt, zorder=2, wrap=True)
        ax.text(cx, r + 0.62, t.get("detail", ""), ha="center", va="center",
                fontsize=8.2, color=txt, zorder=2)
    if title:
        ax.set_title(title, pad=10, color=BLUE)
    save(fig, "validation_grid")

if __name__ == "__main__":
    validation_grid([
        {"name": "RPS rules are right", "detail": "test_game.py — 14/14", "status": "pass"},
        {"name": "ledger math is exact", "detail": "unit tests — 7/7", "status": "pass"},
        {"name": "self-play finds the Nash", "detail": "convergence — 3/3 @ 50k iters", "status": "pass"},
        {"name": "experiments reproduce", "detail": "test_analysis.py — 3/3", "status": "pass"},
        {"name": "input handling works", "detail": "test_players.py — 11/11", "status": "pass"},
        {"name": "continuous integration", "detail": "none configured — local only", "status": "skip"},
    ])
