"""The distillation figures, drawn from `figures/rung4/distill/results.json`.

    uv run python scripts/deepcfr/distill_figures.py \\
        figures/rung4/distill/results.json figures/rung4/distill

That wrote `by_size.png`, `gaps.png`, `curves.png` and `fraction.png` beside
the results file on 2026-11-05. Nothing here is typed in: every number is
read from the file `scripts/deepcfr/distill_teacher.py` wrote.

**`by_size`** — exploitability against parameter count, log on both axes:
one dot per distilled copy, the teacher as the floor line every copy sits
above, Deep CFR's exact average and Π as the lines the copies are read
against. The gap from a dot down to the teacher is what that net cannot
hold; the gap from Deep CFR's line down to the dot at its own size is what
its training lost.

**`gaps`** — the three gaps at the headline size on one log axis: teacher,
its 4 × 256 copy, Deep CFR's 4 × 256 exact average, with each gap's size
and its share of the whole plateau written on the bracket.

**`curves`** — two panels on one step axis: the training loss above, the
reach-weighted error against the teacher below, one line per size arm,
the exact grade written at the end of each line. The lower panel is the
quantity that tracks the grade (`pi_reach_weighted.py`); the upper is what
the optimiser saw.

**`fraction`** — the generalisation arms: the grade against the share of
rows trained on, with the error on the seen and the unseen rows beside it
on one axis.

Colours are the validated categorical palette (blue, orange, aqua, yellow,
violet in fixed order, each size arm keeping its slot in every figure);
references are ink and grey. One axis per panel, thin marks, direct labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1a1a1a"
MUTED = "#667079"
RULE = "#c9ced3"
SURFACE = "#ffffff"
# The validated categorical order; a size arm keeps its slot in every figure.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6")
SIZE_ARMS = ("3x64", "3x128", "4x256", "4x512", "4x1024")
COPY = SLOTS[0]
DEEP = SLOTS[1]
PI = SLOTS[2]

def size_label(arm: str) -> str:
    depth, width = arm.split("-")[0].split("x")
    return f"{depth} × {width}"

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, which="major", color=RULE, linewidth=0.6, alpha=0.7)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)

def _plain_ticks(ax, axis: str, values, labels=None) -> None:
    """Log-axis ticks as plain numbers (0.02, 51k), never 2 × 10⁻²."""
    import matplotlib.ticker as ticker

    which = ax.xaxis if axis == "x" else ax.yaxis
    which.set_major_locator(ticker.FixedLocator(values))
    which.set_major_formatter(ticker.FixedFormatter(labels or [str(v) for v in values]))
    which.set_minor_locator(ticker.NullLocator())

def _thousands(x: float) -> str:
    return f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"

def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# by_size
# ---------------------------------------------------------------------------

def by_size(results: dict, out: Path) -> None:
    arms = results["arms"]
    sizes = [arm for arm in SIZE_ARMS if arm in arms]
    teacher = results["teacher"]
    deep = results["deep_cfr"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = [arms[arm]["parameters"] for arm in sizes]
    ys = [arms[arm]["exploitability"] for arm in sizes]
    ax.plot(xs, ys, color=COPY, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label="distilled copy of the table", zorder=4)
    for arm, x, y in zip(sizes, xs, ys, strict=True):
        ax.annotate(f"{size_label(arm)}\n{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(teacher["exploitability"], color=INK, linewidth=1.4, linestyle="--", label=f"teacher: LCFR table at {teacher['hands_per_seat'] / 1e6:.0f}M hands", zorder=3)
    ax.axhline(deep["exact"]["exploitability"], color=DEEP, linewidth=1.4, linestyle="-", label=f"Deep CFR {deep['depth']} × {deep['width']}, exact average at {deep['hands_per_seat'] / 1e6:.0f}M hands", zorder=3)
    ax.axhline(deep["pi"]["exploitability"], color=PI, linewidth=1.4, linestyle=":", label="Deep CFR, policy net Π", zorder=3)
    ax.plot([deep["parameters"]], [deep["exact"]["exploitability"]], marker="s", color=DEEP, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
    ax.annotate("Deep CFR, same net", (deep["parameters"], deep["exact"]["exploitability"]), textcoords="offset points", xytext=(0, 8), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.margins(x=0.12, y=0.30)
    _plain_ticks(ax, "x", xs + [deep["parameters"]] if deep["parameters"] not in xs else xs, [_thousands(x) for x in (xs + [deep["parameters"]] if deep["parameters"] not in xs else xs)])
    _plain_ticks(ax, "y", [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3])
    ax.set_xlabel("parameters in the net (log)", color=MUTED)
    ax.set_ylabel("exploitability, chips per hand (log)", color=MUTED)
    ax.set_title("Each copy of the table sits between the teacher and Deep CFR; the floor falls with size", loc="left", fontsize=10.5, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _save(fig, out / "by_size.png")

# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------

def gaps(results: dict, out: Path, headline: str = "4x256") -> None:
    teacher = results["teacher"]["exploitability"]
    copy = results["arms"][headline]["exploitability"]
    deep = results["deep_cfr"]["exact"]["exploitability"]
    whole = deep - teacher
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    _style(ax)
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(0, deep * 1.18)
    # A linear axis: the two gaps add up to the plateau, and the picture
    # must say so; a log axis would stretch the lower one.
    marks = [
        (teacher, INK, f"teacher, the LCFR table   {teacher:.4f}"),
        (copy, COPY, f"its {size_label(headline)} copy   {copy:.4f}"),
        (deep, DEEP, f"Deep CFR {size_label(headline)}   {deep:.4f}"),
    ]
    for value, colour, label in marks:
        ax.hlines(value, 0.04, 0.34, color=colour, linewidth=2.4)
        ax.annotate(label, (0.34, value), textcoords="offset points", xytext=(8, 0), va="center", fontsize=9.5, color=INK)
    def bracket(x, low, high, text, colour):
        ax.annotate("", (x, high), (x, low), arrowprops=dict(arrowstyle="<->", color=colour, linewidth=1.3, shrinkA=0, shrinkB=0))
        ax.annotate(text, (x, (low + high) / 2), textcoords="offset points", xytext=(8, 0), ha="left", va="center", fontsize=9, color=INK)
    bracket(0.72, teacher, copy, f"representation: what the net cannot hold\n+{copy - teacher:.4f}, {(copy - teacher) / whole:.0%} of the plateau", COPY)
    bracket(0.72, copy, deep, f"training: what Deep CFR did not find\n+{deep - copy:.4f}, {(deep - copy) / whole:.0%}", DEEP)
    ax.set_yticks([0, teacher, copy, deep])
    ax.set_yticklabels(["0", f"{teacher:.4f}", f"{copy:.4f}", f"{deep:.4f}"])
    ax.set_ylabel("exploitability, chips per hand", color=MUTED)
    ax.set_title(f"The {size_label(headline)} plateau over the table, split into its two parts", loc="left", fontsize=10.5, color=INK, pad=10)
    _save(fig, out / "gaps.png")

# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------

def curves(results: dict, out: Path) -> None:
    arms = results["arms"]
    sizes = [arm for arm in SIZE_ARMS if arm in arms]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax in (top, bottom):
        _style(ax)
    for arm, colour in zip(SIZE_ARMS, SLOTS, strict=True):
        if arm not in arms:
            continue
        curve = arms[arm]["curve"]
        steps = [c["step"] for c in curve]
        top.plot(steps, [c["loss"] for c in curve], color=colour, linewidth=1.8, label=size_label(arm))
        bottom.plot(steps, [c["rw_error"] for c in curve], color=colour, linewidth=1.8)
        bottom.annotate(f"{size_label(arm)}  graded {arms[arm]['exploitability']:.4f}", (steps[-1], curve[-1]["rw_error"]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8.5, color=INK)
    top.set_yscale("log")
    bottom.set_yscale("log")
    _plain_ticks(top, "y", [0.1, 0.15, 0.2, 0.3, 0.4])
    _plain_ticks(bottom, "y", [0.001, 0.003, 0.01, 0.03, 0.1])
    most = max(c["step"] for arm in sizes for c in arms[arm]["curve"])
    ticks = [t for t in range(0, most + 1, 20_000)]
    _plain_ticks(bottom, "x", ticks, [_thousands(t) if t else "0" for t in ticks])
    top.set_ylabel("cross-entropy per row (log)", color=MUTED)
    bottom.set_ylabel("reach-weighted error vs teacher (log)", color=MUTED)
    bottom.set_xlabel("Adam steps of 16,384 rows", color=MUTED)
    top.set_title("Loss and reach-weighted error fall together; the bigger net ends lower and grades lower", loc="left", fontsize=10.5, color=INK, pad=10)
    top.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=len(sizes))
    bottom.set_xlim(0, most * 1.3)
    _save(fig, out / "curves.png")

# ---------------------------------------------------------------------------
# fraction
# ---------------------------------------------------------------------------

def fraction(results: dict, out: Path, headline: str = "4x256") -> None:
    arms = results["arms"]
    names = sorted((arm for arm in arms if arm.startswith(headline + "-f")), key=lambda a: arms[a]["train_fraction"])
    names.append(headline)
    xs = [arms[arm]["train_fraction"] for arm in names]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax in (top, bottom):
        _style(ax)
    top.plot(xs, [arms[a]["exploitability"] for a in names], color=COPY, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for x, arm in zip(xs, names, strict=True):
        top.annotate(f"{arms[arm]['exploitability']:.4f}", (x, arms[arm]["exploitability"]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
    top.axhline(results["teacher"]["exploitability"], color=INK, linewidth=1.2, linestyle="--", label="teacher")
    top.axhline(results["deep_cfr"]["exact"]["exploitability"], color=DEEP, linewidth=1.2, label="Deep CFR exact average")
    top.set_ylabel("exploitability of the copy", color=MUTED)
    top.legend(frameon=False, fontsize=8.5, loc="upper right")
    seen = [arms[a].get("rw_error_seen", arms[a]["rw_error"]) for a in names]
    unseen = [arms[a].get("rw_error_unseen") for a in names]
    bottom.plot(xs, [arms[a]["rw_error"] for a in names], color=COPY, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label="all rows")
    bottom.plot(xs[:-1], seen[:-1], color=PI, linewidth=1.6, marker="o", markersize=6, markeredgecolor=SURFACE, label="rows trained on")
    bottom.plot(xs[:-1], unseen[:-1], color=DEEP, linewidth=1.6, marker="o", markersize=6, markeredgecolor=SURFACE, label="rows never seen")
    bottom.set_ylabel("reach-weighted error vs teacher", color=MUTED)
    bottom.set_xlabel("share of the 6.2M infosets trained on", color=MUTED)
    bottom.set_xscale("log")
    bottom.set_xticks(xs)
    bottom.set_xticklabels([f"{x:.0%}" for x in xs])
    bottom.legend(frameon=False, fontsize=8.5, loc="upper right")
    top.set_title(f"A {size_label(headline)} copy trained on a share of the rows, graded on all of them", loc="left", fontsize=10.5, color=INK)
    _save(fig, out / "fraction.png")

# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    results_path = Path(argv[0]) if argv else Path("figures/rung4/distill/results.json")
    out = Path(argv[1]) if len(argv) > 1 else results_path.parent
    results = json.loads(results_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    by_size(results, out)
    curves(results, out)
    if "4x256" in results["arms"]:
        gaps(results, out)
        if any(arm.startswith("4x256-f") for arm in results["arms"]):
            fraction(results, out)
    print(f"wrote {sorted(p.name for p in out.glob('*.png'))} to {out.resolve()}")

if __name__ == "__main__":
    main()
