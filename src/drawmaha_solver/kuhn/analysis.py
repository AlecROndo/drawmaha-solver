"""Convergence analysis for rung 1: does CFR find Kuhn's known equilibrium?

Runs one solve, recording at log-spaced checkpoints what the two candidate
answers are worth, and renders the three figures that certify the rung:

1. Exploitability of the AVERAGE strategy must fall like 1/sqrt(T) toward
   zero, while the CURRENT strategy stays exploitable forever — the visual
   form of "CFR's guarantee is about the average, not the last iterate".
2. The opening bet frequencies must settle with the king betting three times
   as often as the jack bluffs. Alpha itself is free in [0, 1/3]; the ratio
   is not, so the figure plots 3x the jack's frequency as a target line.
3. The solved strategy must match Kuhn's closed form at whatever alpha it
   happened to find, at all twelve infosets.

Everything is deterministic: vanilla CFR enumerates the whole tree and never
samples, so there is no seed, two runs agree bit for bit, and `figures/rung1`
doubles as a regression fingerprint for `cfr.py` — a change that moves any
number shows up as a dirty working tree.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drawmaha_solver.kuhn.cfr import run_iteration
from drawmaha_solver.kuhn.exploitability import expected_value, exploitability
from drawmaha_solver.kuhn.game import (
    ACTION_SYMBOL,
    CARD_SYMBOL,
    DECISION_HISTORIES,
    DECK,
    Action,
    Card,
    InfoSet,
    action_label,
    all_infosets,
)
from drawmaha_solver.kuhn.infoset_table import (
    average_strategy,
    current_strategy,
    format_strategy_grid,
    new_infoset_table,
)
from drawmaha_solver.plotting import (
    CATEGORICAL,
    GRID,
    MUTED,
    SECONDARY,
    legend,
    new_axes,
    save,
)

# The three cards are the natural three-series encoding for every figure here,
# so they take the three validated categorical slots and keep their colour
# across plots.
CARD_COLOUR = dict(zip(DECK, CATEGORICAL))

# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Trajectory:
    """One solve, sampled at log-spaced checkpoints."""

    iterations: np.ndarray  # (k,): iteration count at each checkpoint
    exploitability_average: np.ndarray  # (k,): the answer's distance from Nash
    exploitability_current: np.ndarray  # (k,): the cycling iterate's, for contrast
    bet: np.ndarray  # (k, 12): average P(BET) everywhere, in all_infosets order
    final_average: dict[InfoSet, np.ndarray]  # all twelve, at the last checkpoint
    game_value: float  # to P0 under the final average; Kuhn's is -1/18

    @property
    def opening(self) -> np.ndarray:
        """The (k, 3) slice for the J/Q/K opening infosets.

        Selected by key rather than by assuming where `all_infosets` puts the
        root row, so the column order cannot silently drift.
        """
        spots = all_infosets()
        return self.bet[:, [spots.index(InfoSet(card, ())) for card in DECK]]

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Solve Kuhn, write the three figures, print the answer sheet."""
    parser = argparse.ArgumentParser(description="Rung-1 convergence analysis")
    parser.add_argument("--iters", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=Path("figures/rung1"))
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the trajectory here, for the rung-1 visualizer",
    )
    args = parser.parse_args()

    trajectory = run(args.iters)

    fig_exploitability(trajectory, args.out / "exploitability.png")
    fig_strategy_convergence(trajectory, args.out / "strategy_convergence.png")
    fig_answer_sheet(trajectory, args.out / "answer_sheet.png")
    if args.json is not None:
        write_json(trajectory, args.json)

    alpha = trajectory.opening[-1, Card.JACK]
    print(f"\nvanilla CFR, {args.iters:,} iterations\n")
    print(format_strategy_grid(trajectory.final_average))
    print(f"\n  alpha (jack's opening bluff)   {alpha:.4f}   family is [0, 1/3]")
    print(f"  king's opening bet             {trajectory.opening[-1, Card.KING]:.4f}"
          f"   closed form 3*alpha = {3 * alpha:.4f}")
    print(f"  game value to P0               {trajectory.game_value:+.5f}"
          f"   closed form {-1 / 18:+.5f}")
    print(f"  exploitability, average        {trajectory.exploitability_average[-1]:.5f} chips/hand")
    print(f"  exploitability, current        {trajectory.exploitability_current[-1]:.5f} chips/hand")

# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run(iterations: int, *, checkpoints: int = 60) -> Trajectory:
    """Solve Kuhn, measuring both candidate answers at log-spaced checkpoints.

    Exploitability is exact but costs 128 tree walks per call, so it is
    sampled at checkpoints rather than every iteration — unlike rung 0, where
    the meter was a max over three numbers and could be taken every round.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")

    marks = _checkpoints(iterations, checkpoints)
    spots = all_infosets()
    table = new_infoset_table()
    expl_average, expl_current, bet = [], [], []

    done = 0
    for mark in marks:
        while done < mark:
            run_iteration(table)
            done += 1
        average = average_strategy(table)
        expl_average.append(exploitability(average))
        expl_current.append(exploitability(current_strategy(table)))
        bet.append([average[spot][Action.BET] for spot in spots])

    return Trajectory(
        iterations=marks,
        exploitability_average=np.array(expl_average),
        exploitability_current=np.array(expl_current),
        bet=np.array(bet),
        final_average=average,
        game_value=expected_value(average)[0],
    )

def _checkpoints(iterations: int, count: int) -> np.ndarray:
    """Log-spaced iteration counts, always including 1 and `iterations`."""
    marks = np.geomspace(1, iterations, count).astype(int)
    return np.unique(np.append(marks, iterations))

# ---------------------------------------------------------------------------
# Export for the visualizer
# ---------------------------------------------------------------------------

def to_json(trajectory: Trajectory) -> dict:
    """The trajectory as plain JSON, keyed by infoset label ("K", "Jpb", ...).

    The rung-1 visualizer renders this rather than re-implementing CFR in
    TypeScript. Rung 0 could safely port its 63-line ledger to the browser;
    porting the reach-weighted tree walk would make a second copy of the one
    routine where a swapped weight converges silently to the wrong answer. So
    the page shows numbers this solver produced, under the same tests.

    Labels are `str(InfoSet)` — the notation the literature uses.
    """
    spots = all_infosets()
    alpha = float(trajectory.opening[-1, Card.JACK])
    return {
        "iterations": [int(t) for t in trajectory.iterations],
        "exploitabilityAverage": [float(x) for x in trajectory.exploitability_average],
        "exploitabilityCurrent": [float(x) for x in trajectory.exploitability_current],
        "bet": {
            str(spot): [float(row[i]) for row in trajectory.bet]
            for i, spot in enumerate(spots)
        },
        # Instantiated at the alpha this run found, since the equilibrium is a
        # family and comparing against any other member would read as an error.
        "closedForm": {str(spot): _closed_form(spot, alpha) for spot in spots},
        "alpha": alpha,
        "gameValue": float(trajectory.game_value),
        "gameValueExact": -1 / 18,
    }

def write_json(trajectory: Trajectory, path: Path) -> None:
    """Write `to_json` to `path`, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(trajectory), indent=1) + "\n")
    print(f"wrote {path.resolve()}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_exploitability(trajectory: Trajectory, out: Path) -> None:
    fig, ax = new_axes(
        "Only the average strategy converges — the current one cycles forever"
    )
    t = trajectory.iterations
    # Anchor the 1/sqrt(T) guide to a mid-run point rather than the first, whose
    # near-uniform average sits far off the eventual trend.
    anchor = len(t) // 3
    guide = trajectory.exploitability_average[anchor] * np.sqrt(t[anchor]) / np.sqrt(t)
    ax.plot(t, guide, color=MUTED, linewidth=1, linestyle="--", label="c/sqrt(T) reference")
    ax.plot(t, trajectory.exploitability_current, color=CATEGORICAL[1], linewidth=1.4,
            label="current strategy")
    ax.plot(t, trajectory.exploitability_average, color=CATEGORICAL[0], linewidth=2.0,
            label="average strategy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CFR iterations", color=MUTED, fontsize=9)
    ax.set_ylabel("exploitability (chips/hand, antes of 1)", color=MUTED, fontsize=9)
    legend(ax, loc="lower left")
    save(fig, out)

def fig_strategy_convergence(trajectory: Trajectory, out: Path) -> None:
    fig, ax = new_axes(
        "The jack's bluff locks to one third of the king's value bet, wherever alpha lands"
    )
    t = trajectory.iterations
    for card in DECK:
        ax.plot(t, trajectory.opening[:, card], color=CARD_COLOUR[card], linewidth=1.8,
                label=f"{CARD_SYMBOL[card]} opening bet")
    # Not a fourth series: the target the king's own line has to sit on.
    ax.plot(t, 3 * trajectory.opening[:, Card.JACK], color=MUTED, linewidth=1,
            linestyle="--", label="3 x jack's frequency")
    ax.set_xscale("log")
    ax.set_xlabel("CFR iterations", color=MUTED, fontsize=9)
    ax.set_ylabel("average P(bet) at the opening", color=MUTED, fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    legend(ax)
    save(fig, out)

def fig_answer_sheet(trajectory: Trajectory, out: Path) -> None:
    alpha = float(trajectory.opening[-1, Card.JACK])
    spots = [InfoSet(card, history) for history in DECISION_HISTORIES for card in DECK]
    solved = [trajectory.final_average[spot][Action.BET] for spot in spots]
    exact = [_closed_form(spot, alpha) for spot in spots]

    fig, ax = new_axes(
        f"All twelve infosets match Kuhn's closed form at the alpha the solver found ({alpha:.3f})",
        figsize=(8, 5.4),
    )
    y = np.arange(len(spots))
    ax.barh(y + 0.19, exact, height=0.34, color=MUTED, label="closed form")
    ax.barh(y - 0.19, solved, height=0.34, color=CATEGORICAL[0], label="solved")
    # Four spots are pure PASS, so both bars have zero length and the row would
    # otherwise read as missing data rather than as an exact agreement at 0.
    for row, value in zip(y, solved):
        ax.text(value + 0.015, row, f"{value:.3f}", va="center", fontsize=8,
                color=SECONDARY)
    ax.set_yticks(y, [_spot_label(spot) for spot in spots], fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("P(bet), which reads as P(call) where the spot faces a bet",
                  color=MUTED, fontsize=9)
    ax.set_xlim(0, 1.12)
    # Below the axes: every horizontal band already carries a bar, so an inset
    # legend lands on data wherever it goes.
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    save(fig, out)

def _spot_label(spot: InfoSet) -> str:
    """e.g. 'Kpb  call' — the infoset plus what BET means there."""
    path = "".join(ACTION_SYMBOL[a] for a in spot.history) or "-"
    return f"{CARD_SYMBOL[spot.card]}{path:<3} {action_label(Action.BET, spot.history)}"

def _closed_form(spot: InfoSet, alpha: float) -> float:
    """Kuhn's equilibrium P(BET) at `spot`, for the given member of the family."""
    exact = {
        (Card.JACK, ()): alpha,
        (Card.QUEEN, ()): 0.0,
        (Card.KING, ()): 3 * alpha,
        (Card.JACK, (Action.PASS,)): 1 / 3,
        (Card.QUEEN, (Action.PASS,)): 0.0,
        (Card.KING, (Action.PASS,)): 1.0,
        (Card.JACK, (Action.BET,)): 0.0,
        (Card.QUEEN, (Action.BET,)): 1 / 3,
        (Card.KING, (Action.BET,)): 1.0,
        (Card.JACK, (Action.PASS, Action.BET)): 0.0,
        (Card.QUEEN, (Action.PASS, Action.BET)): alpha + 1 / 3,
        (Card.KING, (Action.PASS, Action.BET)): 1.0,
    }
    return exact[(spot.card, spot.history)]

if __name__ == "__main__":
    main()
