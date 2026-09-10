"""Convergence analysis for rung 2: does CFR solve a game with no closed form?

Runs one solve, recording at log-spaced checkpoints what the answer is worth,
and renders the three figures that certify the rung:

1. Exploitability of the AVERAGE strategy must fall toward zero while the
   CURRENT strategy stays exploitable — the same guarantee rung 1 showed, on
   a game 460 times larger.
2. The game value must approach the sequence-form LP's answer. This replaces
   rung 1's closed-form check: Kuhn's -1/18 could be derived on paper, and
   Leduc's -0.0856 can only be measured by solving the game exactly, so the
   target line is a number rather than a formula.
3. Three strategic facts that no equilibrium of this game can miss must
   emerge from the ledgers: the nuts check-raise rather than lead, a beaten
   hand folds to a raise, and the king opens for a raise. Rung 1 could check
   all twelve of its infosets against a closed form; here 288 spots have no
   formula, so the certificate is poker facts pinned by dominance instead.

Where rung 1 measured the whole solve in seconds, one CFR iteration here
walks 9,450 nodes and costs about 50 ms, so the default is 2,000 iterations
rather than rung 1's 100,000 — roughly two minutes. Exploitability costs a
further 160 ms per call, which is why it is sampled at checkpoints instead of
every iteration.

Everything is deterministic: vanilla CFR enumerates the whole tree and never
samples, so there is no seed, two runs agree bit for bit, and `figures/rung2`
doubles as a regression fingerprint for `cfr.py` — a change that moves any
number shows up as a dirty working tree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drawmaha_solver.leduc.cfr import run_iteration
from drawmaha_solver.leduc.exploitability import expected_value, exploitability
from drawmaha_solver.leduc.game import (
    LP_VALUE_P0,
    Action,
    InfoSet,
    Rank,
)
from drawmaha_solver.leduc.infoset_table import (
    average_strategy,
    current_strategy,
    format_strategy_tables,
    new_infoset_table,
)
from drawmaha_solver.plotting import (
    BASELINE,
    CATEGORICAL,
    MUTED,
    legend,
    new_axes,
    save,
)

C, R = Action.CALL, Action.RAISE
J, K = Rank.JACK, Rank.KING

# ---------------------------------------------------------------------------
# The three spots figure 3 tracks
# ---------------------------------------------------------------------------

# Each is a spot where one action is forced by dominance, so a correct solve
# must find it whatever else it does. They are the same three `test_cfr.py`
# pins, plotted over training so the reader watches them arrive.
#
# The nuts one is the interesting one: a jack on a jack board holds trips and
# still does NOT bet at the round-2 open. It checks, and raises when bet into
# — leading would fold out every hand it beats, while checking induces the
# bluff it wants to punish. That is a real poker idea falling out of regret
# matching, not something anybody encoded.
TRACKED: tuple[tuple[str, InfoSet, Action], ...] = (
    (
        "trips check-raise",
        InfoSet(rank=J, board=J, betting=((C, C), (C, R))),
        Action.RAISE,
    ),
    (
        "beaten jack folds",
        InfoSet(rank=J, board=K, betting=((C, C), (R,))),
        Action.FOLD,
    ),
    (
        "king opens for a raise",
        InfoSet(rank=K, board=None, betting=((),)),
        Action.RAISE,
    ),
)

# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One solve, sampled at log-spaced checkpoints."""

    iterations: np.ndarray  # (k,): iteration count at each checkpoint
    exploitability_average: np.ndarray  # (k,): the answer's distance from Nash
    exploitability_current: np.ndarray  # (k,): the cycling iterate's, for contrast
    game_value: np.ndarray  # (k,): chips per hand to P0, heading for LP_VALUE_P0
    tracked: np.ndarray  # (k, 3): the three dominance-pinned frequencies
    final_average: dict[InfoSet, np.ndarray]  # all 288, at the last checkpoint


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Solve Leduc, write the three figures, print the answer sheet."""
    parser = argparse.ArgumentParser(description="Rung-2 convergence analysis")
    parser.add_argument("--iters", type=int, default=2_000)
    parser.add_argument("--out", type=Path, default=Path("figures/rung2"))
    args = parser.parse_args()

    trajectory = run(args.iters)

    fig_exploitability(trajectory, args.out / "exploitability.png")
    fig_game_value(trajectory, args.out / "game_value.png")
    fig_strategy_convergence(trajectory, args.out / "strategy_convergence.png")

    print(f"\nvanilla CFR, {args.iters:,} iterations\n")
    print(format_strategy_tables(trajectory.final_average))
    print(f"\n  game value to P0               {trajectory.game_value[-1]:+.5f}"
          f"   exact {LP_VALUE_P0:+.5f}")
    print(f"  exploitability, average        "
          f"{trajectory.exploitability_average[-1]:.5f} chips/hand")
    print(f"  exploitability, current        "
          f"{trajectory.exploitability_current[-1]:.5f} chips/hand")
    for (label, _, _), value in zip(TRACKED, trajectory.tracked[-1]):
        print(f"  {label:<30} {value:.3f}")


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


def run(iterations: int, *, checkpoints: int = 40) -> Trajectory:
    """Solve Leduc, measuring the answer at log-spaced checkpoints.

    Both exploitability readings cost a full best-response pass each, so they
    are sampled at checkpoints rather than every iteration. Checkpoints are
    log-spaced because the interesting motion is all early: the curve falls an
    order of magnitude in the first hundred iterations and then crawls.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")

    marks = _checkpoints(iterations, checkpoints)
    table = new_infoset_table()
    expl_average, expl_current, values, tracked = [], [], [], []

    done = 0
    for mark in marks:
        while done < mark:
            run_iteration(table)
            done += 1
        average = average_strategy(table)
        expl_average.append(exploitability(average))
        expl_current.append(exploitability(current_strategy(table)))
        values.append(expected_value(average)[0])
        tracked.append([_probability(average, spot, action)
                        for _, spot, action in TRACKED])

    return Trajectory(
        iterations=marks,
        exploitability_average=np.array(expl_average),
        exploitability_current=np.array(expl_current),
        game_value=np.array(values),
        tracked=np.array(tracked),
        final_average=average,
    )


def _checkpoints(iterations: int, count: int) -> np.ndarray:
    """Log-spaced iteration counts, always including 1 and `iterations`."""
    marks = np.geomspace(1, iterations, count).astype(int)
    return np.unique(np.append(marks, iterations))


def _probability(
    strategies: dict[InfoSet, np.ndarray], spot: InfoSet, action: Action
) -> float:
    """How often `strategies` plays `action` at `spot`.

    Looks the action up by its POSITION in the spot's legal actions, never by
    its enum value: a ledger is 2 or 3 wide, so column 0 means CALL at an
    opening spot and FOLD facing a raise. Indexing by `Action` would read the
    wrong column at every 2-wide spot in this file.
    """
    return float(strategies[spot][spot.legal_actions().index(action)])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_exploitability(trajectory: Trajectory, out: Path) -> None:
    fig, ax = new_axes(
        "Only the average strategy converges — the current one cycles forever"
    )
    t = trajectory.iterations
    # Anchor the 1/sqrt(T) guide mid-run rather than at the first checkpoint,
    # whose near-uniform average sits far off the eventual trend.
    anchor = len(t) // 3
    guide = trajectory.exploitability_average[anchor] * np.sqrt(t[anchor]) / np.sqrt(t)
    ax.plot(t, guide, color=MUTED, linewidth=1, linestyle="--",
            label="c/sqrt(T) reference")
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


def fig_game_value(trajectory: Trajectory, out: Path) -> None:
    # The title carries the measured gap rather than claiming the line lands
    # on the target: at 2,000 iterations it is still a few thousandths short
    # (0.0018 as last measured), and a figure that said "lands on" would be
    # overstating its own evidence.
    gap = abs(trajectory.game_value[-1] - LP_VALUE_P0)
    fig, ax = new_axes(
        f"{trajectory.iterations[-1]:,} iterations bring the game value within "
        f"{gap:.4f} of the exact answer"
    )
    ax.plot(trajectory.iterations, trajectory.game_value, color=CATEGORICAL[0],
            linewidth=2.0, label="solved game value to P0")
    # Not a second series: the target the solved line has to reach. Drawn as a
    # rule rather than a curve because it does not depend on the x axis.
    ax.axhline(LP_VALUE_P0, color=BASELINE, linewidth=1.2, linestyle="--",
               label=f"sequence-form LP  ({LP_VALUE_P0:+.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("CFR iterations", color=MUTED, fontsize=9)
    ax.set_ylabel("chips per hand to P0", color=MUTED, fontsize=9)
    legend(ax, loc="lower right")
    save(fig, out)


def fig_strategy_convergence(trajectory: Trajectory, out: Path) -> None:
    fig, ax = new_axes(
        "Three plays no equilibrium can miss, arriving out of the regret ledgers"
    )
    for i, (label, _, _) in enumerate(TRACKED):
        ax.plot(trajectory.iterations, trajectory.tracked[:, i],
                color=CATEGORICAL[i], linewidth=1.8, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("CFR iterations", color=MUTED, fontsize=9)
    ax.set_ylabel("frequency in the average strategy", color=MUTED, fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    legend(ax, loc="center right")
    save(fig, out)


if __name__ == "__main__":
    main()
