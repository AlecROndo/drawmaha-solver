"""Convergence analysis for rung 0: does the ledger find the Nash equilibrium?

Runs two experiments and renders the figures that certify rung 0:

1. Self-play (learner vs learner): the average strategy must converge to the
   uniform Nash (1/3, 1/3, 1/3) and its exploitability must fall like
   1/sqrt(T), while the current strategy visibly cycles forever.
2. Learner vs a biased fixed opponent: the average strategy must converge to
   the best response (all paper against a rock-heavy opponent).

"Exploitability" is the one-shot best-response value from `game.py`; every
trajectory is recorded per iteration so the figures can subsample freely.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drawmaha_solver.plotting import CATEGORICAL, MUTED, legend, log_indices, new_axes, save
from drawmaha_solver.regret_matching import RegretMatcher
from drawmaha_solver.rps.game import PAYOFF, Action, best_response_value

# ---------------------------------------------------------------------------
# Series colours
# ---------------------------------------------------------------------------

# Exactly three series, so they take the three validated categorical slots.
SERIES = dict(zip(("rock", "paper", "scissors"), CATEGORICAL))

# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SelfPlayTrajectories:
    """Per-iteration record of one self-play run, seen from player 0."""

    current0: np.ndarray  # (n_iters, 3): strategy in effect each round
    average0: np.ndarray  # (n_iters, 3): running average strategy
    exploitability0: np.ndarray  # (n_iters,): best-response value vs average0
    final_average1: np.ndarray  # (3,): player 1's final average, symmetry check

@dataclass(frozen=True, slots=True)
class VsFixedTrajectories:
    """Per-iteration record of the learner against a fixed opponent."""

    average: np.ndarray  # (n_iters, 3): learner's running average strategy
    payoffs: np.ndarray  # (n_iters,): learner's realized payoff each round

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run both experiments, write the four figures, print the summary."""
    parser = argparse.ArgumentParser(description="Rung-0 convergence analysis")
    parser.add_argument("--iters", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("figures/rung0"))
    args = parser.parse_args()

    self_play = run_self_play(args.iters, seed=args.seed)
    opponent = np.array([0.5, 0.25, 0.25])
    biased = run_vs_fixed(min(args.iters, 20_000), opponent=opponent, seed=args.seed)

    fig_average_strategy(self_play, args.out / "self_play_average_strategy.png")
    fig_current_vs_average(self_play, args.out / "self_play_current_vs_average.png")
    fig_exploitability(self_play, args.out / "self_play_exploitability.png")
    fig_vs_biased(biased, opponent, args.out / "vs_biased_average_strategy.png")

    final_avg = self_play.average0[-1]
    print(f"self-play {args.iters:,} iters:")
    print(f"  p0 average strategy  {np.array2string(final_avg, precision=4)}")
    print(f"  p1 average strategy  {np.array2string(self_play.final_average1, precision=4)}")
    print(f"  exploitability of p0 average  {self_play.exploitability0[-1]:.5f} chips/round")
    print(f"vs fixed {np.array2string(opponent, precision=2)} over {len(biased.payoffs):,} iters:")
    print(f"  average strategy  {np.array2string(biased.average[-1], precision=4)}")
    print(f"  mean payoff {biased.payoffs.mean():+.4f} chips/round (best response earns +0.25)")

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_self_play(n_iters: int, *, seed: int) -> SelfPlayTrajectories:
    """Two regret matchers play each other for n_iters sampled rounds."""
    rng = np.random.default_rng(seed)
    p0, p1 = RegretMatcher(3), RegretMatcher(3)
    current0 = np.empty((n_iters, 3))
    average0 = np.empty((n_iters, 3))
    exploitability0 = np.empty(n_iters)
    for i in range(n_iters):
        s0, s1 = p0.strategy(), p1.strategy()
        a0 = rng.choice(3, p=s0)
        a1 = rng.choice(3, p=s1)
        p0.update(PAYOFF[:, a1])
        p1.update(PAYOFF[:, a0])
        current0[i] = s0
        average0[i] = p0.average_strategy()
        exploitability0[i] = best_response_value(average0[i])
    return SelfPlayTrajectories(
        current0=current0,
        average0=average0,
        exploitability0=exploitability0,
        final_average1=p1.average_strategy(),
    )

def run_vs_fixed(n_iters: int, *, opponent: np.ndarray, seed: int) -> VsFixedTrajectories:
    """The learner against a fixed mixed strategy for n_iters sampled rounds."""
    rng = np.random.default_rng(seed)
    learner = RegretMatcher(3)
    average = np.empty((n_iters, 3))
    payoffs = np.empty(n_iters)
    for i in range(n_iters):
        a0 = rng.choice(3, p=learner.strategy())
        a1 = rng.choice(3, p=opponent)
        learner.update(PAYOFF[:, a1])
        average[i] = learner.average_strategy()
        payoffs[i] = PAYOFF[a0, a1]
    return VsFixedTrajectories(average=average, payoffs=payoffs)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_average_strategy(traj: SelfPlayTrajectories, out: Path) -> None:
    idx = log_indices(len(traj.average0))
    fig, ax = new_axes("Self-play average strategy converges to the uniform Nash (1/3, 1/3, 1/3)")
    ax.axhline(1 / 3, color=MUTED, linewidth=1, linestyle="--")
    ax.text(1.2, 1 / 3 + 0.012, "Nash 1/3", color=MUTED, fontsize=8.5)
    for action in Action:
        name = action.name.lower()
        ax.plot(idx + 1, traj.average0[idx, action], color=SERIES[name], linewidth=1.8, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("average P(action)", color=MUTED, fontsize=9)
    # The early average can spike toward one action before settling; keep the
    # axis just above the observed peak so no series ever clips.
    ax.set_ylim(0, min(1.0, max(0.55, float(traj.average0.max()) + 0.06)))
    legend(ax)
    save(fig, out)

def fig_current_vs_average(traj: SelfPlayTrajectories, out: Path, window: int = 5000) -> None:
    fig, ax = new_axes("The current strategy cycles forever — only the running average converges")
    window = min(window, len(traj.current0))
    t = np.arange(1, window + 1)
    ax.plot(t, traj.current0[:window, Action.ROCK], color=SERIES["rock"], linewidth=0.7, alpha=0.7, label="current P(rock)")
    ax.plot(t, traj.average0[:window, Action.ROCK], color=SERIES["paper"], linewidth=2.2, label="average P(rock)")
    ax.axhline(1 / 3, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("P(rock)", color=MUTED, fontsize=9)
    legend(ax)
    save(fig, out)

def fig_exploitability(traj: SelfPlayTrajectories, out: Path) -> None:
    expl = traj.exploitability0
    # The first iterations' averages are exactly/near uniform, which reads as
    # ~0 exploitability and spikes off the bottom of a log axis — skip them.
    idx = log_indices(len(expl))
    idx = idx[idx >= 9]
    fig, ax = new_axes("Exploitability of the average strategy falls like 1/sqrt(T)")
    anchor = min(100, len(expl))
    c = expl[anchor - 1] * np.sqrt(anchor)
    ax.plot(idx + 1, c / np.sqrt(idx + 1), color=MUTED, linewidth=1, linestyle="--", label="c/sqrt(T) reference")
    ax.plot(idx + 1, expl[idx], color=SERIES["rock"], linewidth=1.8, label="best-response value vs average strategy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("exploitability (chips/round)", color=MUTED, fontsize=9)
    legend(ax)
    save(fig, out)

def fig_vs_biased(traj: VsFixedTrajectories, opponent: np.ndarray, out: Path) -> None:
    idx = log_indices(len(traj.average))
    opp_desc = ", ".join(f"{int(round(p * 100))}% {a.name.lower()}" for a, p in zip(Action, opponent))
    fig, ax = new_axes(f"Against a biased opponent ({opp_desc}), the average converges to the counter: paper")
    for action in Action:
        name = action.name.lower()
        ax.plot(idx + 1, traj.average[idx, action], color=SERIES[name], linewidth=1.8, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("average P(action)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1.05)
    legend(ax)
    save(fig, out)

if __name__ == "__main__":
    main()
