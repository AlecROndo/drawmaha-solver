"""Player interface, concrete players, and the match runner."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from drawmaha_solver.rps.game import N_ACTIONS, PAYOFF, Action
from drawmaha_solver.rps.regret_matching import RegretMatcher

class QuitGame(Exception):
    """Raised by a human player who wants to stop."""

class Player(ABC):
    @abstractmethod
    def act(self, rng: np.random.Generator) -> Action:
        """Choose an action for the next round."""

    def observe(self, own: Action, opp: Action) -> None:
        """See the finished round (default: ignore it)."""

class FixedStrategyPlayer(Player):
    """Plays a fixed mixed strategy forever."""

    def __init__(self, strategy):
        strategy = np.asarray(strategy, dtype=np.float64)
        if strategy.shape != (N_ACTIONS,) or np.any(strategy < 0) or not np.isclose(strategy.sum(), 1.0):
            raise ValueError(f"not a distribution over {N_ACTIONS} actions: {strategy}")
        self.strategy = strategy

    def act(self, rng: np.random.Generator) -> Action:
        return Action(rng.choice(N_ACTIONS, p=self.strategy))

class RegretMatchingPlayer(Player):
    """Samples from the regret matcher's current strategy and feeds every
    finished round back into the ledger."""

    def __init__(self):
        self.learner = RegretMatcher(N_ACTIONS)

    def act(self, rng: np.random.Generator) -> Action:
        return Action(rng.choice(N_ACTIONS, p=self.learner.strategy()))

    def observe(self, own: Action, opp: Action) -> None:
        # PAYOFF[:, opp] = what each of my actions pays against the opponent's
        # revealed action — computable because RPS reveals both moves.
        self.learner.update(PAYOFF[:, opp], own)

class HumanPlayer(Player):
    """Reads r/p/s (or full words) from stdin; q raises QuitGame."""

    PROMPT = "[r]ock / [p]aper / [s]cissors / [q]uit > "
    PARSE = {
        "r": Action.ROCK,
        "rock": Action.ROCK,
        "p": Action.PAPER,
        "paper": Action.PAPER,
        "s": Action.SCISSORS,
        "scissors": Action.SCISSORS,
    }

    def act(self, rng: np.random.Generator) -> Action:
        while True:
            raw = input(self.PROMPT).strip().lower()
            if raw in ("q", "quit", "exit"):
                raise QuitGame
            if raw in self.PARSE:
                return self.PARSE[raw]
            print(f"  didn't understand {raw!r}")

def play_match(p0: Player, p1: Player, n_rounds: int, rng: np.random.Generator) -> np.ndarray:
    """Run n_rounds; both players observe each round. Returns p0's payoffs."""
    payoffs = np.empty(n_rounds)
    for i in range(n_rounds):
        a0 = p0.act(rng)
        a1 = p1.act(rng)
        p0.observe(a0, a1)
        p1.observe(a1, a0)
        payoffs[i] = PAYOFF[a0, a1]
    return payoffs
