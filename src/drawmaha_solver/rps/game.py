"""Rock-paper-scissors rules: actions, payoffs, winner determination.

Rung 0 of the validation ladder. RPS is the smallest game with a
mixed-strategy Nash equilibrium — uniform (1/3, 1/3, 1/3) — which gives the
regret-matching ledger a known exact answer to converge to before any game
tree exists. "Exploitability" here is the one-shot best-response value: what
a perfect adversary who knows the strategy (but plays the same simultaneous
game) earns per round against it.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

# ---------------------------------------------------------------------------
# Actions and payoffs
# ---------------------------------------------------------------------------

class Action(IntEnum):
    # IntEnum (not StrEnum) so actions index payoff matrices and strategy
    # vectors directly.
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

N_ACTIONS = len(Action)

# PAYOFF[a, b] = row player's chips when row plays a and column plays b:
# +1 win, 0 tie, -1 loss. Zero-sum and symmetric: PAYOFF == -PAYOFF.T.
PAYOFF = np.array(
    [
        [0.0, -1.0, 1.0],  # rock:     ties rock, loses to paper, beats scissors
        [1.0, 0.0, -1.0],  # paper:    beats rock, ties paper, loses to scissors
        [-1.0, 1.0, 0.0],  # scissors: loses to rock, beats paper, ties scissors
    ]
)

def payoff(a: Action, b: Action) -> float:
    """Player a's payoff when a meets b."""
    return float(PAYOFF[a, b])

def winner(a: Action, b: Action) -> int | None:
    """0 if action `a` wins, 1 if action `b` wins, None on a tie."""
    p = PAYOFF[a, b]
    if p > 0:
        return 0
    if p < 0:
        return 1
    return None

# ---------------------------------------------------------------------------
# Strategy metrics
# ---------------------------------------------------------------------------

def action_values(strategy: np.ndarray) -> np.ndarray:
    """Expected payoff of each pure action against a mixed strategy."""
    return PAYOFF @ np.asarray(strategy, dtype=np.float64)

def best_response_value(strategy: np.ndarray) -> float:
    """One-shot exploitability: what a best-responding adversary earns per
    round against `strategy`.

    Exactly 0 at the Nash equilibrium (uniform), positive everywhere else.
    Because RPS is symmetric zero-sum, the adversary's payoff matrix is the
    same PAYOFF, so this is just the best entry of action_values.
    """
    return float(np.max(action_values(strategy)))
