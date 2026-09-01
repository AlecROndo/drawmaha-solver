"""Play Kuhn poker against the solved equilibrium in the terminal.

Solves the game first, then deals hands and plays the average strategy — so
the opponent is a genuine Nash equilibrium, not a learner adapting to you.
That is the point of playing it: an equilibrium cannot be beaten in the long
run, and it also will not punish your mistakes any harder than the game value
allows. You cannot win. You can only find out how fast you lose, which is a
more interesting number.

Seats alternate every hand, because Kuhn is not symmetric: the first player's
game value is -1/18, so a fixed seat would confound your errors with the
seat's built-in edge.

The bot sees only its own infoset — its card plus the public history — so it
cannot condition on yours. When it calls your bluff with a queen it is doing
so at the equilibrium frequency, not because it looked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drawmaha_solver.kuhn.cfr import train
from drawmaha_solver.kuhn.exploitability import exploitability
from drawmaha_solver.kuhn.game import (
    CARD_SYMBOL,
    DEALS,
    Action,
    KuhnState,
    action_label,
)
from drawmaha_solver.kuhn.infoset_table import (
    average_strategy,
    format_strategy_grid,
)

SOLVE_ITERATIONS = 20_000

CARD_NAME = {0: "jack", 1: "queen", 2: "king"}

class QuitGame(Exception):
    """Raised when the human asks to stop."""

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Solve Kuhn, then play hands against the equilibrium until quit."""
    print("Kuhn poker against a solved equilibrium.")
    print(f"Solving ({SOLVE_ITERATIONS:,} CFR iterations)...", end=" ", flush=True)
    strategies = average_strategy(train(SOLVE_ITERATIONS))
    print(f"done. Exploitable for {exploitability(strategies):.5f} chips/hand.\n")
    print("Both ante 1. One card each from J/Q/K, one bet of 1. Seats alternate.")
    print("q to quit.\n")

    rng = np.random.default_rng()
    board = Scoreboard()
    while True:
        try:
            play_hand(strategies, rng, human_seat=board.hands % 2, board=board)
        except QuitGame:
            break
    _report(board, strategies)

# ---------------------------------------------------------------------------
# One hand
# ---------------------------------------------------------------------------

def play_hand(strategies, rng, *, human_seat: int, board: Scoreboard) -> None:
    """Deal, alternate actions until terminal, score, and narrate."""
    state = deal(rng)
    print(f"--- hand {board.hands + 1}   you are P{human_seat}, "
          f"holding the {CARD_NAME[state.cards[human_seat]]}")

    while not state.is_terminal():
        if state.current_player == human_seat:
            action = ask(state.history)
            print(f"  you {action_label(action, state.history)}")
        else:
            action = bot_action(state, strategies, rng)
            print(f"  bot {action_label(action, state.history)}s")
        state = state.apply(action)

    board.record(human_seat=human_seat, returns=state.returns())
    chips = state.returns()[human_seat]
    verdict = "you win" if chips > 0 else "bot wins"
    # Always reveal, including on a fold: knowing whether the bot was bluffing
    # is the entire lesson, and the hand is over so it costs nothing.
    print(f"  showdown: you {CARD_SYMBOL[state.cards[human_seat]]} "
          f"bot {CARD_SYMBOL[state.cards[1 - human_seat]]}   "
          f"{verdict} {abs(chips):.0f}   (running {board.chips:+.0f} over {board.hands})\n")

def deal(rng: np.random.Generator) -> KuhnState:
    """A fresh hand: one of the six orderings, uniformly."""
    return KuhnState(cards=DEALS[rng.integers(len(DEALS))])

def bot_action(state: KuhnState, strategies, rng: np.random.Generator) -> Action:
    """Sample from the equilibrium strategy at the bot's own infoset.

    Reads `state.infoset()`, which is keyed on the acting player's card and
    the public history — so nothing about the human's card can reach it.
    """
    probabilities = strategies[state.infoset()]
    return Action(rng.choice(len(probabilities), p=probabilities))

# ---------------------------------------------------------------------------
# Reading the human's move
# ---------------------------------------------------------------------------

def ask(history: tuple[Action, ...]) -> Action:
    """Prompt until the human types a move that is legal at this node."""
    while True:
        typed = input(prompt_for(history))
        action = parse_action(typed, history)
        if action is not None:
            return action
        print(f"  didn't understand {typed.strip()!r} here")

def prompt_for(history: tuple[Action, ...]) -> str:
    """The prompt, naming what the two actions are called at this node."""
    passing, betting = (action_label(a, history) for a in (Action.PASS, Action.BET))
    return f"  [{passing[0]}]{passing[1:]} / [{betting[0]}]{betting[1:]} / [q]uit > "

def parse_action(typed: str, history: tuple[Action, ...]) -> Action | None:
    """The action `typed` names at this node, or None if it names none.

    Deliberately context-strict: "fold" with nothing to call is rejected
    rather than read as a check. The player has misread the spot, and silently
    picking the other action for them would hide that. Raises QuitGame on q.
    """
    word = typed.strip().lower()
    if word in ("q", "quit", "exit"):
        raise QuitGame
    for action in (Action.PASS, Action.BET):
        label = action_label(action, history)
        if word in (label, label[0]):
            return action
    return None

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Scoreboard:
    """Chips and hands from the human's seat, across alternating seats."""

    chips: float = 0.0
    hands: int = 0
    _by_seat: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def record(self, *, human_seat: int, returns: tuple[float, float]) -> None:
        """Bank one finished hand, taking the human's side of the payoff."""
        self.chips += returns[human_seat]
        self._by_seat[human_seat] += returns[human_seat]
        self.hands += 1

    @property
    def per_hand(self) -> float | None:
        """Chips per hand, or None before any hand has been played."""
        if self.hands == 0:
            return None
        return self.chips / self.hands

def _report(board: Scoreboard, strategies) -> None:
    if board.per_hand is None:
        return
    print(f"\n{board.hands} hands. You net {board.chips:+.0f} chips "
          f"({board.per_hand:+.3f} per hand).")
    print("Against an equilibrium the long-run answer is 0.000 per hand from")
    print("alternating seats — anything below that is yours to explain.\n")
    print("What it was playing (P(bet), which reads as P(call) facing a bet):\n")
    print(format_strategy_grid(strategies))

if __name__ == "__main__":
    main()
