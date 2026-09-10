"""Play Leduc poker against the solved equilibrium in the terminal.

Solves the game first, then deals hands and plays the average strategy — so
the opponent is a genuine equilibrium, not a learner adapting to you. That is
the point of playing it: an equilibrium cannot be beaten in the long run, and
it also will not punish your mistakes any harder than the game value allows.
You cannot win. You can only find out how fast you lose.

Two things make this a different game to sit down at than rung 1's Kuhn.
There are three actions rather than two, so a spot offers fold/call/raise or
just call/raise depending on whether anything has been bet — the prompt tracks
that rather than offering a fixed pair. And a public card is turned between
the two betting rounds, so the hand you hold changes value mid-way: a jack is
trash until a jack lands on the board, at which point it is the best hand
possible.

Seats alternate every hand, because Leduc is not symmetric: the first player's
game value is about -0.086, so a fixed seat would confound your errors with
the seat's built-in edge.

The bot sees only its own infoset — its card's RANK, the board, and the public
betting — so it cannot condition on yours, and it cannot tell its own two
jacks apart either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drawmaha_solver.leduc.cfr import train
from drawmaha_solver.leduc.exploitability import Profile, exploitability
from drawmaha_solver.leduc.game import (
    ANTE,
    CARD_SYMBOL,
    DEALS,
    LP_VALUE_P0,
    RANK_SYMBOL,
    Action,
    LeducState,
    action_label,
    legal_actions_for,
)
from drawmaha_solver.leduc.infoset_table import (
    average_strategy,
    format_strategy_tables,
)

# One iteration walks 9,450 nodes at about 50 ms, so this is roughly two
# minutes — the point where the strategy is worth playing against (beatable
# for well under 0.02 chips a hand) without making anyone wait for a solve
# that is merely tidier.
SOLVE_ITERATIONS = 2_000

RANK_NAME = {"J": "jack", "Q": "queen", "K": "king"}


class QuitGame(Exception):
    """Raised when the human asks to stop."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Solve Leduc, then play hands against the equilibrium until quit."""
    print("Leduc poker against a solved equilibrium.")
    print(f"Solving ({SOLVE_ITERATIONS:,} CFR iterations)...", end=" ", flush=True)
    strategies = average_strategy(train(SOLVE_ITERATIONS))
    print(f"done. Exploitable for {exploitability(strategies):.5f} chips/hand.\n")
    print(f"Both ante {ANTE}. Six cards: two each of J, Q, K. One card each, a")
    print("betting round, one public card, another betting round. Bets are 2")
    print("then 4, two raises a round. Seats alternate. q to quit.\n")

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


def play_hand(
    strategies: Profile,
    rng: np.random.Generator,
    *,
    human_seat: int,
    board: Scoreboard,
) -> None:
    """Deal, run both betting rounds around the board card, score, and narrate."""
    state = deal(rng)
    print(f"--- hand {board.hands + 1}   you are P{human_seat}, "
          f"holding the {RANK_NAME[RANK_SYMBOL[state.cards[human_seat].rank]]}")

    while not state.is_terminal():
        if state.is_chance_node():
            state = turn_board(state, rng)
            print(f"  board: {CARD_SYMBOL[state.board]}")
            continue
        line = state.infoset().betting[-1]
        if state.current_player == human_seat:
            action = ask(line)
            print(f"  you {action_label(action, line)}")
        else:
            action = bot_action(state, strategies, rng)
            print(f"  bot {action_label(action, line)}s")
        state = state.apply(action)

    board.record(human_seat=human_seat, returns=state.returns())
    chips = state.returns()[human_seat]
    # Always reveal, including on a fold: knowing whether the bot was bluffing
    # is the entire lesson, and the hand is over so it costs nothing.
    shown = (f"you {CARD_SYMBOL[state.cards[human_seat]]} "
             f"bot {CARD_SYMBOL[state.cards[1 - human_seat]]}")
    if state.board is not None:
        shown += f" board {CARD_SYMBOL[state.board]}"
    print(f"  showdown: {shown}   {verdict(chips)}   "
          f"(running {board.chips:+.0f} over {board.hands})\n")


def verdict(chips: float) -> str:
    """How the hand ended, from the human's seat.

    Three outcomes, not rung 1's two: there are two cards of each rank here,
    so both players can hold the same one and a showdown between them splits
    the pot. Reporting that as "bot wins 0" would call a chop a loss on the
    one line of the hand the player actually reads.
    """
    if chips > 0:
        return f"you win {chips:.0f}"
    if chips < 0:
        return f"bot wins {-chips:.0f}"
    return "split pot"


def deal(rng: np.random.Generator) -> LeducState:
    """A fresh hand: one of the thirty orderings, uniformly."""
    return LeducState(cards=DEALS[rng.integers(len(DEALS))])


def turn_board(state: LeducState, rng: np.random.Generator) -> LeducState:
    """Turn the public card, drawn at the deck's own probabilities.

    Sampled from `chance_outcomes` rather than from the deck directly so the
    human faces exactly the distribution the solver trained against — the two
    cards already in hands are gone, which is why a jack on a jack board is
    the last jack rather than one of two.
    """
    cards, probabilities = zip(*state.chance_outcomes())
    return state.apply_chance(cards[rng.choice(len(cards), p=probabilities)])


def bot_action(state: LeducState, strategies: Profile, rng: np.random.Generator) -> Action:
    """Sample from the equilibrium strategy at the bot's own infoset.

    The sampled index is a POSITION in this spot's legal actions, not an
    `Action` value: at the opening spot the columns are (call, raise), so
    reading the index as an action would turn every check into a fold.
    """
    spot = state.infoset()
    probabilities = strategies[spot]
    return spot.legal_actions()[rng.choice(len(probabilities), p=probabilities)]


# ---------------------------------------------------------------------------
# Reading the human's move
# ---------------------------------------------------------------------------


def ask(line: tuple[Action, ...]) -> Action:
    """Prompt until the human types a move that is legal at this node."""
    while True:
        typed = input(prompt_for(line))
        action = parse_action(typed, line)
        if action is not None:
            return action
        print(f"  didn't understand {typed.strip()!r} here")


def prompt_for(line: tuple[Action, ...]) -> str:
    """The prompt, naming exactly the actions that are legal at this node.

    Built from `legal_actions_for` rather than a fixed list, because the width
    of a spot is the whole difference between rung 1's game and this one:
    facing a raise there are three answers, at the open there are two, and at
    the raise cap there are two again but different ones.
    """
    words = [action_label(a, line) for a in legal_actions_for(line)]
    return "  " + " / ".join(f"[{w[0]}]{w[1:]}" for w in [*words, "quit"]) + " > "


def parse_action(typed: str, line: tuple[Action, ...]) -> Action | None:
    """The action `typed` names at this node, or None if it names none.

    Deliberately context-strict: "fold" with nothing to call is rejected
    rather than read as a check. The player has misread the spot, and silently
    picking another action for them would hide that. Raises QuitGame on q.
    """
    word = typed.strip().lower()
    if word in ("q", "quit", "exit"):
        raise QuitGame
    for action in legal_actions_for(line):
        label = action_label(action, line)
        if word in (label, label[0]):
            return action
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Scoreboard:
    """Chips and hands from the human's seat, across alternating seats.

    Kept per seat as well as in total, because the total is the only number
    that is supposed to approach zero. Each seat separately approaches its own
    game value, and seeing the two straddle -/+0.086 is what shows the
    alternation is cancelling a seat edge rather than hiding one.
    """

    chips: float = 0.0
    hands: int = 0
    seat_chips: list[float] = field(default_factory=lambda: [0.0, 0.0])
    seat_hands: list[int] = field(default_factory=lambda: [0, 0])

    def record(self, *, human_seat: int, returns: tuple[float, float]) -> None:
        """Bank one finished hand, taking the human's side of the payoff."""
        self.chips += returns[human_seat]
        self.seat_chips[human_seat] += returns[human_seat]
        self.seat_hands[human_seat] += 1
        self.hands += 1

    @property
    def per_hand(self) -> float | None:
        """Chips per hand, or None before any hand has been played."""
        if self.hands == 0:
            return None
        return self.chips / self.hands

    def per_hand_in_seat(self, seat: int) -> float | None:
        """Chips per hand in `seat`, or None before that seat has played."""
        if self.seat_hands[seat] == 0:
            return None
        return self.seat_chips[seat] / self.seat_hands[seat]


def _report(board: Scoreboard, strategies: Profile) -> None:
    if board.per_hand is None:
        return
    print(f"\n{board.hands} hands. You net {board.chips:+.0f} chips "
          f"({board.per_hand:+.3f} per hand).")
    for seat in (0, 1):
        rate = board.per_hand_in_seat(seat)
        if rate is None:
            continue
        # +LP_VALUE_P0 in seat 0, -LP_VALUE_P0 in seat 1: what an equilibrium
        # player would earn there, so the two lines are read against different
        # targets and only the total is read against zero.
        target = LP_VALUE_P0 if seat == 0 else -LP_VALUE_P0
        print(f"  as P{seat}: {board.seat_chips[seat]:+.0f} over "
              f"{board.seat_hands[seat]} hands ({rate:+.3f} per hand, "
              f"equilibrium {target:+.3f})")
    print("Against an equilibrium the long-run answer is 0.000 per hand from")
    print("alternating seats — anything below that is yours to explain.\n")
    print("What it was playing:\n")
    print(format_strategy_tables(strategies))


if __name__ == "__main__":
    main()
