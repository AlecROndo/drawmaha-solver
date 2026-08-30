"""Play rock-paper-scissors against the regret-matching learner in the terminal."""

from __future__ import annotations

import numpy as np

from drawmaha_solver.rps.game import Action, payoff, winner
from drawmaha_solver.rps.players import HumanPlayer, QuitGame, RegretMatchingPlayer

def main() -> None:
    print("Rock-paper-scissors vs the regret-matching ledger.")
    print("It starts uniform and learns from every round you play. q to quit.\n")
    rng = np.random.default_rng()
    human = HumanPlayer()
    bot = RegretMatchingPlayer()
    score = {0: 0, 1: 0, None: 0}
    chips = 0.0
    rounds = 0

    while True:
        try:
            a_h = human.act(rng)
        except QuitGame:
            break
        a_b = bot.act(rng)
        human.observe(a_h, a_b)
        bot.observe(a_b, a_h)
        w = winner(a_h, a_b)
        chips += payoff(a_h, a_b)
        score[w] += 1
        rounds += 1
        verdict = {0: "you win", 1: "bot wins", None: "tie"}[w]
        print(
            f"  you: {a_h.name.lower():<8} bot: {a_b.name.lower():<8} -> {verdict}   "
            f"(you {score[0]} – {score[1]} bot, {score[None]} ties)"
        )

    if rounds:
        avg = bot.learner.average_strategy()
        print(f"\n{rounds} rounds. You net {chips:+.0f} chips.")
        print(
            "Bot's average strategy so far: "
            + ", ".join(f"{a.name.lower()} {avg[a]:.0%}" for a in Action)
        )
        print("(Against a non-uniform human it drifts toward the counter of your habits.)")

if __name__ == "__main__":
    main()
