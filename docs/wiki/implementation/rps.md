---
title: Rock-paper-scissors
description: The ledger with nothing around it — the rules, the players, the two experiments and the terminal game in src/drawmaha_solver/rps/.
---

Rock-paper-scissors is the smallest game with a mixed-strategy equilibrium, the uniform (⅓, ⅓, ⅓). The regret ledger therefore has an exact answer to converge to before any game tree exists. There is no tree, no chance and no hidden information here, only the ledger playing itself. That isolates the one thing every later game inherits: regret matching's *average* converges and its *current* strategy does not.

## Files

| File | What it is for |
| --- | --- |
| [`game.py`](repo:src/drawmaha_solver/rps/game.py) | The rules and the meter in one place, because both are a few lines here |
| [`players.py`](repo:src/drawmaha_solver/rps/players.py) | Who plays, and the loop that runs a match |
| [`analysis.py`](repo:src/drawmaha_solver/rps/analysis.py) | The two experiments that certify the game, and their four figures |
| [`play.py`](repo:src/drawmaha_solver/rps/play.py) | Play the learner from the terminal (`uv run rps-play`) |

## `game.py`: payoffs and the one-shot best response

- **`Action`**: rock, paper, scissors.
- **`payoff`** and **`winner`**: who beats whom, as a +1 / 0 / −1 payoff and as a winner index.
- **`action_values`**: what each pure action earns against a mixed strategy. This is the utility vector the ledger learns from, and it doubles as the input to the meter.
- **`best_response_value`**: rock-paper-scissors' exploitability, the most a perfect adversary who knows your mix earns per round against it. It is zero exactly at the uniform mix. See [Exploitability](wiki:theory/exploitability).

## `players.py`: the player interface and the match runner

- **`Player`**: the interface a match needs. `act` chooses the next action and `observe` sees the finished round.
- **`FixedStrategyPlayer`**: plays one mixed strategy forever; the opponent the learner is tested against.
- **`RegretMatchingPlayer`**: the learner. It samples from its ledger's current strategy and feeds every finished round's payoff column back into `update`, with both weights at 1 because every round counts the same.
- **`HumanPlayer`**: reads `r` / `p` / `s` from the keyboard. `q` raises `QuitGame`.
- **`play_match`**: runs n rounds in which both players observe each one, and returns player 0's payoffs.

## `analysis.py`: two experiments, four figures

`uv run rps-analysis` runs both and writes `figures/rung0/`:

1. **Self-play** (**`run_self_play`**): two ledgers play each other. The average strategy must reach the uniform Nash, with exploitability falling like 1/√T, while the current strategy visibly cycles. After 100k iterations the average is (0.334, 0.333, 0.333) and is exploitable for 0.0009 chips a round.
2. **Against a biased fixed opponent** (**`run_vs_fixed`**): the learner must find the best response. Against a 50%-rock opponent it converges to pure paper and earns +0.24 a round, against a best-response ceiling of +0.25.

**`SelfPlayTrajectories`** and **`VsFixedTrajectories`** record every iteration, so the four figure functions can subsample freely.

![Self-play average strategy converges to the uniform Nash](fig:rung0/self_play_average_strategy.png)

![The current strategy cycles; the average settles](fig:rung0/self_play_current_vs_average.png)

## Where it is drawn

`web/rung0-viz`, served at [`/rung0`](/rung0). It is the one app that ports its ledger to TypeScript. The 63-line regret-matching rule is safe to transcribe, which the reach-weighted tree walk of the later games is not. The rung-0 writeup (Markdown, LaTeX and PDF) is under [`docs/writeups/rung0-rps/`](repo:docs/writeups/rung0-rps).

:::cards
- [The regret ledger](wiki:implementation/ledger) The class this game exercises.
- [Regret matching](wiki:theory/regret-matching) Why the average converges and the current strategy does not.
- [Kuhn poker](wiki:implementation/kuhn) The next game: the ledger meets a tree.
:::
