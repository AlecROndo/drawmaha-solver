---
title: Introduction
description: What the Drawmaha solver is, the rules of the game it is built for, and the terms used throughout this wiki.
---

The Drawmaha solver is a game-theory solver for heads-up pot-limit **Drawmaha**, a split-pot hybrid of draw poker and Omaha that no existing solver plays. It does not attack the full game head-on. It is built as a **validation ladder**: a sequence of games, each one adding a single new idea to the last, and each one graded against a known answer before the next is attempted. By the time the method meets the real game, every piece of it has been checked somewhere a mistake would have shown.

The question every step answers is the same: **how much would a perfect adversary win against this strategy?** That number, *exploitability*, is zero exactly at a Nash equilibrium and positive everywhere else, and it is computed exactly by a grader that shares no code with the solver. Training loss is never used as a progress meter, because in self-play it can fall while the strategy gets no better.

## The game: five hole cards, one draw, a pot split two ways

If you have never played Drawmaha, this is the minimum you need to follow the rest of the wiki. The target game, as the README's "The target game" section defines it:

- Each player is dealt **5 hole cards**.
- It is a **bomb pot**: both players ante and the flop comes out with **no preflop betting**. The order is ante → flop (3 cards) → betting → **one draw** (discard 0–5 cards and replace them) → turn → betting → river → betting. Three betting rounds, not Omaha's four.
- **The draw-1 rule:** a player who draws exactly one card is dealt it **face up**; they may keep it or reject it and take the next card face down.
- **Split pot:** half the pot goes to the best **Omaha** hand (exactly 2 hole cards + 3 board cards), half to the best **5-card poker hand made of the hole cards alone**.

The solve's defaults, all configurable: pot-limit with the action set {fold, check/call, pot}; 100 big blinds effective; both players ante, so there are no blinds and no positional asymmetry from them. Bet sizes are a config, so a multi-size solve is a re-solve, not a rewrite.

Why it is hard: a hand is judged two ways at once, the draw changes both of them, and the draw-1 rule leaks information publicly. A strategy has to balance what it holds for the Omaha half against what it holds for the draw half, and the size of the full game rules out the table-based methods that solve the small games below. See [Abstraction](wiki:theory/abstraction) for how the project shrinks it, and [Full Drawmaha](wiki:research/full-drawmaha) for what is still to be done. The live rules page is at [/rules](/rules).

### Poker glossary

| Term | Meaning |
| --- | --- |
| **Heads-up** | Exactly two players. Every game in this project is two-player and zero-sum: what one wins the other loses. |
| **Ante** | A forced bet every player puts in before the cards. In every game here both players ante, which is what makes the pot worth fighting for. |
| **Bomb pot** | A hand where everyone antes and the first board cards come out with no betting before them. |
| **Hole cards** | A player's private cards, seen by them alone. |
| **Board** | Shared face-up cards both players use. |
| **Draw** | Discarding some hole cards and receiving replacements from the deck. |
| **Street / round** | One betting round, after a set of board cards or a draw. |
| **Pot-limit** | The largest bet allowed is the size of the pot. "Pot" in the action set means a pot-sized bet. |
| **Fold / check / call / bet / raise** | Give up the hand / pass with nothing to call / match the bet in front of you / put chips in first / increase a bet. |
| **Showdown** | When betting ends with two players left, the hands are compared. |
| **Split pot** | The pot is divided into halves, each won by a different hand ranking. Winning both halves is a **scoop**. |
| **Omaha hand** | A five-card hand built from exactly two hole cards and exactly three board cards. |
| **Big blind (bb)** | The unit stack sizes are quoted in. With antes only, it is a unit, not a blind anyone posts. |
| **Chips per hand** | The unit every grade is quoted in: chips won or lost on average per hand dealt, with the ante as one chip in the small games. |

### Game-theory glossary

| Term | Meaning |
| --- | --- |
| **Strategy (σ)** | For every situation a player can be in, a probability for each legal action. A strategy that mixes is a *mixed* strategy; one that always picks one action per situation is *pure*. |
| **Information set (infoset)** | Everything the acting player knows at a decision: their own cards and the public history, never the opponent's cards. A strategy is defined per infoset, because a player cannot act differently in two situations they cannot tell apart. |
| **Chance node** | A point where the deck acts rather than a player: a deal, a board card, a draw. |
| **Nash equilibrium** | A pair of strategies where neither player gains by changing their own. In a two-player zero-sum game, playing one guarantees at least the game's value against anyone. |
| **Game value** | What a seat wins per hand when both play an equilibrium. Kuhn poker's first seat is worth −1/18. |
| **Best response (BR)** | The strategy that wins the most against one fixed opponent strategy. |
| **Exploitability** | The average of the two seats' best-response winnings against a strategy pair, (BR₀ + BR₁) / 2. Zero at equilibrium. The project's only grade. |
| **Regret** | How much better an action would have done than what the strategy actually did, accumulated over iterations. |
| **Regret matching** | Play each action in proportion to its positive accumulated regret. |
| **CFR** | Counterfactual regret minimisation: regret matching at every infoset of a game tree at once, with each update weighted by how likely the others make it to reach that infoset. |
| **Average strategy** | The reach-weighted average of every iteration's strategy. It is CFR's answer: the average converges to equilibrium; the current strategy need not. |
| **MCCFR** | Monte Carlo CFR: CFR on sampled hands instead of the whole tree, for games too large to enumerate. |
| **Deep CFR** | CFR with the per-infoset regret tables replaced by a neural network that generalises across infosets. |

## Eight ideas the solver is built from, each linked to its theory

These are the ideas the solver is built from. Each links to its full treatment in the Theory tab.

| Concept | What it is |
| --- | --- |
| **Nash equilibrium and exploitability** | The target, and the meter. Exploitability measures distance from equilibrium in the units that matter, chips an adversary takes. See [Equilibrium](wiki:theory/equilibrium) and [Exploitability](wiki:theory/exploitability). |
| **Regret matching** | The learning rule at the bottom of every game here, unchanged from rock-paper-scissors to mini-drawmaha. One ledger per decision; only what feeds it changes. See [Regret matching](wiki:theory/regret-matching). |
| **Counterfactual regret** | Regret at one infoset in a tree, weighted by the probability that chance and the opponent bring play there. The two reach weights, own and counterfactual, are the classic place to make a silent mistake. See [CFR](wiki:theory/cfr). |
| **Exact best response** | A grader that walks the whole game, sums over the hidden cards *first* and takes the adversary's best action *second*. Reversing the order describes an adversary who sees your cards. See [Exploitability](wiki:theory/exploitability). |
| **External-sampling MCCFR, run in lockstep** | Sample one hand per seat per worker, bank every worker's regret against the same frozen strategy, and get bit-for-bit reproducible parallel training. See [MCCFR](wiki:theory/mccfr). |
| **Regret rules** | Vanilla, LCFR (linear weighting), CFR+ and DCFR: different ways of weighting and flooring regrets over time. LCFR won the race on mini-drawmaha. See [Regret rules](wiki:theory/regret-rules). |
| **Deep CFR** | Replace 6.2 million regret ledgers with one network, refit each iteration from a reservoir of sampled regrets. See [Deep CFR](wiki:theory/deep-cfr). |
| **Abstraction** | Mini-drawmaha keeps Drawmaha's structure (a bomb pot, a draw, a pot that pays twice) on a deck small enough to grade exactly. See [Abstraction](wiki:theory/abstraction). |

:::note
The method's words — rung, ladder — name the steps of the project. The code is organised the same way: one subpackage per rung under `src/drawmaha_solver/`, one figures folder per rung under `figures/`. See [Method](wiki:overview/method) for what each rung tests.
:::

:::cards
- [Method](wiki:overview/method) The ladder, the pre-registered evaluation protocol, and every rung's result.
- [Architecture](wiki:overview/architecture) How the packages, graders, exports and site fit together.
- [Quickstart](wiki:overview/quickstart) Install, run the suite, solve a game, play against it.
- [Theory](wiki:theory/equilibrium) The derivations behind every piece of the solver.
:::
