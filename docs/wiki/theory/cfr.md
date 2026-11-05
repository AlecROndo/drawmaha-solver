---
title: Counterfactual regret minimisation
description: How CFR turns a game tree into one regret-matching ledger per infoset, and why the two reach weights must never be swapped.
---

**Counterfactual regret minimisation (CFR)** runs [regret matching](wiki:theory/regret-matching) at every infoset of a game tree at once, and its average strategy converges to a Nash equilibrium. RPS needs nothing more than the ledger: the opponent shows a move, and the payoff matrix already has a column of utilities for it. A tree doesn't give you that. What "bet with the king" is worth depends on everything both players do afterwards, so the utilities have to be *computed*, by walking the tree. That walk is the one new idea in Kuhn, in [kuhn/cfr.py](repo:src/drawmaha_solver/kuhn/cfr.py), and Leduc and mini-drawmaha keep its shape.

## Two quantities travel through the walk in opposite directions

One CFR iteration is a recursive walk from the root. At every node it carries **reach probabilities down** and returns **values up**.

- **Down: reach.** `reach[p]` is the product of player p's *own* action probabilities on the path so far, and nothing else. Chance is in neither entry.
- **Up: values.** What the node is worth to each seat, as a plain conditional expectation: at a decision node, Σₐ σ(I, a) · v(child a). Values are *not* weighted by reach. Reach weights are applied once, when a ledger is banked.
- The walk returns the value to *both* seats, so nothing is negated on the way up.

At an infoset I owned by player i, the walk has a value vᵢ(I, a) for each action and the mix's value vᵢ(I) = Σₐ σ(I, a)·vᵢ(I, a). It banks:

```eq
R(I, a) += π₋ᵢ(h) · ( vᵢ(h, a) − vᵢ(h) )
S(I, a) += πᵢ(h) · σ(I, a)
```

- **h** is the concrete history being walked. Many histories share one infoset, because they differ only in cards player i cannot see. The bank adds across all of them.
- **π₋ᵢ(h)**, the **counterfactual reach**, is the probability that chance and the opponent bring play to h. Player i's own choices are left out on purpose: it is the reach i *would* have if i played to get there.
- **πᵢ(h)** is player i's own reach, the probability i's own choices lead to h.
- **R** is `cumulative_regret`, **S** is `strategy_sum`. The average strategy is S(I, ·) normalised.

### Why the regret uses the opponent's reach and the average uses yours

Regret asks "had I played to get here, how much would always-a have beaten my mix?" Weighting it by your own reach would silence a line you currently avoid: its regret would be multiplied by roughly zero, and you could never find out it is good. The counterfactual weight keeps every line learning at full strength.

The average asks a different question: "what did I actually play?" A spot you almost never steer into should barely affect your average, so it is weighted by your own reach.

:::warning
**Swap the two weights and every rung still runs.** On Kuhn the swapped walk prints a converged-looking grid that folds the king to a bet 18% of the time, with game value −0.00001 instead of −1/18. Nothing crashes. That is why `RegretMatcher.update` takes `regret_weight=` and `strategy_weight=` as keyword-only arguments.
:::

:::details Why chance is in π₋ᵢ, and why it does not matter in Kuhn
The counterfactual reach is *defined* to include chance's probability. In Kuhn every deal is equally likely (1/6), so the chance factor scales every regret increment by the same constant. Regret matching reads only the *ratio* of positive regrets, so σ, the reach and the average are unchanged; the repo measures the difference at 9e-15 after 5,000 iterations, which is floating-point rounding. The factor is still carried because once a game deals non-uniform chance it stops being a common scale. It is also why the ledgers hold numbers 6× smaller than Neller & Lanctot's reference trace, whose pseudocode leaves the factor out.
:::

## The average converges; the current iterate never does

Kuhn's exploitability, from the README:

| iterations | average | current | game value |
| ---: | ---: | ---: | ---: |
| 10 | 0.09114 | 0.24191 | −0.03597 |
| 100 | 0.02330 | 0.25000 | −0.05472 |
| 1,000 | 0.00650 | 0.27885 | −0.05518 |
| 10,000 | 0.00149 | 0.20359 | −0.05553 |
| 100,000 | 0.00063 | 0.24649 | −0.05555 |

Two lessons, and the repo builds on both:

- **The average falls steadily; the current strategy sits around 0.2 forever.** At 1,000 iterations it is briefly *worse* than at 10. Vanilla CFR's current iterate cycles. Every grader grades the average, and the live surfaces serve averages.
- **Game value is the softer grader.** At 100 iterations the game value is within 0.0009 of −1/18 and looks solved, while the strategy is still 15× more exploitable than it will be at 10,000. Leduc repeats the lesson on a 460-times-larger tree: by 100 iterations its game value is within 0.004 of −0.08561, while the strategy is five times more exploitable than at 2,000.

![Only the average strategy converges; the current one cycles forever](fig:rung1/exploitability.png)

The average falls along the 1/√T guide, which is CFR's convergence rate: exploitability of the average after T iterations is O(1/√T).

## What changes at Leduc: chance in the middle, ragged action sets

[Leduc](wiki:implementation/leduc) keeps the walk and adds two things:

- **A chance node mid-hand.** The board card comes after the first betting round, so the walk meets chance inside the tree rather than enumerating every deal at the top. A jack is trash until a jack hits the board, then it's the best hand in the game.
- **Spots with different action sets.** Facing a raise you may fold, call or raise; at the open you may not fold. A ledger is 2 or 3 wide depending on where it sits, and column 0 means *call* at the open but *fold* facing a raise. Every loop pairs `enumerate(legal_actions())` with the ledger's columns.

288 infosets instead of Kuhn's 12. One iteration walks 9,450 nodes in about 50 ms. Vanilla CFR enumerates the whole tree and never samples, so two runs agree bit for bit, and `figures/rung1/` and `figures/rung2/` double as regression fingerprints.

## Locking a seat turns CFR into a best-response finder

The same walk can hold some infosets fixed. At a locked infoset it plays the supplied probabilities and banks nothing; elsewhere it learns as usual. The locked probabilities still advance the reach, so they still enter the learner's π₋ᵢ. The learner is never told the opponent is fixed. It is simply weighted by how often that opponent brings it to each spot. Minimising regret against an opponent that can't move back converges to a **best response**, not to Nash. Lock a whole seat and CFR becomes an exploit finder with no new algorithm. That is the **Exploit** tab at `/rung1#exploit`, graded against the exact best response (see [Exploitability](wiki:theory/exploitability)).

## Past Leduc, the full walk does not run

Mini-drawmaha has 85.1 billion decision nodes and 6,220,050 infosets. A full walk per iteration is impossible, so rung 3 *samples* the walk instead. The regret weight then becomes the sampling frequency, which is the subject of [MCCFR](wiki:theory/mccfr).

:::cards
- [Exploitability](wiki:theory/exploitability) The exact meter every CFR run is graded by.
- [Monte Carlo CFR](wiki:theory/mccfr) Sampling the deck and the opponent so the walk fits.
- [Kuhn in code](wiki:implementation/kuhn) The walk, the meter and the exploiter, file by file.
:::
