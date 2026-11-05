---
title: Regret matching
description: The learning rule at the heart of every solver in the repo — one ledger per spot, unchanged from rock-paper-scissors to mini-drawmaha.
---

Regret matching (Hart & Mas-Colell 2000) is how a player who repeats a game learns to play it without being told the answer. The repo uses one implementation of it, `RegretMatcher` in [regret_matching.py](repo:src/drawmaha_solver/regret_matching.py), for every game. Rock-paper-scissors feeds it a payoff vector each round. Kuhn, Leduc and mini-drawmaha feed it one vector per infoset per visit. The rule itself is the same in all of them; only what feeds it grows.

## The whole algorithm is three lines

1. **Record regret.** After a visit, for every action a, add how much better always-a would have done than the mix you actually played.
2. **Play in proportion to positive regret.** Next visit, play each action with probability proportional to its accumulated *positive* regret; uniform when none is positive.
3. **Answer with the average.** The running *average* of the strategies played converges to equilibrium in two-player zero-sum games. The current strategy does not.

```eq
r(a) = u(a) − ⟨σ, u⟩
R(a) ← R(a) + r(a)
σ(a) = R⁺(a) / Σₐ′ R⁺(a′),   R⁺ = max(R, 0)
```

- **u(a)** is the payoff action a would have earned this visit. In RPS it is a column of the payoff matrix against what the opponent threw; in a tree it is a counterfactual value computed by the walk (see [CFR](wiki:theory/cfr)).
- **⟨σ, u⟩ = Σₐ σ(a)·u(a)** is what the current mix earned on the same visit — the baseline every action is compared against.
- **r(a)** is the regret of not having played a: positive if a beat the mix, negative if it did worse.
- **R(a)** is `cumulative_regret`, the ledger's memory. **σ** is derived from it on demand and never stored.

### Why positive regret, and why proportional

The rule puts probability where playing more would have helped. An action with negative cumulative regret has, on balance, done worse than the mix, and gets probability zero until it climbs back above zero. Among the positive ones, an action with twice the regret gets twice the probability. When nothing is positive there is no evidence to follow, so the ledger falls back to uniform.

**What moves what.** Feed in one large positive r for an action, and that action's probability jumps next visit. Feed in a long run of small negatives and its probability drops to zero, and stays there until enough positive regret accumulates to cancel the debt. Plain regret matching keeps that debt in full. CFR+, compared in [Regret rules](wiki:theory/regret-rules), floors it at zero instead.

## The average converges; the current strategy cycles

The current σ chases whatever beat it last time, so against an adapting opponent it cycles: rock gets punished by paper, paper by scissors, and round it goes. The average of those strategies converges.

The ledger therefore keeps a second vector, `strategy_sum`, which banks σ after every visit. `average_strategy()` normalises it. In rock-paper-scissors self-play the average reaches (0.334, 0.333, 0.333) after 100,000 rounds, exploitable for 0.0009 chips/round, and its exploitability falls like 1/√T. Against a fixed opponent that throws rock half the time, the same ledger converges to pure paper and earns +0.24/round; the best response earns +0.25.

![Self-play average strategy converges to the uniform Nash](fig:rung0/self_play_average_strategy.png)

:::warning
Every solver in the repo answers with the **average**, and every grader grades the average. Kuhn's table in [CFR](wiki:theory/cfr) shows the current iterate between 0.20 and 0.28 chips/hand exploitable however long the run goes, and Leduc's committed solve ends with the current iterate at 0.97 against the average's 0.011 ([`web/rung2-viz/src/data/solve.json`](repo:web/rung2-viz/src/data/solve.json)). This is also why rung 4 has to rebuild an average explicitly — see [Deep CFR](wiki:theory/deep-cfr).
:::

## Two weights let the same ledger serve a game tree

In RPS every round counts the same. In a tree a spot is only reached some of the time, and the two vectors want different scales:

| Weight | Multiplies | Is | Why |
| --- | --- | --- | --- |
| `regret_weight` | the regret r added to R | the **counterfactual reach** π₋ᵢ: chance's and the opponent's probabilities of reaching the spot, leaving out your own | a line you currently avoid must keep learning at full strength, or you could never discover it is good |
| `strategy_weight` | the σ added to `strategy_sum` | your **own reach** πᵢ | the average must reproduce what you really played, so a spot you rarely steer into counts rarely |

Both default to 1, so RPS callers never mention them.

:::warning
**Swapped, both versions run silently.** On Kuhn, swapping the two weights still completes, still prints a converged-looking grid, and folds the king to a bet 18% of the time, with game value −0.00001 instead of −1/18. That is why `update(utilities, *, regret_weight, strategy_weight)` takes them keyword-only, and why a test pins that the two scales are distinguishable. The repo's root instructions forbid making them positional.
:::

## Four slots per ledger, written in place and never rebound

From mini-drawmaha on, many things touch a ledger: the walk, the regret rules, the checkpoint, the packed table that holds 6.2M of them in flat arrays. The module docstring fixes exactly which slots they may use:

| Slot | Shape | Holds |
| --- | --- | --- |
| `cumulative_regret` | `(n_actions,)` float64 | R |
| `strategy_sum` | `(n_actions,)` float64 | the run's primary average |
| `extra_sums` | `(k, n_actions)` float64 | further averages of the same strategies at other weights (uniform, quadratic), banked for free |
| `stamp` | `(1,)` int64 | the last iteration a lazily applied discount reached (DCFR) |

All four can be **views** into a table's flat buffers (`RegretMatcher.over`), so writers update in place (`+=`, `[:] =`) and never rebind the attribute. Rebinding would silently cut the ledger loose from its table.

:::cards
- [Counterfactual regret minimisation](wiki:theory/cfr) One ledger per infoset, fed by a walk of the tree.
- [The ledger in code](wiki:implementation/ledger) `RegretMatcher`, its tests and its callers.
- [Rock-paper-scissors](wiki:implementation/rps) The first graded game, where the ledger is all there is.
:::
