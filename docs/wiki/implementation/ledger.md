---
title: The regret ledger
description: RegretMatcher, the one learning rule every game in the repo trains, and why its two weights are keyword-only.
---

Every solver in the repo, from rock-paper-scissors to Deep CFR, learns through the same small class: [`RegretMatcher`](repo:src/drawmaha_solver/regret_matching.py) in `regret_matching.py`. One ledger sits at each decision point a player can face. It remembers how much better each action would have done than the mix actually played, and it turns that memory into the next strategy. The games around it grow by eight orders of magnitude, and this class barely changes.

## What one ledger holds: two vectors, nothing derived

A ledger over n actions stores two persistent vectors and two extra slots:

| Slot | Type | What it is |
| --- | --- | --- |
| `cumulative_regret` | float64, n | R(a): accumulated regret for each action |
| `strategy_sum` | float64, n | S(a): the banked strategy mass, the thing that becomes the answer |
| `extra_sums` | float64, k × n | k further averages of the same strategies under other weightings (uniform, quadratic), banked beside the primary one; k = 0 unless a run asks |
| `stamp` | int64, 1 | the last iteration a lazily applied discount has been brought up to (only DCFR uses it) |

The **current** strategy and the **average** strategy are never stored. Both are read off these vectors when asked for, so nothing inside a ledger is ever "already averaged".

## `strategy()`, `average_strategy()` and `update()`, in plain words

- **`strategy()`**: what this spot plays right now. It keeps only the positive regrets and normalises them, so each action is played in proportion to how much it has been regretted. When no regret is positive, it plays uniformly.

```eq
σ(a) = R⁺(a) / Σₐ′ R⁺(a′),   with R⁺(a) = max(R(a), 0)
```

- **`average_strategy()`**: the answer. It normalises `strategy_sum`, and plays uniformly if nothing was ever banked. In two-player zero-sum games this average converges to equilibrium, while the current strategy cycles forever. [Regret matching](wiki:theory/regret-matching) shows both behaviours on rock-paper-scissors.
- **`update(utilities, *, regret_weight, strategy_weight)`**: record one visit. `utilities[a]` is what action a would have earned. It is read off a payoff matrix in rock-paper-scissors and computed by a subtree walk in the poker games.

```eq
R += regret_weight · (u − ⟨σ, u⟩)
S += strategy_weight · σ
```

Read the first line as follows. ⟨σ, u⟩ is what the mix actually earned, so u − ⟨σ, u⟩ is how much better each pure action would have done. Positive entries are regrets, and negative ones say the action did worse than the mix. The second line adds the strategy that was played into the running sum the average is read from.

`update` validates its input before it mutates anything. A wrong-width vector, a NaN utility or a negative weight is refused and leaves the ledger exactly as it was. A NaN in particular would poison both accumulators for good, and every later `strategy()` would quietly return NaN.

## Why the two weights are keyword-only

In rock-paper-scissors every round counts the same, so both weights default to 1. In a game tree a spot is only reached some of the time, and the two accumulators want different scales:

| Weight | Is | Why that scale |
| --- | --- | --- |
| `regret_weight` | the **counterfactual** reach π₋ᵢ: how often chance and the opponent bring play here, excluding your own choices | a line you currently avoid must keep learning at full strength, or it can never be rediscovered |
| `strategy_weight` | your **own** reach πᵢ (times t in linear averaging) | the average has to reproduce what you actually played |

Pass them swapped and every game still converges, quietly, to the wrong answer. Kuhn with the weights transposed reports a converged-looking grid that folds the king to a bet 18% of the time. For that reason both weights are keyword-only, and the root [`AGENTS.md`](repo:AGENTS.md) forbids making them positional. The derivation of the two reaches is in [CFR](wiki:theory/cfr).

:::warning
Never make `regret_weight` or `strategy_weight` positional. A call site that transposes them runs without an error and produces a wrong solve; the only defence is that the transposition cannot be written by accident.
:::

## `over`: a ledger that is only a window

At mini-drawmaha's 6.2 million infosets one Python object per ledger cost 3.4 GB, rising to 5.4 GB once the two extra slots were added. **`RegretMatcher.over(...)`** builds a ledger whose four arrays are *views* into somebody else's flat buffers, with no copy. That somebody is the packed table. An `update` therefore lands in the table itself, and the ledger object can be thrown away after the visit. `over` checks every view's shape and dtype, because a wrong-width view or a float32 buffer would not crash: it would bank into a neighbour's columns or lose precision without a word.

That is also why every writer changes the arrays in place (`+=`, `[:] =`) and never rebinds an attribute. Rebinding would silently cut the ledger loose from its table. [Mini-drawmaha solver](wiki:implementation/minidrawmaha-solver) covers the packed table this serves.

## Four callers, from rock-paper-scissors to Deep CFR

| Caller | How it feeds the ledger |
| --- | --- |
| `rps/players.py` → **`RegretMatchingPlayer`** | samples from `strategy()`, then banks the payoff column of the finished round, both weights 1 |
| `kuhn/cfr.py`, `leduc/cfr.py` → `walk` | computes the utilities by recursion; banks at counterfactual and own reach |
| `minidrawmaha/solver/mccfr.py`, `lockstep.py` | sampled walks; regret weight 1 (the sample *is* the weight), the regret rule decides how r is banked |
| `deepcfr/materialise.py` | writes a net's outputs into `cumulative_regret` so `strategy()` plays the net, without touching this class |

:::cards
- [Regret matching](wiki:theory/regret-matching) Why playing in proportion to positive regret converges on average.
- [CFR](wiki:theory/cfr) Where the two reach weights come from.
- [Rock-paper-scissors](wiki:implementation/rps) The ledger with nothing around it.
:::
