---
title: Monte Carlo CFR, external sampling and lockstep
description: How rung 3 keeps CFR's walk but samples the deck and the opponent, why the regret weight becomes exactly 1, and how W workers share one table.
---

Vanilla CFR walks the whole tree on every iteration. Leduc's tree is small enough for that. Mini-drawmaha's has **85.1 billion decision nodes**, so a full walk isn't slow, it's impossible. **Monte Carlo CFR (MCCFR)** keeps the walk and samples the parts of it that are expensive to enumerate. Each iteration then costs one sampled hand instead of the whole game, and the expected update is still CFR's. The repo uses **external sampling** ([mccfr.py](repo:src/drawmaha_solver/minidrawmaha/solver/mccfr.py)), parallelised as **lockstep** ([lockstep.py](repo:src/drawmaha_solver/minidrawmaha/solver/lockstep.py)).

## External sampling: sample what is outside the traverser

One traversal per seat per iteration, P0 then P1, each from a freshly dealt root. The traversing seat is the one whose regrets are updated. Rung 2's recursion stays — terminal, chance and decision branches, utilities indexed by position in `legal_actions()`, ledgers banked on the way back up — and two branches change:

| Node | Vanilla CFR | External sampling |
| --- | --- | --- |
| Chance (the deck) | average over every outcome | pick **one** outcome by its probability |
| The opponent's spot | enumerate every action, weighted by σ | pick **one** action from the opponent's current mix |
| The traverser's spot | enumerate every action | enumerate every action, because regret compares the actions and a comparison needs all of them |

## The weight is the sample

In vanilla CFR the regret bank is multiplied by π₋ᵢ · π_c, how often the deck and the opponent bring play here. Under sampling, that product is exactly the probability the walk *arrives here at all*. So the update lands with that **frequency** instead of being multiplied by that **coefficient**, and the regret weight at the traverser's spot is exactly 1:

```eq
at my spot:      R(I, ·) += 1 · ( u − ⟨σ, u⟩ )
at their spot:   S(I, ·) += t · σ(I, ·)
```

- **u[k]** is what the traverser's k-th action earned down the sampled path; **⟨σ, u⟩** is what its current mix earned on the same path. Their difference is one unbiased sample of the counterfactual regret.
- **S** is banked at the **opponent's** spots, where the walk samples from σ anyway. Arriving there already happened with the opponent's own reach, so adding σ once per arrival reproduces the own-reach weight. The traverser's own spots bank no strategy: they were enumerated, not played, and counting them would credit every action with having been chosen.
- **t** makes the average **linear**: iteration t's strategy counts t times as much, on the premise that later iterates are better.

:::warning
**Multiplying π₋ᵢ · π_c in on top applies the probability twice.** Nothing crashes. The opponent's reach, which differs across the worlds of one spot because the opponent's cards differ, ends up squared, so the worlds the opponent rarely produces are under-trained and the solve converges somewhere else. On Leduc that walk stalls at 0.14–0.20 exploitability, where the correct one reaches 0.05, and the two look the same for the first few hundred thousand iterations. Pattern-matching `leduc/cfr.py` is how you'd make this mistake, so `traverse` has no reach parameter to thread through.
:::

:::details Why the linear average is not free under vanilla regret
On Leduc under vanilla regret, the **uniform** averaging column beats the linear one at every length: 1.5–2.7× at 10k iterations, and about 4× by a million (0.012 against 0.050). The docstring calls this a measurement, not yet an explanation. It is consistent with vanilla CFR's regret bound applying to the *uniform* average (linear averaging is backed when the regrets are weighted to match, as in LCFR and CFR+), and with a t-weighted average having a smaller effective sample. Under the other three rules the linear column is the better one. On mini-drawmaha the race found linear beats uniform for both vanilla (8.6 against 15.5 per hundred hands) and CFR+ (24.8 against 36.5). Extra averaging columns cost nothing to keep, because nothing reads an average while training.
:::

## Lockstep: W hands per seat against one frozen σₜ

A one-hand walk writes as it goes, so two processes can't share a table. A **lockstep** iteration walks W hands per seat (one per worker) and writes nothing while it walks:

1. Freeze σₜ. Each of the W workers walks one sampled hand for seat s and records what it *would* write.
2. Once all W are in, every touched row is banked **once**, with the sum of its workers' regrets, and the strategy mass is added. The rows are banked in a fixed order.
3. The other seat does the same against the updated regrets (alternating updates).

The W walks of one seat are a **phase**. This is MCCFR with W samples per iteration. In the race W = 10, so one lockstep iteration is 10 hands per seat, and 2,000,000 iterations is 20M hands per seat.

### Why lockstep is the right shape

- **No races.** The table is read-only during a phase, so the walks can't race. The one write per row happens in a fixed order, so the float sums are the same however the walks were scheduled. [parallel.py](repo:src/drawmaha_solver/minidrawmaha/solver/parallel.py) runs the W walks in W processes; `lockstep.py` runs them one after another and is the reference the processes must equal bit for bit.
- **Every regret rule is exact.** CFR+ floors a row once per iteration, on the total of its W sampled regrets. That is the paper's rule with that total as the iteration's regret, not W separately floored one-hand banks.
- **Textbook semantics.** A walk that reaches the same row twice (suit relabelling can do that) reads σₜ both times, instead of seeing its own first bank the second time.

### Counter-based randomness makes resume exact

Worker w's walk for seat s at iteration t draws from `default_rng([seed, t, s, w])`. A run carries no RNG state, so resuming from any checkpoint is the same as never having stopped. Deep CFR inherits the rule: every draw is a function of (seed, t, seat, walk index), never of W, so arms with different worker counts see identical hands.

## The memory: 6.2M ledgers in four flat arrays

6,220,050 infosets, each a `RegretMatcher` 2–4 wide, would be millions of Python objects. [packed_table.py](repo:src/drawmaha_solver/minidrawmaha/solver/packed_table.py) stores them as four flat arrays — the [ledger contract](wiki:theory/regret-matching)'s `cumulative_regret`, `strategy_sum`, `extra_sums`, `stamp` — and hands out ledgers as views into them, which is why ledger writers update in place and never rebind.

:::cards
- [Regret rules](wiki:theory/regret-rules) Vanilla, LCFR, CFR+ and DCFR raced on this sampler.
- [Mini-drawmaha solver](wiki:implementation/minidrawmaha-solver) `mccfr.py`, `lockstep.py`, `parallel.py`, `packed_table.py`.
- [Deep CFR](wiki:theory/deep-cfr) The same lockstep walk with the regret table replaced by a net.
:::
