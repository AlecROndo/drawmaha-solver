---
title: Regret rules — why linear weighting won, twice
description: Vanilla, LCFR, CFR+ and DCFR differ only in what a visit banks. Under sampling, the two rules that test a regret's sign lost to linear weighting, tabular and neural alike.
---

Once the walk has measured one sample of counterfactual regret, r = u − ⟨σ, u⟩, at one of the traverser's spots, something has to decide how r enters the stored total. That decision is the **regret rule**, and it is the only thing that differs between the four algorithms the repo raced. In the code a rule is a function `bank(ledger, r, t)` and nothing else ([regret_rules.py](repo:src/drawmaha_solver/minidrawmaha/solver/regret_rules.py)). The average isn't the rule's business: every rule banks the same played strategies into the same averages.

**The headline:** on mini-drawmaha the two rules designed to beat linear CFR lost to it by 3–4× tabular and by about 17× neural. Both times the cause was the same. A rule that depends on a regret's **sign**, or runs on its own **clock**, stops meaning what its paper meant once rows are sampled. LCFR survives because its weight w(t) = t depends on neither. Linear weighting is the repo's default for anything sampled.

## The four rules, stated on a running total R

| Rule | Bank | In words |
| --- | --- | --- |
| **vanilla** | R ← R + r | every iteration counts the same |
| **LCFR** (linear CFR) | R ← R + t·r | iteration t counts t times iteration 1, so the early, noisy iterations fade; after T iterations the first tenth carries 1% of the weight. Nothing is erased and no sign is tested. |
| **CFR+** | R ← max(R + r, 0) | the floor is on the running **total**: a bad action stops piling up debt and can come back the moment it turns good |
| **DCFR** (α = 1.5, β = 0) | R ← R + r, then positive entries × t^1.5/(t^1.5 + 1), negative entries × ½ | positive regret fades fast early (factor 0.5 at t = 1, 0.99 by t = 22, 0.99997 at t = 1,000); negative regret halves every iteration — a softer CFR+ |

All but vanilla are from Brown & Sandholm 2019 (arXiv 1809.04040). CFR+'s other half, alternating updates, is already how every iteration runs: P0 traverses, then P1.

:::details DCFR discounted lazily, and why that is exact
As written in the paper, DCFR discounts every regret of the traverser on every iteration, visited or not. Under sampling a row is visited once in thousands of iterations, and touching all 6.2M rows per iteration is out of the question. So a row is discounted only when it is next visited, with all the owed iterations at once. That is exact because an unvisited row receives r = 0, and multiplying by a positive factor never changes a sign. Each entry stays positive or negative over the whole gap, so the per-sign factors simply multiply:

```eq
positive entries:  ∏ₖ k^1.5 / (k^1.5 + 1) = exp( L(t−1) − L(s−1) )
negative entries:  (½)^(t − s)
```

**s** is the row's `stamp`, the last iteration it was brought up to. **L** is the running log of the positive factor. The lazy rule equals the eager one bit for bit.
:::

## Rung 3, tabular: keeping negative regret wins by 3–4×

Each rule trained for 2,000,000 lockstep iterations (20M hands per seat) on Modal, graded by the exact walk. Chips per **100** hands, linear average, seed 0:

| hands per seat | vanilla | LCFR | CFR+ | DCFR |
| --- | --- | --- | --- | --- |
| 1M | 114.5 | 113.1 | 156.4 | 156.0 |
| 10M | 16.7 | 12.6 | 36.9 | 37.0 |
| 20M | 8.6 | **6.1** | 24.8 | 25.0 |

![Keeping negative regret wins](fig:rung3/race.png)

### DCFR collapsed onto CFR+ because of its clock

The implementation is correct. The problem is what "per iteration" means once rows are sampled. A row visited once every g iterations pays g iterations of discount between two pieces of evidence:

```eq
negative regret at the next visit = (½)^g × what was banked
```

- **g**, the gap between visits, was large. At 2M iterations the median row was last visited about 37,000 iterations earlier (the reach-weighted median gap was 7,006).
- Float64 reaches exactly zero after about 1,075 halvings. So every negative regret was gone before its row came round again, which is exactly CFR+'s floor.
- The **positive** side failed too. DCFR's positive discount does its work in the first few dozen ticks of the global clock, and 73% of rows were first reached after that window, so their positive regret was never discounted.

In the limit of rare visits DCFR(α, β, γ) → DCFR(∞, −∞, γ), which Brown & Sandholm identify with CFR+. Both runs used the same linear average, and they were within 0.7% at every checkpoint. They aren't identical, though (24.8 against 25.0): DCFR also discounts positive regret.

### LCFR was immune for two reasons

- **Sign-blind.** Weighting by t treats positive and negative regret alike, so idle time can't tilt the balance between an action's good and bad history.
- **Scale-free.** w(t) = t has no built-in time scale. A rare row's updates arrive at t ≈ j/p, and the ratio of two of its weights is tᵢ/tⱼ ≈ i/j whatever the visit rate p is. A rare row gets the same linear-in-evidence weighting as a frequent one.

:::note
**The rule that follows:** decaying untouched rows is harmless exactly when the decay is sign-blind and scale-free. DCFR's discount is neither. CFR+'s floor has no clock, so it is still CFR+ under sampling. Why CFR+ itself lost to LCFR by 3–4× was not isolated. One untested guess: a floored total forgets an action's bad history, and sampling noise is what makes that history worth keeping. LCFR against vanilla (6.1 against 8.6) is one seed, so suggestive, not settled.
:::

## Rung 4, neural: per-sample DCFR and CFR+ lose by about 17×

Deep CFR stores no running regret. Each refit learns the t-weighted **mean** of the regret samples in the reservoir, and regret matching reads that mean's sign pattern. So the only place a rule can act is on the individual samples. The screen (`train.rule_targets`, PR #62) rescales each stored sample by the weight the rule would give it at iteration *now*, using **that sample's own sign**:

```eq
DCFR, positive sample from iteration s:  ∏ₖ₌ₛ^(now−1) k^1.5 / (k^1.5 + 1)
DCFR, negative sample from iteration s:  (½)^(now − s)
CFR+:  positive samples keep weight 1; negative samples count only when s = now
```

Exact average, chips per hand, 4 × 256 net, 4,000-step refits, seed 0 (`figures/rung4/rule_*-256x4.json`):

| hands per seat | linear | DCFR screen | CFR+ screen |
| --- | --- | --- | --- |
| 200k (t = 20) | 0.294 | 2.227 | 2.858 |
| 500k (t = 50) | 0.173 | 2.870 | 3.092 |

Both get **worse** with training (DCFR 1.51 → 2.23 → 2.87 over t = 10, 20, 50); uniform random is 4.84.

### A sample's sign is not its cell's sign

`figures/rung4/rule_sign_flips.json` measures the mechanism. By t = 50 the screen keeps only 4.1% (DCFR) or 2.1% (CFR+) of the memory's negative regret mass, against 82–100% of its positive mass. A genuinely bad action still draws some positive samples from lucky walks. Once the negatives are discounted away and the lucky positives aren't, the fitted mean turns positive. 36–38% of seat 0's cells that are negative under linear weights turn positive by t = 50, up from 8–11% at t = 10. Regret matching plays every positive cell, so these are bad actions the net is being taught to play, and there are more of them every iteration. That fits the rising curve, but no ablation has isolated it as the only cause.

A worked case shows the shape. Two actions' samples: b = {+1, −3, +1, −3}, bad on average at −1 per sample, and c = {+0.5, +0.5, +0.5, +0.5}, mildly good. Clip per sample and both fit to +0.5, so regret matching plays them equally. CFR+ on the *totals* would floor b at zero and play only c. A per-sample rule inflates the noisy action, and sampling produces noisy actions.

:::cards
- [Deep CFR](wiki:theory/deep-cfr) Where per-sample rules meet a net that keeps no total.
- [The rung-4 run log](wiki:research/run-log) Every arm, its grades, and what each one showed.
- [Mini-drawmaha solver](wiki:implementation/minidrawmaha-solver) `regret_rules.py` and the lazy DCFR stamp.
:::
