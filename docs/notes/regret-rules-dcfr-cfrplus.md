# DCFR and CFR+ under sampling: what rungs 3 and 4 taught, and the rule that comes out of it

**Short version.** Twice now, once tabular and once neural, the two rules that are supposed
to beat linear CFR lost to it badly. Both times the cause was the same: under sampling, a
discount that depends on a regret's **sign** and runs on its own **clock** stops meaning what
its paper meant. LCFR survives because its weight w(t) = t depends on neither. Unless one of
the conditions below is fixed first, the default for anything sampled in this repo is linear
weighting, and DCFR/CFR+ need a reason to be tried again.

## What each rule does, stated on a running regret total R

- **Vanilla CFR:** R ← R + r. Every iteration's regret counts the same.
- **LCFR:** R ← R + t·r. Iteration t's regret is weighted by t, so early, noisy iterations fade
  *relatively*, but nothing is ever erased and no sign is ever tested.
- **CFR+:** R ← max(R + r, 0). The floor is on the running **total**, not on each sample. An
  action that has been bad stops accumulating debt, so it can come back the moment it turns
  good.
- **DCFR(α, β, γ)** (Brown & Sandholm 2019; here α = 1.5, β = 0): after each iteration k,
  multiply positive totals by k^α/(k^α + 1) (starts at ½, fades toward 1 within a few dozen
  iterations) and negative totals by k^β/(k^β + 1), which is ½ at β = 0. Like CFR+, it tests the
  sign of the **total**.

Both of the "better" rules make two assumptions: (1) there is a running total whose sign you
can test, and (2) one tick of the clock is one round of evidence for every row. Rung 3 broke
assumption 2. Rung 4 broke assumption 1.

## Rung 3, tabular: DCFR collapsed onto CFR+ because of its clock

Four rules raced to 20M hands per seat by lockstep external-sampling MCCFR (README, "The
race"; `figures/rung3/grades.json`). Exploitability in chips per 100 hands:

| hands per seat | vanilla | LCFR | CFR+ | DCFR |
|---|---|---|---|---|
| 1M | 114.5 | 113.1 | 156.4 | 156.0 |
| 10M | 16.7 | 12.6 | 36.9 | 37.0 |
| 20M | 8.6 | **6.1** | 24.8 | 25.0 |

**Why DCFR tied CFR+.** The implementation is correct: `regret_rules.py` applies DCFR lazily
at a row's next visit, and that equals the eager rule bit for bit. The problem is what "per
iteration" means once rows are sampled. A row visited once every g iterations pays g
iterations' discount between two pieces of evidence:

    negative regret at the next visit = (½)^g × what was banked

The reach-weighted median gap in the race was 7,006 iterations (the Oct-7 paper below). The
README states the related plain statistic: at 2M iterations the median row was last visited
about 37,000 iterations earlier. Halving a float64 about 1,075 times gives exactly zero. So
every negative regret was gone before its row came round again, and that is CFR+'s floor.

The positive side failed too. DCFR's positive discount does its work in the first few dozen
ticks. On the global clock those ticks pass before most rows are ever visited. 73% of rows were
first reached after that window, so for them positive regret was never discounted (α → ∞). The
paper's limit is exact: on a global clock with visit probability p → 0,
DCFR(α, β, γ) → DCFR(∞, −∞, γ), which Brown & Sandholm identify with CFR+. The race used the
same linear average for both, so in the limit the two runs were one algorithm. They were within
0.7% at every checkpoint, which is what that predicts.

**Why LCFR was immune.** Two properties, both necessary:

- **Sign-blind.** Weighting by t never treats positive and negative regret differently, so
  idle time cannot shift the balance between an action's good and bad history.
- **Scale-free.** w(t) = t has no built-in time scale. A rare row's updates arrive at
  t ≈ j/p, and the ratio of two of its weights is t_i/t_j ≈ i/j whatever p is. A rare row gets
  the same linear-in-evidence weighting as a frequent one.

> **The rule:** decaying untouched rows is harmless exactly when the decay is sign-blind and
> scale-free. DCFR's discount is neither. CFR+'s floor is sign-dependent, but it has no clock,
> so under sampling CFR+ is still CFR+. Why CFR+ itself lost 3–4× to LCFR was not isolated.
> One plausible reason, untested here: a floored total forgets an action's bad history, and
> sampling noise is what makes that history worth keeping.

Source: `~/Desktop/Claude/poker/cfr-discounting-under-sampling/` ("Discounting on the Wrong
Clock", 2026-10-07, outside the repo), with its proof, the evidence from the real checkpoints,
and a toy simulation where only the clock changes.

## Rung 4, Deep CFR: per-sample DCFR and CFR+ lose 17× because there is no total

Deep CFR stores no running regret. Each refit learns the (t-weighted) **mean** of every regret
sample the reservoir holds, and regret matching reads that mean's sign pattern. So the only
place a rule can act is on the individual samples. `train.rule_targets` (#62) rescales each
stored sample by the weight the rule would have given it at iteration `now`, using **that
sample's own sign**:

    DCFR, positive sample from iteration s:  ∏ₖ₌ₛ^(now−1) k^1.5 / (k^1.5 + 1)
    DCFR, negative sample from iteration s:  (½)^(now − s)
    CFR+:  positive samples keep weight 1; negative samples count only when s = now

**The result** (4 × 256 net, 4,000-step refits, seed 0; `figures/rung4/rule_*-256x4.json`),
exact average in chips per hand:

| hands per seat | linear | DCFR screen | CFR+ screen |
|---|---|---|---|
| 200k (t = 20) | 0.294 | 2.227 | 2.858 |
| 500k (t = 50) | 0.173 | 2.870 | 3.092 |

Both rules are about 17× worse than linear at t = 50, and both get **worse** with training
(DCFR 1.51 → 2.23 → 2.87 over t = 10, 20, 50). Uniform random is 4.84.

**The mechanism, measured** (`figures/rung4/rule_sign_flips.json`, written by
`scripts/deepcfr/rule_sign_flips.py`). By t = 50 the screen keeps only 4.1% (DCFR) or 2.1%
(CFR+) of the memory's negative regret mass, against 82–100% of its positive mass. A sample's
sign is not its cell's sign. A genuinely bad action still draws some positive samples from
lucky walks. When the negative samples are discounted away and the lucky positives are not, the
fitted mean turns positive. 36–38% of the seat-0 cells that are negative under linear weights
turn positive by t = 50 (24–26% for seat 1), up from 8–11% at t = 10. Regret matching plays
every positive cell, so those are bad actions the net is being taught to play, and more of them
every iteration. That is why the curve rises.

A small worked case shows the shape. Take two actions' samples: b = {+1, −3, +1, −3} (bad on
average, −1 per sample) and c = {+0.5, +0.5, +0.5, +0.5} (mildly good). With per-sample
clipping both fit to +0.5, so regret matching plays them equally. CFR+ on the totals would
floor b at zero and play c. A per-sample rule inflates noisy actions, and it is noisy actions
that sampling produces.

**Rescaling targets vs. reweighting the loss: rescaling is the right analog.** The screen
rescales targets and fits with an unweighted loss, so it learns Σcᵢrᵢ / N per cell, not the
weighted mean Σcᵢrᵢ / Σcᵢ. That is intended. Tabular DCFR's total is Σcᵢrᵢ, with no division,
and every action at one spot shares the same N rows, so Σcᵢrᵢ / N is that total times a
positive per-spot factor. Regret matching is invariant to that factor. The weighted mean would
divide each action by its own Σcᵢ, which depends on that action's sample signs, and that would
change the ratios between actions. The linear path is consistent too: dividing by the batch's
mean t is a common factor.

**Two arms are not what they might sound like.** The CFR+ screen keeps negatives from the
current iteration (DCFR(∞, −∞) read per sample). The seed paper's arm (c),
max(r̃, 0) on every sample, drops even those, so it would be at least as biased.

## What this means for future work

1. **Default to linear weighting** for every sampled trainer here, tabular or neural.
2. **A sign-dependent rule needs a total to test.** Tabular has one, Deep CFR does not. A
   Deep CFR version of CFR+/DCFR would need a learned cumulative (or a per-cell running sum on
   the side, which gives up the point of the net), not a per-sample sign.
3. **A clocked rule needs a visit clock.** Under sampling, count a row's own updates, not
   global iterations. The untried rung-3 follow-up is visit-clock DCFR(1.5, 0, 2) and
   (1.5, 0.5, 2). Its code is one commit on the unpushed branch `rung3-dcfr-visit-clock`
   (fe49de4, in the `drawmaha-lockstep` worktree), and it has never been run.
4. **Still untested at rung 4:** tᵞ sample weights with γ = 0 and 2 (sign-blind and
   scale-free, so this note predicts they behave sanely), and the averaging weight on the
   policy memory. At rung 3, linear averaging beat uniform for both vanilla (8.6 vs 15.5) and
   CFR+ (24.8 vs 36.5).
5. **One seed throughout.** The 3–4× and 17× gaps are far outside seed noise. LCFR's edge
   over vanilla at rung 3 (6.1 vs 8.6) is not settled.
