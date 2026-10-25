# Walkthrough: #62, the DCFR and CFR+ regret-rule screen

Merged as 5673449. The findings, and what they mean beyond this PR, are in
`docs/notes/regret-rules-dcfr-cfrplus.md`. This note covers how the code does it.

## The question it answers

Rung 3 found that linear weighting (LCFR) beat CFR+ and DCFR by 3–4× in the tabular race. Would
the ranking flip for a network? Deep CFR's refit makes that cheap to ask, because the weighting
rule only enters in one place: what the refit is told to fit.

## The picture

    advantage reservoir: rows (spot), t (iteration drawn), targets (regret per action)
        │
        │  rule = linear  →  targets as stored, loss weighted by t          (the spec's path)
        │  rule = dcfr    →  rule_targets: each sample × DCFR's weight at `now`, loss weight 1
        │  rule = cfr+    →  rule_targets: negatives kept only if t == now,  loss weight 1
        ▼
    fit_from_scratch → net → materialise → regret matching → the next iteration's strategy

The screen changes nothing else: same walks, same reservoirs, same seed, same grader. Any
difference in exploitability comes from the rule.

## Why "screen" and not "the rules"

DCFR and CFR+ test the sign of the **running total** of regret. Deep CFR keeps no total, only
samples, so the screen tests each **sample's** sign instead. That is a different rule, and the
docstring of `rule_targets` says so. The screen's job was to find out whether that
approximation is good enough to be worth pursuing. It is not: it loses 17× and gets worse with
training.

## The pieces, by purpose

- **`rule_targets(targets, t, *, now, rule)`** (`train.py`). It turns a regret rule into a
  per-sample multiplier. For DCFR, a positive sample from iteration s is multiplied by the
  product of every later positive discount (computed as one reversed cumulative sum of logs, so
  the whole memory is rescaled in a single vectorised pass), and a negative one by (½)^(now−s).
  For CFR+, positives keep 1 and negatives keep 1 only if they are from this iteration. An
  unknown rule raises rather than silently fitting linear.
- **`DeepConfig.rule`** (`train.py`). It chooses the path. Under `dcfr` or `cfr+` the refit is
  handed t = 1, because the weight has already been folded into the targets. Weighting by t as
  well would apply the rule twice.
- **`--rule {linear,dcfr,cfr+}`** (`scripts/deepcfr/run_deep_cfr.py`). The command-line face.
- **`seat_counts`** (`scripts/deepcfr/rule_sign_flips.py`). It explains the loss. It reads a
  run's saved advantage memory at each mark, applies the rule exactly as the refit did, and
  counts two things: how much of the memory's negative and positive regret mass survives, and
  how many (spot, action) cells that are bad under linear weights (Σ t·r < 0) the rule turns
  good. A cell that turns good is an action regret matching will now play. Cells whose samples
  are all zero (actions the spot does not offer) are skipped, so padding cannot dilute the
  share.

## What the measurement showed

At t = 50, DCFR keeps 4.1% of the negative mass and CFR+ keeps 2.1%, while 82–100% of the
positive mass survives. 36–38% of seat 0's bad cells and 24–26% of seat 1's have flipped to
good, up from 8–11% at t = 10. The flips grow every iteration, so the strategy degrades every
iteration, which matches the rising exploitability (DCFR 1.51 → 2.23 → 2.87).

## Outputs and how they are made

    uv run --extra deep python scripts/deepcfr/run_deep_cfr.py --out runs/rule-dcfr-256x4 --rule dcfr ...
    uv run --extra deep --with modal modal run scripts/deepcfr/modal_rung4.py::grade \
        --local runs/rule-dcfr-256x4 --grades figures/rung4/rule_dcfr-256x4.json
    uv run --extra deep python scripts/deepcfr/rule_sign_flips.py runs/rule-dcfr-256x4:dcfr runs/rule-cfrplus-256x4:cfr+

`figures/rung4/rule_dcfr-256x4.json`, `figures/rung4/rule_cfrplus-256x4.json` and
`figures/rung4/rule_sign_flips.json` are those commands' outputs.

## Open review notes, settled here

- *Rescaling targets with an unweighted loss fits Σcᵢrᵢ / N, not the weighted mean
  Σcᵢrᵢ / Σcᵢ.* That is intended, and it is the right analog of a tabular total. See the regret
  rules note, "Rescaling targets vs. reweighting the loss".
- *The CFR+ screen keeps the current iteration's negatives, while the seed paper's arm (c) is
  max(r̃, 0) on every sample.* These are different rules. The one implemented is the
  DCFR(∞, −∞) limit read per sample. Arm (c) would discard strictly more negative mass, so by
  the sign-flip mechanism it should do no better. That is a prediction, not a measurement.
- *Is `now=seed[1]` the current iteration?* Yes. Both call sites pass
  `seed=(config.seed, t, p)` (`train.py`), so `seed[1]` is the iteration t being fitted.
