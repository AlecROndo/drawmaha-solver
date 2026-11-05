---
title: The Deep CFR run log, digested
description: Every Deep CFR arm trained on mini-drawmaha, its settings, its grades, and the nine findings they add up to.
---

The lab notebook for Deep CFR is [`docs/notes/rung4-run-log.md`](repo:docs/notes/rung4-run-log.md): every arm, where it ran, with which net and refit, what the exact grader said. This page is a reader's digest of it. When the two disagree, the notebook wins — and the notebook itself loses to the files under `figures/rung4/`, which `tests/deepcfr/test_rung4_quotes.py` checks it against.

## The settings every arm shares

All arms are **seed 0**, **K = 10,000 walks per seat per iteration**, 50M-row reservoirs and 10,000-row minibatches. So iteration t has seen 10,000·t hands per seat, exactly what LCFR saw at lockstep iteration 1,000·t, and the two can be compared at equal hands.

The rows a run collects never depend on the worker count W: every walk is drawn from (seed, t, seat, walk index). A Mac run and a Modal run of the same config therefore see identical data, and only the refit's floating-point order can differ. That is what lets arms trained on different machines sit in one table.

## The scoreboard: the exact average at each mark

Net is depth × width; "÷ pot^k" is how the regret targets were scaled before the refit (0 = raw chips). Blank means not graded.

| arm | net | refit steps | lr | ÷ pot^k | 100k | 200k | 500k | 1M | 2M | 5M | 10M | 20M |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LCFR (the table)** | — | — | — | — | 3.048 | 2.697 | 1.876 | 1.131 | 0.630 | 0.264 | 0.126 | **0.061** |
| 3 × 64, Mac | 3 × 64 | 4,000 | 1e-3 | 0 | 0.581 | 0.444 | 0.396 | | | | | |
| sweep 3 × 64, Modal | 3 × 64 | 4,000 | 1e-3 | 0 | | | | 0.382 | | | | |
| sweep 4 × 64, Modal | 4 × 64 | 4,000 | 1e-3 | 0 | | | | 0.375 | | | | |
| sweep 3 × 128, Modal | 3 × 128 | 4,000 | 1e-3 | 0 | | | | 0.243 | | | | |
| 3 × 128 to t = 200 | 3 × 128 | 4,000 | 1e-3 | 0 | | | | | 0.234 | | | |
| 3 × 128, double fit | 3 × 128 | 8,000 | 1e-3 | 0 | | | 0.213 | 0.184 | | | | |
| 4 × 128 | 4 × 128 | 4,000 | 1e-3 | 0 | | | 0.220 | 0.221 | | | | |
| 4 × 256 | 4 × 256 | 4,000 | 1e-3 | 0 | | 0.294 | 0.173 | 0.152 | | | | |
| 4 × 256, double fit | 4 × 256 | 8,000 | 1e-3 | 0 | | 0.330 | 0.167 | 0.134 | | | | |
| 4 × 256, 4× fit, half rate | 4 × 256 | 16,000 | 5e-4 | 0 | 0.643 | 0.330 | 0.169 | 0.133 | 0.115 | 0.108 | | |
| 4 × 256, 4× fit, full rate | 4 × 256 | 16,000 | 1e-3 | 0 | 0.850 | | | 0.127 | 0.104 | | | |
| … half rate, ÷ √pot | 4 × 256 | 16,000 | 5e-4 | ½ | 0.879 | | | **0.108** | 0.086 | | | |
| **… half rate, ÷ pot, to t = 2,000** | 4 × 256 | 16,000 | 5e-4 | 1 | | | | 0.114 | **0.084** | **0.071** | **0.068** | 0.068 |
| DCFR screen | 4 × 256 | 4,000 | 1e-3 | 0 | 1.515 | 2.227 | 2.870 | | | | | |
| CFR+ screen | 4 × 256 | 4,000 | 1e-3 | 0 | 2.539 | 2.858 | 3.092 | | | | | |

Each arm's file under `figures/rung4/` also holds Π, the current iterate, each seat's best response, the training clock and the full config. The JSON names spell nets width × depth: `sweep_256x4` is four hidden layers of 256.

## The nine findings

### 1. Width was the lever that mattered

At 1M hands, going from 3 × 64 to 3 × 128 cut the exact average from 0.382 to 0.243 (−36%). An extra layer did far less: −2% at 64 wide, −9% at 128 wide. The sweep's rule was fixed before its numbers existed — stay at 3 × 64 unless another size is lower by more than 10% — and it picked 3 × 128. Later arms on the Mac went wider still, to 4 × 256 (0.152).

### 2. A longer refit helps, by less than a wider net

Doubling the refit from 4,000 to 8,000 Adam steps took 3 × 128 from 0.243 to 0.184 (−24%) and 4 × 256 from 0.152 to 0.134 (−12%). The gain shrinks as the net grows — what you would expect if the small net was under-fit and the large one is near its fit's floor. Early on a longer fit can be *worse* (4 × 256 at 200k hands: 0.330 against 0.294), so arms are compared at 500k hands or later.

### 3. The net is far ahead early, then flattens

At 1M hands the best raw arm is 0.133 against the table's 1.131 — an eighth. But from 1M to 5M hands the 16,000-step arm moved 0.133 → 0.115 → 0.108 (−19% over five times the hands) while LCFR fell 1.131 → 0.630 → 0.264 (−77%). That arm was stopped at t = 506; its two averages at t = 500 are the release `rung4-deep-t500` that [`/solver`](/solver) shows. Finding 8 measures where the curves cross.

### 4. The three graded objects usually order exact < Π < current

The exact average beats Π at every graded mark of every linear arm. The current iterate is usually worst but not always — the 16,000-step arm's current iterate beats both averages at 100k hands. CFR promises convergence of the average only, so none of these orderings is a bug.

### 5. Per-sample DCFR and CFR+ lose to linear by 17×, and get worse with training

At 500k hands the two screens grade 2.870 and 3.092 against linear's 0.173, and both *rise* with t. A sampled trainer has no running regret total whose sign the rules could test, so they test each sample's sign instead, and a bad action's positive samples outvote its negative regret. The mechanism, at both rungs, is written up in [`docs/notes/regret-rules-dcfr-cfrplus.md`](repo:docs/notes/regret-rules-dcfr-cfrplus.md) and on [regret rules](wiki:theory/regret-rules). Linear is the only rule worth running further.

### 6. The refit is the iteration

The two refits are about 95% of an iteration's time. Moving them to Apple's GPU (`--device mps`, PR #61) cut a 4 × 256, 8,000-step refit from 127 s to 34 s per seat. The 16,000-step arms cost about 70 s per seat per refit, about 152 s an iteration.

### 7. Dividing each regret target by its spot's pot moved the plateau: −25% at 2M hands

Three 16,000-step arms differ only in the divisor k:

| targets | 1M | 2M |
| --- | ---: | ---: |
| raw chips (k = 0) | 0.133 | 0.115 |
| ÷ √pot (k = ½) | 0.108 | 0.086 |
| ÷ pot (k = 1) | 0.114 | 0.084 |

**What problem it solves.** One net fits every spot's regrets with one accuracy budget, and the squared-error loss decides where that budget goes. A regret is in chips, so its size scales with the pot; squared error therefore weighs a spot by how often it is reached × pot². The 52-chip spots count 676 times as much as the 2-chip ante spots reached equally often, and the ante spots — where every hand begins — are fitted last.

**Why dividing is free.** The divisor is the pot at the decision, shared by every sample at that spot, so it scales the spot's whole regret row by one constant. Regret matching reads only the row's signs and ratios, so it plays the same strategy from the scaled regrets as from the raw ones. What changes is only where the net's *errors* land.

**What k moves.** Dividing by pot^k leaves a weight of how often × pot^(2 − 2k). k = ½ weighs a spot by the chips a mistake there costs; k = 1 weighs every spot alike. ÷ √pot is the default for new runs since 2026-10-27.

**The caveats.** One seed each, and the only noise estimate is two raw runs 0.005 apart; the pot arms' gain is five to six times that. The two weightings are tied at 2M hands. The targets are not otherwise normalised, so dividing also shrinks the gradients the clip of 1.0 sees — part of the gain may be a changed clipping regime rather than the reweighting.

### 8. Run to the race's endpoint, the ÷ pot net flattens at 0.068, and the table passes it between 10M and 20M

| hands per seat | 1M | 2M | 5M | 10M | 20M |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact average | 0.114 | 0.084 | 0.071 | 0.068 | 0.068 |
| Π | 0.152 | 0.115 | 0.098 | 0.114 | 0.103 |
| current iterate | 0.197 | 0.214 | 0.175 | 0.163 | 0.191 |
| LCFR | 1.131 | 0.630 | 0.264 | 0.126 | 0.061 |
| training clock, hours | 6.2 | 10.6 | 22.5 | 51.7 | 100.7 |

To four figures the last two exact grades are 0.0679 and 0.0685: the second 10M hands bought nothing. The plateau of finding 3 is real for this weighting too — 37% lower than the raw arm's, not removed. Π does not track the exact average at these marks.

## The arms that did not count

- **The Modal main run.** A 3 × 64 run with target 500 was started on Modal on 2026-10-23 and stopped at t = 64, six minutes after the sweep picked 3 × 128. Its marks are kept on the volume `drawmaha-rung4` and ungraded; they are only worth grading as a Modal-vs-Mac reproducibility check.
- **The 16,000-step arm's stop rule.** Before its marks were graded, the rule said: go past 1M hands only if the exact average there is below 0.121 (10% under the double-fit arm's 0.134). It came in at 0.133, so by the rule the longer fit bought nothing; it was run on anyway to see the curve's shape, which is what finding 3 reads.

## How an arm is run and graded

A Mac arm, as the ÷ √pot one would be launched today (÷ √pot is now the default):

```bash
uv run --extra deep python scripts/deepcfr/run_deep_cfr.py --out runs/pot0.5-256x4-s16000-lr5e-4 \
    --target 200 --walks 10000 --workers 16 --width 256 --depth 4 --steps 16000 \
    --lr 5e-4 --device mps --save-every-s 600
```

Each mark is graded as it lands, locally and without Modal spend, through the Modal app's grade entry point:

```bash
uv run --extra deep --with modal modal run scripts/deepcfr/modal_rung4.py::grade \
    --local runs/<arm> --marks 200 --grades figures/rung4/<file>.json
```

A grade takes several minutes and competes with a live trainer for cores, so grade one mark at a time. Add the arm's row to the notebook when it is graded. Running on Modal itself is covered in [Modal](wiki:ops/modal).

### 9. Distilling the table into the net splits the plateau into capacity and training

The LCFR table's average at iteration 10,000,000 (graded 0.0157) fitted straight into the rung-4 net by reach-weighted cross-entropy grades 0.0389 at 4 × 256 and 0.0175 at 4 × 512, against Deep CFR's 0.0685. So at Deep CFR's size about 44% of the plateau is what the net cannot hold and 56% is what its training did not find, and one doubling of the width removes almost all of the first part. [Distilling the table into the net](wiki:research/distillation) has the method, the ablations and the caveats.

:::cards
- [Where the project stands](wiki:research/status) The scoreboard in one page.
- [Open questions](wiki:research/open-questions) The arms not yet run.
- [Deep CFR](wiki:theory/deep-cfr) The algorithm the arms train.
- [Training Deep CFR](wiki:implementation/deepcfr-training) The loop, the launchers and the flags.
:::
