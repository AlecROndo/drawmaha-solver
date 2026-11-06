---
title: Where the project stands
description: The tabular answer to mini-drawmaha, the Deep CFR net beside it at equal hands, and what comes next.
---

Mini-drawmaha is solved twice: once by a table (LCFR, rung 3) and once by a network (Deep CFR, rung 4), both graded by the same exact best-response walk. Full Drawmaha, the game the project exists for, is not started. This page is the scoreboard. The arm-by-arm record is the [run log](wiki:research/run-log), and the next unknowns are under [open questions](wiki:research/open-questions).

Every number below is **exploitability**: what a perfect adversary wins against the strategy, averaged over the two seats, in chips per hand with an ante of 1. Zero means equilibrium. For scale, playing uniformly at random on mini-drawmaha is exploitable for **4.842** (`figures/rung3/grades.json`, key `uniform_random`).

## Every graded game, one row each

| Game | Method | Graded against | Result |
| --- | --- | --- | --- |
| Rock-paper-scissors | regret matching | the uniform Nash | 0.0009 chips/round after 100k rounds |
| Kuhn poker | vanilla CFR | the closed-form equilibrium; all 64 pure strategies | 0.00063 chips/hand; game value −0.05555 against −1/18 |
| Leduc poker | vanilla CFR | an OpenSpiel solve, frozen as a fixture | 0.011 chips/hand; game value −0.0874 against −0.0856 |
| Mini-drawmaha, 6.2M infosets | lockstep MCCFR, four regret rules raced | the exact best-response walk | LCFR wins: 0.061 at 20M hands per seat |
| Mini-drawmaha | Deep CFR | the same walk, and LCFR's answer | 0.068 at 10M hands per seat (LCFR 0.126); still 0.068 at 20M, where LCFR is 0.061 |
| Full Drawmaha | Deep CFR at scale | a head-to-head checkpoint ladder and a trained exploiter | not started |

The table is the README's results table; [Benchmarks](/benchmarks) draws each row's chart.

## LCFR wins the four-rule race on mini-drawmaha

Rung 3 trained four regret rules for 2,000,000 lockstep iterations each — ten hands per seat per iteration, so 20M hands per seat — and graded every checkpoint exactly (`figures/rung3/grades.json`, linear average, seed 0):

| hands per seat | vanilla | LCFR | CFR+ | DCFR |
| ---: | ---: | ---: | ---: | ---: |
| 100k | 3.021 | 3.048 | 3.418 | 3.442 |
| 1M | 1.145 | 1.131 | 1.564 | 1.560 |
| 5M | 0.313 | 0.264 | 0.553 | 0.553 |
| 10M | 0.167 | 0.126 | 0.369 | 0.370 |
| 20M | 0.086 | **0.061** | 0.248 | 0.250 |

The two rules that keep negative regret (vanilla, LCFR) beat the two that throw it away (CFR+, DCFR) by 3–4×. Under sampling a row is revisited so rarely that DCFR's halving of negative regret drives it to exactly zero between visits, which is CFR+'s floor — why that happens, and why linear weighting is immune, is the subject of [regret rules](wiki:theory/regret-rules).

The winner is frozen as the project's tabular answer: LCFR's linear average at 2M iterations, 14.25M probabilities, released as `rung3-lcfr-2m` and graded at **0.061346** (BR₀ −0.034421, BR₁ +0.157114). In self-play the first seat is worth −0.0906 a hand: the second seat acts last in both rounds and at the draw, and that is worth about a tenth of an ante.

## Deep CFR vs the table

Rung 4 replaces the 6.2M ledgers with a network and asks one question at every point of the race's grid:

```eq
price = exploitability(Deep CFR's average) ÷ exploitability(LCFR's average), at equal hands per seat
```

Read it as *how many times more exploitable the net is than the table after both have seen the same hands*. Below 1 the net is ahead; above 1 the table is. The hands are made equal on purpose — Deep CFR walks K = 10,000 hands per seat per iteration, so its iteration t has seen what LCFR saw at lockstep iteration 1,000·t, and the eight marks t = 10 … 2,000 land exactly on the race's checkpoints. Equal hands is not equal compute: a Deep CFR iteration spends most of its time refitting two nets (see [finding 6 of the run log](wiki:research/run-log)).

The best arm — a 4 × 256 net, a 16,000-step refit at learning rate 5e-4, each regret target divided by its spot's pot — was run to the race's endpoint (`figures/rung4/pot_1-256x4-s16000-lr5e-4.json`):

| hands per seat | 1M | 2M | 5M | 10M | 20M |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deep CFR, exact average | 0.114 | 0.084 | 0.071 | 0.068 | 0.068 |
| LCFR, linear average | 1.131 | 0.630 | 0.264 | 0.126 | 0.061 |
| price | 0.10 | 0.13 | 0.27 | 0.54 | 1.12 |
| training clock, hours | 6.2 | 10.6 | 22.5 | 51.7 | 100.7 |

Three readings, all on one seed and one net:

- **Early, the net is about a tenth as exploitable as the table.** At 1M hands the table has visited only a sliver of its 6.2M rows; one learned function covers spots the table has not banked yet. That is consistent with generalisation, but the share of rows LCFR had visited at 1M hands is not measured.
- **The net has a floor and the table does not.** The second 10M hands, 49 hours of training, moved the net from 0.0679 to 0.0685. The table is still falling on its ~1/√N line, so it passes the net somewhere between 10M and 20M hands. That is a measurement, not an extrapolation.
- **Dividing regret targets by the pot moved the floor, not removed it.** The same net with raw-chip targets flattened at 0.108; ÷ pot flattens at 0.068 (−37%). Why the divisor is free and what it reweights is [the pot-normalised targets](wiki:theory/deep-cfr).

![Deep CFR ÷ LCFR at equal hands](fig:rung4/price.png)

:::note
The best graded Deep CFR object — this arm's exact average at 10M hands, 0.0679 — is **not released**. The site shows two earlier releases: `rung4-deep-t500` (raw targets, 5M hands; Π 0.136, exact average 0.108) on [`/solver`](/solver) and the Learn tab, and `rung4-deep-pot0.5-t200` (÷ √pot, 2M hands; exact average 0.086) in the Play tab at [`/play`](/play). Moving either to the 0.068 object is a catalogue edit, an asset upload and a re-export — see [releases](wiki:ops/releases).
:::

## Three objects are graded at each mark, and they usually order exact < Π < current

| Object | What it is | Why it is graded |
| --- | --- | --- |
| **exact average** | CFR's actual answer: every iteration's strategy, weighted by t and by the player's own reach, summed in a table as training runs | the strategy CFR's theory promises converges |
| **Π**, the policy net | a fresh net fit at the mark on the stored strategy samples | the only average a game *without* a table can keep — full Drawmaha will have to use it |
| **current iterate** | regret matching over the net's latest regrets | CFR promises nothing about it; graded to show how far it sits from the average |

The exact average beats Π at every graded mark of every linear arm, so Π's fit costs exploitability (at 20M hands, 0.103 against 0.068). The current iterate is usually worst but not always; none of those exceptions is a bug, because CFR's guarantee is about the average only.

## Next

- **Release the 0.068 object** for `/solver`, Learn and Play — the best graded Deep CFR strategy is sitting frozen under `runs/` unpublished.
- **Move the floor.** A bigger net, a cross-entropy Π, ÷ √pot at the full learning rate, a second seed — the untried arms are listed in [open questions](wiki:research/open-questions).
- **Full Drawmaha.** The engine, the encoder and an evaluation that works without an exact grader: a head-to-head checkpoint ladder plus a trained exploiter. [`/solver`](/solver) already runs on the full game with an illustrative action mix, built ahead of the network. See [full Drawmaha](wiki:research/full-drawmaha).
- **A PPO chapter**: train a PPO agent by self-play and measure what the CFR solution wins against it.
- **A test job in CI.** The only workflow is an automated code review, so the ~1,600-test suite is run by hand.

:::cards
- [The run log](wiki:research/run-log) Every Deep CFR arm, its settings and what its grades say.
- [Open questions](wiki:research/open-questions) What has not been run, and what the literature audit says is missing.
- [Full Drawmaha](wiki:research/full-drawmaha) The target game, what changes from mini, and how it will be graded.
- [The research publication](/research) The same results as nine chapters with figures.
:::
