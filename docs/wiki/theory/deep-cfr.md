---
title: Deep CFR — the regret table replaced by a network
description: Why full Drawmaha needs a net instead of a table, how rung 4 swaps one in without touching the walk or the grader, and what it has measured so far.
---

A table needs one row per infoset. Mini-drawmaha's 6.2M rows fit in memory; full Drawmaha's don't come close. **Deep CFR** (Brown, Lerer, Gross & Sandholm 2019) runs CFR with the regret table replaced by a neural network that predicts each spot's regrets from features of the spot. Rung 4 ([deepcfr/](repo:src/drawmaha_solver/deepcfr)) puts it on mini-drawmaha, a game where the exact answer and an exact grader already exist. The cost of the net can then be read off at every point:

```eq
price = exploitability(Deep CFR's average) ÷ exploitability(LCFR's linear average),  at equal hands per seat
```

- **The numerator** is graded by the same exact meter as rung 3 (see [Exploitability](wiki:theory/exploitability)).
- **The denominator** is the rung-3 race's LCFR at the same number of sampled hands.
- **Equal hands**: Deep CFR at K = 10,000 walks per seat per iteration has seen at iteration t the same hands LCFR had at lockstep iteration 1,000·t (LCFR's W = 10). So the eight marks t = 10, 20, 50, 100, 200, 500, 1,000, 2,000 land exactly on the race's 100k–20M checkpoints.
- There is no pass mark. The ratio is recorded and read.

## One iteration is rung 3's lockstep with the regret half swapped out

For each seat in turn ([train.py](repo:src/drawmaha_solver/deepcfr/train.py)):

1. **Materialise.** Write both seats' nets into the packed table's regret slots ([materialise.py](repo:src/drawmaha_solver/deepcfr/materialise.py)). The walk then plays regret matching over the table as before, with `RegretMatcher.strategy()` unchanged.
2. **Average.** Let the exact average see the iteration's strategy while it sits in the table ([average.py](repo:src/drawmaha_solver/deepcfr/average.py)).
3. **Walk.** K hands on W processes ([collect.py](repo:src/drawmaha_solver/deepcfr/collect.py)), returning the traverser's regret samples and the opponent's strategy samples as rows.
4. **Remember.** Merge the rows into fixed-size reservoirs, 50M rows each ([reservoir.py](repo:src/drawmaha_solver/deepcfr/reservoir.py)).
5. **Refit from scratch.** Throw the seat's net away and fit a fresh one to its whole memory ([fit.py](repo:src/drawmaha_solver/deepcfr/fit.py)).

### Materialise writes eq. (3) into the slots

Deep CFR plays regret matching over the net's outputs when any output is positive, and the single best action when none is (eq. 3 of the paper). Plain regret matching would read all-negative as *uniform*, so materialise writes the verdict in a form the ledger already reads correctly:

| Net outputs at a spot | Written to the slots | The ledger then plays |
| --- | --- | --- |
| some positive | the outputs as they are | regret matching (negatives clipped) |
| all ≤ 0, not all equal | a one-hot at the argmax, shared by ties | the best action(s) |
| all equal (an untrained net, an exact tie, no net) | zeros | uniform |

It is 141 batched forward passes, one per public point, each written with one slice assignment.

### The net and its loss

The net is a plain MLP ([net.py](repo:src/drawmaha_solver/deepcfr/net.py)): 136 input features describing what the actor can see, then hidden layers, then **7 signed outputs**, one per action in a fixed order, so a column means the same thing at every spot. The head is linear because the outputs stand in for signed regrets; a ReLU or softmax would throw away the sign materialise needs. The default is 3 × 64 (17,543 parameters); the graded arms went to 3 × 128 and 4 × 256.

The refit minimises a t-weighted squared error:

```eq
loss = mean over rows of  ( t / mean t ) · Σ_legal a ( net(I)[a] − target[a] )²
```

- **target** is a sampled regret r = u − ⟨σ, u⟩ at row I's legal columns. Padding columns are masked out *before* the subtraction.
- **t / mean t** is Linear CFR's weight carried into the fit: early iterations' regrets were sampled against near-random strategies and say the least about the game. Dividing by the batch's mean t keeps the loss on one scale as the run lengthens.
- **Squared error** means the net learns the t-weighted *mean* regret of each spot, which is what regret matching needs.
- Adam, gradient norm clipped at 1.0, rows drawn uniformly with replacement from the memory.

:::details Why a reservoir, and why Algorithm R
Over 2,000 iterations the walks offer about 280M advantage rows per seat, and the memory keeps C = 50M. The kept rows should be a uniform sample of everything offered, so no iteration is favoured by when it happened. Algorithm R does that in one pass: the first C offers fill the slots, and offer g ≥ C draws j uniformly from [0, g] and overwrites slot j if j < C. By induction every offer is resident with probability C/(g+1). The draws come from a counter-based stream seeded by (seed, t, seat, memory), and the offer count n is saved with the memory, so a resumed run merges exactly as the uninterrupted one would. A row is 24 bytes: row (int32), t (int32), four float32 targets.
:::

## Three averages are graded, because they answer different questions

CFR's guarantee is about the average strategy, and Deep CFR throws the table away every iteration, so nothing keeps the average for free. Rung 4 keeps it two ways and grades both, plus the current iterate:

| Object | What it is | Why grade it |
| --- | --- | --- |
| **Exact average** | Σₜ t · own_reachₜ(I) · σₜ(I, a), summed per column while each iteration's strategy sits in the table (Single Deep CFR) | CFR's actual answer: no sampling or fitting error in the averaging itself |
| **Π, the policy net** | a fresh net fit on the policy memory, the (row, t, σ) rows the opponent's walks copied | the only average a game *without* a table can keep, so it is what full Drawmaha will use |
| **Current iterate** | regret matching over the materialised table | CFR never promises it converges; graded to see by how much |

The exact average's weight is the textbook own reach: the sum, over the histories a key stands for, of seat q's *own* action probabilities on the way there. The opponent's probabilities are left out because the average is of q's play, not of how the opponent happened to be playing. A post-draw key has three pre-draw predecessors, all of which count, so own reach is pushed through a predecessor table over the 141 public points.

Π gets the same weighting by regression. At a spot with memory rows σ₁, σ₂, σ₃ at weights t₁, t₂, t₃, the output y minimising Σ tᵢ·(y − σᵢ)² is Σ tᵢσᵢ / Σ tᵢ, the t-weighted mean, even though no single row held it. Reach is carried by the **sample counts**, since walks visit a spot in proportion to how often play arrives. Π carries two errors the exact average doesn't: the reservoir's sampling and the fit's. The exact average beat Π at every graded mark of every linear arm.

## Dividing each target by the pot spreads the fit's accuracy by chip cost

One net fits every spot with one accuracy budget, and the loss decides where it goes. A regret is in chips, so a spot's regrets are about as large as its pot, and a relative error ε there costs (ε · pot)² per sample. Raw squared error therefore weighs a spot by **how often × pot²**. The 52-chip spots count 676 times the 2-chip spots they are reached as often as, and the ante spots, where every hand begins, are fitted last.

Divide every target at a spot by pot^k before the fit:

- **It's free for the regrets themselves.** The divisor is shared by every action and every sample at the spot, so the spot's mean regret is scaled by one constant. Regret matching reads only signs and ratios, so it plays the same σ from scaled regrets as from raw ones (`pot_scaled` in `train.py`; a test pins the invariance). Only the net's *errors*, and so the σ it actually plays, move.
- **What k does.** The weight left is how often × pot^(2 − 2k). k = ½ weighs by how often × pot, the chips a σ mistake at that spot actually costs, and is the default. k = 1 weighs every spot alike.

Measured on one seed, 4 × 256 at 16,000 steps and lr 5e-4: at 2M hands raw 0.115, ÷ √pot 0.086, ÷ pot 0.084 chips/hand. The two pot weightings are tied within the noise.

:::warning
**One confound is unresolved.** `fit.py` doesn't normalise its targets, so dividing by up to √52 ≈ 7.2 also shrinks the gradients that the 1.0 clip sees. Part of the gain may come from a different clipping regime rather than the reweighting. A raw arm with a looser clip would separate the two, and it hasn't been run.
:::

## The net leads early, flattens, and the table passes it

The ÷ pot arm, run to the race's endpoint (`figures/rung4/pot_1-256x4-s16000-lr5e-4.json`), exact average in chips per hand:

| hands per seat | 1M | 2M | 5M | 10M | 20M |
| --- | --- | --- | --- | --- | --- |
| Deep CFR, exact average | 0.114 | 0.084 | 0.071 | 0.068 | 0.068 |
| LCFR at the same hands | 1.131 | 0.630 | 0.264 | 0.126 | 0.061 |

- **Early, the net generalises.** At 1M hands it is a tenth as exploitable as the table: one learned function covers spots the table hasn't banked yet. Equal hands isn't equal compute, though. Two large refits make an iteration cost tens of seconds to minutes, and the refits are about 95% of the time.
- **Then it flattens.** The last 10M hands, 49 hours of training, moved it from 0.0679 to 0.0685. The table, still on its ~1/√N line, passes it between 10M and 20M hands. This is a measurement on one seed and one net, not an extrapolation.
- **Width was the lever.** 3 × 64 → 3 × 128 cut the 1M-hand grade by 36%. Depth did 2–9%, and doubling the refit 12–24%.

![Deep CFR ÷ LCFR at equal hands](fig:rung4/price.png)

The [exploitability decomposition](wiki:theory/exploitability) shows *where* the two differ. The net leaks about three times the table per visit at the commonest spots and a fifth of it at the rarest.

:::cards
- [Deep CFR in code](wiki:implementation/deepcfr) The modules, their contracts and their tests.
- [Training runs](wiki:implementation/deepcfr-training) The launchers, the Mac runner and how arms are graded.
- [The rung-4 run log](wiki:research/run-log) Every arm, every grade, what is not yet run.
:::
