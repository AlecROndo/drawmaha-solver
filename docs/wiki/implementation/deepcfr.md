---
title: Deep CFR package
description: Eight modules that replace the regret half of mini-drawmaha's 6.2 million ledgers with a network — without changing the walk, the ledger or the grader.
---

Mini-drawmaha's tabular solve keeps one ledger per infoset. That works for 6.2 million infosets and is impossible for full Drawmaha, whose table could never exist. [`deepcfr/`](repo:src/drawmaha_solver/deepcfr) replaces the ledgers' **regret** half with a network that reads what the player can see, the 136 features from `rules/features.py`, and outputs one number per action. Everything else stays the same: the walk is mini-drawmaha's `lockstep.walk`, the ledger's `strategy()` is unchanged, and the exact grader is untouched. Mini-drawmaha's answer is the yardstick. At equal hands per seat, how exploitable is Deep CFR compared with LCFR?

## The trick: write the net into the table, don't teach the walk about nets

The walk already plays regret matching over `cumulative_regret`. So instead of changing the walk, **`materialise`** writes each net's outputs *into* the regret slots before every phase, and the walk plays the net through the ledger it always read. The table becomes a cache of the net's opinion, rewritten twice per iteration.

## Eight modules, one concern each

| Module | What it is for |
| --- | --- |
| [`net.py`](repo:src/drawmaha_solver/deepcfr/net.py) | The MLP, chunked prediction into NumPy, and its save file |
| [`materialise.py`](repo:src/drawmaha_solver/deepcfr/materialise.py) | Two seats' nets into the table's regret slots: Deep CFR's eq. (3) obtained by what is written |
| [`fit.py`](repo:src/drawmaha_solver/deepcfr/fit.py) | The from-scratch refit on (row, t, target) arrays |
| [`reservoir.py`](repo:src/drawmaha_solver/deepcfr/reservoir.py) | The memories the refit reads: Algorithm R, with an exact resume |
| [`collect.py`](repo:src/drawmaha_solver/deepcfr/collect.py) | K lockstep walks for one seat on W processes, returned as memory rows |
| [`train.py`](repo:src/drawmaha_solver/deepcfr/train.py) | The loop that calls them in order, its resume state, marks and stop, behind a `Learner` seam |
| [`average.py`](repo:src/drawmaha_solver/deepcfr/average.py) | The exact average strategy: own reach through a predecessor table, summed every phase |
| [`policy.py`](repo:src/drawmaha_solver/deepcfr/policy.py) | Π, the average learned by regression on the policy memory: the only average a game without a table can keep |

## `net.py`: 136 → 64 → 64 → 64 → 7, linear head

**`Net`** has `depth` hidden layers of `width` ReLU units and a linear head of 7, one output per `Action` in enum order, so an output column means the same thing at every spot. A spot reads its 2, 3 or 4 legal columns through `legal_gather`; the rest are trained on nothing and read by nobody. The head is linear because the outputs stand in for cumulative regrets, which are signed, and `materialise` needs the sign. Width and depth are parameters so one trainer can run the size sweep. The default 3 × 64 has 17,543 parameters, a number a test pins. The best arms ran 4 × 256. **`predict`** runs chunks of 262,144 rows under `inference_mode` and returns float32 NumPy. **`save_net`** / **`load_net`** store the size and weights, and loading uses `weights_only=True` so a file cannot execute code.

## `materialise.py`: eq. (3) by the write

Deep CFR plays regret matching over the net's outputs when any is positive, and the single best action when none is. The ledger cannot express the second clause, since it reads all-negative as uniform. So **`regret_slots_from_outputs`** writes:

| Net outputs at a spot | Written into the slots | The ledger then plays |
| --- | --- | --- |
| some positive | the outputs as they are | regret matching (negatives clipped) |
| all ≤ 0, not all equal | a one-hot at the argmax (shared by ties) | the best action |
| all equal (untrained net, exact tie, no net) | zeros | uniform |

**`materialise`** does this for every row of the whole game, one pass per public point, because each point's rows are a contiguous slice beside one public feature row.

## `fit.py`: throw the net away, fit a new one

Deep CFR does not update a net between iterations. It fits a fresh one to everything in memory. **`fit_from_scratch`** takes three equal-length arrays: `rows` (which table row; turned into the 136 features, never seen as a number), `t` (the iteration that offered the sample) and `targets` (4 wide, padded). **`batch_loss`** is per-row squared error over the row's legal columns, with padding masked *before* the subtraction so a padded NaN never reaches a gradient. Each row is weighted by t over the batch's mean t, borrowed from Linear CFR, so later and better samples outvote early ones. **`FitConfig`** is the net's size, Adam's steps, batch, rate, clip, threads and device (`cpu` or `mps`). Nothing here knows what a target *is*, so the same fit serves the regret nets and Π.

## `reservoir.py`: a uniform sample of everything ever offered

Over 2,000 iterations the walks offer about 280 million advantage rows per seat. A memory keeps C = 50 million. **`Reservoir`** uses Algorithm R so that every offer is resident with probability C/n and no iteration is favoured by *when* it happened. The first C offers fill the slots. Offer g ≥ C draws j uniformly from [0, g] and overwrites slot j if j < C. **`merge`** offers one phase's rows at once. **`contents`** returns read-only `(rows, t, targets)` for the refit, and **`size`**, **`offered`** and **`capacity`** report the counts. **`reservoir_stream`** seeds each merge from (seed, t, seat, **`Memory`**), and **`save`** / **`load`** resume exactly. A row is 24 bytes (`int32` row, `int32` t, `float32 × 4` target), so a memory is 1.2 GB.

## `collect.py`: walks in, memory rows out

**`Collector`** owns the run's table and W spawned workers for the whole run. **`collect`** runs walks 0..K−1 of one seat at iteration t against the table as it stands. The walk records, per traverser visit, `(key, slot, r)` with r = u − ⟨σ, u⟩, and per opponent visit the strategy it sampled from. **`rows_of`** reshapes one walk's record into a **`Collected`**: *advantage* rows (traverser rows and their r, padded to 4) and *policy* rows (opponent rows and their σ). **`joined`** lays K of them end to end. σ is read from the walk's `uniform` averaging column, whose weight is t⁰ = 1, so it is σ bit for bit; dividing `t · σ` by t would not be. Walk k's stream is `lockstep.stream(seed, t, seat, k)`, so the rows never depend on W. A Mac run and a Modal run of one config see identical hands.

:::warning
Anything handed to the collector reaches a spawned worker by pickle, so it must be importable by module name in a fresh process: a module-level function, never a lambda or a closure. A learner's `write` assigns *into* `table.cumulative_regret` and never rebinds it, because the workers read that array through a shared block.
:::

## `average.py`: CFR's actual answer, rebuilt exactly

The strategy played at iteration t never settles; only the average converges. The tabular solver got the average for free in `strategy_sum`, and Deep CFR throws the table away every iteration. **`ExactAverage`** rebuilds it (Single Deep CFR). While each iteration's whole strategy sits materialised in the table, **`accumulate`** adds

```eq
sums(I, a) += t · own_reachₜ(I) · σₜ(I, a)
```

and **`frozen`** normalises each row for the grader. Own reach is the weight because the average strategy must reproduce what the seat actually played, so a spot it rarely steers into counts for little. **`own_reach`** pushes reach through the 141 public points using **`predecessors`** (a **`Pred`** per point) and **`current_strategy`**. A post-draw key has three pre-draw predecessors, since the key forgets which card was drawn, and all of them count, with multiplicity. A brute-force walk in `tests/deepcfr/test_average.py` is the gate. **`save`** / **`load`** make it part of the resume set, because the sum cannot be rebuilt later except by replaying every net.

## `policy.py`: Π, the average a net can carry

At a game whose table cannot exist, `ExactAverage` is impossible. The original Deep CFR keeps the average anyway. Every opponent visit copies `(row, t, σ)` into a policy reservoir, and a fresh net is fit to it with the same t-weighted squared error. The output minimising Σ tᵢ(y − σᵢ)² at a spot is Σ tᵢσᵢ / Σ tᵢ, the t-weighted mean, and the reach weight rides in how often each spot was sampled. **`fit_policy`** fits Π at a mark under **`policy_seed`**. **`policy_strategy`** freezes it over every key: clipped, normalised, uniform if all zero. Π carries two errors the exact average does not, the reservoir's sampling and the fit's. Measuring both on mini-drawmaha is the reason it is graded.

## Three objects graded at every mark

| Object | Built by | Answers |
| --- | --- | --- |
| exact average | `average.ExactAverage` | what CFR's guarantee is about |
| Π | `policy.fit_policy` on the mark's policy memory | what full Drawmaha will have to use |
| current iterate | `train.current_iterate`, regret matching over the materialised table | what CFR never promises |

:::cards
- [Deep CFR training](wiki:implementation/deepcfr-training) The loop, its resume state, ÷ pot, and how runs are launched and graded.
- [Deep CFR](wiki:theory/deep-cfr) The algorithm and why the exact average and Π differ.
- [Mini-drawmaha rules](wiki:implementation/minidrawmaha-rules) The 136 features the net reads.
:::
