---
title: Deep CFR training
description: One iteration of the loop, the oracle pin that proves it is CFR, the resume state, dividing regret targets by the pot, and how runs are launched on a Mac or on Modal and graded at eight grid marks.
---

[`deepcfr/train.py`](repo:src/drawmaha_solver/deepcfr/train.py) is the order the other seven modules are called in, plus everything a 100-hour run needs around them: a resume state that survives a crash, grid-mark saves, a progress line per iteration and a graceful stop. It adds no new math. This page follows one iteration, then the run around it, then how a run is launched and graded.

## One iteration: two phases, each a write, a walk and a refit

Seats alternate, as in lockstep. For each seat p, in order:

```text
for p in (0, 1):
    learner.write(table, nets)                    # both seats' current nets into the regret slots
    for observe in observers: observe(table, t, 1 - p)   # the exact average sees σₜ
    rows = collector.collect(seat=p, t=t, walks=K)        # K walks for seat p
    advantage[p].merge(rows.advantage_rows, rows.advantage_targets, t=t, rng=...)
    policy.merge(rows.policy_rows, rows.policy_targets, t=t, rng=...)
    nets[p] = learner.fit(*advantage[p].contents(), seed=(seed, t, p)).model
```

| Step | Module | What it does in the run |
| --- | --- | --- |
| write | `materialise` | turns both nets into regret slots, so the unchanged walk plays them |
| observe | `average.ExactAverage` | adds t · own reach · σₜ while the whole strategy sits in the table |
| collect | `collect.Collector` | K = 10,000 walks for the seat on W spawned processes |
| merge | `reservoir.Reservoir` | the regret samples into seat p's advantage memory, the opponent's σ into the policy memory |
| refit | `fit.fit_from_scratch` | a fresh net on seat p's memory: 4,000 to 16,000 Adam steps of 10,000 rows |

Materialise runs before **every** phase, twice per iteration, because seat 1's walks at iteration t must sample seat 0's *freshly refit* net. **`step`** runs one iteration in that order and returns **`StepStats`**: seconds per stage per seat, the refits' losses and the rows offered. The refit is about 95% of the time. Moving it to Apple's GPU (`--device mps`) cut a 4 × 256, 8,000-step refit from 127 s to 34 s per seat.

## The learner seam, and the oracle that proves the loop is CFR

The loop never names torch. It talks to a **`Learner`** with four methods, `fit`, `write`, `save` and `load`. **`NetLearner`** is the real one; it imports `fit`, `materialise` and `net` inside those methods, and **`fit_config`** builds its `FitConfig` once torch is first needed. **`Fitted`** is what a refit returns, the model and its loss on a held batch.

**`OracleLearner`** is a "perfect net". Its model (**`OracleModel`**) is a copy of the memory, and its `write` puts at every row the t-weighted sum of every sample there, in LCFR's own float order. With it, Deep CFR collapses into the tabular solver's LCFR. `tests/deepcfr/test_train.py` pins the loop bit for bit over three iterations against LCFR built from lockstep's own `walk`, `flatten` and `apply_arrays`, with each sample rounded through float32 first. Every possible join error breaks that pin: a stale table, the wrong t, the wrong seat, a dropped or reordered row. The test runs in the default suite with no torch.

:::warning
The oracle pin is the correctness gate. A change to the loop's order, to the float32 cast in `collect.rows_of`, or to `regret_rules.bank_rows` breaks it, and if it breaks, the loop is wrong, not the test.
:::

## Dividing each regret target by the pot

**`pot_scaled`** exists because one net fits every spot's mean regret with a single accuracy budget, and the loss decides where that budget goes. Raw chips spend it badly.

- **What each piece is.** A regret is in chips, so a spot's regrets are about the size of its pot. A relative error ε in the net's output there costs (ε · pot)² per sample. Raw squared error therefore weighs a spot by *how often it is reached × pot²*. A 52-chip spot counts 676 times as much as a 2-chip spot reached equally often, and the ante spots where every hand begins are fitted last.
- **Why dividing is free.** The divisor is the pot at the *decision*, which every sample at that spot shares. Each spot's mean regret is scaled by one constant across its actions, and regret matching reads only a row's sign pattern and ratios, so it plays the same σ from scaled regrets as from raw ones. What changes is what the *fitted* net gets wrong, and so the σ it actually plays. A divisor drawn from the sampled hand's outcome would reweight samples *within* a spot and move its mean, which is a different algorithm.
- **What the power k does.** Dividing by potᵏ leaves a weight of how often × pot^(2 − 2k). k = 0 is raw chips (pot²). k = ½ leaves how often × pot, the chips a σ mistake there actually costs. k = 1 weighs every spot alike, whatever its pot.

**`spot_pots`** gives the pot at each of the 141 public points. The default `pot_power` is ½ since 2026-10-27. Measured on one seed at 2M hands per seat, the exact average was 0.115 raw, 0.086 with ÷ √pot and 0.084 with ÷ pot. Run on to 20M hands, the ÷ pot arm flattens at 0.068. `docs/notes/rung4-run-log.md` (findings 7 and 8) has the caveats, including a gradient-clipping confound that no arm has separated yet.

**`rule_targets`** is the screen behind the DCFR and CFR+ arms. It rescales each sample by what the rule would weigh it at iteration `now`. Because Deep CFR stores no running total, the sign it tests is each *sample's* own rather than the cumulative regret's. That is why per-sample DCFR and CFR+ lost to linear by about 17×. See [Regret rules](wiki:theory/regret-rules).

## The run: config, resume state, marks

**`DeepConfig`** is everything a run is:

| Field | Default | Is |
| --- | --- | --- |
| `seed` | 0 | every stream derives from it |
| `walks` (K) | 10,000 | walks per seat per iteration |
| `workers` (W) | 8 | processes; changes nothing in the sample |
| `capacity` (C) | 50,000,000 | rows per reservoir |
| `width`, `depth` | 64, 3 | the net (the best arms ran 256, 4) |
| `steps`, `batch`, `lr`, `clip` | 4,000, 10,000, 1e-3, 1.0 | the refit |
| `threads`, `device` | 8, `cpu` | the refit's arithmetic; `device` may be `mps` |
| `rule` | `linear` | the regret rule; others are the screen |
| `pot_power` | 0.5 | the k above |
| `grid` | 10, 20, 50, 100, 200, 500, 1,000, 2,000 | iterations at which a full mark is kept |
| `save_every_s` | 1,800 | seconds between resume saves |

**`new_run`** builds a fresh **`DeepRun`**: iteration 0, no nets, three empty memories. **`save_run`** writes the three memories, the two latest nets (**`net_name`**: `p{seat}-t{t:05d}.pt`), every **`StatefulObserver`**'s file (the exact average's) and, *last and atomically*, `state.json`. That file is the commit point: a crash anywhere in a save leaves the previous `state.json` naming the previous, complete files. **`load_run`** restores a run, **`saved_config`** reads a save's config from `state.json` alone, and **`config_from_saved`** fills any field in **`ADDED_LATER`** with the value old runs had. `pot_power` is the field that made the rule: runs saved before it existed fitted raw chips, and they must not resume under ½.

**`require_same_run`** refuses a resume that changes the seed, K, C, the net's size, any refit field or the thread count, naming each one. All of these change the sample or the fit. W may change because rows depend on (seed, t, seat, k) and never on W. The device may change on purpose so a run can move to the GPU, at the cost of no longer being reproducible from its seed.

**`run_deep`** trains `out` up to a target, a deadline or a signal: SIGTERM or Ctrl-C finishes the iteration and saves. At each grid mark it keeps a full save under `marks/` (**`mark_name`**: `t{t:05d}`). **`current_iterate`** freezes regret matching over the materialised table for the grader.

## Launching a run

| Where | How |
| --- | --- |
| this Mac | [`scripts/deepcfr/run_deep_cfr.py`](repo:scripts/deepcfr/run_deep_cfr.py): argparse over `DeepConfig` (**`parse`**, **`config_from_args`**); re-running the same `--out` resumes; `--device mps` is checked (**`mps_available`**) before any walk is spent |
| Modal | [`scripts/deepcfr/modal_rung4.py`](repo:scripts/deepcfr/modal_rung4.py): deploy once, then `::smoke`, `::sweep`, `::train`, `::status`, `::grade`; the trainer (one 33-core container) claims its directory on the `drawmaha-rung4` Volume, saves `resume/` every 30 minutes and a mark at each grid point, and spawns its own continuation at 23 hours |

The smoke command, from the script's docstring:

```bash
caffeinate -i uv run --extra deep python scripts/deepcfr/run_deep_cfr.py \
    --out runs/deep-smoke --target 3 --walks 1000
```

On Modal, **`train`** spawns **`train_run`**. **`smoke`** runs **`benchmark`**, which trains three iterations at K = 1,000, resumes one more, grades the mark and prints seconds, hours and dollars for a full run. **`sweep`** launches 3 × 64, 4 × 64 and 3 × 128 to 100 iterations. With `--grade`, **`sweep_winner`** applies the pre-registered rule: stay at 3 × 64 unless another size grades lower by more than 10%. It picked 3 × 128. **`status`** / **`read_status`** print each run's last line.

:::warning
Every `modal run` starts paid containers, and only a person launches one. The Mac runs are the same loop, and their rows are identical to a Modal run's of the same config.
:::

## Grading a mark: three objects, one untouched meter

**`grade`** (via **`grade_mark`** on Modal, or `::grade --local DIR` for a Mac run) grades every kept mark (**`read_marks`**). **`grade_mark_dir`** freezes three objects from a mark directory and grades each with the exact meter in `minidrawmaha/exploitability.py`: the exact average's sums, Π fit fresh on the mark's policy memory (sharing the run's `FitConfig`), and the current iterate over the mark's nets. Results go to `figures/rung4/grades.json` or an arm's own file, keyed `deep/<t>/<column>`. `uv run deep-analysis` then draws the price figure and the answer sheet (see [Mini-drawmaha answer](wiki:implementation/minidrawmaha-answer)).

At K = 10,000, iteration t has seen 10,000 · t hands per seat, the same as LCFR at lockstep iteration 1,000 · t. The eight grid marks therefore sit exactly on the race's 100k to 20M checkpoints, and the price is read at equal hands:

```eq
price = exploitability(Deep CFR's average) ÷ exploitability(LCFR's linear average)
```

![Deep CFR ÷ LCFR at equal hands](fig:rung4/price.png)

## The studies beside the loop

| Script | Question it answered |
| --- | --- |
| [`pi_loss_sweep.py`](repo:scripts/deepcfr/pi_loss_sweep.py) | Π stalled while the exact average kept falling: memory or fit? It refits Π on one stored memory under other losses, heads and schedules (**`loss_fn`**, **`fit`**, **`freeze`**) and grades each. The answer was the fit; cross-entropy beat squared error |
| [`pi_reach_weighted.py`](repo:scripts/deepcfr/pi_reach_weighted.py) | Which refit error tracks exploitability? Per refit, the reach-weighted mean absolute difference from the exact average (**`own_reach`**). That measure tracks it; a count of zeroed bluffs does not |
| [`rule_sign_flips.py`](repo:scripts/deepcfr/rule_sign_flips.py) | How far does a per-sample rule bend the memory? **`seat_counts`** measures mass kept and bad actions turned good under DCFR and CFR+ |

Every arm and its grades is in `docs/notes/rung4-run-log.md`; [Run log](wiki:research/run-log) summarises it.

:::cards
- [Deep CFR package](wiki:implementation/deepcfr) What each of the eight modules does.
- [Deep CFR](wiki:theory/deep-cfr) The algorithm, eq. (3), and the three graded objects.
- [Modal](wiki:ops/modal) Volumes, costs and who may launch what.
- [Run log](wiki:research/run-log) Every arm trained so far.
:::
