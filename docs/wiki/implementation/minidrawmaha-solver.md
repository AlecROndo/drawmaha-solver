---
title: Mini-drawmaha solver
description: The packed table, external-sampling MCCFR, the four regret rules, lockstep's determinism contract, W processes over shared memory, and the run driver that survives being killed.
---

Mini-drawmaha has 6.2 million infosets, about 21,000 times Leduc's 288, and a tree of 85 billion decision nodes, so a full pass per iteration does not run at all. [`solver/`](repo:src/drawmaha_solver/minidrawmaha/solver) keeps the shape of the earlier walks and changes three things. It stores the ledgers in four flat arrays, samples the deck and the opponent instead of enumerating them, and walks W sampled hands per seat against one frozen strategy so that W processes can share one table. The race this machinery ran, four regret rules for 20M hands per seat on Modal, is what picked LCFR as the tabular answer.

## Files

| File | What it is for |
| --- | --- |
| [`regret_rules.py`](repo:src/drawmaha_solver/minidrawmaha/solver/regret_rules.py) | What one visit banks: vanilla, LCFR, CFR+, DCFR, and the extra averaging columns |
| [`packed_table.py`](repo:src/drawmaha_solver/minidrawmaha/solver/packed_table.py) | The only table: every ledger as a window into four flat arrays |
| [`infoset_table.py`](repo:src/drawmaha_solver/minidrawmaha/solver/infoset_table.py) | The ledger-per-infoset interface over the packed table, and the readouts |
| [`mccfr.py`](repo:src/drawmaha_solver/minidrawmaha/solver/mccfr.py) | External sampling, one hand per seat per iteration |
| [`lockstep.py`](repo:src/drawmaha_solver/minidrawmaha/solver/lockstep.py) | W hands per seat against one frozen σₜ: the reference |
| [`parallel.py`](repo:src/drawmaha_solver/minidrawmaha/solver/parallel.py) | The same in W processes over shared memory, equal bit for bit |
| [`lockstep_run.py`](repo:src/drawmaha_solver/minidrawmaha/solver/lockstep_run.py) | Drive a run to its target: resume, grid checkpoints, progress log |
| [`exploitability.py`](repo:src/drawmaha_solver/minidrawmaha/exploitability.py) (one level up) | The exact meter; imports `rules/` only |

## The packed table: 0.3 GB instead of 5.4 GB

The first version held one `RegretMatcher` object per key: 3.4 GB and half a minute to build. Ten of those cannot share a machine. **`PackedTable`** keeps the numbers and drops the objects, in four public arrays laid out in `all_infosets()` order:

| Array | Shape | What it holds |
| --- | --- | --- |
| `cumulative_regret`, `strategy_sum` | float64, total width | every ledger's columns end to end |
| `extra_sums` | float64, k × total width | the extra averaging columns |
| `stamp` | int64, one per *ledger* | DCFR's discount stamp |

`table[key]` returns a ledger *window* (`RegretMatcher.over`) onto that key's slice. **`whole_game`** builds the full table, **`listed`** builds one over chosen keys (the tests use it), and **`over`** lays the index over buffers someone else owns: shared memory, or a loaded checkpoint. **`row_of`** and **`slot_of`** locate a key's row and columns. **`widths`**, one per row, is the table's fingerprint.

`infoset_table.py` keeps the familiar interface on top: **`new_infoset_table`**, plus **`ledger_widths`**, which counts ledgers by width without allocating. A ledger is 2 wide at an open or a shove, 3 facing a bet and **4 at the draw**. Column 0 is CHECK_CALL at the open, FOLD facing a bet and THROW_NONE at the draw. A uniform 4-wide table would bank regret for throwing a card at a betting spot, and that phantom column could win the normalisation. **`StrategyView`**, **`current_strategy`** and **`average_strategy`** compute strategies lazily, and **`spots`** and **`format_strategy_tables`** print one hand's strategy for eyeballing.

## External sampling: sample the deck and the opponent, enumerate yourself

`mccfr.py` keeps Leduc's recursion and changes two branches. A chance node follows *one* outcome drawn by its probability, and an opponent's spot follows *one* action drawn from the opponent's current mix. The traverser's own spots are still fully enumerated, because regret is a comparison between actions and a comparison needs all of them. Since the walk arrives at a spot exactly as often as π₋ᵢ · π_c, the sample *is* the weight:

```eq
at my spot:     R(I, ·) += 1 · (u − ⟨σ, u⟩)
at their spot:  S(I, ·) += t · σ(I, ·)
```

**`traverse`** is that walk. **`run_iteration`** deals a fresh hand for P0 to traverse and another for P1, alternating. **`Solve`** is everything needed to continue a run (table, t, RNG, rule, columns), with **`new_solve`** and **`train`** around it, and **`save_solve`** / **`load_solve`** / **`read_run`** for flat `.npz` checkpoints. **`State`** is the nine-member interface the walk reads, which `LeducState` and `MiniState` both satisfy, so the same walk was calibrated on Leduc first. See [MCCFR](wiki:theory/mccfr).

## The four regret rules

`regret_rules.py` is the only thing that differs between the rules the race compared. Each `bank_*` function takes one sampled regret r = u − ⟨σ, u⟩:

| Rule | Bank | In plain words |
| --- | --- | --- |
| vanilla (**`bank_vanilla`**) | R += r | every iteration counts the same |
| LCFR (**`bank_linear`**) | R += t · r | iteration t counts t times iteration 1, so noisy early iterations fade |
| CFR+ (**`bank_cfr_plus`**) | R ← max(R + r, 0) | negative regret is floored at zero, so a late-good action is played at once |
| DCFR (**`bank_discounted`**) | pay the discounts owed since the stamp, then add | positive and negative regret are discounted at different rates |

Under sampling a row can go 37,000 iterations between visits, so DCFR's per-iteration discount cannot be applied eagerly. The ledger's `stamp` records how far the discount has been paid, **`settled_regret`** reads a row's regret as of iteration t without touching it, and **`cumulative_log_discount`** keeps the product in log space. **`bank_rows`** banks many rows at once for the packed path. **`RegretRule`** and **`Average`** name the rules and averaging weights (1, t or t²) a checkpoint records. **`validate_averages`**, **`extra_weights`** and **`column_average`** handle the extra columns, which cost nothing and cannot change play. The mechanism behind why CFR+ and DCFR lost is in [Regret rules](wiki:theory/regret-rules).

:::note
A traverser row can be banked twice in one traversal, through the suit relabelling, so "at most once per iteration" is false here. DCFR therefore defers its discount to the row's next visit, and CFR+ floors after every bank. Do not "simplify" either to the textbook once-per-iteration form.
:::

## Lockstep: W walks against one frozen σₜ

`lockstep.py` makes one iteration W sampled hands per seat, one per worker, all reading the same σₜ and writing nothing while they walk. **`walk`** returns a **`Recorded`**, the writes it *would* have made. **`apply`** then banks each touched row once, in worker order, with the sum of its regrets. The W walks of one seat are a *phase*, and seats alternate. Two consequences matter:

- **It is exact MCCFR with W samples per iteration.** CFR+ floors once per iteration, on the W-sample total.
- **It is a determinism contract.** **`stream`** derives each walk's random stream from (seed, t, seat, k), never from W or scheduling. A run's floats are therefore the same however the walks were run.

**`LockstepSolve`**, **`new_lockstep`** and **`train_lockstep`** wrap a run. **`Flat`**, **`flatten`** and **`apply_arrays`** are the array form a worker hands back, banked vectorised. **`save_lockstep`** writes atomically, **`load_lockstep`** resumes, and **`is_lockstep`** / **`read_lockstep`** inspect a checkpoint without loading it.

## `parallel.py`: W processes, no locks

**`ParallelLockstep`** runs each walk in its own process over one packed table in shared memory (**`packed_table_for`** builds it). One phase:

1. The parent writes (t, seat) into a control block and posts each worker's `go` semaphore.
2. Each worker walks its hand, reading σₜ and writing nothing, copies its record into its own scratch block and posts `done`.
3. Once `done` has been taken W times, the parent banks the W records in worker order while every worker waits.

The table is only written while nobody reads it, so no lock is needed. Semaphores are used rather than `multiprocessing.Barrier`, whose crossing cost about 2 ms on macOS, more than a walk. `test_parallel.py` holds **`train`** and **`step`** equal to `lockstep` bit for bit at the same W.

## `lockstep_run.py`: a run that survives being killed

**`run_to(out, rule=..., target=..., workers=...)`** keeps everything in `out/`: `resume.npz`, rewritten every few minutes; `iter-<t>.npz` at each grid mark (**`kept_name`**), never overwritten; and `progress.jsonl`, one line per save. Each save is a temp file renamed into place, then its progress line, then one commit (a Modal Volume's commit, or nothing locally). SIGTERM and SIGINT finish the iteration and save. Calling `run_to` again resumes, and because the streams are counter-based a resumed run equals one that never stopped. A resume refuses a different W or rule, by name.

## The meter: vectors over the public tree

`exploitability.py` asks the same question as Leduc's meter, sum over hidden worlds first and then one max, but it cannot tag 8.5 × 10¹⁰ concrete worlds. It walks the 141-point *public* tree instead, carrying a vector over the opponent's private states. At each responder point it carries a vector of values over the responder's states:

```eq
v[b, r] = Σₕ w[b, h] · M_b[r, h]     (a terminal, board b)
v[b, r] = maxₐ vₐ[b, r]                (a responder spot)
```

r is the responder's private state and h the opponent's. w is the reach vector, how often the deck and the frozen opponent bring h here. M_b is the showdown matrix on board b, zero where the hands share a card. **`minidrawmaha_game`** compiles the rules into a **`VectorGame`** of arrays once: **`Node`**, **`Edge`**, **`DrawMap`**, **`Level`**, **`KeyMap`** and **`Showdown`** are its parts. **`exploitability`**, **`best_response_value`** and **`expected_value`** answer as before. **`gains_by_infoset`** splits a seat's exploitability over infosets, together with how often each is reached; this is what `minidraw-exploit-by-reach` bins. A grade holds about 1.5 GB of showdown tables and takes about two minutes.

## Running it

| Where | Command |
| --- | --- |
| this machine | `caffeinate -i uv run python scripts/minidrawmaha/run_lockstep.py lcfr --out runs/lockstep-lcfr` ([`run_lockstep.py`](repo:scripts/minidrawmaha/run_lockstep.py)) |
| Modal, the race | [`modal_rung3.py`](repo:scripts/minidrawmaha/modal_rung3.py): `::smoke`, `deploy`, `::main --target 2000000 --rules ...`, `::status`; one 11-core container per rule (10 workers + parent) on the `drawmaha-rung3` Volume, self-continuing past Modal's 24-hour cap |
| grade one checkpoint | `uv run python scripts/minidrawmaha/grade_checkpoint.py <ckpt.npz> [--column quadratic]` ([`grade_checkpoint.py`](repo:scripts/minidrawmaha/grade_checkpoint.py)), about 2 GB |
| grade the race | `uv run minidraw-analysis --grade-runs runs/modal` adds new checkpoints to `figures/rung3/grades.json` |

:::warning
Modal costs money, and its launchers are started by a person, never by an agent on its own. The race's checkpoints are 555 MB each and live on the Modal Volume, never in git. See [Modal](wiki:ops/modal).
:::

![Keeping negative regret wins](fig:rung3/race.png)

:::cards
- [MCCFR](wiki:theory/mccfr) External sampling and why the sample is the weight.
- [Regret rules](wiki:theory/regret-rules) Why LCFR beat CFR+ and DCFR under sampling.
- [Mini-drawmaha answer](wiki:implementation/minidrawmaha-answer) What was frozen from the race, and how the site reads it.
:::
