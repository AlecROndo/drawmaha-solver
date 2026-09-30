# Rung 3: lockstep parallel MCCFR, four rules at 10M iterations on Modal

Agreed with Alec 2026-09-30. Follows #40 (packed table) and #41 (regret rules).

## Goal

Race vanilla, CFR+, LCFR and DCFR on mini-drawmaha, each for **10M lockstep iterations**
on **10 cores**, all four at once (40 worker cores), saving checkpoints on a log
scale so `analysis.py` (the next PR) can grade each rule × each average (1, t, t²) ×
the current iterate with the exact grader, and pick the rule for the 100M+ run.

## Semantics: one lockstep iteration

Iteration t, seat 0 then seat 1 (alternating updates, as `run_iteration` does today):

1. The shared table is frozen at σₜ for the whole phase.
2. Each of W workers deals from its own RNG stream and runs one external-sampling
   traversal for the seat, reading σₜ from the shared table and **writing nothing to
   it**. It records into its own scratch buffer: the traverser's regret vector per
   visited row, and σ at each opponent row (the primary sum banks t·σ, column j
   banks wⱼ(t)·σ).
3. Barrier. The parent sums the W buffers in worker order (fixed, so the float sums
   are reproducible) and applies each touched row's total ONCE through the run's
   rule, vectorised. Strategy mass is added. Barrier.

This is MCCFR with W samples per iteration: every sample of iteration t sees σₜ and
each row is banked once with its total, so every rule is exact — including CFR+,
whose per-bank floor (#41's documented approximation) becomes a once-per-iteration
floor. A row reached twice by one traversal (suit relabelling) now reads σₜ both
times instead of seeing its own first bank; that is the textbook semantics.

One run = one rule = one table. A traversal is on-policy for one σ, so it cannot
train another rule's table; the averaging columns are free because they never feed
back into play.

## Components

- **`PackedTable.attach(regret, strategy_sum, extra_sums, stamp, index)`** —
  a table over caller-owned arrays (shared memory), same index, same windows.
- **`regret_rules.bank_rows(rule, table, rows, regret_by_column, t)`** — the four
  rules applied to many rows at once with NumPy (DCFR: owed discount per row from
  its stamp, vectorised `cumulative_log_discount`). Must equal the per-ledger rules.
- **`minidrawmaha/lockstep.py`** — the phase logic with no processes: `Recorder`
  (a table-shaped reader that records deltas instead of writing), `run_lockstep_iteration`
  (W traversals in sequence, then apply). This is the reference the processes must
  match bit for bit.
- **`minidrawmaha/parallel.py`** — W processes (`spawn`), four shared-memory arrays,
  scratch buffers per worker, two barriers per phase; worker streams from
  `SeedSequence(seed).spawn(W)`. `ParallelSolve` exposes `train(iterations)`,
  `save(path)`, `load(path)`; clean shutdown on error or signal (shared memory
  unlinked).
- **Checkpoint**: the #41 format plus `workers` and the W RNG states. Resume refuses
  another worker count. A serial checkpoint cannot be resumed in parallel and vice versa.
- **`scripts/modal_rung3.py`** — Modal app: image from `uv.lock`, Volume
  `drawmaha-rung3`, one function per rule (`cpu=11`, ~6 GB, `timeout=24h`). Each call
  trains until ~23 h of wall clock, saves, and `spawn`s its own continuation, so a run
  survives the 24 h cap with nobody watching (`modal run --detach`). Kept checkpoints
  at 10k, 30k, 100k, 300k, 1M, 3M, 10M (~530 MB each, ~15 GB total), plus a rolling
  resume checkpoint every ~30 min. Progress lines (iteration, it/s, ETA) in the logs.

## Tests (TDD)

- W = 1 lockstep ≡ today's serial learner, byte for byte, on Leduc, every rule and
  columns. (Not on mini-drawmaha: there a traversal that reaches a row twice reads σ
  after its own first bank in the serial learner and σₜ in lockstep — the intended
  difference above.)
- W processes ≡ the in-process lockstep reference with the same W, byte for byte
  (proves no races and a fixed summation order).
- `bank_rows` ≡ the per-ledger rule for every rule, incl. DCFR across the table's end
  and CFR+ on a row with two banks' worth of regret.
- Leduc gate: every rule with W = 4 reaches the LP value at 50k iterations within the
  bounds #41 uses.
- Checkpoint round trip: save at 30, resume, 30 more ≡ 60 straight (parallel).
- Shared memory is released after a normal finish and after a worker raises.

## Launch sequence

1. Local: W = 10 on the Mac for a few minutes → measured it/s and speedup vs W = 1.
2. Modal smoke test (~10 min, ~$1): one rule, measure it/s on Modal's CPUs.
3. Launch all four detached. Expected ~15–23 h per rule (5.6 ms/iteration measured
   locally × Modal's per-core factor); cost ≈ 4 × 11 cores × ~23 h × $0.047 ≈ $50,
   plus memory ≈ $5. If the smoke test says > 30 h per rule, stop and report before
   spending.

## Out of scope

`analysis.py` (grading the kept checkpoints), the homepage chart, a compiled traversal.
