# Rung 4, milestone 1: Deep CFR on mini-drawmaha, graded exactly, run on Modal

Draft for Alec's review, 2026-10-07. Follows #45 (lockstep MCCFR, Modal race) and
#46 (frozen LCFR strategy). Alec's 2026-10-07 decisions are requirements here; the
one change since the handoff breadcrumb is that **the run goes on Modal, not the
Mac** (Alec, 2026-10-07: "please use Modal, I'd prefer Modal over local compute").

## The question: how much more exploitable is a network than the table, at equal hands

Rung 3 solved mini-drawmaha tabularly: LCFR, 2M lockstep iterations x 10 hands per
seat = 20M hands per seat, exploitability 0.061 chips/hand by the exact grader.
Milestone 1 trains Deep CFR on the same game with the same walk, and reports

    price = exploitability(Deep CFR's average) / exploitability(LCFR's average)

**at equal hands per seat**, with wall-clock beside it. There is no go/no-go
threshold; the number is recorded and Alec decides what it means.

Equal hands is a grid, not one point. The race kept LCFR at 10k, 20k, 50k, 100k,
200k, 500k, 1M and 2M iterations (x 10 hands); Deep CFR at K = 10,000 walks per seat
per iteration lands on the same hands at iterations **10, 20, 50, 100, 200, 500,
1,000 and 2,000**. Every Deep CFR mark is graded three ways: the exact average
(Single Deep CFR), the distilled policy net Pi, and the current iterate.

| hands/seat | tabular iteration | Deep CFR iteration | LCFR linear average |
|-----------:|------------------:|-------------------:|--------------------:|
| 100k | 10,000 | 10 | 3.048 |
| 200k | 20,000 | 20 | 2.697 |
| 500k | 50,000 | 50 | 1.876 |
| 1M | 100,000 | 100 | 1.131 |
| 2M | 200,000 | 200 | 0.630 |
| 5M | 500,000 | 500 | 0.264 |
| 10M | 1,000,000 | 1,000 | 0.126 |
| 20M | 2,000,000 | 2,000 | 0.061 |

(LCFR numbers from `figures/rung3/grades.json`; uniform random grades 4.842.)

## What is reused unchanged, and what each reused piece gives us

- `mccfr.traverse` + `lockstep.walk`/`_Recorder`/`Recorded`/`Flat`: one walk against
  a frozen table returns, per traverser visit, `(key, slot, r)` with
  `r = u - <sigma, u>`, and per opponent visit `(key, slot, t*sigma, extra)`. Those
  ARE the advantage-memory and policy-memory rows. Asking the walk for the
  `uniform` extra column makes `extra[0] = sigma` exactly, so sigma is never
  recovered by dividing by t. The walk's RNG stream is `default_rng([seed, t, seat,
  k])`, so K walks indexed k = 0..K-1 are the same sample however many processes run
  them.
- `PackedTable.whole_game()`: 6,220,050 rows in `all_infosets()` order; `slot_of`,
  `widths`, `cumulative_regret`. Materialising the net means writing this array.
- `RegretMatcher.strategy()`: positive regrets normalised, uniform when none is
  positive. Deep CFR's eq. (3) differs only in its all-negative clause, which is
  obtained by WHAT IS WRITTEN (a one-hot at the argmax), not by changing the ledger.
- `exploitability.py` and `strategy.FrozenStrategy`/`average_rows`/`save_strategy`:
  every graded object is frozen as a `FrozenStrategy` file and graded by the
  untouched grader, exactly as the release strategy was.
- `scripts/modal_rung3.py` + `lockstep_run.run_to`: the Volume, claim/heartbeat,
  23-hour continuation, `progress.jsonl`, `smoke` benchmark and `status` pattern.
- `kuhn_deep_cfr.py` (Desktop): the reference for the reservoir, the from-scratch
  refit with t'-weighted MSE, the argmax clause and the exact own-reach average.

## The loop, in rung-3 names

```
theta = {0: None, 1: None}                    # None = untrained = uniform play
for t in 1..T:
    for p in (0, 1):                          # alternating updates, as lockstep does
        materialise(table, theta)             # eq. (3) into table.cumulative_regret, both seats
        accumulate_exact_average(table, t, seat=1-p)   # the strategy seat 1-p is about to be sampled from
        rows_V, rows_Pi = collect(table, K, seat=p, t)   # K walks of lockstep.walk, W processes
        M_V[p].merge(rows_V); M_Pi.merge(rows_Pi)        # Algorithm R, one process
        theta[p] = fit_from_scratch(M_V[p], seed=(seed, t, p))   # fresh init, Adam, clip 1
        save(theta[p], t)                                        # every checkpoint kept
    at grid marks: save memories, exact-average sums, Pi fit on M_Pi, current iterate
```

Why the exact average is accumulated during training rather than rebuilt at the
end: the loop already materialises every iteration's strategy, so adding
`t * own_reach_t(I) * sigma_t(I)` into a running numerator and `t * own_reach_t(I)`
into a running denominator costs one pass per iteration and no replay. Every
checkpoint is still kept, and a test rebuilds the average from the checkpoints of a
short run and checks it equals the running sums (Single Deep CFR, both ways).

Which strategy counts as "iteration t's" for seat q: the one the OTHER seat's walks
sampled from at iteration t, because that is what the tabular `strategy_sum` banks
and what the policy memory records. With alternating updates that is seat 1's
strategy during seat 0's phase and seat 0's (freshly refit) strategy during seat 1's.

## Inputs: card, discard, board, draw and betting features, never a key id

One fixed-length vector per key, built from the key's two halves. Nothing in it
identifies the key; two keys that share cards and lines share inputs.

| group | encoding | size |
|---|---|---:|
| hole (3 cards) | card multi-hot (15) + rank counts (5) + suit counts (3) | 23 |
| discarded (0 or 1 card) | same three parts, zeros when nothing thrown | 23 |
| board card 1 | same | 23 |
| board card 2 | same, zeros before it is dealt | 23 |
| draw signals | P0: {not yet, 0, 1}; P1: {not yet, 0, 1} | 6 |
| betting | per round, 5 slots x {fold, check/call, pot} one-hot, zeros for unused slots | 30 |
| chips | pot / 26, each seat's chips behind / 26 | 3 |
| seat to act, stage | one-hot 2 + one-hot {round 1, draw, round 2} | 5 |
| **total** | | **136** |

Rank and suit counts are the one-hot analogue of Brown et al.'s rank + suit + card
embeddings: they let the net share what it learns about "any pair" or "two of a
suit" across cards. Suits are already canonical in the key. The private half
(groups 1 to 4) is precomputed per shape (970 / 10,170 / 100,400 keys), the public
half (groups 5 to 8) per public point (141); a key's row is `concat(private[shape]
[position], public[point])`, so no 6.2M x 136 matrix is ever stored, and a memory
row needs only the key's row number.

## The network: a 3 x 64 MLP with one output per Action

`136 -> 64 -> 64 -> 64 -> 7`, ReLU, about 17,000 parameters, as the DEFAULT: the
hidden width and the number of hidden layers are run parameters (`--width`,
`--depth`), so one trainer serves the size sweep below. The head is 7 wide, one
output per member of `game.Action` (fold, check/call, pot, the four throws), so an
output column means one thing at every spot. A spot's ledger columns are a gather of
its `legal_actions()` into that head (2 or 3 of the first three at a betting spot,
the last four at the draw); the gather is per public point (141 x 4 int8) and never
per key. The policy net Pi has the same shape and its own weights.

Fresh random initialisation every iteration, Adam at lr 1e-3, gradient norm
clipped to 1, loss = squared error over the legal columns, each row weighted by its
iteration t' (divided by the batch's mean weight, as the Kuhn reference does, so the
scale does not drift with t). 4,000 steps of 10,000 rows per refit (Brown et al.,
decided 2026-10-07). Seeded
`torch.manual_seed` per (seed, t, seat), fixed thread count, CPU only.

Measured on this Mac (8 threads, synthetic data of the right shapes; one forward
pass over the 6.2M keys / one refit of 4,000 x 10,000): 3x32 0.18 s / 8.2 s;
**3x64 0.32 s / 10.8 s**; 4x64 0.37 s / 12.4 s; 3x128 0.81 s / 18.6 s; 3x256
2.5 s / 48 s. The refit, not the forward pass, is the cost, and below 3x64 the
per-step overhead dominates, so a smaller net buys nothing.

## Materialising: the net's outputs become regret-slot values

For each public point, run the seat's net over the point's keys, gather the legal
columns, and write into `cumulative_regret[start:start+width]`:

- some output positive: the outputs as they are (negatives included; the ledger
  clips them);
- every output <= 0 and not all equal: a one-hot at the argmax (eq. 3's clause,
  obtained by the write);
- all equal (the untrained net, or exact ties): zeros, which the ledger reads as
  uniform.

`ledger.strategy()` on that table is then Deep CFR's eq. (3) and the walk is
unchanged. An untrained seat (iteration 1) writes zeros everywhere: uniform.

## Memories: three reservoirs of 24-byte rows, merged in one process

A row is `(row: int32, t: int32, target: float32 x 4)`, the target padded to the
widest ledger. Advantage memory per seat, policy memory shared. Vitter's Algorithm R
with a counter-based stream per phase, so a resumed run merges the same way. The
count of rows ever offered, `n`, is part of the checkpoint. Capacity C = 50M rows
per memory (1.2 GB each, 3.6 GB for the three; decided 2026-10-07). An advantage
memory is offered about 140,000 rows per iteration, so it holds the complete
history until about iteration 360 and, at 2,000 iterations, an 18% uniform sample
of the ~280M rows ever offered. The refit's cost does not depend on C: every step
samples 10,000 rows by index and builds their features on the fly, so no
C x 136 matrix exists. What C does cost is checkpoint traffic — 3.6 GB per save of
the memories — so the resume file is rewritten every 30 minutes rather than
rung 3's 10, and the net checkpoints (tens of KB) every iteration.

## Collecting: K walks per seat per iteration on W processes

`collect(table, seat, t, seed, walks=K, workers=W)` runs walk k on process k mod W
and returns the concatenated `Flat` arrays plus, for policy rows, the row number
recovered from the column start (`searchsorted` over the table's cumulative widths).
Processes fork on Linux (Modal) and share the read-only table copy-on-write; on the
Mac they go through the same shared-memory blocks `parallel.py` uses. The sample is
a function of (seed, t, seat, k), never of W.

## The exact average needs up to three predecessor rows per key

Own reach of seat p at key I under sigma_t is the sum over the physical histories
in I of the product of p's own action probabilities on the way. On mini-drawmaha
that is a pass over the 141 public points in walk order, because every
predecessor point precedes its successors, with a precomputed predecessor table:

| key | its previous own decision | rows |
|---|---|---:|
| round-1 betting, first own action | none (reach 1) | 0 |
| round-1 betting, later | same hole, same board card, line prefix two actions shorter | 1 |
| the draw | the seat's last round-1 decision | 1 |
| round-2, stood pat | the draw key with the same hole, board[:1] | 1 |
| round-2, threw a card | the draw key for each of the three pre-draw holes (hole minus one card, plus the thrown card), with the throw column of that card in its canonical draw order | 3 |
| round-2, later in the line | same key, line prefix two actions shorter | 1 |

The three-row case is why the plan's `previous_own_row` (one row per key) is not
enough: the key deliberately forgets which card was drawn, so three pre-draw holes
lead to it, and all three count (verified 2026-10-07 on a sample of 101 post-draw
discard keys: 96 have three distinct predecessors, 5 have two distinct ones that
still count as three physical histories). Chance factors (deal, replacement) are
equal across the three and constant over t, so they cancel in the per-row
normalisation; only own reach moves with t. A test checks the vectorised pass
against a brute-force own reach walked through `MiniState` for a handful of keys.

## Three graded objects per grid mark, all as FrozenStrategy files

- **exact average**: numerator / denominator from the running sums (uniform where
  the denominator is zero), `average_rows` -> `FrozenStrategy`.
- **Pi**: a fresh policy net fit on M_Pi as it stands at the mark (t'-weighted MSE on
  the recorded sigma), run over every key, clipped at 0 and renormalised (uniform
  if all zero) -> `FrozenStrategy`. Trained at every mark, not only at the end, so
  the curve has Pi at each point.
- **current iterate**: `RegretMatcher.strategy` over the materialised table ->
  `FrozenStrategy`.

Each is graded by `exploitability.py` unchanged. The grades land in
`figures/rung4/grades.json` keyed `deep/<iteration>/<exact|pi|current>` with
`hands_per_seat`, `wall_clock_s` and the run's config beside them.

## Modal: one trainer container, graders fanned out, everything on a Volume

`scripts/modal_rung4.py`, on the rung-3 pattern:

- Volume `drawmaha-rung4`: `nets/p{seat}-t{iteration}.pt` (every checkpoint),
  `marks/t{iteration}/{memories.npz, exact.npz, pi.npz, current.npz}`,
  `resume.npz` (memories, running sums, t), `progress.jsonl`, `owner.json`.
- Image: `debian_slim` + numpy + torch (CPU wheel) + `drawmaha_solver`.
- `train`: one container, W + 1 cores and 16 GB (default W = 32; `smoke`
  measures), the claim/heartbeat/23-hour continuation of rung 3, resumable from
  `resume.npz`, which is rewritten every 30 minutes (it carries the 3.6 GB of
  memories); the eight marks' `memories.npz` are about 29 GB on the Volume.
- `grade`: one container per grid mark (2 GB for the grader), grades the three
  files, returns the JSON; `main --grade` fans them out and writes `grades.json`.
- `smoke`: 3 iterations at K = 1,000, reports seconds per step.
- `sweep`: three trainers at once, one per net size (3 x 64, 4 x 64, 3 x 128),
  100 iterations each under `sweep/<size>/`, then `grade` on each at iteration
  100; the size the 2,000-iteration run uses is the clear winner, else 3 x 64.

Budget (to be replaced by `smoke`'s measurement; Modal measured 10.5 ms a walk):

| step, per iteration | seconds on 33 cores |
|---|---:|
| materialise both seats, 141 batched passes each | ~3 |
| 20,000 walks on 32 workers | ~7 |
| two refits, 4,000 x 10,000, 8 threads each | ~15 |
| reservoir merges, exact-average pass | ~1 |
| **one iteration** | **~26** |

So 400 iterations is about 3 hours and 2,000 about 15 hours, at Modal's listed CPU
rate about $14 and $70 respectively (plus a few dollars of memory). The Mac would
be 2 to 4x faster per core, but Alec wants Modal, and the 23-hour continuation
already exists.

## Correctness gates, run before the paid run

1. **Exactness pin, no torch.** With an oracle fit (the memory unbounded, the "net"
   returning each key's t'-weighted regret sum) the loop after one iteration must
   equal `train_lockstep(rule="lcfr", workers=K)` after one iteration bit for bit
   on the whole table. This pins recorder -> memory -> materialise -> walk.
2. **Reservoir uniformity**: every row offered is resident with probability C/n
   (chi-square over trials), and a resumed reservoir merges identically.
3. **Materialise unit tests**: positive outputs pass through, all-negative writes
   the argmax one-hot, ties write zeros; `ledger.strategy()` then equals eq. (3).
4. **Own reach**: the vectorised pass equals a brute-force walk for sampled keys of
   every row type in the table above.
5. **Short real run on mini-drawmaha** (torch, behind the slow flag): a few
   iterations of the real loop at a small K on the whole table; the exact
   average's exploitability must fall below uniform play's 4.842 and below the
   current iterate's, and the rebuilt-from-checkpoints average must equal the
   running sums. This is the end-to-end check; it has no outside referee, which
   is the price of decision 4 below (the Leduc gate was dropped, so the deep
   package is written against `PackedTable` directly, with no game adapter).
6. **Modal smoke**: 3 iterations at K = 1,000 on the real container; the timing
   report picks W and confirms the resume path.

The default suite stays fast and passes without torch (`pytest.importorskip
("torch")` in every deep test); the short real run and any whole-game deep run sit
behind `MINIDRAWMAHA_FULL_DEEP=1`.

## PRs, one concern each, each with a code-reading walkthrough PDF

| PR | files | what lands |
|---|---|---|
| 1 | `minidrawmaha/features.py`, `pyproject.toml` (`deep` optional group: torch) | the 136-feature encoder, built per shape and per point; row -> (point, position); the per-point legal gather |
| 2 | `deepcfr/net.py`, `deepcfr/materialise.py` | the MLP, `regret_slots_from_outputs` (eq. 3 by the write), `materialise(table, nets)` |
| 3 | `deepcfr/reservoir.py`, `deepcfr/collect.py` | Algorithm R rows and their checkpoint; K walks on W processes returning rows |
| 4 | `deepcfr/fit.py`, `deepcfr/train.py`, `scripts/run_deep_cfr.py` | from-scratch refit; the loop with checkpoints, resume, grid marks, exactness pin, the short real run |
| 5 | `deepcfr/average.py`, `deepcfr/policy.py` | predecessor table and own-reach pass; running exact average; Pi fit and freeze; current iterate freeze |
| 6 | `scripts/modal_rung4.py`, `minidrawmaha/analysis.py` (rung-4 figure), `figures/rung4/`, `README.md` | the Modal launcher and grader fan-out, the net-size sweep (3 x 64, 4 x 64, 3 x 128 at 100 iterations, graded exactly), the run, the equal-hands figure, the answer sheet, the README section |

Each PR: tests first, driven to merge-ready, a walkthrough PDF under
`/Users/alec/Desktop/Claude/poker/code-walkthroughs/pr<N>-<name>/`, and an
intuition-first chat explanation. PyTorch constructs get a syntax block the first
time they appear.

## Open questions for Alec, to be asked one at a time

1. **Run length and marks: DECIDED 2026-10-07.** The full 2,000 iterations (20M
   hands/seat, about 15 h and $70 on Modal) with all eight marks graded, so every
   tabular grid point has an equal-hands Deep CFR point and the headline is at the
   race's own endpoint (0.061).
2. **Refit size: DECIDED 2026-10-07.** Brown et al.'s 4,000 Adam steps on
   minibatches of 10,000 rows, per refit.
3. **Net size: DECIDED 2026-10-07.** 3 x 64 (the SD-CFR Leduc scale) is the
   default, with width and depth as run parameters. PR 6's analysis adds a size
   sweep — 3 x 64, 4 x 64 and 3 x 128, 100 iterations each on Modal, graded
   exactly at equal hands — and if a larger net is clearly better at 100
   iterations, the 2,000-iteration run uses it.
4. **The Leduc gate: DECIDED 2026-10-07, dropped.** Alec: "drop leduc, only
   focus on minidrawmaha". No Leduc encoder, no game adapter; gate 5 is the short
   real run on mini-drawmaha instead, and the exactness pin (gate 1) is the only
   bit-for-bit check of the loop.
5. **Reservoir capacity: DECIDED 2026-10-07.** 50M rows per memory (Alec: "lets
   do 50m"), above the paper's 40M; the memories section above has the cost.

## Out of scope for M1

Full Drawmaha's engine and encoder, GPU refits, querying the net inside the walk
(approach B), a `StrategySource` interface (approach C), Deep Discounted CFR
weighting, and any dashboard work.
