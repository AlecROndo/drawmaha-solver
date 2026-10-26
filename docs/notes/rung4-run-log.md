# Rung 4 run log: every Deep CFR arm trained so far, and what each one showed

A lab notebook, not a design. The spec (`docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`)
says what rung 4 is meant to measure. This file records what has actually been run, with
which settings, and what the exact grader said. Every number below is read from a committed
file under `figures/rung4/` (or `figures/rung3/grades.json` for LCFR). They are copied by hand,
and `tests/deepcfr/test_rung4_quotes.py` checks the scoreboard and the README's rung-4 table
against those files: if a number here disagrees with its file, the file wins and this note is the bug.

Exploitability is in chips per hand, ante 1, from the exact best-response walk. Uniform random
play is 4.842. All runs are seed 0, K = 10,000 walks per seat per iteration, 50M-row
reservoirs and 10,000-row minibatches, so iteration t has seen 10,000·t hands per seat, the
same hands LCFR had seen at lockstep iteration 1,000·t. Rows do not depend on the worker count
W (`collect.py` draws every walk from (seed, t, seat, walk)), so a Mac run and a Modal run of
the same config see identical data. Only the refit's arithmetic order can differ.

## The scoreboard: the exact average at each mark

| arm | where | net | refit steps | Adam lr | t=10 | t=20 | t=50 | t=100 | t=200 | file |
|---|---|---|---|---|---|---|---|---|---|---|
| **LCFR (rung 3, tabular)** | Modal | table | — | — | 3.048 | 2.697 | 1.876 | 1.131 | 0.630 | `figures/rung3/grades.json` |
| 3 × 64 | Mac, W = 8 | 3 × 64 | 4,000 | 1e-3 | 0.581 | 0.444 | 0.396 | | | `figures/rung4/grades.json` |
| sweep 3 × 64 | Modal | 3 × 64 | 4,000 | 1e-3 | | | | 0.382 | | `figures/rung4/sweep.json` |
| sweep 4 × 64 | Modal | 4 × 64 | 4,000 | 1e-3 | | | | 0.375 | | `figures/rung4/sweep.json` |
| sweep 3 × 128 | Modal | 3 × 128 | 4,000 | 1e-3 | | | | 0.243 | | `figures/rung4/sweep.json` |
| 3 × 128 to t = 200 | Mac | 3 × 128 | 4,000 | 1e-3 | | | | | 0.234 | `figures/rung4/sweep_128x3_t200.json` |
| 3 × 128, double fit | Mac | 3 × 128 | 8,000 | 1e-3 | | | 0.213 | 0.184 | | `figures/rung4/sweep_128x3-s8000.json` |
| 4 × 128 | Mac | 4 × 128 | 4,000 | 1e-3 | | | 0.220 | 0.221 | | `figures/rung4/sweep_128x4.json` |
| 4 × 256 | Mac | 4 × 256 | 4,000 | 1e-3 | | 0.294 | 0.173 | 0.152 | | `figures/rung4/sweep_256x4.json` |
| **4 × 256, double fit** | Mac (cpu, then mps from t = 37) | 4 × 256 | 8,000 | 1e-3 | | 0.330 | 0.167 | **0.134** | | `figures/rung4/sweep_256x4-s8000.json` |
| 4 × 256, 4× fit, half rate (to t = 500: 0.108) | Mac, mps | 4 × 256 | 16,000 | 5e-4 | 0.643 | 0.330 | 0.169 | 0.133 | 0.115 | `figures/rung4/sweep_256x4-s16000-lr5e-4.json` |
| DCFR screen | Mac, mps | 4 × 256 | 4,000 | 1e-3 | 1.515 | 2.227 | 2.870 | | | `figures/rung4/rule_dcfr-256x4.json` |
| CFR+ screen | Mac, mps | 4 × 256 | 4,000 | 1e-3 | 2.539 | 2.858 | 3.092 | | | `figures/rung4/rule_cfrplus-256x4.json` |

"Net" is depth × width (the JSON files spell it width x depth: `sweep_256x4` is four hidden
layers of 256). Every file also holds Π (the policy net) and the current iterate at each mark,
plus each seat's best response, the training clock and the full config.

## What the arms say

**1. Width is the lever that mattered.** At 1M hands per seat (t = 100), 3 × 64 → 3 × 128 cut
the exact average from 0.382 to 0.243 (−36%). Depth did less at either width: 4 × 64 is 0.375
against 3 × 64's 0.382 (−2%), and 4 × 128 is 0.221 against 3 × 128's 0.243 (−9%, just under the
sweep's 10% bar). The widest net, 4 × 256,
reached 0.152. The sweep's pre-registered rule (stay at 3 × 64 unless another size is lower by
more than 10%) picked 3 × 128, and the later Mac arms went wider still.

**2. A longer refit helps, by less than a wider net.** Doubling the refit from 4,000 to 8,000
Adam steps took 3 × 128 from 0.243 to 0.184 at t = 100 (−24%) and 4 × 256 from 0.152 to 0.134
(−12%). The 3 × 128 baseline is the Modal sweep and the 4 × 256 one a Mac run; the rows do
not depend on W, so each pair saw the same hands. The gain shrinks as the net grows, which is what you would expect if the small net was
under-fit and the large one is closer to its fit's floor. Early on the longer fit can be worse
(4 × 256 at t = 20: 0.330 against 0.294), so compare arms at t ≥ 50.

**3. Deep CFR is far ahead of the table early, and the one arm graded past t = 100 may
be flattening.** At equal hands the best arm is 0.134 against LCFR's 1.131 at t = 100, an eighth of
the table's exploitability. The one arm graded at t = 200, 3 × 128, moved from 0.243 (the Modal
sweep) to 0.234 (a Mac run of the same config, whose own t = 100 mark
is not graded) between t = 100 and t = 200, a 4% change on one
seed, while LCFR fell 44% (1.131 → 0.630). The 16,000-step 4 × 256 arm, the one net graded
to 5M hands, says the same thing more plainly: its exact average went 0.133 → 0.115 → 0.108
from t = 100 to t = 200 to t = 500 (−19% over five times the hands) while LCFR went
1.131 → 0.630 → 0.264 (−77%). At t = 500 Deep CFR is still 2.4× better at equal hands, but LCFR
reaches 0.061 at t = 2,000, so if the net stays near 0.108 the table passes it somewhere between
t = 500 and t = 2,000. That is one seed, and whether a bigger net or a different fit moves the
floor is the open question; the arm was stopped at t = 506 and its two averages at t = 500 are
what `/solver` shows beside the table (below).

**4. The three graded objects usually order exact < Π < current, not always.** The exact
average beats Π at every graded mark of every linear arm. The current iterate is usually worst
but not always: in the Modal sweep, 4 × 64's current iterate (0.418) beats its Π (0.444), and in
4 × 256 at t = 20 with the double fit the current iterate and the exact average tie (0.3299 and
0.3300). The 16,000-step arm's current iterate beats both averages at t = 10 (0.528 against
exact 0.643 and Π 0.663). CFR promises nothing about the current iterate, so none of these is a bug.

**5. Per-sample DCFR and CFR+ lose to linear by 17× and get worse with training.** See
`docs/notes/regret-rules-dcfr-cfrplus.md` for the mechanism. The linear rule is the only one
worth running further.

**6. The refit is the iteration.** Per iteration, the two refits are about 95% of the time
(`progress.jsonl` → `seconds.fit`). The move to Apple's GPU (#61, `--device mps`) cut a
4 × 256, 8,000-step refit from 127 s to 34 s per seat. The 16,000-step arm costs about 70 s per
seat per refit on mps, about 152 s an iteration.

## The Modal main run: started, then stopped

The Modal trainer's directory `deep/` on the volume `drawmaha-rung4` holds a 3 × 64 run with
`--target 500`, started 2026-10-23 16:40 UTC on 32 workers (about 127 s per iteration). It was
stopped at t = 64, six minutes after the sweep's verdict (written 20:07 UTC) picked 3 × 128
over 3 × 64. Its marks t = 10, 20 and 50 and a resume at t = 64 are kept there, and none of
them has been graded. Because the sweep superseded its net size, it is a dead arm. Its marks
would only be worth grading as a Modal-vs-Mac reproducibility check of
`figures/rung4/grades.json`, which is the same config.

## The 16,000-step arm: 4 × 256 at half the learning rate, run to t = 506, graded to t = 500, flat

`runs/sweep-256x4-s16000-lr5e-4`, started 2026-10-25 02:11 EDT on this Mac and stopped at
t = 506 on 2026-10-26 once the t = 500 grade showed the curve had flattened:

    uv run --extra deep python scripts/deepcfr/run_deep_cfr.py --out runs/sweep-256x4-s16000-lr5e-4 \
        --target 1000 --walks 10000 --workers 16 --width 256 --depth 4 --steps 16000 \
        --lr 5e-4 --device mps --save-every-s 600

It asks whether the plateau in finding 3 is fit error: four times the steps of the 4,000-step
arm at half its rate, so each refit can settle closer to the weighted mean it is fitting. At
about 152 s an iteration, 1,000 iterations is roughly 42 hours.

**Stop rule, fixed before its marks are graded.** Keep going past t = 100 only if its exact
average at t = 100 is below 0.121, which is more than 10% below the double-fit arm's 0.134 (the
same 10% bar the size sweep used). If it is not, the extra fit buys nothing at this net size
and the remaining ~38 hours are better spent elsewhere. t = 50 (bar: 0.167) is an early warning
only, because a longer fit has been worse early before (finding 2).

**What it said.** The exact average at t = 100 was 0.133 — under the double-fit arm's 0.134 by
less than 1%, nowhere near the 0.121 bar — so by the rule the longer fit buys nothing at this
net size. The arm was run on anyway to see the shape of the curve past 1M hands, which no other
net had shown: 0.115 at t = 200 and 0.108 at t = 500 (Π 0.136, the current iterate 0.265), and
it was stopped at t = 506. Finding 3 reads the slope. The marks t = 10, 20 and 50 were graded
after the fact so the file holds every kept mark.

**What shipped from it.** Its two averages at t = 500 are the release `rung4-deep-t500`
(`deep-256x4-s16000-lr5e-4-t500-pi.npz` and `…-exact.npz`, the files `grade` wrote under
`runs/…/frozen/`), and `uv run minidraw-ranges --release deep-pi` / `--release deep-exact`
export them for `/solver`, where a switch shows either beside rung 3's tabular strategy with
the grade quoted from this file (see `figures/AGENTS.md`).

Grade a mark as it lands, without Modal spend:

    uv run --extra deep --with modal modal run scripts/deepcfr/modal_rung4.py::grade \
        --local runs/sweep-256x4-s16000-lr5e-4 --marks 20 \
        --grades figures/rung4/sweep_256x4-s16000-lr5e-4.json

A grade takes several minutes and competes with the trainer for cores, so grade one mark at a
time while a trainer is live.

## Not yet run

- The 2,000-iteration Modal run at the chosen size. It was the spec's headline. The
  16,000-step arm answered the fit-size question in the negative (a longer refit does not move
  the 4 × 256 floor), so the run's net and refit are Alec's call from the arms above.
- The tᵞ sample-weight arm (γ = 0 vanilla, γ = 2 quadratic). Linear is γ = 1.
- A second seed for any arm. Every comparison above is one seed, so differences under ~10% are
  suggestive, not settled.
