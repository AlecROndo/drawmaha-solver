# Friction (RTCFR+-style) on sampled mini-drawmaha: a mini experiment

**Goal.** Does a friction term toward a moving reference make the *current* strategy of
sampled tabular MCCFR converge on mini-drawmaha, and which inner rule carries it best:
CFR+ (W = 10), mini-batch CFR+ (W = 100), or LCFR (W = 10)?

**Done when.** Nine lockstep runs (3 rules × μ ∈ {0, 0.1, 1}) reach 1M hands per seat
in parallel, each graded at 50k / 100k / 250k / 500k / 1M hands on its current strategy
and its linear average, and the results sit in one JSON per run plus a curves figure.
μ = 0 is each rule's no-friction control.

## The friction (Meng et al. 2023, arXiv 2308.11256, Eq. 7–8, adapted to sampling)

RTCFR+ adds μ·(x̂ − r̂) to each player's sequence-form loss: x̂(I,a) is the player's own
reach of sequence (I,a), r̂ the reference's. Its counterfactual value at (I,a) is the
payoff value minus μ·(x̂(I,a) − r̂(I,a)), plus the children's (already transformed) values.

Under external sampling the payoff part of a banked regret is, in expectation, weighted
by the opponent-and-chance reach of I. We apply the friction undivided on each visit
(choice B), so its expected weight carries the same reach: friction and payoff keep the
same ratio at every infoset. The paper's unweighted form would need a 1/reach factor and
explode at rare rows. γ (the paper's 1e-10 mix toward uniform) is numerically inert at
this scale and is left out. Every R hands per seat the reference is reset to the current
strategy; regret totals are kept (the paper: resetting them stops convergence).

## Pieces

1. `solver/friction.py`: `Friction(mu, table, reference)`; `Friction.new(table, mu)`
   (reference = current strategy, uniform at the start); `of(key)`; `reset()`.
   Vectorised current strategy over a packed table.
2. `mccfr.traverse`: optional `friction`, carrying the traverser's own reach and the
   reference's reach down its own actions; utilities at its spots minus μ(x − r).
   `friction=None` is today's walk, bit for bit.
3. `lockstep`: `LockstepSolve.friction` handed to every walk; `save_lockstep` and
   `ParallelLockstep` refuse a run with friction (not checkpointed, not parallelised).
4. `scripts/minidrawmaha/friction_experiment.py`: one run; resets, grades, JSON lines.
5. Tests (`tests/minidrawmaha/test_friction.py`), written first:
   - friction with μ = 0 banks today's regrets bit for bit (Leduc, lockstep W = 4);
   - a reference equal to the current strategy along with equal reaches cancels: at t = 1
     (all σ uniform, reference uniform) any μ banks today's regrets bit for bit;
   - the hand-computed adjustment at one spot;
   - the vectorised reset equals `RegretMatcher.strategy` row by row (packed slice);
   - save and parallel refuse friction by name;
   - Leduc gate (slow): friction + CFR+ drives the current strategy's exploitability
     below no-friction CFR+'s current strategy.

## As run (9 Nov 2026)

The machine could not hold nine 1M-hand runs at once (12 concurrent runs land on the
efficiency cores at about 35 hands per seat per second), so the grid ran in waves:
μ ∈ {0, 10} for all three rules to 1M; μ = 0.1 stopped at 50k (indistinguishable
from μ = 0); μ ∈ {1, 100} for CFR+ and LCFR at W = 10 to 500k. The record is
`figures/rung3/friction_mini.jsonl`. Outcome: μ = 10 ends 2–4% below its control at
1M in all three pairs and is 5–11% worse at 50k; μ = 100 is 20–56% worse; the
current strategy beats the linear average at every mark of every arm; LCFR beats
CFR+ by 30%; W = 100 tracks W = 10 within 1.6%.
