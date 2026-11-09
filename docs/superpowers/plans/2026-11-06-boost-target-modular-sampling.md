# Boost target: modular sampling from a high-noise row — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn PR #89's exact target into a sampling framework. At a targeted row, it runs M extra play-outs from that row. Each source of randomness below the row is sampled k-of-n or enumerated, chosen per call. The result is unbiased for the regret the walk's samples estimate. Exact enumeration remains as the all-enumerate special case and the test oracle.

**Architecture:** One value type, `Sampling`, names a count per source: opponent range, board cards, opponent draws, hero draws, opponent actions, and the number of play-outs. The existing downward recursion (`_value`, `_action_values`, `_push`) takes a private `_Draws` object. That object either returns everything (exact) or a random subset with its unbiasing scale. The upward back-solve (`_reach`) and the arrival stay exact always: the arrival is the cheap half, and an exact denominator under a sampled numerator keeps the ratio unbiased. Randomness is one `default_rng([seed, t, seat, row, TAG])` per row, so a row's result never depends on how rows are batched or split across workers.

**Tech Stack:** Python 3.12, NumPy 2.5 (`Generator.multinomial` with 3-D `pvals`), pytest, `uv`.

**Spec:** the chat design agreed 2026-11-06 (Alec): "sample extra starting from node x because node x is high variance … modular … board, opponent draws, hero draws, opp actions … unbiased". This builds on `~/Desktop/Claude/poker/targeted-sampling-plan/targeted-sampling-plan.md` §3 (row weight K · arrival · t, unchanged).

## Global Constraints

- The module imports `rules/` and the packed table only: never `exploitability.py`, `deepcfr`, or torch. The static import test stays.
- `Sampling` fields: `playouts: int = 1`, `opponent_range: int | None = None`, `board_cards: int | None = 1`, `opponent_draws: int | None = 1`, `hero_draws: int | None = 1`, `opponent_actions: int | None = 1`. `None` means enumerate all. `EXACT = Sampling(1, None, None, None, None, None)`.
- Chance sources (board, opponent draws, hero draws) sample k of n **without** replacement with scale n/k; k ≥ n is enumeration.
- Opponent actions are k i.i.d. draws from σ per opponent state, weighted count/k; `None` means split by σ.
- Opponent range is k i.i.d. hands per entry, drawn from that entry's compatible weight, each carrying W/k; `None` means the whole range.
- The arrival is always exact; `targets()` returns the same arrival as `arrivals()`, bit for bit.
- `targets(..., sampling=EXACT)` equals the pre-change output bit for bit.
- Stream: `np.random.default_rng([seed, t, seat, row, _STREAM_TAG])`; a non-EXACT `sampling` without `seed` and `t` is refused by name.
- Rename: `solver/exact_target.py` → `solver/boost_target.py`, `ExactTargets` → `BoostTargets`, `tests/minidrawmaha/test_exact_target.py` → `test_boost_target.py`; update `minidrawmaha/AGENTS.md`.
- The default suite stays torch-free and under its current time budget. Sampled unbiasedness tests use the restricted `SMALL`/`WIDE` deal sets.

## Review Focus

1. A sampled dimension that never actually samples, because the row has nothing of that kind below it. Every unbiasedness case must also assert that the sampled spread is > 0.
2. Hero-draw sampling must not leak into the hero's own **actions**. Hero actions are always enumerated under the hero's mix; only the replacement card is sampled.
3. Pruning a zero-weight opponent branch must not change the EXACT path. This is pinned by the bit-for-bit test.
4. `opponent_range = k` at a row whose compatible weight is zero (an unreachable row) must return zero regret, not NaN or a sampling error.
5. The same row requested twice in one call, or in two calls with rows in a different order, must give identical sampled results.

---

### Task 1: rename and the `Sampling` value type

**Files:**
- Rename: `src/drawmaha_solver/minidrawmaha/solver/exact_target.py` → `boost_target.py`; `tests/minidrawmaha/test_exact_target.py` → `test_boost_target.py`
- Modify: `src/drawmaha_solver/minidrawmaha/AGENTS.md` (the `exact_target` line)

**Interfaces — Produces:** `Sampling` (frozen dataclass, fields above), `EXACT`, `BoostTargets` (the renamed class). `BoostTargets.targets(sigma, *, seat, rows, sampling: Sampling, seed: int | None = None, t: int | None = None) -> tuple[np.ndarray, np.ndarray]`. `sampling` is required, so every caller states its choice.

- [ ] Step 1: Snapshot today's outputs (not committed). Save `targets` on `random_table(9)` for 40 rows spread over round-1, draw and round-2 points, both seats, to `.context/exact_snapshot.npz`.
- [ ] Step 2: `git mv` both files, update imports and the AGENTS.md line, and add `sampling=EXACT` at every existing test call site. Run `uv run pytest tests/minidrawmaha/test_boost_target.py tests/test_agent_docs.py`: PASS.
- [ ] Step 3: Failing test `test_sampling_refuses_counts_that_are_not_positive_ints`: `Sampling(board_cards=0)`, `Sampling(hero_draws=-1)`, `Sampling(opponent_actions=True)`, `Sampling(playouts=0)` and `Sampling(playouts=2, …all None)` each raise `ValueError`, and the message names the field.
- [ ] Step 4: Failing test `test_a_sampled_call_without_seed_and_t_is_refused`.
- [ ] Step 5: Implement `Sampling.__post_init__` and the refusal in `targets`, then run the tests: PASS. Commit `feat(rung4): the boost target names what it samples — one Sampling value, EXACT the oracle`.

### Task 2: the `_Draws` seam and the exact path through it

**Files:** `boost_target.py` (`_value`, `_action_values`, `_push`, `targets`)

**Interfaces — Produces:** `_Draws(sampling, rng)` with:
- `board(kids: (nb,14)) -> (kids', scale)`
- `replacements(n_rows, fan=12) -> (pick | None, scale)`, used by both hero and opponent draws with their own field
- `actions(pvals: (nb,S,w)) -> weights (nb,S,w)`
- `range(u, mask, k) -> u'`

On `EXACT`, every method is the identity, with no RNG and no array copy.

- [ ] Step 1: Failing test `test_exact_sampling_reproduces_the_snapshot_bit_for_bit`. This is a temporary local test reading `.context/exact_snapshot.npz`; it's deleted before the commit, and its committed successor is Step 3 of this task.
- [ ] Step 2: Thread `draws: _Draws` through `_action_values` and `_value`, and give `_push` an optional `pick`. Skip an opponent child whose pushed weight has no positive entry (it returns zeros). Run: PASS bit for bit.
- [ ] Step 3: Committed test `test_a_count_at_or_above_the_fan_is_enumeration`: `Sampling(board_cards=14, opponent_draws=12, hero_draws=12, opponent_actions=None, opponent_range=None)` equals `EXACT` with `np.array_equal`, on a draw row and a round-1 row.
- [ ] Step 4: Commit `feat(rung4): the play-out takes its draws from one seam; exact is the seam returning everything, bit for bit`.

### Task 3: the five samplers, each unbiased and non-vacuous

**Files:** `boost_target.py` (`_Draws` bodies, the playouts loop, the per-row stream), `test_boost_target.py`

- [ ] Step 1: Failing test `test_each_sampled_source_is_unbiased_and_actually_samples`, parametrized over `opponent_range=1`, `board_cards=1`, `opponent_draws=1`, `hero_draws=1`, `opponent_actions=1`, and `Sampling()` (all at once). Each case uses a row where that source is live, under `build(holes=WIDE)`, a random positive table, 400 seeds (t = seed). Assertions:
  - for every action column, |mean − exact| < 4 · s.e.
  - `assert_pooled_fair` on the pooled z
  - the per-column spread is > 0 for at least one column (Review Focus 1)
  - `arrival` equals `arrivals()` exactly
- [ ] Step 2: Implement the samplers:
  - **Board:** `rng.permuted` over the 14 children per batch row, take the first k, scale 14/k into `u`.
  - **Replacements:** k of 12 per (row, hole) or (row, hero state), via argsort of uniforms, scale 12/k.
  - **Actions:** `rng.multinomial(k, pvals) / k`; rows of invalid states get uniform `pvals`, and their weight is already zero.
  - **Range:** per entry, mask the states that clash with the hero's cards, draw k states ∝ `u·mask` with `rng.choice`, and put W/k on each draw (with multiplicity); W = 0 returns zeros.
  - **Play-outs:** loop M times per row, averaging the numerator; divide once by the exact arrival.
  
  Run: PASS.
- [ ] Step 3: Failing test, then passing: `test_a_rows_sample_depends_on_seed_t_seat_row_only`. The same rows in reversed order, split over two calls, and with a duplicate give equal results; t + 1 gives a different result (Review Focus 5).
- [ ] Step 4: Failing test, then passing: `test_more_playouts_shrink_the_spread`. Across 200 seeds, the variance at `playouts=8` is within [1/16, 1/4] of the variance at `playouts=1`.
- [ ] Step 5: Failing test, then passing: `test_an_unreachable_row_is_zero_under_any_sampling`, with `opponent_range=1` (Review Focus 4).
- [ ] Step 6: Commit `feat(rung4): any source below a boosted row can be sampled k of n — each unbiased against the exact sum, on a per-row stream`.

### Task 4: cost table, docstring, PR, and PR 2

- [ ] Step 1: Extend the benchmark test to print ms per row for `EXACT` vs `Sampling()` at round-1, draw and round-2 rows. Keep the existing round-2 ceiling, applied to `EXACT`.
- [ ] Step 2: Rewrite the module docstring: what is sampled, why the arrival stays exact (ratio bias), why k-of-n without replacement for chance and i.i.d. for σ, the stream, and the measured costs. Every number in it must be recomputed by a test or printed by the benchmark.
- [ ] Step 3: `uv run pytest`, the full default suite: PASS. Push to `AlecROndo/canary-handoff-pr-1-build`, retitle #89 and rewrite its body.
- [ ] Step 4: PR 2: in a temporary detached worktree on `origin/AlecROndo/pr-2-handoff-prompt`, `git merge origin/AlecROndo/canary-handoff-pr-1-build` (merge, never rebase), run `uv run pytest`, and push `HEAD:AlecROndo/pr-2-handoff-prompt`. Workspace `bariloche`'s local branch then needs a `git pull`; say so.
- [ ] Step 5: Drive #89 and #91 to merge-ready: checks green, `claude-review` findings dispositioned.
