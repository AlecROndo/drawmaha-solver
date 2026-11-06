# Targeted sampling, PR 3: targeting — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, test first. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a trainer-side module that turns "where does the strategy leak, and how noisy are the samples there" into a frozen list of table rows to boost, drawn reproducibly and never twice in one draw; a grader-backed refresher that feeds it from the exact meter, outside the package; and a committed leak-by-line figure keyed by the same groups. Nothing is wired into `train.step` yet (PR 4).

**Done when:** `uv run pytest` passes without torch; `tests/deepcfr/test_targeting.py` pins the groups (141 lines, 3,870 line × hand-class groups), the noise score, the selection rule, the draws (reproducible in (seed, t, seat, stream), no repeats, inclusion probabilities of successive sampling), the save/load round trip, and that no module under `src/drawmaha_solver/deepcfr/` imports the grader; the oracle pin passes unchanged; `figures/rung3/exploit_by_line.json` and `.png` are written by `uv run minidraw-exploit-by-reach --release tabular --by-line` and a test holds the JSON to the committed `exploit_by_reach.json`.

**Architecture:** `deepcfr/targeting.py` is pure NumPy over whole-game per-row arrays. It knows the groups (from `rules/` alone), the noise score (from the advantage memories' 4-tuple), the selection rule, and the draws. It never sees the grader: the leak and the frequency arrive as arrays, and the draw weight arrives as a callable `arrival(seat, rows)`. The script `scripts/deepcfr/targeting_refresh.py` is the only place the two meet: `GraderRefresher` grades a profile with `gains_by_infoset` (both seats), scatters gains and frequencies onto table rows with `analysis.gains_by_row`, binds `BoostTargets.arrivals` under `BoostTargets.sigma(table)` as the draw weight, and calls `targeting.target_set`. `analysis.py --by-line` uses the same scatter and the same groups.

**Tech stack:** Python 3.12+, NumPy, matplotlib (figure only). No torch anywhere in this PR.

**Spec:** the targeted-sampling plan, kept outside the repo on Alec's machine: §3 Idea 2 (the four arms and the random control), §6 (open decisions), and the appendix contracts "PR 3" and "PR 4".

## Decisions provisional (Alec asleep; to confirm)

- **B = 2,000 infosets per seat per iteration, R = 25 iterations between refreshes**: the plan's proposals, as the defaults of `TargetSpec.budget` and `TargetSpec.refresh`. Every caller can override them.
- **Leak of what**: `GraderRefresher(leak="average")`, the frozen exact average (what is graded), is the default; `leak="current"` grades the current iterate (what drives the regrets) instead. Both cost the same two grader walks.
- **α is not settled.** `TargetSpec.alpha` exists because the noise score needs it (below) and has NO default: every caller states it.

Two further choices this plan makes where the appendix was silent, also listed in the PR body:

- **Reach for draws = the walk's arrival** (`BoostTargets.arrivals` under the table's σ at the refresh), not the grader's play frequency. The arrival is what the boost's row weight `(1 − α) · K · arrival` multiplies, so drawing ∝ arrival spends the budget where the most sampled weight is replaced; it needs no grader, so the random arm's refresh needs none either. Cost measured on this Mac: about 0.06 ms a row when a point's rows go in one call, so about 190 s per seat for the random arm's whole seat and a few seconds for a line pool.
- **Eligible = reached** (grader frequency > 0) for the three aimed arms, so the pool's size rule (reached infosets ≥ 20 × B) counts exactly the rows that can be drawn; the random arm has no grader and takes every row of the seat with positive arrival.

### Found during the build: B counts rows, and rows differ in cost a hundredfold

On LCFR-2M's leak the line arm's pool is about 45,000 rows a seat, and a third
of its arrival weight sits on round-1 rows and 41–53% on draw rows, because the
leakiest lines are round 1's and the walk arrives at early rows most often. An
exact target costs about 1–2 ms at a round-2 row but 0.07–0.27 s at a draw or
round-1 row, so B = 2,000 drawn by the plan's rule costs minutes per
iteration, not the plan's 1.5%. PR 3 keeps the plan's rule as the default and
adds one opt-in, `TargetSpec.round_two_only`, so the choice between it, a cost
budget, a smaller B for one-card rows, or a sampled target there is Alec's
before PR 4.

## Contract (what PR 4 consumes)

```python
Granularity = Literal["random", "infoset", "line", "line_class"]   # "off" is DeepConfig.target = None, not a fifth value

@dataclass(frozen=True, kw_only=True)
class TargetSpec:
    granularity: Granularity
    alpha: float                 # no default: PR 4 settles it
    budget: int = 2_000          # B, rows per seat per iteration
    refresh: int = 25            # R, iterations a TargetSet serves
    noise_cap: float = 20.0
    pool: int = 20               # line / line_class: groups taken until their reached rows ≥ pool × B
    stream: int = 0x7A52
    round_two_only: bool = False # opt-in, added during the build (below): leave one-board-card rows out

@dataclass(frozen=True, eq=False)
class TargetSet:
    spec: TargetSpec
    built_at: int                          # serves t in [built_at, built_at + refresh)
    rows: tuple[np.ndarray, np.ndarray]    # per seat, int64, sorted: the pool (infoset: the B rows themselves)
    weight: tuple[np.ndarray, np.ndarray]  # per seat, float64 ≥ 0, the draw weight of each pool row (∝ arrival)
    groups: tuple[np.ndarray, np.ndarray]  # per seat, int64 group ids chosen, best first
    def rows_for(self, t: int, seat: int, seed: int) -> np.ndarray: ...  # ≤ B distinct rows, sorted
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> TargetSet: ...

class Refresher(Protocol):
    def __call__(self, *, t: int, table: PackedTable, average: ExactAverage,
                 memories: tuple[Reservoir, Reservoir]) -> TargetSet: ...
```

Departures from the appendix sketch, each for a reason: no 6.2M-entry `group_of_row` in the set (the pool's rows and weights are all a draw needs, a few hundred KB instead of 25 MB per saved set); the spec rides inside the set so `rows_for` can refuse an iteration outside its window and PR 4 reads R from the set it holds; "none" is not a granularity, because the run already has one way to say "off".

## The rules, stated once

- **Groups**, one id per table row of the seat: `infoset` = the row; `line` = the public point (141 in the whole game, both seats); `line_class` = point × `InnerCategory` of the three private cards × `OuterCategory` with both board cards, with outer 0 at the 970-key one-board-card shape (round 1 and the draw), 3,870 non-empty groups in the whole game; `random` = one group per seat.
- **Noise** of a group g, from the seat's advantage memory: natural rows only (weight 1, or α — the walk's own rows, turned down at a targeted infoset), because an exact row is a mean, not a sample. a* = the column of largest |mean|; E[r²] and E[r] at a* over g's natural rows; p(g) = visits per walk = (n_g / size) · offered / (K · t_last), where t_last is the latest t in the memory. noise(g) = √(E[r²] / p) / |E[r]|, which is √((σ² + μ²)/p)/|μ|, the relative standard error of one walk's banked total at g. Fewer than 2 natural rows, or E[r] = 0, gives the cap.
- **Score** = leak(g) · min(noise(g), cap) / reached(g), leak = Σ gains over g, reached = rows with frequency > 0; a group with reached 0 is not eligible.
- **Selection**: `infoset` takes the B eligible rows of highest score (ties to the lower row), fixed for the set's R iterations. `line` / `line_class` take groups best first until their reached rows reach pool × B (the group that crosses is in), and the pool is those reached rows. `random`'s pool is every row of the seat.
- **Draws** (all but `infoset`): B rows without replacement ∝ the pool's arrival weight, by exponential keys (Efraimidis–Spirakis: key = E / w, E ~ Exp(1), the B smallest), which is successive sampling ∝ w; a row of weight 0 is never drawn; a pool with fewer positive rows than B gives all of them. The stream is `default_rng([seed, t, seat, stream, "TARG"])`, five words with a tag no reservoir or walk stream carries.

## Global constraints

- `src/drawmaha_solver/deepcfr/` never imports `minidrawmaha/exploitability.py`; a static test walks every module's import block. Only `scripts/deepcfr/targeting_refresh.py` and `answer/analysis.py` touch the grader.
- No change to `train.step`, `DeepConfig`, `ADDED_LATER` or `SAMPLE_FIELDS`. The oracle pin passes unchanged.
- PR 1's names as they will be after its rework: module `minidrawmaha/solver/boost_target.py`, class `BoostTargets`; this PR reads only `BoostTargets.build()`, `sigma(table)` and `arrivals(sigma, *, seat, rows)`.
- Figures regenerated by the named command, never edited. Public repo: no private paths, nothing from the private notes.
- Default suite: no torch, no Modal, no network; the end-to-end grader refresh is gated by `MINIDRAWMAHA_FULL_GRADE`.
- Commit subjects `feat(rung4): <the finding>`; branch `AlecROndo/pr-3-targeting` from #91's head; merge (never rebase) a new #91 head.

## Review focus

1. A row drawn twice in one `rows_for` (PR 4 would give it two exact rows and double its expected weight) — Task 4, the no-repeat test over many seeds and a pool barely larger than B.
2. Draw probabilities that are not successive sampling ∝ arrival — Task 4, exact inclusion probabilities enumerated on a 4-row pool against 20,000 seeds within 4 s.e.
3. Exact rows (PR 4) counted as samples in the noise score — Task 2, a memory mixing weight-1, weight-α and exact-weight rows.
4. The gains scattered onto the wrong rows (point order of the grader's decisions vs the table's) — Task 6, a coded synthetic `per_node` decoded back to (point, position).
5. The trainer package importing the grader through any module — Task 8.

---

### Task 1: the spec and the groups

**Files:** create `src/drawmaha_solver/deepcfr/targeting.py`; create `tests/deepcfr/test_targeting.py`.

- [ ] Failing tests: `TargetSpec` refuses an unknown granularity, `"none"`, a budget < 1 or non-int, refresh < 1, alpha outside [0, 1] or missing, noise_cap ≤ 0, pool < 1; `groups("line", seat)` gives each seat's rows its point index and −1 on the other seat's rows; line ids over both seats are 141; line_class over both seats is 3,870 non-empty groups; outer is 0 exactly on the one-board-card shape; `infoset` ids are the row numbers; `random` is one id per seat.
- [ ] Implement `TargetSpec.__post_init__` validation, `_layout()` (row offsets per point from `public_decision_points` and `private_keys`, cached), `_classes()` (inner/outer per key per shape, cached), `groups(granularity, seat)`.
- [ ] Run, pass, commit: `feat(rung4): targeting names four granularities of group over the table's rows — 141 lines and 3,870 line × hand classes, from the rules alone`.

### Task 2: the noise score

- [ ] Failing tests: a synthetic advantage memory with hand-chosen targets at two groups gives noise = √(E[r²]/p)/|E[r]| at the largest-|mean| column, p = (n_g/size) · offered/(K · t_last); a group with one natural row, with none, or with mean 0 gets the cap; rows of weight 1 and α count, a row of any other weight (an exact row) does not; an empty memory gives the cap everywhere.
- [ ] Implement `noise(memory, group_of_row, *, walks, alpha, cap)` with `np.bincount` per column over the natural rows.
- [ ] Commit: `feat(rung4): the noise score is one walk's relative error at a group, from the memory's natural rows only`.

### Task 3: scores and selection

- [ ] Failing tests: score = leak · min(noise, cap)/reached; unreached groups are never chosen; `infoset` returns the B best eligible rows with ties to the lower row; `line` stops at the first group whose reached rows bring the pool to ≥ pool × B and the pool holds only reached rows of chosen groups; `random` pools every seat row.
- [ ] Implement `choose(spec, *, seat, leak, frequency, noise_of_group) -> (pool_rows, chosen_groups)`.
- [ ] Commit: `feat(rung4): the aimed arms take groups by leak × noise per reached infoset until the pool is 20 × B`.

### Task 4: the frozen set and its draws

- [ ] Failing tests: `rows_for` is equal for equal (seed, t, seat), differs across t and seat, never repeats a row (200 seeds, pool of B + 3), returns min(B, positive rows) rows, never a weight-0 row, sorted int64; inclusion probabilities on a 4-row pool with B = 2 match the exact successive-sampling enumeration within 4 s.e. over 20,000 seeds; `infoset` returns its fixed rows at every t; a t outside [built_at, built_at + refresh) and a seat outside {0, 1} are refused by name; save/load round-trips every field and refuses a file that is not a target set.
- [ ] Implement `TargetSet` (validation in `__post_init__`), `target_stream`, `rows_for`, `save` (atomic, `allow_pickle=False`), `load`.
- [ ] Commit: `feat(rung4): a target set draws B rows a seat without replacement ∝ arrival, reproducible from (seed, t, seat, stream) and never the same row twice`.

### Task 5: the orchestrator and the protocol

- [ ] Failing tests: `target_set(spec, t=, leak=, frequency=, memories=, walks=, arrival=)` over synthetic whole-game arrays and a fake `arrival` returns a set whose pool weights are the fake's values on the pool rows, calls `arrival` only for pool rows (counted), and for `infoset` never calls it.
- [ ] Implement `target_set` and the `Refresher` protocol.
- [ ] Commit: `feat(rung4): one call turns leak, frequency and the memories into a frozen target set; the Refresher protocol is the seam PR 4 calls`.

### Task 6: gains on table rows, and the leak by line

**Files:** modify `src/drawmaha_solver/minidrawmaha/answer/analysis.py`; create `tests/minidrawmaha/test_exploit_by_line.py`; write `figures/rung3/exploit_by_line.json`, `.png`.

- [ ] Failing tests: `scatter_by_row(per_node, game, seat=)` on a coded `per_node` (value = 1,000 × point + position) puts every value at row_offset[point] + position; `line_profile(gain, frequency)` partitions rows into the 141 lines, shares sum to the total, each line lists its line_class count; the committed JSON's exploitability equals `exploit_by_reach.json`'s to 1e-9 and its group counts are 141 and 3,870.
- [ ] Implement `gains_by_row(strategy, game=)` (two `gains_by_infoset` walks), `scatter_by_row`, `line_profile`, `fig_by_line`, and `--by-line` (with `--replot`) in `reach_main`.
- [ ] Run `uv run minidraw-exploit-by-reach --release tabular --by-line`; commit code, tests and both files: `feat(rung4): the leak by betting line — <the finding read off the JSON>`.

### Task 7: the grader-backed refresher

**Files:** create `scripts/deepcfr/targeting_refresh.py`; create `tests/deepcfr/test_targeting_refresh.py` (loads the script by path).

- [ ] Failing tests: `GraderRefresher(spec, walks=, leak=)` refuses an unknown leak source; `profile()` returns the exact average's freeze for "average" and the current iterate for "current"; the random arm never calls the grader; gated (`MINIDRAWMAHA_FULL_GRADE`) end to end on a uniform table and empty memories: both seats' pools are reached rows, draws reproducible.
- [ ] Implement the class (grader walks, scatter, `BoostTargets.arrivals` bound to `sigma(table)`), and a `main` that refreshes once from a run's mark for inspection.
- [ ] Commit: `feat(rung4): the grader-backed refresher, the only importer of the meter on the trainer's side`.

### Task 8: the layering test and the docs

- [ ] Failing test: no module under `src/drawmaha_solver/deepcfr/` imports anything named `exploitability` (AST over every file).
- [ ] Docs: `deepcfr/__init__.py` (nine modules), `deepcfr/AGENTS.md` (the targeting rule), `scripts/AGENTS.md` (`targeting_refresh.py`), `figures/AGENTS.md` (`exploit_by_line`), README's exploit-by-reach paragraph (one sentence on the by-line figure).
- [ ] `uv run pytest` whole; fresh-context reviewer; push; PR; claude-review rounds to merge-ready.
