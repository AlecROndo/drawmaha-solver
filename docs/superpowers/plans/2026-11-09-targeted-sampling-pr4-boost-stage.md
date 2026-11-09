# Targeted sampling, PR 4: the boost stage — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, test first. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** wire PRs 1–3 into one Deep CFR iteration: after a seat's walks and before the merge, draw this iteration's target rows from the frozen `TargetSet`, compute one SAMPLED boosted row per target under the σ the walks just read, turn the walk's own rows at those infosets down to α, and offer the boosted rows after the natural ones at weight (1 − α)·K·arrival, so every targeted infoset's expected banked weight and regret equal plain sampling's. The refresher runs every R iterations and its set rides in the resume state. `--boost none` (no `TargetSpec`) is today's run bit for bit.

**Done when:** `uv run pytest` passes without torch; the oracle pin `tests/deepcfr/test_train.py::test_the_oracle_loop_is_lcfr_bit_for_bit` passes unchanged; `tests/deepcfr/test_boost.py` pins the injection (order, weights, the skip rules, the refusal of an exact `Sampling`) and the unbiasedness of the whole stage on a restricted-deal game with the oracle learner (paired over seeds, within 4 s.e. at every targeted row); `tests/deepcfr/test_train.py` pins the config fields, the refresh cadence, the resume of the target set and the progress line; a whole-game smoke behind `MINIDRAWMAHA_FULL_DEEP=1` records `seconds.boost`; the fresh-context review is dispositioned and the PR is merge-ready.

**Architecture:** one new module, `deepcfr/boost.py`, holds the stage's arithmetic (`inject`) and its fixed parts for a run (`Boost`: the refresher and the boost oracle `BoostTargets`). `train.step` orchestrates: materialise → observe → (refresh if due, seat 0 only) → collect → inject → merge → fit. The table the boost reads σ from is the collector's table right after `learner.write`, the same array the walks read. `DeepConfig` gains two fields, `target: TargetSpec | None` (the aim, PR 3's spec plus a `leak` field) and `boost: Sampling` (what one boosted row samples; an exact `Sampling` is refused). The grader stays outside the package: `GraderRefresher` in `scripts/deepcfr/targeting_refresh.py` now takes the `ExactAverage` at construction, and the `Refresher` protocol no longer carries an `average` argument, so the loop passes only what it has (t, the table, the two advantage memories).

**Tech stack:** Python 3.12+, NumPy. No torch in this PR; the whole default suite runs with the oracle learner.

**Spec:** the targeted-sampling plan outside the repo (`~/Desktop/Claude/poker/targeted-sampling-plan/targeted-sampling-plan.md`, §3 Idea 1 and the appendix "PR 4 — the boost stage in `train.step`"), the breadcrumb beside it dated 2026-11-09, and Alec's decisions of 2026-11-09 recorded below. PR 3's plan is `docs/superpowers/plans/2026-11-06-targeted-sampling-pr3-targeting.md`; PR 1's is `2026-11-06-boost-target-modular-sampling.md`.

## Decisions settled by Alec on 2026-11-09

- **α = 0.** The walk's own rows at a targeted infoset are offered at weight 0: inert in the fit, still offered so the reservoir sees the same offers as a run without the boost.
- **B = 500** rows per seat per iteration (the plan's 2,000 costs about 92 s per iteration under sampling, five table iterations; 500 is about 23 s, one table iteration). **R = 25.** Both become `TargetSpec`'s defaults.
- **Draws ∝ arrival** inside the pool, and the **leak is measured on the frozen exact average**. PR 3's provisional choices, confirmed. The leak source is now a `TargetSpec` field, `leak`, so a resume can refuse a change to it by name.
- **No EXACT anywhere in the boost stage.** Every boosted row is a sampled play-out: `Sampling()` by default — the whole opponent range in one vectorised sum, one second board card, one replacement per throw, one opponent action line. `DeepConfig` refuses an exact `Sampling`, and `inject` refuses one again. `EXACT` survives only as the tests' oracle. The opponent-range count is a flag, so the mini-walk form (one dealt hand per play-out, M play-outs) is one argument away. Alec's guardrail: boosting is "extra sampling at high-variance nodes", never "brute-force the tree from node x".
- **Flags are `--boost …`**, not the appendix's `--target …`: `run_deep_cfr.py` already uses `--target` for the iteration to reach.

## Global constraints

- `src/drawmaha_solver/deepcfr/` never imports `minidrawmaha/exploitability.py` or `answer.analysis`; `tests/deepcfr/test_targeting.py` reads every import block. `boost_target` is rules-only and may be imported.
- `--boost none` ⇒ `step` calls `merge(rows, targets, t=, rng=)` with no `weight`, as today. The oracle pin must pass unchanged.
- Injected rows are offered AFTER the natural rows in the same `merge` call, so natural rows keep their stream positions. Natural rows of targeted infosets are still offered (weight α).
- A boosted row's float32 weight must never equal 1 or α: PR 3's noise score tells walk rows from boosted rows by weight alone. A drawn row whose weight would, or whose arrival is 0, is left untreated (its natural rows keep weight 1, no boosted row) and counted as skipped. Treatment is all-or-nothing per row, so skipping is unbiased.
- The boost's σ is `BoostTargets.sigma(collector.table)` read after this phase's `learner.write` and before `inject`; never a σ from an earlier phase.
- Boost draws use PR 1's own stream (`[seed, t, seat, row, "BOOST"]`) and the target draws PR 3's (`"TARG"`); neither touches the walk's or the reservoir's stream.
- New `DeepConfig` fields go in `ADDED_LATER` (`target: None`, `boost: Sampling()`) and `SAMPLE_FIELDS` (both).
- The policy memory never receives a boosted row or a weight.
- torch stays optional; the default suite passes on `uv sync` alone. Heavy tests are gated by `MINIDRAWMAHA_FULL_DEEP` or `MINIDRAWMAHA_FULL_TARGET`, named in the skip reason.
- Never `git stash`; never rebase; never launch Modal; the repo is public, so nothing from the Desktop plan folder's private notes is quoted.
- Commit subjects read `feat(rung4): <the finding>` or `fix(rung4): …`.

## Review focus

Inputs the spec implies but no appendix test names, each pinned in the task that owns the code:

1. **A drawn row no walk visited this phase, at α = 0.** Nothing is turned down, the boosted row is still added. Task 2 (`test_a_drawn_row_without_natural_rows_still_gets_its_boosted_row`).
2. **A stop inside a refresh window and a resume after it.** The saved set serves the remaining iterations without a re-grade; a resume whose saved set's window has ended refreshes before its first walks. Task 4.
3. **A refresher that hands back a set built for another spec or another iteration** (a mis-built launcher). Refused by name before any walk is spent. Task 4.
4. **`rows_for` hands back fewer than B rows** (a pool with fewer than B rows of positive weight). `rows.boosted` says how many, nothing is padded. Task 2 + Task 4's stats.
5. **An old `state.json` without the two fields.** Reads as `target=None, boost=Sampling()` and resumes under today's defaults with no refusal; a saved config with a `target` whose `leak` is missing is refused (it was written by code this code does not know). Task 3.

---

## File structure

- Create `src/drawmaha_solver/deepcfr/boost.py` — the stage: `inject` (pure arithmetic over one phase's rows), `Injected` (its result), `Boost` (a run's refresher + boost oracle, `due`/`refresh`/`phase`).
- Modify `src/drawmaha_solver/deepcfr/targeting.py` — `TargetSpec.leak`, defaults `budget=500`, `refresh=25`; `Refresher.__call__(*, t, table, memories)`.
- Modify `scripts/deepcfr/targeting_refresh.py` — `GraderRefresher(spec, *, walks, average=None, seed=0, exact=None)`, reads `spec.leak`.
- Modify `src/drawmaha_solver/deepcfr/train.py` — `DeepConfig.target`, `DeepConfig.boost`, `ADDED_LATER`, `SAMPLE_FIELDS`, `config_from_saved`; `DeepRun.targets`; `StepStats.refresh/boost/boosted/skipped`; `step(..., boost=None)`; `save_run`/`load_run` carry `targets-t{t:05d}.npz`; `run_deep(..., boost=None)`; the progress line.
- Modify `scripts/deepcfr/run_deep_cfr.py` — the `--boost…` flags; `main` builds `ExactAverage`, `GraderRefresher`, `Boost`.
- Create `tests/deepcfr/test_boost.py`; modify `tests/deepcfr/test_train.py`, `tests/deepcfr/test_targeting.py`, `tests/deepcfr/test_targeting_refresh.py`.
- Docs: `src/drawmaha_solver/deepcfr/AGENTS.md`, `src/drawmaha_solver/deepcfr/__init__.py`, `scripts/AGENTS.md`, `train.py`'s module docstring, `README.md` (the rung-4 targeting sentence, if present).

---

### Task 1: `TargetSpec.leak`, the new defaults, and the refresher's `average` at construction

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/targeting.py` (`TargetSpec`, `Refresher`)
- Modify: `scripts/deepcfr/targeting_refresh.py` (`GraderRefresher.__init__`, `profile`, `__call__`, `main`)
- Test: `tests/deepcfr/test_targeting.py`, `tests/deepcfr/test_targeting_refresh.py`

**Interfaces:**
- Produces: `TargetSpec(granularity, alpha, budget=500, refresh=25, noise_cap=20.0, pool=20, stream=0x7A52, round_two_only=False, leak="average")`, `leak ∈ LEAK_SOURCES = ("average", "current")` (moved into `targeting.py`; the script imports it).
- Produces: `class Refresher(Protocol): def __call__(self, *, t: int, table: PackedTable, memories: tuple[Reservoir, Reservoir]) -> TargetSet`.
- Produces: `GraderRefresher(spec: TargetSpec, *, walks: int, average: ExactAverage | None = None, seed: int = 0, exact: BoostTargets | None = None)`; `profile(*, t, table) -> FrozenStrategy`; `__call__(*, t, table, memories) -> TargetSet`. `average` is required (refused by name as `None`) when `spec.leak == "average"`, ignored for `"current"`.

- [ ] **Step 1: Write the failing tests**

In `tests/deepcfr/test_targeting.py`, beside the spec's refusal tests:

```python
def test_the_spec_names_which_profiles_leak_it_aims_at():
    assert TargetSpec(granularity="line", alpha=0.0).leak == "average"
    assert TargetSpec(granularity="line", alpha=0.0, leak="current").leak == "current"
    with pytest.raises(ValueError, match="leak"):
        TargetSpec(granularity="line", alpha=0.0, leak="policy")

def test_the_defaults_are_the_settled_budget_and_refresh():
    spec = TargetSpec(granularity="line", alpha=0.0)
    assert (spec.budget, spec.refresh) == (500, 25)
```

The existing save/load round-trip test gets one line: a set saved with `leak="current"` loads with `leak == "current"` (the field list is read off the dataclass, so this is automatic; the assertion pins it). Any test that asserted `budget == 2000` or `refresh == 25` by number is updated to the new defaults.

In `tests/deepcfr/test_targeting_refresh.py`: every `GraderRefresher(..., leak=...)` becomes a `TargetSpec(..., leak=...)`; every `refresher(t=, table=, average=FakeAverage(), memories=)` becomes `GraderRefresher(..., average=FakeAverage())` at construction and `refresher(t=, table=, memories=)` at the call; `profile(t=, table=)`. Add:

```python
def test_the_average_is_required_only_when_the_spec_aims_at_it(refresh):
    with pytest.raises(ValueError, match="average"):
        refresh.GraderRefresher(TargetSpec(granularity="line", alpha=0.0), walks=10)
    refresh.GraderRefresher(TargetSpec(granularity="line", alpha=0.0, leak="current"), walks=10)   # no average needed
    refresh.GraderRefresher(TargetSpec(granularity="random", alpha=0.0), walks=10)                 # grades nothing
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/deepcfr/test_targeting.py tests/deepcfr/test_targeting_refresh.py -q -p no:cacheprovider`
Expected: FAIL on `leak` (unexpected keyword) and on the changed signatures.

- [ ] **Step 3: Implement**

`targeting.py`: add `LEAK_SOURCES: tuple[str, ...] = ("average", "current")` and the field `leak: str = "average"` to `TargetSpec`, refused by name in `__post_init__` when not in `LEAK_SOURCES`; change `budget` to 500 and `refresh` to 25 and the docstring's numbers (the pool rule's "20 × 2,000 = 40,000" becomes "20 × 500 = 10,000"); change `Refresher.__call__` to `(self, *, t, table, memories)` and its docstring (the average is the refresher's own, handed to it by the script that builds it). `_FILE_KEYS` picks `leak` up through `fields(TargetSpec)`; `save` writes it as a 0-d string array and `load`'s `.item()` reads it back.

`targeting_refresh.py`: import `LEAK_SOURCES` from `targeting` (delete the local copy); `__init__(spec, *, walks, average=None, seed=0, exact=None)` stores `average` and refuses `None` when `spec.leak == "average"` with a message naming `average` and the leak; `profile(*, t, table)` reads `self.spec.leak`; `__call__(*, t, table, memories)`; `main` builds `TargetSpec(..., leak="current")` and `GraderRefresher(spec, walks=1, seed=args.seed)`. Update the module docstring's items 1 and "the boost stage will" tense (PR 4 is this PR).

- [ ] **Step 4: Run the two files and the full targeting pins**

Run: `uv run pytest tests/deepcfr/test_targeting.py tests/deepcfr/test_targeting_refresh.py tests/minidrawmaha/test_exploit_by_line.py -q -p no:cacheprovider`
Expected: all pass (one gated).

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr/targeting.py scripts/deepcfr/targeting_refresh.py tests/deepcfr/test_targeting.py tests/deepcfr/test_targeting_refresh.py
git commit -m "feat(rung4): the aim names which profile's leak it follows, B = 500 and R = 25 are the defaults, and the refresher carries its own exact average"
```

---

### Task 2: `boost.inject` — one phase's rows with the boosted rows appended

**Files:**
- Create: `src/drawmaha_solver/deepcfr/boost.py`
- Test: `tests/deepcfr/test_boost.py`

**Interfaces:**
- Consumes: `BoostTargets.targets(sigma, *, seat, rows, sampling, seed, t) -> (regret (n, 4) float64, arrival (n,) float64)`, `Sampling.is_exact`, `reservoir.TARGET_WIDTH`.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class Injected:
    rows: np.ndarray      # int32 (a + b,): the natural rows, then the b boosted rows in drawn order
    targets: np.ndarray   # float32 (a + b, TARGET_WIDTH)
    weight: np.ndarray    # float32 (a + b,): α at natural rows of treated infosets, 1 at the rest, (1 − α)·K·arrival at boosted rows
    boosted: int          # b
    skipped: int          # drawn rows left untreated: zero arrival, or a weight that would read as 1 or α

def boosted_weight(arrival: np.ndarray, *, alpha: float, walks: int) -> np.ndarray   # float64 (1 − α)·K·arrival

def inject(
    rows: np.ndarray, targets: np.ndarray, *,
    drawn: np.ndarray, exact, sigma, seat: int, t: int, seed: int,
    sampling: Sampling, alpha: float, walks: int,
) -> Injected
```

`exact` is anything with `BoostTargets.targets`' signature (tests pass a stand-in; `Boost` passes the real one). `inject` refuses, by name: `sampling.is_exact` ("the boost is extra sampling, never the exact sum"); `drawn` not 1-D, not integer, not strictly increasing (a repeat would double a row's expected weight); `alpha` outside [0, 1]; `walks < 1`.

Algorithm (the one piece the signature does not determine): `regret, arrival = exact.targets(sigma, seat=seat, rows=drawn, sampling=sampling, seed=seed, t=t)`; `w = boosted_weight(arrival, alpha=alpha, walks=walks)`; `w32 = w.astype(np.float32)`; `treated = (arrival > 0) & (w32 != np.float32(1)) & (w32 != np.float32(alpha))`; natural weight = ones, set to `alpha` where `np.isin(rows, drawn[treated])`; concatenate natural then `drawn[treated]` as int32, `regret[treated]` as float32, `w32[treated]`; `boosted = treated.sum()`, `skipped = drawn.size − boosted`.

- [ ] **Step 1: Write the failing tests**

`tests/deepcfr/test_boost.py` with a stand-in:

```python
class FakeExact:
    """`targets` hands back the regret and arrival this test set per row, and records the call."""
    def __init__(self, regret: dict[int, np.ndarray], arrival: dict[int, float]): ...
    def targets(self, sigma, *, seat, rows, sampling, seed, t): ...   # records (sigma, seat, rows, sampling, seed, t)

def test_boosted_rows_follow_the_natural_rows_at_the_settled_weight():
    # natural rows [5, 9, 5, 12]; drawn [5, 12]; K = 100, α = 0; arrivals 0.02 and 0.5
    out = inject(rows, targets, drawn=np.array([5, 12]), exact=fake, sigma="σ", seat=0, t=3, seed=7, sampling=Sampling(), alpha=0.0, walks=100)
    np.testing.assert_array_equal(out.rows, [5, 9, 5, 12, 5, 12])
    np.testing.assert_array_equal(out.weight, np.float32([0, 1, 0, 0, 2.0, 50.0]))
    np.testing.assert_array_equal(out.targets[4:], fake_regret_rows_as_float32)
    assert (out.boosted, out.skipped) == (2, 0)
    assert out.rows.dtype == np.int32 and out.targets.dtype == np.float32 and out.weight.dtype == np.float32
    assert fake.calls == [("σ", 0, [5, 12], Sampling(), 7, 3)]

def test_alpha_half_leaves_half_on_the_natural_rows():
    # same rows, α = 0.5: natural weights at 5 and 12 are 0.5, boosted weights are 1.0 and 25.0

def test_a_drawn_row_without_natural_rows_still_gets_its_boosted_row():
    # natural rows [9]; drawn [5]; the boosted row for 5 is appended, weights [1, (1−α)·K·arrival]

def test_a_zero_arrival_row_is_skipped_and_its_natural_rows_keep_weight_one():
    # drawn [5, 12], arrival of 12 is 0.0: rows == natural + [5], weight at natural 12 stays 1, skipped == 1

@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_a_row_whose_weight_would_read_as_a_walk_row_is_left_untreated(alpha):
    # arrival = 1 / (K · (1 − α)) makes (1 − α)·K·arrival exactly 1.0 in float32; and arrival = α / (K · (1 − α)) makes it α (α = 0.5 only;
    # at α = 0 that is the zero-arrival case). The row is not boosted, its natural rows weigh 1, skipped counts it.

def test_an_exact_sampling_is_refused_before_any_target_is_computed():
    with pytest.raises(ValueError, match="extra sampling"):
        inject(..., sampling=EXACT, ...)
    assert fake.calls == []

def test_repeated_or_unsorted_drawn_rows_are_refused():
    # drawn [5, 5] and drawn [12, 5] each raise ValueError naming "drawn"

def test_fewer_drawn_rows_than_the_budget_are_simply_fewer():
    # drawn of 3 rows → boosted 3; nothing padded

def test_the_module_imports_no_grader():
    # restates test_targeting's import check for boost.py alone: no "exploitability", no "answer.analysis"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/deepcfr/test_boost.py -q -p no:cacheprovider`
Expected: FAIL, `ModuleNotFoundError: drawmaha_solver.deepcfr.boost`.

- [ ] **Step 3: Implement `boost.py`**

Module docstring: what the stage is (plan §3 Idea 1 in the module's own words), why the weight is (1 − α)·K·arrival (the expected sampled weight it replaces; the fit multiplies by t), why natural rows are turned down rather than dropped (the reservoir sees the same offers), why a row is treated all-or-nothing (skipping is unbiased, half-treating is not), why the float32 weight must avoid 1 and α (PR 3's noise score), and why EXACT is refused (Alec's guardrail, quoted as "extra sampling at high-variance nodes, never brute-forcing the tree from a node"). Section banners: the result, the weight, the injection. `Boost` is Task 4's; leave a banner placeholder out, add it then.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/deepcfr/test_boost.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr/boost.py tests/deepcfr/test_boost.py
git commit -m "feat(rung4): inject appends one sampled boosted row per target after the walk's rows, at (1 − α)·K·arrival, turns the walk's rows there down to α, and skips a row whose weight would read as a walk row"
```

---

### Task 3: `DeepConfig.target` and `DeepConfig.boost`, saved and resumed honestly

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/train.py` (`DeepConfig`, `ADDED_LATER`, `SAMPLE_FIELDS`, `config_from_saved`)
- Test: `tests/deepcfr/test_train.py`

**Interfaces:**
- Produces: `DeepConfig(..., target: TargetSpec | None = None, boost: Sampling = Sampling())`. `__post_init__` refuses by name: `target` not `None`/`TargetSpec`; `boost` not a `Sampling`; `boost.is_exact` ("boost must sample: the boost stage is extra sampling, never the exact sum; EXACT is the tests' oracle").
- Produces: `ADDED_LATER = {"device": "cpu", "pot_power": 0.0, "target": None, "boost": Sampling()}`; `SAMPLE_FIELDS` gains `"target", "boost"`.
- Produces: `config_from_saved` rebuilds a `Mapping` under `target` as `TargetSpec(**mapping)` and under `boost` as `Sampling(**mapping)`; `asdict(DeepConfig)` already flattens both to dicts for `state.json`.

- [ ] **Step 1: Write the failing tests**

In `tests/deepcfr/test_train.py`:

```python
def test_config_defaults_are_the_specs():   # extend
    assert config.target is None and config.boost == Sampling()

@pytest.mark.parametrize("field, value", [..., ("target", "line"), ("boost", EXACT), ("boost", "walk")])
def test_config_refuses_bad_values_by_name(field, value): ...   # the existing parametrised test; `EXACT` must raise naming "boost"

def test_the_boost_fields_are_sample_fields_added_later():
    assert ADDED_LATER["target"] is None and ADDED_LATER["boost"] == Sampling()
    assert {"target", "boost"} <= set(SAMPLE_FIELDS)
    with pytest.raises(ValueError, match="target"):
        require_same_run(DeepConfig(), DeepConfig(target=TargetSpec(granularity="line", alpha=0.0)))

def test_a_config_with_a_target_and_a_sampling_round_trips_through_json():
    config = DeepConfig(target=TargetSpec(granularity="line_class", alpha=0.0, budget=7, leak="current"), boost=Sampling(opponent_range=3, playouts=2))
    assert config_from_saved(json.loads(json.dumps(asdict(config)))) == config

def test_a_state_written_before_the_boost_existed_resumes_without_one(tmp_path):
    # as `test_a_state_written_before_the_device_existed_loads_on_the_cpu`: delete "target" and "boost" from the saved config;
    # saved_config(...).target is None, .boost == Sampling(); drive(tmp_path, 2).t == 2 under DRIVEN

def test_a_saved_target_missing_a_field_is_refused_by_name():
    carried = {**asdict(DeepConfig(target=TargetSpec(granularity="line", alpha=0.0))), "grid": list(GRID)}
    del carried["target"]["leak"]
    with pytest.raises(TypeError):   # TargetSpec(**mapping) without a required field; or ValueError if config_from_saved names it — pick ValueError naming "leak" and assert that
        config_from_saved(carried)
```

Pick the second: `config_from_saved` checks the nested mapping's keys against `fields(TargetSpec)` / `fields(Sampling)` and raises `ValueError` naming the missing or unknown field, as it does for the top level.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/deepcfr/test_train.py -q -p no:cacheprovider -k "config or boost or state_written"`
Expected: FAIL on the unknown fields.

- [ ] **Step 3: Implement**

`train.py` imports `TargetSpec` from `deepcfr.targeting` and `Sampling` from `minidrawmaha.solver.boost_target` (neither is the grader; `targeting` is NumPy-only). Field docstrings in `DeepConfig`: `target` is the aim and `None` is off; `boost` is read only when `target` is set, and is a sample field all the same (so a resume of an untargeted run that changes it is refused, which is conservative and said). Update `DeepConfig`'s docstring sentence on `ADDED_LATER` ("…`None` for the target, the walk's own sampling for the boost").

- [ ] **Step 4: Run the file**

Run: `uv run pytest tests/deepcfr/test_train.py -q -p no:cacheprovider`
Expected: PASS, the oracle pin included.

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr/train.py tests/deepcfr/test_train.py
git commit -m "feat(rung4): a run carries its aim and what a boosted row samples; both are sample fields, both read as off for a run saved before they existed, and an exact sampling is refused"
```

---

### Task 4: the stage in `step`, the refresh cadence, the target set in the resume state, the progress line

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/boost.py` (`Boost`)
- Modify: `src/drawmaha_solver/deepcfr/train.py` (`DeepRun`, `StepStats`, `step`, `save_run`, `load_run`, `run_deep`, `_Saves.after_iteration`, module docstring's iteration sketch)
- Test: `tests/deepcfr/test_train.py`, `tests/deepcfr/test_boost.py`

**Interfaces:**
- Consumes: Task 2's `inject`; Task 1's `Refresher`; `TargetSet.rows_for(t, seat, seed)`, `TargetSet.save/load`, `TargetSet.built_at`, `TargetSet.spec`; `BoostTargets.build()`, `.sigma(table)`.
- Produces, in `boost.py`:

```python
class Boost:
    """A run's boost: the refresher that picks rows and the oracle that computes them, built once."""
    def __init__(self, refresher: Refresher, *, exact: BoostTargets | None = None) -> None   # exact None → BoostTargets.build() on first use
    @property
    def exact(self) -> BoostTargets
    def due(self, targets: TargetSet | None, t: int) -> bool          # targets is None or t >= built_at + spec.refresh
    def refresh(self, *, t: int, table, memories, spec: TargetSpec) -> TargetSet
        # calls the refresher; refuses by name a set whose built_at != t or whose spec != `spec`
    def phase(self, targets: TargetSet, offered, *, table, seat: int, t: int, seed: int, sampling: Sampling, walks: int) -> Injected
        # sigma = self.exact.sigma(table); drawn = targets.rows_for(t, seat, seed); inject(offered.advantage_rows, offered.advantage_targets, ...)
```

- Produces, in `train.py`: `DeepRun.targets: TargetSet | None = None` (a dataclass field with default, last); `StepStats` gains `refresh: float` (seconds, 0.0 when none ran), `boost: tuple[float, float]`, `boosted: tuple[int, int]`, `skipped: tuple[int, int]`; `step(run, collector, learner, observers=(), boost: Boost | None = None)`; `save_run` writes `targets-t{t:05d}.npz` when `run.targets is not None` and names it under `files["targets"]` (`None` otherwise); `load_run` reads it when named; `run_deep(..., boost: Boost | None = None)` refuses by name `config.target` set with `boost is None` and `boost` given with `config.target is None`; the progress line gains `seconds.refresh`, `seconds.boost`, `rows.boosted`, `rows.skipped` (always present; zeros without a target).

`step`'s order per seat p: write → observe → (p == 0 and boost and `boost.due(run.targets, t)`: `run.targets = boost.refresh(...)`, timed) → collect → (boost: `injected = boost.phase(...)`, timed; else none) → advantage merge with `injected.rows, injected.targets, weight=injected.weight`, or the natural arrays with no `weight` when there is no boost → policy merge unchanged → fit.

- [ ] **Step 1: Write the failing tests**

In `tests/deepcfr/test_boost.py`:

```python
class SpyRefresher:
    """Hands back the set it was given, records every call's t."""
    def __init__(self, sets: dict[int, TargetSet]): ...
    def __call__(self, *, t, table, memories): ...

def test_due_when_there_is_no_set_or_its_window_has_ended():
    # None → True; built_at=1, refresh=25: t=1..25 False, t=26 True

def test_refresh_refuses_a_set_built_for_another_iteration_or_another_spec():
    # built_at 3 asked at t=4 → ValueError naming "built_at"; a set with a different budget → ValueError naming "spec"

def test_phase_reads_sigma_from_the_table_it_is_handed_and_draws_this_iterations_rows():
    # FakeExact with `sigma(table)` recording id(table) and `targets` recording rows; phase(...) at t=2, seat=1 asks rows_for(2, 1, seed)
```

In `tests/deepcfr/test_train.py`, with a `fixed_set(rows0, rows1, *, built_at, spec)` helper building an `infoset` `TargetSet` by hand (weights all 1) and a `FakeBoost`-free path (use the real `Boost` with `exact=BoostTargets.build()` — the full-game oracle is cached per process, and `Sampling()` on a few round-2 rows costs milliseconds):

```python
BOOSTED = replace(DRIVEN, target=TargetSpec(granularity="infoset", alpha=0.0, budget=3, refresh=2))

def test_the_refresh_runs_at_the_first_iteration_and_every_r_after(tmp_path):
    # drive 5 iterations with a SpyRefresher: calls at t = 1, 3, 5 only; run.targets.built_at == 5

def test_boosted_rows_land_after_the_natural_rows_and_the_policy_memory_is_untouched():
    # one step with the boost vs one without, same seed: advantage memory's first a rows (rows, t, targets) equal the baseline's; the next
    # `boosted` rows are the drawn rows in order with weight (1 − α)·K·arrival; natural rows at the drawn rows weigh 0; policy memories equal
    # bit for bit (same_memory)

def test_the_boost_reads_sigma_after_this_phases_write():
    # a Spy learner whose `write` fills the table with a value that encodes (t, p); a FakeExact whose `sigma` records the table's first entry
    # at each phase; the recorded values are those of (t, 0) and (t, 1) in order

def test_stats_and_the_progress_line_carry_the_boost(tmp_path):
    # with the boost: seconds.refresh > 0 at t = 1 and == 0.0 at t = 2; len(seconds.boost) == 2; rows.boosted == [3, 3] when the pool holds 3 rows of
    # each seat; rows.skipped == [0, 0]; without: all zeros, keys present

def test_the_target_set_rides_the_resume_state_and_is_not_regraded(tmp_path):
    # drive 3 with refresh=4 and a SpyRefresher; stop; resume to 6 with a NEW SpyRefresher: it is called at t = 5 only (the saved set served t = 4);
    # `same_run` extended to compare targets (rows, weight, built_at); state.json's files["targets"] names "targets-t00003.npz"; the mark dir holds it

def test_a_resume_whose_saved_set_has_expired_refreshes_before_its_first_walks(tmp_path):
    # drive 2 with refresh=2 (built_at 1 serves 1–2); resume to 3: the refresher is called at t = 3 before collect (SpyRefresher records memories' offered counts: unchanged from the save)

def test_a_run_stopped_and_restarted_with_the_boost_equals_one_straight_run(tmp_path):
    # as the existing test, under BOOSTED with the same SpyRefresher sets: same_run, written_by equal

def test_run_deep_refuses_a_target_without_a_boost_and_a_boost_without_a_target(tmp_path):
    # ValueError naming "boost" / "target" before any walk (the collector is not opened: assert no `nets/` directory)

def test_an_old_state_without_a_target_file_loads_with_none(tmp_path):
    # drive 1 under DRIVEN; state.json has files["targets"] is None; load_run(...).targets is None
```

Also extend `same_run` to compare `targets` (both `None`, or equal `built_at`, `spec`, rows and weights per seat).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/deepcfr/test_boost.py tests/deepcfr/test_train.py -q -p no:cacheprovider -x`
Expected: FAIL (`Boost` missing, `step` rejects `boost=`).

- [ ] **Step 3: Implement**

`Boost` in `boost.py` as specified. In `train.py`: the `_timed` blocks for refresh and boost; `StepStats` fields; the merge branch; `save_run`/`load_run` with `_targets_name(t) -> f"targets-t{t:05d}.npz"`, the file in `keep` and in the mark; `run_deep`'s two refusals before `_open`; the progress line. Module docstring: the iteration sketch gains two lines (`targets = boost.refresh(...)  # every R iterations, before seat 0's walks` and `rows = boost.phase(...)  # B sampled boosted rows after the walk's, at (1 − α)·K·arrival`), and a paragraph "**The boost.**" stating where σ is read, why the injected rows come after the natural ones, why the policy memory is untouched, and that the set is resume state. Keep the timings paragraph; Task 7 appends the measured boost cost.

- [ ] **Step 4: Run the rung-4 suite**

Run: `uv run pytest tests/deepcfr -q -p no:cacheprovider`
Expected: PASS, the oracle pin unchanged, torch tests skipped.

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr/boost.py src/drawmaha_solver/deepcfr/train.py tests/deepcfr/test_boost.py tests/deepcfr/test_train.py
git commit -m "feat(rung4): the boost stage runs in every iteration — a refresh every R before seat 0's walks, B sampled boosted rows after each seat's, the set saved with the run — and a run without an aim is today's run bit for bit"
```

---

### Task 5: the unbiasedness gate on a restricted-deal game with the oracle learner

**Files:**
- Test: `tests/deepcfr/test_boost.py`

**Interfaces:**
- Consumes: `BoostTargets.build(holes=(seat0_holes, seat1_holes))`, `Collector(seed=, workers=0, deal=restricted_deal)` with a module-level `restricted_deal` (restated from `tests/minidrawmaha/test_boost_target.py`, as `tests/AGENTS.md` requires), `step`, `OracleLearner`, `EXACT` (the oracle, tests only).

- [ ] **Step 1: Write the test**

```python
HOLES_0 = [parse_cards("2c 3c 5d"), parse_cards("2d 3d 5c")]
HOLES_1 = [parse_cards("2d 4h 6h"), parse_cards("3h 4c 4d"), parse_cards("2c 2h 6c"), parse_cards("4s 5h 6s")]
_ALLOWED: list = []
def restricted_deal(rng): ...   # module-level, as the collector pickles it

def banked_at(memory: Reservoir, row: int, width: int) -> np.ndarray:
    """Σ t · weight · target over the memory's rows at `row`, the oracle's sum, per legal column."""

def test_the_boost_leaves_expected_banked_regret_and_weight_unchanged_at_every_targeted_row():
    # Pilot: 2,000 walks of seat 0 at t = 1 under a fixed positive table → the 6 seat-0 rows with the largest arrival (≥ 0.05), which include the
    # two opening rows (arrival 0.5 each) and draw / round-2 rows; the fixed set is `infoset`, budget 6, refresh 100, α = 0, built_at 1.
    # For seeds 0..N−1 (N = 50): run one `step` with OracleLearner and K = 40 walks (a) without the boost and (b) with it (same seed → same walks);
    # read `banked_at` for every targeted row in both; the paired difference d_s = (b) − (a) has mean within 4 s.e. of 0 at every legal column,
    # and the total weight at the row (Σ weight) differs from the baseline's count by a mean within 4 s.e. of 0.
    # Not vacuous: (b)'s variance at the opening rows is below (a)'s (the whole range removes the hidden hand), and `rows.boosted == 6`.
    # Also: the mean of (b)'s banked regret over seeds equals K · arrival · EXACT regret within 4 s.e. — the oracle, used here only.
```

Seed count and K: start at N = 50, K = 40; if the test exceeds 30 s on this Mac, keep N = 20 in the default suite and the 50-seed version behind `MINIDRAWMAHA_FULL_TARGET=1`, both pinned by the same function with the count as a parameter. Say the measured time in the test's comment.

- [ ] **Step 2: Run it and record the time**

Run: `uv run pytest tests/deepcfr/test_boost.py -q -p no:cacheprovider -k unchanged --durations=1`
Expected: PASS; note the seconds.

- [ ] **Step 3: Mutation check (not committed)**

Temporarily drop the `(1 − α)` factor (α = 0 makes that invisible: instead temporarily use `walks + 1`), then temporarily leave natural rows at weight 1: each must fail the test. Restore.

- [ ] **Step 4: Commit**

```bash
git add tests/deepcfr/test_boost.py
git commit -m "feat(rung4): on a restricted deal with the oracle learner the boosted iteration banks the same expected regret and weight as plain sampling at every targeted row, paired over 50 seeds"
```

---

### Task 6: the command line

**Files:**
- Modify: `scripts/deepcfr/run_deep_cfr.py`
- Test: `tests/deepcfr/test_train.py` (the CLI tests, which load the script by path)

**Interfaces:**
- Produces flags, each with a help line: `--boost {none,random,infoset,line,line_class}` (default none; "none" ⇒ `target=None`), `--boost-budget` (B, default `TargetSpec` default 500), `--boost-refresh` (R, 25), `--boost-alpha` (0.0), `--boost-leak {average,current}` (average), `--boost-pool` (20), `--boost-noise-cap` (20.0), `--boost-round-two-only` (store_true); the sampling of one boosted row: `--boost-playouts` (1), `--boost-opponent-range` (default `all`), `--boost-board-cards` (1), `--boost-opponent-draws` (1), `--boost-hero-draws` (1), `--boost-opponent-actions` (1), each an int or the word `all`.
- Produces: `config_from_args` builds `target=None if args.boost == "none" else TargetSpec(granularity=args.boost, alpha=args.boost_alpha, budget=..., refresh=..., noise_cap=..., pool=..., round_two_only=..., leak=args.boost_leak)` and `boost=Sampling(playouts=..., opponent_range=..., ...)`; `DeepConfig` refuses an exact sampling and `parse` turns that into `parser.error`.
- Produces: `main` builds `average = ExactAverage()`; when `config.target` is set, `refresher = GraderRefresher(config.target, walks=config.walks, average=average, seed=config.seed)` and `boost = Boost(refresher)`; `run_deep(..., observers=(average,), boost=boost)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_cli_parses_the_boost_flags_into_the_aim_and_the_sampling():
    args = cli.parse(["--out", "y", "--boost", "line", "--boost-budget", "7", "--boost-alpha", "0", "--boost-leak", "current",
                      "--boost-opponent-range", "3", "--boost-playouts", "2"])
    config = cli.config_from_args(args)
    assert config.target == TargetSpec(granularity="line", alpha=0.0, budget=7, leak="current")
    assert config.boost == Sampling(opponent_range=3, playouts=2)
    assert cli.config_from_args(cli.parse(["--out", "y"])).target is None
    assert cli.config_from_args(cli.parse(["--out", "y"])).boost == Sampling()
    assert cli.config_from_args(cli.parse(["--out", "y", "--boost-opponent-range", "all"])).boost.opponent_range is None

def test_cli_refuses_an_exact_boost_at_the_command_line():
    with pytest.raises(SystemExit):
        cli.parse(["--out", "y", "--boost", "line", "--boost-board-cards", "all", "--boost-opponent-draws", "all",
                   "--boost-hero-draws", "all", "--boost-opponent-actions", "all"])

def test_main_hands_the_loop_a_boost_only_when_an_arm_is_named(monkeypatch):
    # monkeypatch cli.run_deep with a recorder and cli.GraderRefresher with a stub; `--boost none` → boost is None;
    # `--boost random` → a Boost whose refresher is the stub built with (config.target, walks=config.walks, average=the ExactAverage in observers)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/deepcfr/test_train.py -q -p no:cacheprovider -k cli`
Expected: FAIL (unknown flags).

- [ ] **Step 3: Implement**

An `int_or_all(text) -> int | None` argparse type in the script. The script's docstring gains the arm command as its second line: `… --boost line --boost-budget 500` and one sentence on `--boost` naming the arm and `--boost-*` its budget, refresh, α, leak and the sampling of each boosted row. `main` imports `GraderRefresher` from the sibling script by path-free import (`from targeting_refresh import GraderRefresher` works when run from `scripts/deepcfr/`? No: the script is run from the repo root, so load it the way the tests do, `importlib.util.spec_from_file_location` on `Path(__file__).with_name("targeting_refresh.py")`) and `Boost` from `drawmaha_solver.deepcfr.boost`.

- [ ] **Step 4: Run the CLI tests and the script's help**

Run: `uv run pytest tests/deepcfr/test_train.py -q -p no:cacheprovider -k cli && uv run python scripts/deepcfr/run_deep_cfr.py --help | grep -c boost`
Expected: PASS; the count is 14 or more.

- [ ] **Step 5: Commit**

```bash
git add scripts/deepcfr/run_deep_cfr.py tests/deepcfr/test_train.py
git commit -m "feat(rung4): --boost names the arm; its budget, refresh, α, leak and the sampling of each boosted row are flags, and an exact sampling is refused at the command line"
```

---

### Task 7: the whole-game smoke, the measured cost, and the docs

**Files:**
- Test: `tests/deepcfr/test_train.py` (gated)
- Modify: `src/drawmaha_solver/deepcfr/train.py` (timings paragraph), `src/drawmaha_solver/deepcfr/AGENTS.md`, `src/drawmaha_solver/deepcfr/__init__.py`, `scripts/AGENTS.md`, `README.md` (if the rung-4 section carries PR 3's targeting sentence)

- [ ] **Step 1: Write the gated smoke**

```python
@pytest.mark.skipif(not os.environ.get("MINIDRAWMAHA_FULL_DEEP"), reason="a real refresh (two grader walks, ~2 min) and 2 boosted iterations on the whole game; set MINIDRAWMAHA_FULL_DEEP=1")
def test_two_boosted_iterations_on_the_whole_game_record_their_cost(tmp_path):
    # the line arm, B = 500, K = 200, workers 0, OracleLearner, the real GraderRefresher(leak="current") so no ExactAverage is needed;
    # progress.jsonl has 2 lines with seconds.refresh > 0 at t = 1 only, seconds.boost[0] and [1] > 0, rows.boosted summing to ≥ 500 per line;
    # print the three numbers so the run log can quote them
```

- [ ] **Step 2: Run it**

Run: `MINIDRAWMAHA_FULL_DEEP=1 uv run pytest tests/deepcfr/test_train.py -q -p no:cacheprovider -k whole_game -s`
Expected: PASS; copy `seconds.refresh` and `seconds.boost` into `train.py`'s timings paragraph with the date, B, K and the arm. Expect the boost near 23 s per iteration at B = 500 (PR 3's per-row costs scaled); if it is far off, say so in the PR body rather than adjust B.

- [ ] **Step 3: Docs**

- `deepcfr/AGENTS.md`: the opening paragraph says the boost stage is in (`boost` module); a rule "**The boost never computes the exact sum.** `DeepConfig` refuses an exact `Sampling` and `inject` refuses it again; `EXACT` is the tests' oracle. A boosted row is a sampled play-out at the weight the walk's samples would have carried, (1 − α)·K·arrival, offered after the natural rows, which are turned down to α and still offered. A drawn row whose float32 weight would equal 1 or α, or whose arrival is 0, is left untreated and counted as skipped: the noise score tells walk rows from boosted rows by weight alone."; the resume-state rule gains "and the current target set, `targets-t{t:05d}.npz`, when the run has an aim"; the `DeepConfig` rule's example list gains `target` and `boost`.
- `deepcfr/__init__.py`: "Eleven modules"; a `boost` entry; `boost` in the NumPy-only list.
- `scripts/AGENTS.md`: `run_deep_cfr.py`'s entry mentions `--boost`; `targeting_refresh.py`'s entry says the refresher carries the exact average.
- `train.py`: the module docstring's measured paragraph gains the smoke's numbers.
- `README.md`: `grep -n -i "leak\|targeted\|aim" README.md`; if PR 3's sentence is in the rung-4 section, add one sentence after it saying the boost stage exists, what a boosted row is, and that no run has been raced yet (PR 5).
- `tests/test_agent_docs.py` must pass (every path named in backticks exists).

- [ ] **Step 4: Run the whole default suite**

Run: `uv run pytest -q -p no:cacheprovider`
Expected: all pass without torch (torch tests skipped); note the counts for the PR body.

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr/AGENTS.md src/drawmaha_solver/deepcfr/__init__.py src/drawmaha_solver/deepcfr/train.py scripts/AGENTS.md tests/deepcfr/test_train.py README.md
git commit -m "feat(rung4): the boost's measured cost at B = 500 on the whole game, and the instruction files say the stage is in and never exact"
```

---

### Task 8: fresh-context review, PR, merge-ready loop

- [ ] **Step 1:** Dispatch a fresh-context reviewer subagent with the diff against `origin/AlecROndo/pr-3-targeting` (PR 3's head merged with main), this plan, and the decisions section; it re-derives the weight arithmetic, the skip rules and the σ timing, and checks every number quoted in docstrings against the tests. Disposition each finding (fix, or answer with the reason) before the first push.
- [ ] **Step 2:** Push `AlecROndo/pr-4-boost-stage`, open the PR against `main` with the body: what it adds (per file), the decisions and who settled them, departures from the appendix (the `--boost` flags, `Refresher` without `average`, `TargetSpec.leak`, no EXACT), the measured cost, the tests and their counts, the reviewer's findings dispositioned, and "PR 5 (the race) is not in this PR".
- [ ] **Step 3:** Watch `claude-review` (one to two minutes after each push), fix or answer every finding, re-push until the tree is clean and no finding is open. Do not merge; Alec merges.

---

## Self-review

- **Spec coverage.** Idea 1's weight and turn-down: Task 2. "Natural rows first": Task 2 (order) and Task 4 (merge). σ_t the same materialised table the walk read: Task 4's σ-timing test. `DeepConfig` field + `ADDED_LATER` + `SAMPLE_FIELDS`: Task 3. Flags: Task 6. `progress.jsonl` keys: Task 4. Refresher passed in from the script layer, `run_deep` taking it: Tasks 4 and 6. Set saved per refresh and resume reloading rather than re-grading: Task 4. The appendix's four tests: `--target none` pin (unchanged, Task 4 runs it), draws without replacement (PR 3's; Task 2 refuses a repeat), the restricted-deal expectation over 50 seeds (Task 5), the gated whole-game smoke recording `seconds.boost` (Task 7). PR 3's carried finding (weight 1 or α): Task 2. Alec's no-EXACT rule: Tasks 2, 3, 6. Plan numbering note: the PR body.
- **Types.** `Injected.weight` float32 everywhere; `inject` takes `drawn` int, `Boost.phase` passes `rows_for`'s int64 sorted array; `Refresher.__call__(*, t, table, memories)` in Tasks 1 and 4; `GraderRefresher(spec, *, walks, average=None, seed=0, exact=None)` in Tasks 1 and 6; `DeepRun.targets` in Tasks 4 and 5.
- **Review focus.** Items 1 and 4 in Task 2, 2 and 3 in Task 4, 5 in Task 3.
- **Proportion.** Bodies appear only for `inject`'s algorithm and `step`'s order, which the signatures do not fix.
