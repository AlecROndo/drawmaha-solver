# Deep CFR M1, PR 4: the loop — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the Deep CFR iteration loop over PRs 1–3's parts — materialise, collect, merge, refit, both seats in turn — with an exact resume, grid-mark saves, a progress log, signal and deadline handling, a local CLI, the exactness pin (gate 1) and the short real run (gate 5, current iterate only).

**Architecture:** one new module `deepcfr/train.py`. `DeepConfig` is everything a run is and travels in every checkpoint; `DeepRun` is the mutable state (t, two nets, three memories); `step` is one iteration in the spec's order; `Learner` is the single seam between the loop and the model — `NetLearner` wraps PR 2's `fit_from_scratch` + `materialise` behind lazy torch imports, `OracleLearner` is the "perfect net" that turns the loop into rung 3's LCFR for gate 1; `save_run`/`load_run` are the resume state with `state.json` as the commit point; `run_deep` is the driver, modelled on `lockstep_run.run_to`. `scripts/run_deep_cfr.py` is argparse over `DeepConfig`. No new math: the exact average and Π are PR 5, which plugs into `observers`.

**Tech Stack:** Python 3.12+, NumPy; torch only inside `NetLearner` (the `deep` extra). `train.py` imports no torch at module level, so gate 1 and every fake-learner test run in the default suite.

**Spec:** `docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`, sections "The loop, in rung-3 names", "Memories", "Collecting", "Modal" (the `out/` layout and save cadence), correctness gates 1 and 5, and PR 4 of the PR table. The Conductor prompt for this PR is `/Users/alec/Desktop/Claude/poker/rung4-context/CONDUCTOR-PROMPT-deep-cfr-pr4.md`.

## Global Constraints

- The iteration, in order, per seat p in (0, 1): `learner.write(table, nets)`; observers `(table, t, 1 − p)`; `rows = collector.collect(seat=p, t=t, walks=K)`; `advantage[p].merge(…, t=t, rng=reservoir_stream(seed, t, p, ADVANTAGE))`; `policy.merge(…, rng=reservoir_stream(seed, t, p, POLICY))`; `nets[p] = learner.fit(*advantage[p].contents(), seed=(seed, t, p))`. Materialise runs before EVERY phase.
- Defaults: K = 10,000 walks, C = 50,000,000 rows per memory, grid marks (10, 20, 50, 100, 200, 500, 1,000, 2,000), `save_every_s` = 1,800. Net and refit defaults are `FitConfig`'s (3 × 64, 4,000 × 10,000, lr 1e-3, clip 1, threads 8).
- A resume with a different seed, K, C or net/refit size is refused BY NAME. W, the grid and the save cadence may change.
- Determinism: everything is a function of (seed, t, seat, k); no RNG state is ever checkpointed.
- `out/` layout: `nets/p{seat}-t{t:05d}.pt` every iteration, never overwritten; `marks/t{t:05d}/` at each grid mark; `resume/` the resume state; `progress.jsonl` one line per iteration.
- Every save is a temp file + fsync + `os.replace`; `state.json` is written LAST.
- Do not change merged code: `lockstep.py`, `lockstep_run.py`, `mccfr.py`, `parallel.py`, `packed_table.py`, `regret_rules.py`, `exploitability.py`, `strategy.py`, `features.py`, `deepcfr/{net,fit,materialise,reservoir,collect}.py`.
- Repo style: long explanatory module docstring, 75-hyphen section banners, WHY-only comments, `from __future__ import annotations`, keyword-only args where ints could be transposed. Commits `feat(rung4): …`.

## Decisions this plan makes that the spec and prompt leave open

- **Gate 1 is pinned twice, because the memory is float32.** `collect.rows_of` casts each regret sample to float32 (PR 3's row format), and about half of a walk's samples are not float32-exact (measured 2026-10-20: 1,022 of 2,067 entries exact on a random table, 1,656 of 2,230 on the uniform table). So the loop cannot equal `train_lockstep` bit for bit through the memory. Two pins: (a) BIT FOR BIT, across 3 iterations, against LCFR built from rung 3's own `lockstep.walk` → `flatten` → `apply_arrays(rule="lcfr")` with each walk's `regret_values` rounded through float32 first — the only difference from `train_lockstep` is that one cast, and every join of the loop (stale table, wrong t, wrong seat, dropped rows, wrong order) still breaks it; (b) to tolerance (`rtol=1e-5, atol=1e-6`, plus exact zeros on never-seen rows) against the untouched `train_lockstep(new_lockstep(seed, workers=K, rule="lcfr"), N)`.
- **`DeepConfig` carries the fit parameters flat** (`width, depth, steps, batch, lr, clip, threads`) rather than a `fit: FitConfig` field: `fit.py` imports torch at module level, so a `FitConfig` default would pull torch into `train.py`'s import. `NetLearner` builds the `FitConfig` lazily.
- **The `Learner` seam has four methods, not two**: `fit`, `write`, and `save`/`load` of a model, because the resume state must round-trip whatever the learner produced (a net, or the oracle's arrays). `fit` returns `Fitted(model, loss)` so the progress line can carry the refit's loss without the loop knowing torch.
- **The refit's reported loss** is `fit.batch_loss` on one batch of `config.batch` rows drawn with replacement from the memory just fitted by a generator seeded with the fit's own seed: the same rule every iteration, so the number is comparable across the run. The oracle reports `None`.
- **Resume files are versioned by t.** `save_run(run, dir)` writes `advantage-p0-t{t:05d}.npz`, `advantage-p1-t…`, `policy-t….npz`, `p0-t….pt`, `p1-t….pt`, then `state.json` naming them, then removes every other file in `dir`. A crash before `state.json` leaves the previous set named by the previous `state.json`; overwriting fixed names would not (a crash between two memories would mix iterations under one `state.json`). A grid mark is `save_run` into `marks/t…/`, so a mark is loadable as a run.
- **`OracleLearner` lives in `train.py`**, in its own section: it is the clearest statement of what a `Learner` is, it is NumPy only, and the walkthrough can show it next to the pin.
- **`_stop_on_signals` is copied** from `lockstep_run.py` (it is private there and the prompt says not to edit that module). `lockstep._columns` is imported for the oracle's column expansion, as `collect.py` imports `reservoir._is_int`.
- **`current_iterate(table, *, iteration=0, seed=0)`** moves into this PR from PR 5 so gate 5 has something to grade; PR 5 keeps the exact average and Π.
- **The stop is decided after each iteration**: target reached, a signal, or the deadline. So the last progress line carries `"stopped"` and the final resume save happens inside the same save as that iteration; the deadline is checked after an iteration, not before, so a run always completes at least one.
- **One `commit()` per iteration** (the net files and the progress line; marks and resume when written), rather than rung 3's one per save, because the net files are tens of KB and PR 6's Volume should hold every iteration.

## Review Focus

1. A learner that leaves the table unwritten between phases (a stale σ) must fail gate 1 — pinned in Task 2 by the bit-for-bit pin across 3 iterations (`test_the_oracle_loop_is_lcfr_bit_for_bit`), where seat 1's phase at t walks against seat 0's freshly refit sum.
2. A resumed run must take the same reservoir draws as the straight run once the memory is over capacity — pinned in Task 3 (`test_resume_equals_uninterrupted`) with capacity 200 so eviction happens inside the test.
3. A crash between writing a memory file and `state.json` must leave the old state loadable, with no half-written set mistaken for it — pinned in Task 3 (`test_a_save_interrupted_before_state_json_keeps_the_previous_state`).
4. A `state.json` that names a file which is missing must be refused by name, not read as an empty memory — pinned in Task 3 (`test_load_refuses_a_missing_state_or_a_missing_file`).
5. A run restarted with the same command after a signal must continue, and one restarted with a different seed, K, C or net size must be refused by the field's name — pinned in Task 4 (`test_sigterm_finishes_the_iteration_saves_and_resumes_exactly`, `test_a_resume_refuses_a_changed_config_by_name`).

---

### Task 1: the config, the run, the learner seam, and one iteration

**Files:**
- Create: `src/drawmaha_solver/deepcfr/train.py`
- Test: `tests/deepcfr/test_train.py`

**Interfaces:**
- Consumes: `Reservoir(capacity)`, `.merge(rows, targets, *, t, rng)`, `.contents()`, `.offered`, `.size`; `reservoir_stream(*, seed, t, seat, memory)`, `Memory.ADVANTAGE/POLICY`; `Collector(seed=, workers=)` with `.table` and `.collect(seat=, t=, walks=) -> Collected`; `lockstep.SEATS`.
- Produces:
  - `GRID = (10, 20, 50, 100, 200, 500, 1_000, 2_000)`.
  - `@dataclass(frozen=True) DeepConfig`: `seed: int = 0`, `walks: int = 10_000`, `workers: int = 8`, `capacity: int = 50_000_000`, `width: int = 64`, `depth: int = 3`, `steps: int = 4_000`, `batch: int = 10_000`, `lr: float = 1e-3`, `clip: float = 1.0`, `threads: int = 8`, `grid: tuple[int, ...] = GRID`, `save_every_s: float = 1_800.0`. `__post_init__` refuses by name: a negative or bool seed, `walks < 1`, `workers < 0`, `capacity < 1`, any of `width, depth, steps, batch, threads < 1`, `lr`/`clip` not `> 0`, a grid that is not strictly increasing positive ints, `save_every_s < 0`.
  - `SAMPLE_FIELDS = ("seed", "walks", "capacity", "width", "depth", "steps", "batch", "lr", "clip", "threads")`: the fields a resume may not change.
  - `require_same_run(saved: DeepConfig, asked: DeepConfig) -> None`: `ValueError` listing every `SAMPLE_FIELDS` difference as `name: saved → asked`.
  - `@dataclass(frozen=True, slots=True) Fitted`: `model: object`, `loss: float | None`.
  - `class Learner(Protocol)`: `fit(self, rows, t, targets, *, seed: tuple[int, int, int]) -> Fitted`; `write(self, table: PackedTable, models: Sequence[object | None]) -> None`; `save(self, model, path: Path) -> None`; `load(self, path: Path) -> object`.
  - `Observer = Callable[[PackedTable, int, int], None]` — `(table, t, seat)`.
  - `@dataclass(slots=True) DeepRun`: `config: DeepConfig`, `t: int`, `nets: list[object | None]` (length 2), `advantage: tuple[Reservoir, Reservoir]`, `policy: Reservoir`.
  - `new_run(config: DeepConfig) -> DeepRun`: t = 0, nets `[None, None]`, three `Reservoir(config.capacity)`.
  - `@dataclass(frozen=True, slots=True) StepStats`: `t: int`, `materialise: tuple[float, float]`, `collect: tuple[float, float]`, `merge: tuple[float, float]`, `fit: tuple[float, float]`, `seconds: float`, `loss: tuple[float | None, float | None]`, `rows: tuple[int, int]` (advantage rows offered per phase).
  - `step(run: DeepRun, collector, learner: Learner, observers: Sequence[Observer] = ()) -> StepStats`: the iteration block above at `t = run.t + 1`, with `collector.table` as the table, then `run.t = t`. `write` is handed `tuple(run.nets)`.

- [ ] **Step 1: Write the failing tests**

Helpers at the top of `tests/deepcfr/test_train.py`:

```python
class Spy:
    """One event list shared by a learner, a collector wrapper and an observer."""
    def __init__(self, inner: Collector): self.inner = inner; self.events = []
    # learner
    def fit(self, rows, t, targets, *, seed):
        self.events.append(("fit", seed, (rows.copy(), t.copy(), targets.copy())))
        return Fitted(("model", seed), None)
    def write(self, table, models): self.events.append(("write", tuple(models)))
    def save(self, model, path): ...   # json of the seed tuple
    def load(self, path): ...
    # collector
    @property
    def table(self): return self.inner.table
    def collect(self, *, seat, t, walks):
        self.events.append(("collect", seat, t, walks)); return self.inner.collect(seat=seat, t=t, walks=walks)
    # observer
    def observe(self, table, t, seat): self.events.append(("observe", t, seat))

@pytest.fixture(scope="module")
def collector():
    with Collector(seed=3, workers=0) as run: yield run

SMALL = DeepConfig(seed=3, walks=4, workers=0, capacity=100_000)
```

```python
def test_config_defaults_are_the_specs():
    c = DeepConfig()
    assert (c.walks, c.capacity, c.grid, c.save_every_s) == (10_000, 50_000_000, GRID, 1_800.0)
    assert (c.width, c.depth, c.steps, c.batch, c.lr, c.clip, c.threads) == (64, 3, 4_000, 10_000, 1e-3, 1.0, 8)

def test_config_refuses_bad_values_by_name():
    for field, value in [("seed", -1), ("seed", True), ("walks", 0), ("workers", -1), ("capacity", 0),
                         ("width", 0), ("steps", 0), ("lr", 0.0), ("clip", float("nan")),
                         ("grid", (10, 10)), ("grid", (20, 10)), ("save_every_s", -1.0)]:
        with pytest.raises(ValueError, match=field):
            DeepConfig(**{field: value})

def test_require_same_run_names_every_changed_sample_field():
    with pytest.raises(ValueError, match="seed.*walks.*width") as caught:
        require_same_run(DeepConfig(), DeepConfig(seed=1, walks=5, width=32, workers=2, grid=(1,)))
    assert "workers" not in str(caught.value)
    require_same_run(DeepConfig(), DeepConfig(workers=2, grid=(1,), save_every_s=0.0))   # allowed

def test_new_run_is_empty(): ...   # t == 0, nets [None, None], three memories of capacity C with offered == 0

def test_one_iteration_runs_both_phases_in_the_specs_order(collector):
    spy = Spy(collector); run = new_run(SMALL)
    stats = step(run, spy, spy, observers=(spy.observe,))
    kinds = [e[0] for e in spy.events]
    assert kinds == ["write", "observe", "collect", "fit"] * 2
    assert [e[1:] for e in spy.events if e[0] == "collect"] == [(0, 1, 4), (1, 1, 4)]
    assert [e[1:] for e in spy.events if e[0] == "observe"] == [(1, 1), (1, 0)]
    assert [e[1] for e in spy.events if e[0] == "fit"] == [(3, 1, 0), (3, 1, 1)]
    writes = [e[1] for e in spy.events if e[0] == "write"]
    assert writes == [(None, None), (("model", (3, 1, 0)), None)]
    assert run.t == 1 and run.nets == [("model", (3, 1, 0)), ("model", (3, 1, 1))]
    assert stats.t == 1 and len(stats.materialise) == 2 and stats.seconds > 0

def test_the_second_iteration_writes_both_nets_and_the_fresh_one_before_seat_one(collector):
    # after two steps the writes are (None,None), (m10,None), (m10,m11), (m20,m11)

def test_fit_receives_the_memory_as_it_stands_after_the_merge(collector):
    # for each seat's last fit event, its arrays equal run.advantage[p].contents() bytewise

def test_each_merge_uses_its_own_stream_and_the_policy_memory_takes_both_seats(collector):
    # capacity 100 (so draws matter), K = 8, two steps with the Spy (write is a no-op, so the table
    # stays uniform and the walks can be replayed by hand). Rebuild by hand:
    #   for t in (1, 2): for p in (0, 1): rows = collector.collect(seat=p, t=t, walks=8)
    #       adv[p].merge(rows.advantage_rows, rows.advantage_targets, t=t, rng=reservoir_stream(seed=3, t=t, seat=p, memory=Memory.ADVANTAGE))
    #       pol.merge(rows.policy_rows, rows.policy_targets, t=t, rng=reservoir_stream(..., memory=Memory.POLICY))
    # assert every memory's contents and `offered` equal bytewise; assert pol.offered == sum of both seats' policy rows.
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/deepcfr/test_train.py -v` → `ModuleNotFoundError: drawmaha_solver.deepcfr.train`.

- [ ] **Step 3: Implement `train.py`** with the module docstring (what the loop is, why materialise runs before every phase, why the memory's float32 makes gate 1 two pins, why the Learner seam exists), the sections "The run" (`GRID`, `DeepConfig`, `SAMPLE_FIELDS`, `require_same_run`, `DeepRun`, `new_run`), "The learner seam" (`Fitted`, `Learner`, `Observer`), "One iteration" (`StepStats`, `step`, a `_timed()` context manager or `time.perf_counter` pairs).

- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/deepcfr/test_train.py -v` → PASS; `uv run python -c "import sys, drawmaha_solver.deepcfr.train; assert 'torch' not in sys.modules"` prints nothing.

- [ ] **Step 5: Commit** — `git commit -m "feat(rung4): the Deep CFR run, the learner seam, and one iteration in the spec's order"`

---

### Task 2: the oracle and the exactness pin (gate 1)

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/train.py` (new section "The oracle: gate 1's perfect net")
- Test: `tests/deepcfr/test_train.py`

**Interfaces:**
- Consumes: Task 1; `lockstep.walk`, `lockstep.flatten`, `lockstep.apply_arrays`, `lockstep._columns`, `lockstep.new_lockstep`, `lockstep.train_lockstep`; `RegretRule.LCFR`; `PackedTable.whole_game()`, `.widths()`.
- Produces:
  - `@dataclass(frozen=True, slots=True) OracleModel`: `rows: np.ndarray` (int32), `t: np.ndarray` (int32), `targets: np.ndarray` (float32, (n, 4)) — a copy of the memory at fit time.
  - `class OracleLearner`: `fit` returns `Fitted(OracleModel(copies), None)`; `write(table, models)` sets `table.cumulative_regret[...]` to, per model, `R = 0; for t′ in increasing order: columns = _columns(starts[rows], widths[rows]) for that t′'s rows in memory order; totals by np.unique + np.add.at; R[touched] += int(t′) * totals` — LCFR's `bank_rows` ops in LCFR's order; `save`/`load` are `np.savez` into the given path (opened as a file, so no `.npz` is appended) and `np.load`.

- [ ] **Step 1: Write the failing tests**

```python
def lcfr_rounded(seed, walks, iterations):
    """LCFR from rung 3's own walk/flatten/apply_arrays, each sample rounded through float32 as rows_of does. Yields the table after each iteration."""
    table = PackedTable.whole_game()
    for t in range(1, iterations + 1):
        for seat in SEATS:
            flats = [flatten(walk(table, random_deal, seed, t, seat, k, ()), 0) for k in range(walks)]
            flats = [replace(f, regret_values=f.regret_values.astype(np.float32).astype(np.float64)) for f in flats]
            apply_arrays(table, flats, RegretRule.LCFR, t)
        yield table

def oracle_loop(seed, walks, workers, iterations):
    """Yields collector.table after each iteration of the oracle loop."""

@pytest.mark.parametrize("workers", [0, 2])
def test_the_oracle_loop_is_lcfr_bit_for_bit(workers):
    for deep, ref in zip(oracle_loop(5, 40, workers, 3), lcfr_rounded(5, 40, 3)):
        np.testing.assert_array_equal(deep.cumulative_regret, ref.cumulative_regret)
    # and the memory never evicted: run.advantage[p].offered <= capacity for both seats

def test_the_oracle_loop_equals_train_lockstep_to_float32_precision():
    for deep, (t, solve) in zip(oracle_loop(5, 40, 0, 3), lockstep_exact(5, 40, 3)):
        exact = solve.table.cumulative_regret
        np.testing.assert_allclose(deep.cumulative_regret, exact, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(deep.cumulative_regret == 0.0, exact == 0.0)   # never-seen rows exactly 0
        assert not np.array_equal(deep.cumulative_regret, exact)   # the float32 cast is why there are two pins

def test_the_uniform_column_leaves_the_walks_regrets_untouched():
    # walk(..., ()) and walk(..., (Average.UNIFORM,)) record identical (key, slot, r) lists for 20 walks

def test_oracle_models_round_trip(tmp_path): ...   # save then load: three arrays bytewise equal
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/deepcfr/test_train.py -k oracle -v` → `ImportError: OracleLearner`.
- [ ] **Step 3: Implement `OracleModel`, `OracleLearner`** in `train.py`. `write` computes `starts = cumsum(widths) − widths` from `table.widths()` once per table (cache by `id(table)`), zeros a fresh array the size of `cumulative_regret`, applies every model's sums (the seats' columns are disjoint), then assigns INTO `table.cumulative_regret[...]` — never rebinds (the workers read the shared block).
- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/deepcfr/test_train.py -v` → PASS, whole file under ~60 s.
- [ ] **Step 5: Commit** — `git commit -m "feat(rung4): the oracle learner, and gate 1 — the loop with a perfect net is rung 3's LCFR bit for bit"`

---

### Task 3: the resume state

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/train.py` (section "The resume state")
- Test: `tests/deepcfr/test_train.py`

**Interfaces:**
- Consumes: Tasks 1–2; `Reservoir.save/load`.
- Produces:
  - `STATE = "state.json"`.
  - `save_run(run: DeepRun, directory: Path, *, learner: Learner) -> None`: `mkdir -p`; write `advantage-p{p}-t{t:05d}.npz` ×2, `policy-t{t:05d}.npz`, `p{p}-t{t:05d}.pt` per non-None net (each atomic: `Reservoir.save` already is; nets go to `name.tmp` then `os.replace`); then `state.json` via `_write_state(directory, state: dict)` (temp + fsync + replace) holding `{"t", "config": asdict(config) (grid as list), "files": {"advantage": [..], "policy": .., "nets": [.. or None]}}`; then remove every file in `directory` that is not `state.json` and not named in `files`.
  - `load_run(directory: Path, *, learner: Learner) -> DeepRun`: refuses by name a missing `state.json` (`ValueError("… has no state.json")`) and a named file that is absent (`ValueError` naming the file); rebuilds `DeepConfig(**config, grid=tuple(...))`; loads the memories and nets.

- [ ] **Step 1: Write the failing tests**

```python
def same_memory(a, b): contents bytewise, offered, capacity
def same_run(a, b): t, config, memories, oracle nets' arrays

def test_save_and_load_round_trip(tmp_path): step twice with the oracle; save; load; same_run; no *.tmp; exactly the named files + state.json remain

def test_resume_equals_uninterrupted(tmp_path):
    # DeepConfig(seed=5, walks=8, workers=0, capacity=200): 3 steps straight vs 2 steps, save, load into a fresh
    # run, 1 step; same_run, and after OracleLearner().write on two fresh whole-game tables the cumulative_regret
    # arrays are bytewise equal; both advantage memories ended over capacity (offered > 200).

def test_a_save_interrupted_before_state_json_keeps_the_previous_state(tmp_path, monkeypatch):
    # save at t=1; step; monkeypatch train._write_state to raise; save raises; load_run -> t == 1 and same_run as before;
    # then a real save at t=2 succeeds and the t=1 files are gone.

def test_load_refuses_a_missing_state_or_a_missing_file(tmp_path):
    # empty dir -> ValueError match "state.json"; save, delete policy file -> ValueError naming "policy-t00001.npz"
```

- [ ] **Step 2: Run to verify they fail** — `ImportError: save_run`.
- [ ] **Step 3: Implement** `save_run`, `load_run`, `_write_state`, `_memory_name(kind, seat, t)`, `_net_name(seat, t)`.
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(rung4): the resume state — three memories, two nets, and state.json written last as the commit point"`

---

### Task 4: the driver

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/train.py` (sections "Stopping on a signal", "The driver")
- Test: `tests/deepcfr/test_train.py`

**Interfaces:**
- Consumes: Tasks 1–3; `Collector`.
- Produces:
  - `NETS = "nets"`, `MARKS = "marks"`, `RESUME = "resume"`, `PROGRESS = "progress.jsonl"`; `net_name(seat, t) -> str` (`p{seat}-t{t:05d}.pt`), `mark_name(t) -> str` (`t{t:05d}`).
  - `run_deep(out: Path, config: DeepConfig, *, target: int, learner: Learner | None = None, deadline_s: float | None = None, commit: Callable[[], None] = lambda: None, log: Callable[[str], None] = print, observers: Sequence[Observer] = ()) -> DeepRun`. `learner=None` builds `NetLearner(config)` (Task 5). Steps: open (`load_run(out/RESUME)` + `require_same_run`, else `new_run`); return at once if `run.t >= target`; under `_stop_on_signals()` and one `Collector(seed=config.seed, workers=config.workers)`, loop: `stats = step(...)`; `why = "done" | stop.why | "deadline" | ""`; `kept = run.t == next mark`; save both nets to `out/nets/`; if kept `save_run(out/marks/mark_name(t))`; if kept or `save_every_s` elapsed or `why`: `save_run(out/resume)`; append the progress line; `commit()`; stop if `why`.
  - Progress line keys: `time, t, target, seconds: {materialise: [s0, s1], collect: [..], merge: [..], fit: [..], iteration}, rows: {advantage: [offered0, offered1], policy: offered, resident: {advantage: [..], policy: ..}}, loss: [l0, l1], iterations_per_s, eta_h, wrote: [names]`, plus `stopped` on the last line.
  - `_stop_on_signals()` and `_StopRequest`, copied from `lockstep_run.py`.

- [ ] **Step 1: Write the failing tests** (all with `OracleLearner()`, `DeepConfig(seed=5, walks=5, workers=0, capacity=100_000, grid=(4, 8))`, `log=lambda _: None`, `target=10`):

```python
def test_a_run_keeps_its_grid_its_nets_and_its_log_and_ends_at_its_target(tmp_path):
    commits = []; run = run_deep(tmp_path, CFG, target=10, learner=OracleLearner(), commit=lambda: commits.append(1), log=...)
    assert run.t == 10
    for t in (4, 8): assert load_run(tmp_path / MARKS / mark_name(t), learner=OracleLearner()).t == t
    assert sorted(p.name for p in (tmp_path / NETS).iterdir()) == [net_name(s, t) for t in range(1, 11) for s in (0, 1)] sorted
    assert load_run(tmp_path / RESUME, learner=OracleLearner()).t == 10
    lines = [json.loads(l) for l in (tmp_path / PROGRESS).read_text().splitlines()]
    assert [l["t"] for l in lines] == list(range(1, 11)) and lines[-1]["stopped"] == "done"
    assert all(len(l["seconds"]["materialise"]) == 2 and l["seconds"]["iteration"] > 0 for l in lines)
    assert len(commits) == 10 and not list(tmp_path.rglob("*.tmp"))
    assert same_run(load_run(tmp_path / MARKS / mark_name(8), ...), straight(CFG, 8))

def test_a_run_stopped_and_restarted_equals_one_straight_run(tmp_path):
    run_deep(..., target=6); run = run_deep(..., target=10); same_run(run, straight(CFG, 10)); run_deep(..., target=10) again writes no new line

def test_sigterm_finishes_the_iteration_saves_and_resumes_exactly(tmp_path):
    # log hook raises SIGTERM on the first "{" line, as rung 3's test; last line stopped == "SIGTERM"; resume at run.t;
    # run_deep again to 10 -> same_run(straight(CFG, 10))

def test_a_deadline_stops_the_run_with_its_state_saved(tmp_path): deadline_s=0.0 -> 0 < run.t < 10; resume t == run.t; last line stopped == "deadline"

def test_a_resume_refuses_a_changed_config_by_name(tmp_path):
    run_deep(..., target=2); with pytest.raises(ValueError, match="walks"): run_deep(tmp_path, replace(CFG, walks=6), ...)
    run_deep(tmp_path, replace(CFG, workers=0, save_every_s=0.0), target=3, ...)   # allowed
```

- [ ] **Step 2: Run to verify they fail** — `ImportError: run_deep`.
- [ ] **Step 3: Implement** `_stop_on_signals`, `_StopRequest`, `run_deep`, `_open`, `_Saves` (dataclass: `out, target, commit, log, opened_at, opened_t, resume_at`), `_train`.
- [ ] **Step 4: Run to verify they pass** — the file still under ~90 s.
- [ ] **Step 5: Commit** — `git commit -m "feat(rung4): run_deep — the driver with grid marks, a resume every 30 minutes, a progress line per iteration, and a graceful stop"`

---

### Task 5: the real learner, the current iterate, and the torch tests

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/train.py` (sections "The real learner", "The current iterate")
- Test: `tests/deepcfr/test_train.py` (`current_iterate`, torch-free), `tests/deepcfr/test_train_torch.py` (new, `pytest.importorskip("torch")`)

**Interfaces:**
- Consumes: `fit.FitConfig`, `fit.fit_from_scratch`, `fit.batch_loss`, `materialise.materialise`, `net.save_net`, `net.load_net` — all imported INSIDE `NetLearner`'s methods; `strategy.FrozenStrategy`, `strategy.StrategyInfo`, `strategy.average_rows`, `strategy.uniform_strategy`; `exploitability.exploitability`.
- Produces:
  - `class NetLearner`: `__init__(self, config: DeepConfig)` builds `FitConfig(width, depth, steps, batch, lr, clip, threads)` lazily; `fit` → `fit_from_scratch(...)` then the held-batch loss (`torch.Generator().manual_seed(folded seed)`, `config.batch` indices with replacement, `batch_loss(...).item()` under `torch.inference_mode()`); `write` → `materialise(table, models)`; `save`/`load` → `save_net`/`load_net`.
  - `current_iterate(table: PackedTable, *, iteration: int = 0, seed: int = 0) -> FrozenStrategy`: `FrozenStrategy(average_rows(np.maximum(table.cumulative_regret, 0.0), table.widths()), table.widths(), StrategyInfo(rule="deep", column="current", iteration=iteration, seed=seed, workers=0))`.

- [ ] **Step 1: Write the failing tests**

In `test_train.py`:
```python
def test_current_iterate_is_regret_matching_row_by_row():
    # whole-game table; write at 60 sampled keys (both seats, every width): a positive mix, a one-hot, all negative,
    # all zero; frozen = current_iterate(table, iteration=7, seed=5); for each key
    # np.testing.assert_array_equal(frozen[key], table[key].strategy().astype(np.float32)); frozen.info == StrategyInfo("deep", "current", 7, 5, 0)
```
In `test_train_torch.py`:
```python
torch = pytest.importorskip("torch")
TINY = DeepConfig(seed=9, walks=20, workers=0, capacity=100_000, steps=5, batch=64, threads=2)

def test_net_learner_resume_equals_uninterrupted_bytewise(tmp_path):
    # straight: 3 steps. resumed: 2 steps, save_run, load_run(learner=NetLearner(TINY)), 1 step.
    # every state_dict tensor torch.equal; memories same; a write on two whole-game tables bytewise equal.

def test_saved_nets_load_back_and_reproduce_the_write(tmp_path):
    # run_deep(TINY, target=2); for each nets/p{s}-t00002.pt: load_net; materialise(table, loaded) equals materialise(table, run.nets)

def test_fit_reports_a_finite_loss(): Fitted.loss is a finite float after one step

@pytest.mark.skipif(not FULL_DEEP, reason="a few real iterations and one exact grade, ~10 min; set MINIDRAWMAHA_FULL_DEEP=1")
def test_gate_five_the_current_iterate_beats_uniform(tmp_path):
    # DeepConfig(seed=0, walks=1_000, workers=8, capacity=10_000_000, steps=1_000, threads=8); run_deep target=5;
    # table = PackedTable.whole_game(); NetLearner(cfg).write(table, run.nets); grade = exploitability(current_iterate(table, ...))
    # assert grade < 4.842; print the grade and the wall clock (for the PR body)
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `NetLearner`, `current_iterate`; make `run_deep(learner=None)` build `NetLearner(config)`.**
- [ ] **Step 4: Run** — `uv run pytest tests/deepcfr/test_train.py -v` (no torch) PASS; `uv run --extra deep pytest tests/deepcfr/test_train_torch.py -v` PASS; `MINIDRAWMAHA_FULL_DEEP=1 uv run --extra deep pytest tests/deepcfr/test_train_torch.py -k gate_five -s` → record the grade and the wall clock.
- [ ] **Step 5: Commit** — `git commit -m "feat(rung4): NetLearner over fit_from_scratch and materialise, the current iterate frozen for grading, and gate 5"`

---

### Task 6: the CLI, the package docstring, the spec note, timings

**Files:**
- Create: `scripts/run_deep_cfr.py`
- Modify: `src/drawmaha_solver/deepcfr/__init__.py` (add `train` to the module list), `docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md` (gate 1: the float32 note and the two pins; PR table: `current_iterate` lands in PR 4, 2026-10-20)
- Test: `tests/deepcfr/test_train.py` (`test_cli_parses_the_flags_into_a_config` via `scripts.run_deep_cfr.config_from_args`)

**Interfaces:**
- Produces: `scripts/run_deep_cfr.py` with `parse(argv) -> argparse.Namespace`, `config_from_args(args) -> DeepConfig`, `main()`; flags `--out` (required) `--target` (2000) `--walks --workers --capacity --width --depth --steps --batch --threads --seed`, defaults from `DeepConfig()`. Re-running resumes.

- [ ] **Step 1: Write the failing test** — `config_from_args(parse(["--out", "x", "--walks", "7", "--width", "32"])) == DeepConfig(walks=7, width=32)`.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement the script** in the style of `scripts/run_lockstep.py`, docstring with the smoke example `caffeinate -i uv run --extra deep python scripts/run_deep_cfr.py --out runs/deep-smoke --target 3 --walks 1000`.
- [ ] **Step 4: Run the suites** — `uv run pytest -q` (no torch; `test_shared_chrome.py` is known-failing on main since #54 and is not this PR's) and `uv run --extra deep pytest tests/deepcfr -q`.
- [ ] **Step 5: Measure** — `scripts/run_deep_cfr.py --target 2 --walks 1000` and `--walks 10000` on this Mac at W = 8; read the stage seconds from `progress.jsonl`; put them in the module docstring and the PR body beside the spec's ~26 s on 33 Modal cores.
- [ ] **Step 6: Commit** — `git commit -m "feat(rung4): run_deep_cfr.py, the local runner; the package and the spec name the loop"`
