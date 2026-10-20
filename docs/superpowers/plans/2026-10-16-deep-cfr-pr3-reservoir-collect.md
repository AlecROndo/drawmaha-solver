# Deep CFR M1, PR 3: the memories and the walk collector — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the data side of Deep CFR. `reservoir.py` is Vitter's Algorithm R over 24-byte (row, t, target) rows, with a counter-based stream per phase and a checkpoint that resumes exactly. `collect.py` runs K lockstep walks for one seat on W processes over a shared, materialised table and returns the phase's advantage rows and policy rows. The result is a function of (seed, t, seat, k), never of W.

**Architecture:** both modules are NumPy only; no torch anywhere in this PR. `Reservoir` owns three preallocated arrays (rows int32, t int32, targets float32 × 4) and `merge` is vectorised Algorithm R. `Collector` is a context manager that holds the run's whole-game `PackedTable`, with `cumulative_regret` in a shared-memory block, plus a spawn-context process pool whose workers attach to that block. `collect(seat=, t=, walks=)` hands contiguous chunks of walk indices to the pool and concatenates the chunks in k order. The walk is rung 3's `lockstep.walk`, unchanged, asked for the `uniform` extra column so each opponent visit's `extra[0]` is σ exactly. The plan's output plugs into PR 2's `fit_from_scratch(rows, t, targets, …)` through `Reservoir.contents()`.

**Tech Stack:** Python 3.12+, NumPy, `multiprocessing` (spawn) + `concurrent.futures.ProcessPoolExecutor` + `multiprocessing.shared_memory`.

**Spec:** `docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`, sections "What is reused unchanged" (first bullet), "Memories", "Collecting", correctness gate 2, and PR 3 of the PR table.

## Global Constraints

- A row is "`(row: int32, t: int32, target: float32 x 4)`, the target padded to the widest ledger. Advantage memory per seat, policy memory shared. Vitter's Algorithm R with a counter-based stream per phase, so a resumed run merges the same way. The count of rows ever offered, `n`, is part of the checkpoint. Capacity C = 50M rows per memory."
- The advantage target is the walk's banked regret `r = u − ⟨σ, u⟩` per traverser visit. The policy target is σ per opponent visit, read from the `uniform` extra column ("`extra[0] = sigma` exactly, so sigma is never recovered by dividing by t").
- "The sample is a function of (seed, t, seat, k), never of W." Walk k of a phase is `lockstep.walk(table, deal, seed, t, seat, k, (Average.UNIFORM,))`, unchanged.
- `lockstep.py`, `mccfr.traverse`, `parallel.py`, `PackedTable`, `features.py` are merged code: import from them, change nothing in them.
- No torch in `reservoir.py` or `collect.py`, so their tests run in the DEFAULT suite (no `importorskip`).
- Repo conventions: long explanatory module docstring, 75-hyphen section banners, one blank line between top-level defs, WHY-only comments, `from __future__ import annotations`, keyword-only arguments where ints could be transposed, atomic saves (temp file + fsync + `os.replace`, as `lockstep.save_lockstep`).

## Decisions this plan makes that the spec leaves open

- **Spawn everywhere, not fork on Linux.** The spec allows fork on Modal. Spawn plus one shared block for `cumulative_regret` behaves the same on the Mac and on Modal, and avoids forking a parent that may hold torch threads (PR 4's loop imports torch). Each worker rebuilds the whole-game index (about a second) and lays it over the shared block. Its own `strategy_sum` / `extra_sums` / `stamp` are untouched lazy zeros, because `_Recorder` writes only into scratch.
- **Rows come from the slot, not from `rows_of_columns`.** Each `Recorded` visit already carries `slot = (row, start, width)`, so the policy row is `slot[0]`. That gives the same answer as the spec's searchsorted route, with one less moving part.
- **Padding is 0.0.** PR 2's loss ignores padding. Zeros keep `Collected` comparable with `np.array_equal`; NaN never compares equal.
- **Duplicate slots inside one merge: the later offer wins**, which is what sequential Algorithm R does. It's done explicitly (last occurrence per slot via `np.unique` on the reversed slot array), not by relying on NumPy's unspecified repeated-index assignment.
- **The reservoir stream** is `np.random.default_rng([seed, t, seat, memory_id, 0x52455356])`. The `memory_id` is 0 for advantage and 1 for policy; the last word is the ASCII constant "RESV". Five entropy words can never equal a walk's four-word `[seed, t, seat, k]` stream.
- **Chunking:** `ceil(walks / (4 · W))` walks per task, so a slow chunk can't stall the phase. Chunks are concatenated in k order.

## Review Focus

1. A run resumed from a reservoir checkpoint must merge the next phase exactly as the uninterrupted run would: pinned in Task 1 (`test_a_resumed_reservoir_merges_identically`).
2. Changing W, or the chunk size, must not change a single byte of `Collected`: pinned in Task 2 (`test_collect_is_the_same_on_one_and_three_workers`).
3. The collector must read the regrets as they are AT `collect` time: a parent that materialises new numbers between two collects must see the workers walk the new σ, not a stale copy. Pinned in Task 2 (`test_workers_see_regrets_written_after_start`).
4. A worker that crashes, or a `walks < 1` / bad seat, must raise in the parent with the reason, never hang or return a short phase: pinned in Task 2 (`test_collect_refuses_bad_arguments`, `test_a_failing_deal_raises_in_the_parent`).
5. Leaving the `with` block, also by an exception, must unlink the shared block and stop the pool. No `/dev/shm` or `resource_tracker` leak: pinned in Task 2 (`test_close_unlinks_the_block_even_after_an_error`).

---

### Task 1: the reservoir

**Files:**
- Create: `src/drawmaha_solver/deepcfr/reservoir.py`
- Create `src/drawmaha_solver/deepcfr/__init__.py` ONLY if PR 2 has not merged. If it has, leave PR 2's file as it is.
- Test: `tests/deepcfr/test_reservoir.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `TARGET_WIDTH = 4` (equals `features.MAX_WIDTH`; asserted equal in the test, not imported, so `reservoir.py` doesn't depend on the encoder).
  - `class Memory(IntEnum)`: `ADVANTAGE = 0`, `POLICY = 1`.
  - `reservoir_stream(*, seed: int, t: int, seat: int, memory: Memory) -> np.random.Generator`.
  - `class Reservoir`: `__init__(self, capacity: int)`; properties `capacity`, `offered` (n, ever offered), `size` (= min(n, capacity)); `merge(self, rows: np.ndarray, targets: np.ndarray, *, t: int, rng) -> None`, where `rng` is anything with numpy `Generator.integers` semantics; `contents(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]` gives read-only views `(rows[:size], t[:size], targets[:size])`; `save(self, path: Path) -> None`; `@classmethod load(cls, path: Path) -> Reservoir`.

- [ ] **Step 1: Write the failing tests**

```python
def test_below_capacity_everything_is_kept_in_order():
    memory = Reservoir(capacity=10)
    memory.merge(np.arange(6), targets(6), t=1, rng=np.random.default_rng(0))
    rows, t, _ = memory.contents()
    assert memory.offered == 6 and memory.size == 6
    np.testing.assert_array_equal(rows, np.arange(6)); assert (t == 1).all()

def test_never_exceeds_capacity_and_counts_every_offer():
    memory = Reservoir(capacity=10)
    for t in range(1, 6):
        memory.merge(np.arange(7) + 100 * t, targets(7), t=t, rng=np.random.default_rng(t))
    assert memory.offered == 35 and memory.size == 10
    rows, t_col, _ = memory.contents()
    assert set(rows.tolist()) <= {r + 100 * t for t in range(1, 6) for r in range(7)}
    np.testing.assert_array_equal(t_col, rows // 100)          # each row kept its own t

def test_every_offer_is_resident_with_probability_c_over_n():
    # 4,000 trials, C = 10, 100 offers in four merges of 25. Residency count per item
    # must pass chi-square against 400 each: df = 99, the 0.999 critical value is 148.23.
    ...
    assert chi_square < 148.23

def test_the_later_offer_wins_a_shared_slot():
    memory = Reservoir(capacity=2)
    memory.merge(np.array([0, 1]), targets(2), t=1, rng=np.random.default_rng(0))
    memory.merge(np.array([7, 8]), targets(2), t=2, rng=FixedDraws([1, 1]))   # both land in slot 1
    assert memory.contents()[0].tolist() == [0, 8]

def test_a_resumed_reservoir_merges_identically(tmp_path):
    straight, resumed = Reservoir(capacity=50), None
    for t in (1, 2): straight.merge(offers(t), targets(40), t=t, rng=reservoir_stream(seed=3, t=t, seat=0, memory=Memory.ADVANTAGE))
    straight.save(tmp_path / "m.npz"); resumed = Reservoir.load(tmp_path / "m.npz")
    for memory in (straight, resumed):
        memory.merge(offers(3), targets(40), t=3, rng=reservoir_stream(seed=3, t=3, seat=0, memory=Memory.ADVANTAGE))
    for a, b in zip(straight.contents(), resumed.contents()): np.testing.assert_array_equal(a, b)
    assert straight.offered == resumed.offered

def test_streams_differ_by_memory_and_never_equal_a_walk_stream():
    a = reservoir_stream(seed=1, t=2, seat=0, memory=Memory.ADVANTAGE).integers(0, 2**62, 4)
    b = reservoir_stream(seed=1, t=2, seat=0, memory=Memory.POLICY).integers(0, 2**62, 4)
    w = lockstep.stream(1, 2, 0, 0).integers(0, 2**62, 4)
    assert not np.array_equal(a, b) and not np.array_equal(a, w)

def test_merge_refuses_bad_rows():
    memory = Reservoir(capacity=4)
    with pytest.raises(ValueError, match="same length"):  memory.merge(np.arange(3), targets(2), t=1, rng=rng0)
    with pytest.raises(ValueError, match="4"):            memory.merge(np.arange(2), np.zeros((2, 3)), t=1, rng=rng0)
    with pytest.raises(ValueError, match="t"):            memory.merge(np.arange(2), targets(2), t=0, rng=rng0)
    with pytest.raises(ValueError, match="capacity"):     Reservoir(capacity=0)

def test_contents_are_read_only_and_feed_the_fit():
    rows, t, tg = filled_memory().contents()
    assert rows.dtype == np.int32 and t.dtype == np.int32 and tg.dtype == np.float32 and tg.shape[1] == 4
    with pytest.raises(ValueError): rows[0] = 1

def test_save_is_atomic_and_load_refuses_a_foreign_file(tmp_path):
    # save leaves no .tmp behind; load of a lockstep checkpoint (no "offered" key) raises ValueError naming the file.
    ...

def test_target_width_matches_the_encoder():
    from drawmaha_solver.minidrawmaha.features import MAX_WIDTH
    assert TARGET_WIDTH == MAX_WIDTH
```

Helpers: `targets(n)` = `np.zeros((n, 4), np.float32)` with column 0 = arange. `FixedDraws(values)` is a stub whose `integers(low, high, size=None)` returns the given values. `offers(t)` = 40 distinct row numbers per t.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/deepcfr/test_reservoir.py -v`
Expected: FAIL with `ModuleNotFoundError: drawmaha_solver.deepcfr.reservoir`

- [ ] **Step 3: Implement `reservoir.py`**

`merge`, vectorised Algorithm R:

```
g = offered + arange(m)                      # each offer's global index
slot = where(g < C, g, -1)
late = g >= C
draw = rng.integers(0, g[late] + 1)          # one draw per late offer, inclusive of g
slot[late] = where(draw < C, draw, -1)
keep the LAST occurrence of each slot >= 0   # np.unique(slot[::-1], return_index=True)
write rows/t/targets at those slots; offered += m
```

The three arrays are allocated `np.zeros` at construction; the kernel pages them lazily, so a 50M memory costs nothing until filled. `save` writes `rows[:size]`, `t[:size]`, `targets[:size]`, `offered`, `capacity`. `load` refuses a file without `offered`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/deepcfr/test_reservoir.py -v` → all PASS in under 10 s.

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr/reservoir.py tests/deepcfr/test_reservoir.py
git commit -m "feat(rung4): the reservoir — Algorithm R over 24-byte rows, a stream per phase, an exact resume"
```

---

### Task 2: the collector

**Files:**
- Create: `src/drawmaha_solver/deepcfr/collect.py`
- Test: `tests/deepcfr/test_collect.py`

**Interfaces:**
- Consumes: `lockstep.walk(table, deal, seed, t, seat, worker, averages) -> Recorded` (`.regret`: list of `(key, slot, r)`; `.strategy`: list of `(key, slot, t·σ, extra)` with `extra` shaped (1, width) under `(Average.UNIFORM,)`); `PackedTable.whole_game(extra_averages=1)`, `.over(...)`; `game.random_deal`; `TARGET_WIDTH` (Task 1).
- Produces:
  - `@dataclass(frozen=True) Collected`: `advantage_rows: int32 (a,)`, `advantage_targets: float32 (a, 4)`, `policy_rows: int32 (p,)`, `policy_targets: float32 (p, 4)`, `walks: int`.
  - `rows_of(recorded: Recorded) -> Collected` (with `walks=1`): one walk's record as rows, in visit order. Public so tests and PR 4's exactness pin can compare against it.
  - `class Collector` (context manager): `__init__(self, *, seed: int, workers: int, deal: RootSampler = random_deal)`; `.table: PackedTable`, the whole-game table whose `cumulative_regret` lives in the shared block (PR 4 materialises into it); `.collect(self, *, seat: int, t: int, walks: int) -> Collected`. `workers == 0` means walk in this process, with no pool and no shared block: the reference.

- [ ] **Step 1: Write the failing tests**

```python
def test_rows_of_one_walk_are_its_visits():
    table = PackedTable.whole_game(extra_averages=1)
    recorded = lockstep.walk(table, random_deal, 5, 1, 0, 0, (Average.UNIFORM,))
    got = rows_of(recorded)
    assert got.advantage_rows.tolist() == [slot[0] for _, slot, _ in recorded.regret]
    for i, (_, (_, _, width), r) in enumerate(recorded.regret):
        np.testing.assert_array_equal(got.advantage_targets[i, :width], r.astype(np.float32))
        assert not got.advantage_targets[i, width:].any()
    for i, (_, (_, _, width), _, extra) in enumerate(recorded.strategy):
        np.testing.assert_allclose(got.policy_targets[i, :width].sum(), 1.0, rtol=1e-6)   # σ, a distribution

def test_advantage_rows_are_the_seats_and_policy_rows_the_opponents(collected_seat_1):
    points = features.layout().locate_rows(collected_seat_1.advantage_rows)[0]
    assert all(features.layout().points[p].player == 1 for p in points)
    points = features.layout().locate_rows(collected_seat_1.policy_rows)[0]
    assert all(features.layout().points[p].player == 0 for p in points)

def test_collect_in_process_equals_walking_k_by_k():
    with Collector(seed=11, workers=0) as run:
        got = run.collect(seat=0, t=3, walks=20)
        expect = concat([rows_of(lockstep.walk(run.table, random_deal, 11, 3, 0, k, (Average.UNIFORM,))) for k in range(20)])
    assert_collected_equal(got, expect) and got.walks == 20

def test_collect_is_the_same_on_one_and_three_workers():
    results = []
    for workers in (1, 3):
        with Collector(seed=11, workers=workers) as run:
            write_some_regrets(run.table)                    # a fixed non-uniform σ, same in both
            results.append(run.collect(seat=1, t=2, walks=37))   # 37: chunks don't divide evenly
    assert_collected_equal(*results)

def test_workers_see_regrets_written_after_start():
    with Collector(seed=4, workers=2) as run:
        before = run.collect(seat=0, t=1, walks=10)
        write_some_regrets(run.table)
        after = run.collect(seat=0, t=1, walks=10)
    with Collector(seed=4, workers=0) as reference:
        write_some_regrets(reference.table)
        assert_collected_equal(after, reference.collect(seat=0, t=1, walks=10))
    assert not collected_equal(before, after)

def test_collect_refuses_bad_arguments():
    with Collector(seed=1, workers=0) as run:
        with pytest.raises(ValueError, match="walks"): run.collect(seat=0, t=1, walks=0)
        with pytest.raises(ValueError, match="seat"):  run.collect(seat=2, t=1, walks=1)
        with pytest.raises(ValueError, match="t"):     run.collect(seat=0, t=0, walks=1)
    with pytest.raises(ValueError, match="workers"):   Collector(seed=1, workers=-1)

def test_a_failing_deal_raises_in_the_parent():
    with Collector(seed=1, workers=2, deal=exploding_deal) as run:   # module-level, picklable
        with pytest.raises(RuntimeError, match="boom"):
            run.collect(seat=0, t=1, walks=4)

def test_close_unlinks_the_block_even_after_an_error():
    with pytest.raises(KeyError):
        with Collector(seed=1, workers=2) as run:
            name = run._block.name
            raise KeyError("caller failed")
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)
```

Helpers: `concat(list[Collected]) -> Collected`; `assert_collected_equal` / `collected_equal` compare all four arrays bytewise plus `walks`; `write_some_regrets(table)` writes `np.random.default_rng(0).standard_normal(size)` into `table.cumulative_regret[:]`; `exploding_deal(rng)` raises `RuntimeError("boom")`. Pool tests use W ≤ 3 and ≤ 40 walks. Each spawns workers, so the file runs in tens of seconds, not minutes.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/deepcfr/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError: drawmaha_solver.deepcfr.collect`

- [ ] **Step 3: Implement `collect.py`**

- `Collector.__enter__` with `workers ≥ 1`:
  - Allocate `PackedTable.whole_game(extra_averages=1)`.
  - Create one `SharedMemory` the size of its `cumulative_regret`.
  - Set `self.table = table.over(cumulative_regret=<view on the block>, strategy_sum=…, extra_sums=…, stamp=…)`; the other three are the table's own arrays.
  - Start `ProcessPoolExecutor(max_workers=W, mp_context=get_context("spawn"), initializer=_attach, initargs=(block name, seed, deal))`. `_attach` builds the worker's table over the block once and keeps it in a module global.
- `collect`:
  - Validate the arguments.
  - Split `range(walks)` into chunks of `ceil(walks / (4·W))`.
  - `pool.map(_walk_chunk, chunks)`, which returns the chunks in submission order.
  - Each chunk returns the `concat` of its walks' `rows_of`. The parent `concat`s the chunks.
  - A worker exception re-raises through `map` with its message. `BrokenProcessPool` is re-raised as `RuntimeError("a collector worker died")`.
- `workers == 0`: walk the chunks in-process over a private table, with no pool and no block.
- `__exit__`: `pool.shutdown(cancel_futures=True)`, drop the views, `close()` and `unlink()` the block (always, in a `finally`).

- [ ] **Step 4: Run to verify they pass; time a real phase**

Run: `uv run pytest tests/deepcfr/test_collect.py -v` → all PASS.
Run: `uv run python -c "…with Collector(seed=0, workers=8) as run: time run.collect(seat=0, t=1, walks=10_000)…"`. Record the seconds and the row counts in the module docstring. The spec expects about 14 advantage rows and 22 policy rows per walk, so about 140k advantage rows per phase, and about 2.72 ms a walk on one Mac core.

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr/collect.py tests/deepcfr/test_collect.py
git commit -m "feat(rung4): the collector — K lockstep walks on W spawned workers over one shared regret block"
```

---

### Task 3: spec note and the full suite

**Files:**
- Modify: `docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`, "Collecting". Replace "Processes fork on Linux (Modal) … on the Mac they go through the same shared-memory blocks" with the spawn-everywhere decision and its reason, and say that the policy row is the visit's `slot[0]`.

- [ ] **Step 1: Run the default suite (no torch).** `uv run pytest -q` passes, and `tests/deepcfr/test_reservoir.py` + `test_collect.py` RUN (they need no torch).
- [ ] **Step 2: Check for leaked shared blocks.** Run the collect tests under `-W error::UserWarning`; the `resource_tracker` "leaked shared_memory" warning must not appear.
- [ ] **Step 3: Commit.** `git commit -am "docs(rung4): the collector spawns everywhere and reads the policy row from the slot"`
