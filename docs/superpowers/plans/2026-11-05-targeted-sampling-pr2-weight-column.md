# Targeted sampling, PR 2: the weight column — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every memory row carries a float32 `weight`; the refit's loss and the oracle weigh a row by `t × weight`; weight ≡ 1 reproduces today's run bit for bit, and a memory file written before the column existed loads with weights of one.

**Architecture:** the column is a fourth array in `Reservoir`, riding the same slot as its row through Algorithm R's eviction, save and load; `contents()` grows from three arrays to four and that tuple flows unchanged through `step` into `Learner.fit`. The loss multiplies `t` by `weight` before the batch-mean normalisation; the oracle multiplies each sample by its weight before LCFR's per-t′ bank. Nothing consumes weights other than one yet: `collect.rows_of` keeps producing unweighted rows, `merge` without a weight stamps ones, and PR 4's boost stage will be the first caller to pass the column.

**Tech Stack:** Python 3.12+, NumPy; torch only inside `fit.py`, `policy.py` bodies and the torch-gated tests.

**Spec:** the targeted-sampling plan, outside the repo, `/Users/alec/Desktop/Claude/poker/targeted-sampling-plan/targeted-sampling-plan.md` — §3 (why an exact row's weight is `(1 − α) · K · arrival · t` and why the expected weight must be unchanged) and the appendix sections "PR 2 — weight column" (the contract) and "PR 4 — the boost stage" (the consumer).

## Global Constraints

- The oracle pin `tests/deepcfr/test_train.py::test_the_oracle_loop_is_lcfr_bit_for_bit` passes unchanged. The oracle banks `Σ t′ · (w · r)` per t′ in increasing order with `np.add.at` in memory order; with w ≡ 1 the multiplication is exact and the bits do not move.
- The loss is `(t · w) / mean(t · w)` per row; with w ≡ 1 this is `t / mean(t)` bit for bit (`t · 1.0 = t` in float32). The weight is a sample weight, never a divisor of the target: `pot_scaled` composes with it untouched.
- `weight ≥ 0` and finite; a row of weight 0 is inert in the loss and in the oracle's bank; a weight of 2 equals the same row offered twice (oracle). Rows stay in offer order.
- A reservoir file with no `weight` key loads with ones (a saved run resumes as the run it was); a file with a `weight` of the wrong length or dtype is refused by name.
- No new `DeepConfig` field: nothing to add to `ADDED_LATER` or `SAMPLE_FIELDS`. The resume state's file set does not change; the weight lives inside each memory's `.npz`.
- torch stays optional; the default suite passes without it. `uv run pytest` is the gate (CI runs no tests).
- Every prose claim that says a row is 24 bytes, a 50M-row memory is 1.2 GB, or the row is `(row, t, target)` is corrected in the same change: `reservoir.py`, `fit.py`, `train.py`, `policy.py`, `deepcfr/__init__.py`, `deepcfr/AGENTS.md`, `scripts/deepcfr/modal_rung4.py:155`, `tests/deepcfr/test_reservoir.py`.
- Commit subjects `feat(rung4): <the finding>`; branch from PR 1's head (`23cd464`), never rebased onto main. No `git stash`, no Modal.

## Review Focus

1. A minibatch whose every row has weight 0 (small memory, α = 0 in PR 4): the loss must be finite, not 0/0 = NaN that ends the fit as "diverged" — pinned in Task 2 (`test_a_batch_of_weightless_rows_has_a_finite_loss`).
2. A memory whose every row has weight 0: refused by name before the fit, like the empty memory — Task 2 (`test_fit_refuses_bad_weights`).
3. A weight landing in the wrong slot during eviction (the column assigned with a different index than the row's) — Task 1 (`test_a_weight_rides_its_row_through_eviction`).
4. A `weight` array saved as float64 or of the wrong length being poured into the float32 column in silence — Task 1 (`test_load_refuses_a_weight_of_the_wrong_length_or_type`).
5. The grader's Π fit and the sweep script unpacking three arrays from a `contents()` that now returns four — Task 2 (`fit_policy` takes the fourth positionally; `pi_loss_sweep.py` unpacks four).

---

### Task 1: the column in the reservoir

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/reservoir.py` (module docstring, `_FILE_KEYS`, `Reservoir.__init__`, `merge`, `contents`, `save`, `load`, `_validate_offer`)
- Modify: `src/drawmaha_solver/deepcfr/AGENTS.md:72-73`, `src/drawmaha_solver/deepcfr/__init__.py:16-18`, `scripts/deepcfr/modal_rung4.py:155`
- Test: `tests/deepcfr/test_reservoir.py`

**Interfaces:**
- Produces: `Reservoir.merge(rows, targets, *, t, rng, weight=None)` — `weight` (n,) float, finite, ≥ 0; `None` stamps every row 1.0. `Reservoir.contents() -> (rows int32, t int32, targets float32 (n, 4), weight float32 (n,))`, read-only views. `save` writes a `weight` key; `load` reads it when present, else ones; `_WEIGHT_KEY = "weight"` beside `_FILE_KEYS`, which stays the required set.

- [ ] **Step 1: Write the failing tests** in `tests/deepcfr/test_reservoir.py`

```python
def test_rows_offered_without_a_weight_weigh_one():
    rows, t, kept, weight = filled_memory().contents()
    assert weight.dtype == np.float32 and weight.tolist() == [1.0] * 5

def test_a_weight_rides_its_row_through_eviction():
    memory = Reservoir(capacity=5)
    memory.merge(np.arange(3), targets(3), t=1, rng=np.random.default_rng(0), weight=np.array([0.5, 1.0, 1.5]))
    memory.merge(np.arange(4) + 10, targets(4), t=2, rng=FixedDraws([0, 7]), weight=np.array([2.0, 3.0, 4.0, 5.0]))
    rows, _, _, weight = memory.contents()
    assert rows.tolist() == [12, 1, 2, 10, 11]
    assert weight.tolist() == [4.0, 1.0, 1.5, 2.0, 3.0]

def test_merge_refuses_bad_weights():
    # negative, NaN, wrong length, 2-D: each a ValueError matching "weight"

def test_a_file_written_before_the_weight_existed_loads_with_ones(tmp_path):
    # np.savez with rows/t/targets/offered/capacity only → load → weight == ones(size), and the next merge still works

def test_load_refuses_a_weight_of_the_wrong_length_or_type(tmp_path):
    # weight of length size − 1 → ValueError matching the file name; weight float64 → "weight must be float32"
```

Extend `test_contents_are_read_only_and_feed_the_fit` (four views; `weight` float32 `(5,)`, read-only) and `test_never_exceeds_capacity_and_counts_every_offer` (weights given as `rows % 100` so the fourth column moves with the other three). The round trip in `test_save_is_atomic_and_load_refuses_a_foreign_file` already zips `contents()` strictly.

- [ ] **Step 2: Run them to make sure they fail**: `uv run pytest tests/deepcfr/test_reservoir.py -q -p no:cacheprovider` — the new tests fail on `TypeError` (unexpected `weight`) and on unpacking three values into four.

- [ ] **Step 3: Implement** — a fourth `np.zeros(capacity, float32)` array; `_validate_offer` grows a `weight` parameter and returns it as float32 (ones when `None`); `merge` assigns `self._weight[slot[winners]] = weight[winners]`; `contents` returns four views; `save` writes `weight=`; `load` reads `saved[_WEIGHT_KEY]` when present (shape `(size,)`, dtype float32, both refused by name) else `np.ones(size, float32)`. Docstrings: the row is `(row, t, target, weight)`, 28 bytes, a 50M-row memory 1.4 GB; the weight's meaning (how many sampled rows this row stands for; 1 for every row the walk offers, PR 4's exact rows carry `(1 − α) · K · arrival`); why a missing key reads as one.

- [ ] **Step 4: Run the reservoir tests and the default rung-4 suite**: `uv run pytest tests/deepcfr -q -p no:cacheprovider` — `test_train.py` fails at `Spy.fit` and `test_fit_receives_the_memory_as_it_stands_after_the_merge` (four arrays into a three-way zip) until Task 3; everything else passes.

- [ ] **Step 5: Commit** `feat(rung4): every memory row carries a float32 weight, one unless a caller says otherwise; a file without the column loads as ones`

### Task 2: the loss weighs a row by t × weight

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/fit.py` (module docstring, `batch_loss`, `_weighted_loss`, `fit_from_scratch`, `_fit_on_device`)
- Modify: `src/drawmaha_solver/deepcfr/policy.py` (`fit_policy`, its docstring at line 24), `scripts/deepcfr/pi_loss_sweep.py:225`
- Test: `tests/deepcfr/test_fit.py` (torch-gated)

**Interfaces:**
- Consumes: `Reservoir.contents()` four-tuple (Task 1).
- Produces: `batch_loss(net, rows, t, targets, weight=None) -> Tensor`; `_weighted_loss(outputs, gather, targets, t, weight) -> Tensor` with `scale = t * weight; (scale / scale.mean().clamp(min=tiny) * error).mean()`; `fit_from_scratch(rows, t, targets, weight=None, *, seed, config)` refusing a `weight` that is not 1-D, not the memory's length, non-finite, negative, or with no positive entry ("no row carries weight: nothing to fit"); `_fit_on_device(rows, t, targets, weight, folded, config, *, device)`; `fit_policy(rows, t, targets, weight=None, *, seed, iteration, config)`.

- [ ] **Step 1: Write the failing tests** in `tests/deepcfr/test_fit.py`

```python
def test_weight_one_is_the_unweighted_fit_bit_for_bit():
    rows, t, targets = real_rows(300, seed=4)
    plain = fit_from_scratch(rows, t, targets, seed=(0, 1, 0), config=SMALL)
    ones = fit_from_scratch(rows, t, targets, np.ones(300, np.float32), seed=(0, 1, 0), config=SMALL)
    for a, b in zip(plain.parameters(), ones.parameters(), strict=True):
        assert torch.equal(a, b)

def test_a_row_of_weight_zero_is_inert():
    # one row twice: +1 at weight 0, −1 at weight 1, both t = 1 → prediction −1 within 0.05

def test_the_loss_weighs_a_row_by_t_times_weight():
    # one row twice: +1 at (t = 1, w = 3) and −1 at (t = 3, w = 1): (3·1 − 3·1)/6 = 0; swap to w = (1, 1): (1 − 3)/4 = −0.5

def test_a_batch_of_weightless_rows_has_a_finite_loss():
    # batch_loss with weight = zeros → 0.0, finite, backward finite

def test_fit_refuses_bad_weights():
    # negative → "weight"; NaN → "finite"; length n − 1 → "length"; all zero → "nothing to fit"; 2-D → "1-D"
```

Pass `weight=np.ones(...)` through `test_the_device_loop_on_the_cpu_is_the_cpu_fit_bit_for_bit`'s `_fit_on_device` call.

- [ ] **Step 2: Run to make sure they fail** — torch is absent on this Mac, so run under the `deep` extra once: `uv run --extra deep pytest tests/deepcfr/test_fit.py -q -p no:cacheprovider`. Expected: the new tests fail on `TypeError`.

- [ ] **Step 3: Implement** the signatures above. In `fit_from_scratch`, `weight = np.ones(len(rows), np.float32)` when `None`, else `np.asarray(weight, dtype=np.float32)` validated after `t`. In `NetLearner` (Task 3) and the cpu loop, index `weight[picked]` beside `t[picked]`. Docstring: the loss paragraph says `t · weight`; the device paragraph counts the weight in the copy.

- [ ] **Step 4: Run** `uv run --extra deep pytest tests/deepcfr/test_fit.py tests/deepcfr/test_policy.py -q -p no:cacheprovider` → PASS; then `uv sync` to drop torch again and `uv run pytest tests/deepcfr/test_fit.py -q` → skipped.

- [ ] **Step 5: Commit** `feat(rung4): the refit weighs a row by t × weight; weight one is today's loss bit for bit and a row of weight zero is inert`

### Task 3: the learner seam and the oracle

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/train.py` (module docstring lines 15, 36–37, 55–65; `Learner.fit`; `step`'s docstring; `NetLearner.fit`; `OracleModel`; `OracleLearner.fit/write/save/load`)
- Modify: `src/drawmaha_solver/deepcfr/AGENTS.md` (the oracle-pin rule names the weight)
- Test: `tests/deepcfr/test_train.py`, `tests/deepcfr/test_train_torch.py`

**Interfaces:**
- Consumes: Task 1's `contents()`, Task 2's `fit_from_scratch` and `batch_loss`.
- Produces: `Learner.fit(rows, t, targets, weight, *, seed) -> Fitted` (weight required at the seam, so no learner can ignore the column by accident); `OracleModel(rows, t, targets, weight)` with `weight: float32 (n,)`; `OracleLearner.write` banks `summed[touched] += int(t) * totals` where `totals` sums `values * repeat(weight, width)` in float64.

- [ ] **Step 1: Write the failing tests** in `tests/deepcfr/test_train.py`

```python
def test_the_oracle_weighs_a_row_by_its_weight_and_a_weight_of_two_is_the_row_twice():
    # three OracleModels over one real row r = (0.5, −0.25, 0, 0) at t = 3:
    #   twice at weight 1  == once at weight 2  (array_equal)
    #   once at weight 1 + once at weight 0 == once at weight 1  (array_equal)
def test_oracle_models_round_trip  # extend names to ("rows", "t", "targets", "weight")
def test_an_oracle_model_saved_without_a_weight_loads_with_ones(tmp_path)
```

Update `Spy.fit(self, rows, t, targets, weight, *, seed)` to record four arrays; `test_fit_receives_the_memory_as_it_stands_after_the_merge` zips four. In `test_train_torch.py` pass `np.ones(n, np.float32)` as the fourth positional at lines 110, 112, 149, 154, 159 and in the `spy` there (`spy(rows, t, targets, weight=None, *, seed, config)`).

- [ ] **Step 2: Run to make sure they fail**: `uv run pytest tests/deepcfr/test_train.py -q -p no:cacheprovider -x`.

- [ ] **Step 3: Implement.** `step` is untouched but for its docstring (natural rows weigh one; PR 4's boost stage passes the column). `NetLearner.fit` threads `weight` into `fit_from_scratch` and the held `batch_loss`. The oracle: `values = model.targets[at][legal].astype(np.float64) * np.repeat(model.weight[at].astype(np.float64), width)`.

- [ ] **Step 4: Run** `uv run pytest tests/deepcfr -q -p no:cacheprovider` → all pass, the oracle pin among them; then `uv run --extra deep pytest tests/deepcfr -q -p no:cacheprovider` for the torch half, then `uv sync`.

- [ ] **Step 5: Commit** `feat(rung4): the learner seam and the oracle take the weight; the oracle banks Σ t′ · w · r in LCFR's order, so weight one keeps the pin bit for bit`

### Task 4: the whole suite, a fresh-context review, the PR

- [ ] `uv run pytest -q -p no:cacheprovider` (about seven minutes; the LCFR 20M run is live, so `nice`).
- [ ] Dispatch a read-only reviewer subagent over `git diff origin/main...` with the plan's Global Constraints and Review Focus; fix what it finds.
- [ ] Push, open the PR against `main` with the house body (what it is, why the weight is where it is, the tests, rulings against the appendix, not in this PR), then watch `claude-review` and answer or fix every finding until none remain.
