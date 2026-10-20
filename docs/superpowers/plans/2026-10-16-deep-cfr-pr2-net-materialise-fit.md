# Deep CFR M1, PR 2: the net, materialise, and the from-scratch fit — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the PyTorch half of Deep CFR on mini-drawmaha: the MLP that reads PR 1's 136 features and writes 7 outputs, `materialise` that turns two seats' nets into the packed table's `cumulative_regret` (Deep CFR's eq. 3 by the write) so rung 3's walk plays the net's strategy unchanged, and `fit_from_scratch` that trains a fresh net on (row, t, target) arrays.

**Architecture:** a new package `src/drawmaha_solver/deepcfr/` with three modules, each one concern. `net.py` is the only module that defines a `torch.nn.Module`; `materialise.py` runs a net over the table point by point through `features.layout()` and writes regret slots; `fit.py` samples minibatches by index from row arrays, builds their features on the fly with `Layout.features`, and minimises t-weighted squared error over each row's legal columns. Nothing here knows about reservoirs or the training loop: `fit_from_scratch` takes plain arrays, so PR 3's reservoir and PR 4's loop plug in without changing it.

**Tech Stack:** Python 3.12+, NumPy, torch ≥ 2.4 (the `deep` optional extra, already in `pyproject.toml`; 2.14.1 is in the lockfile).

**Spec:** `docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`, sections "The network", "Materialising", correctness gate 3, and PR 2 of the PR table. `fit.py` is listed under PR 4 in the spec; Alec asked on 2026-10-16 for "the pytorch code and the code that actually makes/trains the model" together, so the refit moves into this PR and PR 4 keeps only `train.py`, the loop and the script.

## Global Constraints

- Network: "`136 -> 64 -> 64 -> 64 -> 7`, ReLU, about 17,000 parameters, as the DEFAULT: the hidden width and the number of hidden layers are run parameters".
- The head is indexed by `int(Action)`: fold 0, check/call 1, pot 2, throw-none 3, throw-low 4, throw-mid 5, throw-top 6. A ledger's columns are `features.legal_gather()[point, :width]` into it.
- Refit: "Fresh random initialisation every iteration, Adam at lr 1e-3, gradient norm clipped to 1, loss = squared error over the legal columns, each row weighted by its iteration t' (divided by the batch's mean weight …). 4,000 steps of 10,000 rows per refit." "Seeded `torch.manual_seed` per (seed, t, seat), fixed thread count, CPU only."
- Materialise, per spot: some output positive → the outputs as they are; every output ≤ 0 and not all equal → a one-hot at the argmax; all equal → zeros. An untrained seat (`None`) writes zeros: uniform.
- The default suite passes WITHOUT torch: every test module in `tests/deepcfr/` starts with `torch = pytest.importorskip("torch")`, and nothing outside `deepcfr/` imports `deepcfr`.
- `PackedTable`, `features.py`, `lockstep.py`, `regret_matching.py` are merged code: import from them, change nothing in them.
- Repo conventions: long explanatory module docstring, 75-hyphen section banners, one blank line between top-level defs, WHY-only comments, `from __future__ import annotations`, keyword-only arguments where two ints could be transposed.

## Decisions this plan makes that the spec leaves open

- **Ties under eq. 3.** All ≤ 0, not all equal, with several maxima tied: write 1.0 at EVERY tied maximum, so the ledger plays them uniformly. One-hot at the first index would make the strategy depend on Action order.
- **The loss's scale.** Per row: the sum over its legal columns of (prediction − target)², times its weight t' / mean(t' in the batch); the batch loss is the mean over rows. A 2-wide and a 4-wide spot both count as one row.
- **Minibatches** sample row indices uniformly WITH replacement from a `torch.Generator` seeded with the fit's seed, so a memory smaller than 10,000 rows still makes a full batch.
- **Seed** is a tuple of ints, `(seed, t, seat)` in PR 4; it is folded into one 63-bit int with `np.random.SeedSequence(seed).generate_state(2, np.uint32)` so different tuples cannot collide by summing.
- **Prediction batching:** `predict` runs in chunks of 262,144 rows under `torch.inference_mode()`, so the 100,400-row shape never builds a graph or a huge temporary.

## Review Focus

1. A target's padding past a spot's width (the 4-float target of a 2-wide betting row) must never reach the loss — PR 3 may pad with anything: pinned in Task 3 (`test_padding_never_reaches_the_loss`, padding = NaN, loss finite, weights unchanged by the padding's value).
2. `materialise` handed a table that is not the whole game (a `listed` table) must refuse, not write a whole-game layout into a smaller array: pinned in Task 2 (`test_materialise_refuses_a_listed_table`).
3. Rows with mismatched lengths (`rows`, `t`, `targets`) or an empty memory must refuse by name before any training starts: pinned in Task 3 (`test_fit_refuses_mismatched_or_empty_rows`).
4. A spot whose seat's net is `None` while the other seat has a net must still be uniform, and a seat's points must be scored by THAT seat's net, never the other: pinned in Task 2 (`test_each_point_is_scored_by_its_own_seats_net`).
5. Two fits with the same seed must produce identical weights (resume and the exactness tests depend on it), and different seeds different weights: pinned in Task 3 (`test_same_seed_same_weights`).

---

### Task 1: the net, its prediction, and its save file

**Files:**
- Create: `src/drawmaha_solver/deepcfr/__init__.py` (docstring only: what the package is, that it needs the `deep` extra)
- Create: `src/drawmaha_solver/deepcfr/net.py`
- Test: `tests/deepcfr/test_net.py`

**Interfaces:**
- Consumes: `features.FEATURE_WIDTH = 136`, `features.HEAD_WIDTH = 7`.
- Produces:
  - `class Net(torch.nn.Module)`: `__init__(self, *, width: int = 64, depth: int = 3)`; attributes `width`, `depth`; `forward(x: Tensor (n, 136) float32) -> Tensor (n, 7)`. `depth` hidden `Linear` layers of `width`, ReLU after each, then a `Linear(width, 7)` head with no activation (regrets are signed).
  - `predict(net: Net, features: np.ndarray) -> np.ndarray` — (n, 7) float32, chunked, `inference_mode`, net put in `eval()`.
  - `save_net(net: Net, path: Path) -> None` / `load_net(path: Path) -> Net` — one `torch.save` of `{"width", "depth", "state_dict"}`; `load_net` uses `weights_only=True`.

- [ ] **Step 1: Write the failing tests**

```python
torch = pytest.importorskip("torch")

def test_default_net_is_three_by_sixty_four():
    net = Net()
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(136, 64), (64, 64), (64, 64), (64, 7)]
    assert sum(p.numel() for p in net.parameters()) == 17_543

def test_width_and_depth_are_parameters():
    net = Net(width=128, depth=4)
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 5 and linears[0].out_features == 128

def test_net_refuses_a_nonpositive_size():
    with pytest.raises(ValueError, match="width"):
        Net(width=0)
    with pytest.raises(ValueError, match="depth"):
        Net(depth=0)

def test_predict_matches_forward_and_chunks(monkeypatch):
    net = Net()
    x = np.random.default_rng(0).standard_normal((1_000, 136)).astype(np.float32)
    monkeypatch.setattr(net_module, "PREDICT_CHUNK", 300)   # forces 4 chunks
    out = predict(net, x)
    assert out.shape == (1_000, 7) and out.dtype == np.float32
    np.testing.assert_allclose(out, net(torch.from_numpy(x)).detach().numpy(), rtol=1e-6)

def test_predict_refuses_the_wrong_width():
    with pytest.raises(ValueError, match="136"):
        predict(Net(), np.zeros((3, 92), dtype=np.float32))

def test_save_and_load_round_trip(tmp_path):
    net = Net(width=32, depth=2)
    save_net(net, tmp_path / "net.pt")
    back = load_net(tmp_path / "net.pt")
    assert (back.width, back.depth) == (32, 2)
    x = np.ones((5, 136), dtype=np.float32)
    np.testing.assert_array_equal(predict(back, x), predict(net, x))
```

(17,543 = 136·64+64 + 2·(64·64+64) + 64·7+7; confirm the arithmetic when writing the test.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra deep pytest tests/deepcfr/test_net.py -v`
Expected: FAIL with `ModuleNotFoundError: drawmaha_solver.deepcfr`

- [ ] **Step 3: Implement `Net`, `predict`, `save_net`, `load_net`, and `PREDICT_CHUNK = 262_144` in `net.py`**

`Net` builds a `torch.nn.Sequential`; refuse `width < 1` / `depth < 1` with `ValueError` naming the argument. `predict` refuses a non-2-D input or one whose second dimension is not `FEATURE_WIDTH`.

- [ ] **Step 4: Run to verify they pass, and that the default suite still skips cleanly without torch**

Run: `uv run --extra deep pytest tests/deepcfr/test_net.py -v` → all PASS
Run: `uv run pytest tests/deepcfr -q` (no extra) → all SKIPPED, exit 0

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr tests/deepcfr/test_net.py
git commit -m "feat(rung4): the 3 x 64 net over the 136 features, chunked prediction, and its save file"
```

---

### Task 2: materialise — two nets into the table's regret slots

**Files:**
- Create: `src/drawmaha_solver/deepcfr/materialise.py`
- Test: `tests/deepcfr/test_materialise.py`

**Interfaces:**
- Consumes: `Net`, `predict` (Task 1); `features.layout() -> Layout` (fields `points`, `row_offset`, `column_offset`, `width`, `private_offset`, `private`, `public`), `features.legal_gather() -> (141, 4) int8`; `PackedTable.whole_game()`, `.cumulative_regret`, `.widths()`, `len(table)`; `PublicPoint.player`.
- Produces:
  - `regret_slots_from_outputs(outputs: np.ndarray) -> np.ndarray` — (n, w) → (n, w) float64, eq. 3 by the write (rules in Global Constraints + the tie decision).
  - `materialise(table: PackedTable, nets: Sequence[Net | None]) -> None` — `nets[seat]` for seat 0 and 1; writes every row's slots into `table.cumulative_regret` in place; touches no other array.

- [ ] **Step 1: Write the failing tests**

```python
torch = pytest.importorskip("torch")

def test_some_positive_passes_through():
    out = np.array([[0.5, -2.0, 0.1]])
    np.testing.assert_array_equal(regret_slots_from_outputs(out), out)

def test_all_nonpositive_writes_the_argmax_one_hot():
    np.testing.assert_array_equal(
        regret_slots_from_outputs(np.array([[-3.0, -0.5, -1.0, -2.0]])), [[0.0, 1.0, 0.0, 0.0]]
    )

def test_tied_maxima_share_the_one_hot():
    np.testing.assert_array_equal(
        regret_slots_from_outputs(np.array([[-1.0, -0.5, -0.5]])), [[0.0, 1.0, 1.0]]
    )

def test_all_equal_writes_zeros():
    np.testing.assert_array_equal(regret_slots_from_outputs(np.full((2, 3), -0.7)), np.zeros((2, 3)))

def test_ledger_strategy_is_eq_three():
    # Over a batch of random rows: RegretMatcher.strategy() on the written slots equals
    # max(out,0)/sum when some is positive, else uniform over the argmax set.
    ...

def test_untrained_seats_write_uniform(whole_table):
    whole_table.cumulative_regret[:] = 5.0
    materialise(whole_table, (None, None))
    assert not whole_table.cumulative_regret.any()

def test_materialise_equals_the_per_key_path(whole_table, sampled_keys):
    nets = (seeded_net(0), seeded_net(1))
    materialise(whole_table, nets)
    for key in sampled_keys:          # ~200 keys over every shape and both seats
        _, start, width = whole_table.slot_of(key)
        point_gather = expected_gather(key)        # the key's legal_actions() as ints
        expect = regret_slots_from_outputs(predict(nets[key.player], encode(key)[None])[:, point_gather])
        np.testing.assert_allclose(whole_table.cumulative_regret[start:start + width], expect[0], rtol=1e-6)

def test_each_point_is_scored_by_its_own_seats_net(whole_table, sampled_keys):
    materialise(whole_table, (seeded_net(0), None))
    for key in sampled_keys:
        _, start, width = whole_table.slot_of(key)
        written = whole_table.cumulative_regret[start:start + width]
        assert written.any() == (key.player == 0) or not written.any()
    # seat 1's rows are all exactly zero (uniform); seat 0's are not all zero.

def test_materialise_touches_only_cumulative_regret(whole_table):
    whole_table.strategy_sum[:] = 2.0
    materialise(whole_table, (seeded_net(0), seeded_net(1)))
    assert (whole_table.strategy_sum == 2.0).all() and not whole_table.stamp.any()

def test_materialise_refuses_a_listed_table(sampled_keys):
    with pytest.raises(ValueError, match="whole game"):
        materialise(PackedTable.listed(sampled_keys), (None, None))

def test_materialise_refuses_wrong_number_of_nets(whole_table):
    with pytest.raises(ValueError, match="two"):
        materialise(whole_table, (None,))
```

Fixtures: `whole_table` = a module-scoped `PackedTable.whole_game()` (reset `cumulative_regret[:] = 0` per test); `sampled_keys` = a fixed-seed sample of `all_infosets()` covering every shape and both seats, as `tests/minidrawmaha/test_features.py` builds it; `seeded_net(s)` = `torch.manual_seed(s); Net()`. The whole-game tests run the real 6.2M-row materialise (~1 s each); keep them to these four.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra deep pytest tests/deepcfr/test_materialise.py -v`
Expected: FAIL with `ModuleNotFoundError: drawmaha_solver.deepcfr.materialise`

- [ ] **Step 3: Implement `regret_slots_from_outputs` and `materialise` in `materialise.py`**

`regret_slots_from_outputs` is vectorised over rows (no Python loop): masks for "any positive" and "all equal", `out == out.max(axis=1, keepdims=True)` for the one-hot. `materialise` refuses unless `len(nets) == 2`, `len(table) == layout.row_offset[-1]` and `table.cumulative_regret.size == layout.column_offset[-1]`; then per point `p`: rows of its shape are `layout.private[private_offset[p] : private_offset[p] + n]` with `layout.public[p]` broadcast beside them; `outputs = predict(nets[points[p].player], X)[:, gather[p, :width[p]]]`; `cumulative_regret[column_offset[p] : column_offset[p+1]] = regret_slots_from_outputs(outputs).ravel()`. A `None` net writes zeros for its points without building features.

- [ ] **Step 4: Run to verify they pass; time one whole-game materialise**

Run: `uv run --extra deep pytest tests/deepcfr/test_materialise.py -v` → all PASS
Run: `uv run --extra deep python -c "…time materialise(PackedTable.whole_game(), (Net(), Net()))…"` → record the time in the module docstring (spec's measured forward pass is 0.32 s; expect ~1 s with features and the write).

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr/materialise.py tests/deepcfr/test_materialise.py
git commit -m "feat(rung4): materialise writes two nets into the regret slots, eq. 3 by the write"
```

---

### Task 3: fit_from_scratch — the refit on (row, t, target) arrays

**Files:**
- Create: `src/drawmaha_solver/deepcfr/fit.py`
- Test: `tests/deepcfr/test_fit.py`

**Interfaces:**
- Consumes: `Net` (Task 1); `features.layout()` → `Layout.features(rows) -> (n, 136)`, `Layout.locate_rows(rows) -> (point, position)`; `features.legal_gather()`, `features.NO_ACTION = -1`, `features.MAX_WIDTH = 4`.
- Produces:
  - `@dataclass(frozen=True) FitConfig`: `width: int = 64`, `depth: int = 3`, `steps: int = 4_000`, `batch: int = 10_000`, `lr: float = 1e-3`, `clip: float = 1.0`, `threads: int = 8`.
  - `fit_from_scratch(rows: np.ndarray, t: np.ndarray, targets: np.ndarray, *, seed: tuple[int, ...], config: FitConfig = FitConfig()) -> Net` — `rows` (n,) int, `t` (n,) int ≥ 1, `targets` (n, 4) float32 in ledger-column order, padding ignored. Returns the trained net in `eval()` mode.
  - `batch_loss(net: Net, rows: np.ndarray, t: np.ndarray, targets: np.ndarray) -> torch.Tensor` — the scalar loss of one batch (what each step minimises); public so the tests and PR 4's logging read the same number.

- [ ] **Step 1: Write the failing tests**

```python
torch = pytest.importorskip("torch")
SMALL = FitConfig(steps=300, batch=256, threads=1)

def test_fit_memorises_a_small_memory():
    rows, t, targets = real_rows(n=200, seed=0)            # real table rows, random targets in [-1, 1]
    net = fit_from_scratch(rows, t, targets, seed=(0, 1, 0), config=FitConfig(steps=1_500, batch=200, threads=1))
    assert batch_loss(net, rows, t, targets).item() < 0.05 * batch_loss(Net(), rows, t, targets).item()

def test_later_iterations_weigh_more():
    # One row offered twice: target +1 at t=3 and -1 at t=1; the weighted least-squares answer is +0.5.
    rows = np.array([r, r]); t = np.array([3, 1]); targets = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]], np.float32)
    net = fit_from_scratch(rows, t, targets, seed=(0, 1, 0), config=FitConfig(steps=1_500, batch=64, threads=1))
    np.testing.assert_allclose(legal_prediction(net, r), 0.5, atol=0.05)

def test_padding_never_reaches_the_loss():
    rows, t, targets = real_rows(n=64, seed=1, widths={2, 3})   # betting rows only
    nan_padded = targets.copy(); nan_padded[:, 3] = np.nan       # and col 2 for 2-wide rows
    assert torch.isfinite(batch_loss(Net(), rows, t, nan_padded))
    assert batch_loss(Net_seeded, rows, t, nan_padded).item() == batch_loss(Net_seeded, rows, t, zero_padded).item()

def test_same_seed_same_weights():
    a = fit_from_scratch(*real_rows(64, 2), seed=(7, 3, 1), config=SMALL)
    b = fit_from_scratch(*real_rows(64, 2), seed=(7, 3, 1), config=SMALL)
    c = fit_from_scratch(*real_rows(64, 2), seed=(7, 3, 0), config=SMALL)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values()))

def test_fit_refuses_mismatched_or_empty_rows():
    rows, t, targets = real_rows(10, 3)
    with pytest.raises(ValueError, match="same length"):
        fit_from_scratch(rows, t[:-1], targets, seed=(0,), config=SMALL)
    with pytest.raises(ValueError, match="empty"):
        fit_from_scratch(rows[:0], t[:0], targets[:0], seed=(0,), config=SMALL)
    with pytest.raises(ValueError, match="t"):
        fit_from_scratch(rows, np.zeros_like(t), targets, seed=(0,), config=SMALL)   # t must be >= 1

def test_fit_refuses_rows_outside_the_table():
    rows, t, targets = real_rows(10, 4)
    rows[0] = features.layout().row_offset[-1]
    with pytest.raises(IndexError):
        fit_from_scratch(rows, t, targets, seed=(0,), config=SMALL)

def test_gradient_norm_is_clipped():
    # Huge targets (1e6): after one step every parameter moved by at most lr * ~1 per Adam's bound,
    # and no parameter is non-finite.
    ...
```

Helpers: `real_rows(n, seed, widths=None)` draws `n` row numbers from `features.layout()` (optionally only from points whose width is in `widths`), `t` uniform in 1..100, targets uniform in [−1, 1] with padding zeros. `legal_prediction(net, r)` = the net's outputs at row `r`'s legal columns.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra deep pytest tests/deepcfr/test_fit.py -v`
Expected: FAIL with `ModuleNotFoundError: drawmaha_solver.deepcfr.fit`

- [ ] **Step 3: Implement `FitConfig`, `batch_loss`, `fit_from_scratch` in `fit.py`**

Validation first (lengths, emptiness, `t >= 1`, `targets.shape[1] == MAX_WIDTH`), then `torch.set_num_threads(config.threads)`, fold `seed` into one int (decision above), `torch.manual_seed` it, build a fresh `Net(width, depth)`, a `torch.Generator` with the same seed for the minibatch indices, and `torch.optim.Adam(lr)`. Each of `steps` steps: draw `batch` indices with replacement, `loss = batch_loss(...)`, backward, `clip_grad_norm_(…, config.clip)`, step. `batch_loss`: features from `layout.features(rows)`, the batch's gather `legal_gather()[point]` (b, 4), mask = gather ≠ −1, predictions = `outputs.gather(1, gather.clamp(min=0))`; `torch.where(mask, (pred − target)², 0)` (so a NaN in padding never propagates), summed over columns, times `t / t.mean()`, mean over rows.

- [ ] **Step 4: Run to verify they pass; time one full-size refit**

Run: `uv run --extra deep pytest tests/deepcfr/test_fit.py -v` → all PASS, under ~60 s total
Run: one `fit_from_scratch` at the default config on 1M random real rows → record the time in the module docstring (spec measured 10.8 s for the bare 3×64 refit; the on-the-fly features add to it — if it exceeds ~30 s, say so in the PR rather than optimising here).

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/deepcfr/fit.py tests/deepcfr/test_fit.py
git commit -m "feat(rung4): the from-scratch refit — t-weighted squared error over the legal columns"
```

---

### Task 4: spec note and the full suite

**Files:**
- Modify: `docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md` (PR table: move `deepcfr/fit.py` from row 4 to row 2, with "moved 2026-10-16 at Alec's request")

- [ ] **Step 1: Run the default suite without torch** — `uv run pytest -q` → passes, `tests/deepcfr` skipped
- [ ] **Step 2: Run the deep tests with torch** — `uv run --extra deep pytest tests/deepcfr -q` → all pass
- [ ] **Step 3: Commit** — `git commit -am "docs(rung4): the refit moves into PR 2"`
