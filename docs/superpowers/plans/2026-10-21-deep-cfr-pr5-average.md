# Deep CFR M1, PR 5: the average strategy — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the two averages Deep CFR must rebuild because it threw the table's `strategy_sum` away — the EXACT running average (Single Deep CFR: own reach pushed through the materialised table every phase, `t · own_reach · σ` summed per column) and Π (the policy net fit on the policy memory) — each frozen into a `FrozenStrategy` the unchanged grader scores; gate 4 (own reach equals a brute-force walk) and the exact half of gate 5.

**Architecture:** two new modules. `deepcfr/average.py` builds, once, a predecessor table over the 141 public points and the three private-key shapes (every post-draw throw key maps to its THREE pre-draw draw keys and throw columns), pushes a seat's own reach through it in walk order, and keeps one float64 per table column as the running sum; `ExactAverage` is PR 4's `Observer` with `save`/`load`, and `train.py` gains one narrow seam — an observer that carries state is saved beside the memories and named in `state.json` — so the sums survive a resume and sit in every mark. `deepcfr/policy.py` fits Π with PR 2's `fit_from_scratch` under a documented seed and freezes it by running the net over every key. No loop order changes; gate 1 stays bit for bit.

**Tech Stack:** Python 3.12+, NumPy; torch only inside `policy.py`'s function bodies (lazy imports) and the torch test file. `average.py` imports no torch, so gate 4 and the replay pin run in the default suite.

**Spec:** `docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`, sections "The loop, in rung-3 names" (and the paragraph after it), "The exact average needs up to three predecessor rows per key", "Three graded objects per grid mark", "Modal" (the marks layout), gates 4 and 5, PR 5's row. The Conductor prompt is `/Users/alec/Desktop/Claude/poker/rung4-context/CONDUCTOR-PROMPT-deep-cfr-pr5.md`.

## Global Constraints

- The observer hook is the only entry: `observers(table, t, 1 − p)` runs right after each `learner.write`, before `collect`; seat `1 − p`'s σ is the one about to be sampled. Each seat is accumulated once per iteration, seat 1 first (at p = 0, from the previous refit) then seat 0 (at p = 1, freshly refit).
- Exact average per row: `numerator(I) += t · own_reach_t(I) · σ_t(I)`; `average(I) = numerator(I) / Σ_a numerator(I, a)`, uniform where that is 0. The spec's `denominator(I) += t · own_reach_t(I)` is the numerator's row sum exactly (every σ row sums to 1), so it is not stored — `strategy.average_rows` normalises, as it does for rung 3's `strategy_sum`.
- Own reach is a sum over physical histories of the product of the seat's OWN action probabilities; chance and the opponent are per-row constants over t and cancel in the normalisation. The predecessor table (spec): first own action → none (reach 1); later in round 1 → the same key with the line two actions shorter; the draw → the seat's last round-1 decision; round 2 stood pat → the draw key (hole, board[:1]) at the stand-pat column; round 2 after a throw → the draw key of EACH of the three pre-draw holes (hole − one card + the thrown card) at the column `draw_order` gives the thrown card, listed WITH multiplicity (two distinct keys still count as three histories); later in round 2 → the same key with the line two actions shorter.
- Every graded object is a `FrozenStrategy` through `minidrawmaha/exploitability.py` unchanged. Uniform grades 4.842; PR 4 measured the current iterate at 1.391 at t = 5, K = 1,000.
- Π: a FRESH net fit on the policy memory's `contents()` (`rows, t, targets`; targets are σ rows zero-padded to `TARGET_WIDTH = 4`, masked by `batch_loss` exactly as advantage rows are) with seed `(seed, t, POLICY_TAG)`, `POLICY_TAG = 0x5049` ("PI"), distinct from `NetLearner`'s `(seed, t, p)` and its held-batch tag; frozen as outputs at the legal columns, clipped at 0, renormalised, uniform if all zero.
- Determinism: everything is a function of (seed, t, seat, k); nothing random is added here.
- Do not change merged code except `deepcfr/train.py` (the stateful-observer seam, documented in the PR body) and `scripts/deepcfr/run_deep_cfr.py` (passes `ExactAverage()` so a real run accumulates). Untouched: lockstep.py, lockstep_run.py, packed_table.py, regret_rules.py, exploitability.py, strategy.py, features.py, deepcfr/{net,fit,materialise,reservoir,collect}.py.
- Repo style: long explanatory module docstring, 75-hyphen section banners, WHY-only comments, `from __future__ import annotations`, keyword-only args where ints could be transposed, atomic saves (temp + fsync + `os.replace`), fail-loud validation by name. Commits `feat(rung4): …`.
- Tests: `tests/deepcfr/test_average.py` in the default suite (no torch); `tests/deepcfr/test_policy.py` behind `pytest.importorskip("torch")`; the gate-5 extension behind `MINIDRAWMAHA_FULL_DEEP=1`.

## Decisions this plan makes that the spec and prompt leave open

- **"Exact = tabular" is NOT a test.** Rung 3's `strategy_sum` is banked at the opponent's visits of K SAMPLED walks (`lockstep._Recorder`, `apply_arrays`), so it equals `Σ t · own_reach · σ` only in expectation, with sampling noise in the reach weight and nothing at unvisited rows. The honest pins are (a) gate 4, the brute-force walk through `MiniState` on a random non-uniform table for every predecessor case, and (b) the REPLAY pin: the sums accumulated inside an oracle run equal the sums rebuilt afterwards from the per-iteration models by `write` + `accumulate`, bit for bit — the spec's "Single Deep CFR, both ways" — done torch-free with `OracleLearner` in the default suite and again with the real nets under gate 5.
- **Own reach lives in key space, not the grader's physical frame.** `exploitability._push` spreads reach over physical (hole, thrown) states with the board as a batch axis because it carries chance and the opponent. Own reach needs neither, and the quantity the average weights by is per CANONICAL key, so the pass is a per-point gather over a static `(positions, columns)` map from a point's keys to its predecessor point's keys. The grader is mirrored in idea (a precomputed draw map, edges grouped so the three throws are one push) and not called.
- **The sums are float64**, one per column (14,253,840 × 8 B = 114 MB), on the table's own dtype; reach products of up to six own decisions times t ≤ 2,000 are far inside float64, and the replay pin is bitwise, which float32 accumulation order would not survive. Task 2 measures how far a float32 copy of the sums would move the frozen average, for the PR body.
- **The resume carries the sums through a stateful observer, not a new `DeepRun` field.** `train.py` gains `StatefulObserver` (an observer with `name`, `save(path)`, `load(path)`); `save_run`/`load_run` take `observers=` and write/read `{name}-t{t:05d}.npz` beside the memories, named under `state["files"]["observers"]`; `_Saves` and `_open` pass them through. A resumed run whose state names an observer it was not given is refused by name in `_open` (resuming without the average would silently drop it); `load_run` alone does not refuse, so PR 6's grader can open a mark for its memories and nets without the observer.
- **The accumulator refuses out-of-order calls.** It keeps the last t accumulated per seat and requires each call to be `last[seat] + 1`; a resume from the wrong mark, a skipped phase or a double call is a `ValueError` naming the seat and both t's, not a silently wrong average.
- **Marks: PR 5 exposes the freezes; PR 6 writes the three files.** A mark already holds the sums (`exact-t….npz`), the policy memory and both nets, which is everything `exact`, `pi` and `current` need; fitting Π inside the loop would put torch and 15 s into every mark for a file the grader container can build itself. `ExactAverage.frozen(...)`, `policy.fit_policy(...)` and `policy.policy_strategy(...)` are the functions PR 6 calls.
- **The whole-game layout only.** `own_reach` and `ExactAverage` refuse a table that is not the whole game by the same fingerprint `materialise` uses (row count, column count, widths), restated rather than imported from a private name.

## Review Focus

1. A post-draw throw key whose three pre-draw holes canonicalise to only two draw keys must still count three histories — pinned in Task 1 (`test_own_reach_equals_the_brute_force_walk` includes a key with two distinct predecessors, found by scanning the map).
2. A throw column read off the wrong frame (physical suit order instead of `draw_order`'s canonical order) would give a plausible but wrong reach at every post-draw key — pinned in Task 1 by the brute force, which chooses the throw through `MiniState.apply` and never names a column.
3. A resumed run that was saved before the observer existed, or resumed without it, must not continue with a silently empty or dropped average — pinned in Task 3 (`test_a_resume_refuses_a_state_without_the_observers_file`, `test_a_resume_refuses_to_drop_a_saved_observer`).
4. An accumulate call at the wrong t (a skipped iteration, a double call, a resume from an older mark) must raise rather than weight the wrong σ — pinned in Task 2 (`test_accumulate_refuses_a_t_out_of_order`).
5. A Π freeze with every output non-positive at some key must play uniform there, not NaN — pinned in Task 4 (`test_policy_strategy_is_a_distribution_and_matches_the_net_key_by_key` includes a net forced to all-negative outputs via a bias edit, and `FrozenStrategy`'s own row check).

---

### Task 1: the predecessor table and the own-reach pass (gate 4)

**Files:**
- Create: `src/drawmaha_solver/deepcfr/average.py`
- Test: `tests/deepcfr/test_average.py`

**Interfaces:**
- Consumes: `features.layout()` (`points, row_offset, column_offset, width`), `enumeration.private_keys(board_cards=, discards=)`, `enumeration.public_decision_points()`, `game.canonical_picture`, `game.draw_order`, `game.DRAW_ACTIONS`, `game.Action`, `PackedTable` (`cumulative_regret`, `widths()`, `slot_of`), `strategy._row_starts`-equivalent arithmetic (restated: `np.cumsum(widths) − widths`).
- Produces:
  - `@dataclass(frozen=True, slots=True, eq=False) Pred`: `point: int` (the predecessor public point's index), `positions: np.ndarray` int64 `(n_keys, m)` — for each key of this point, the position(s) in the predecessor point's shape; `columns: np.ndarray` int64 `(n_keys, m)` — the ledger column taken at each. `m` is 1, or 3 for a round-2 point after a throw.
  - `predecessors() -> tuple[Pred | None, ...]` (`@cache`): one entry per public point in walk order; `None` for a seat's first own action.
  - `current_strategy(table: PackedTable) -> np.ndarray`: float64 per column, regret matching row by row as `RegretMatcher.strategy` reads it (positive part normalised, uniform when the positive sum is 0).
  - `own_reach(table: PackedTable, *, seat: int) -> np.ndarray`: float64 per ROW; 1.0 at `seat`'s first-action keys, `Σ_pred reach[pred] · σ[pred, column]` elsewhere, 0.0 at the other seat's rows. Refuses a non-whole-game table and a seat not in (0, 1) by name.

- [ ] **Step 1: Write the failing tests**

Helpers at the top of `tests/deepcfr/test_average.py` (restated, as the suite's rule is):

```python
def key_on(point, position) -> InfoSet: ...   # InfoSet from a public point and a private key
def random_table(seed) -> PackedTable:        # whole game; cumulative_regret ~ N(0, 1), so some rows are all-negative (uniform)
def brute_force_own_reach(table, key) -> float:
    """Σ over the seat's physical pre-histories of Π own σ, walked through MiniState.

    The opponent's hole, throw and replacement are FIXED to one choice disjoint from
    every card the seat has seen (so the sum is over the seat's own histories only, up
    to the per-row constant the average cancels). For a post-draw key with a discard,
    the three pre-draw holes are `hole − d + discarded` for d in hole; the walk follows
    key.betting and key.draws, chooses the seat's throw by `MiniState.apply` producing
    `discards[seat] == key.discarded`, deals `d` back, and multiplies `table[state.infoset()].strategy()[column]`
    at every own decision until `state.infoset() == key`.
    """
```

```python
def test_predecessors_cover_the_six_cases():
    preds = predecessors(); points = public_decision_points()
    assert len(preds) == len(points) == 141
    for index, (point, pred) in enumerate(zip(points, preds)):
        if pred is None:
            assert point.board_cards == 1 and not point.is_draw_decision and len(point.betting[0]) == point.player
        else:
            assert pred.point < index                                   # parents before children in walk order
            assert points[pred.point].player == point.player           # an OWN decision
            n = len(private_keys(board_cards=point.board_cards, discards=point.discards))
            assert pred.positions.shape == pred.columns.shape == (n, 3 if (point.board_cards == 2 and point.discards == 1 and len(point.betting[1]) < 2) else 1)
            assert pred.columns.min() >= 0 and pred.columns.max() < points[pred.point].width
            assert pred.positions.max() < len(private_keys(board_cards=points[pred.point].board_cards, discards=points[pred.point].discards))

def test_a_throw_keys_three_predecessors_are_its_three_pre_draw_holes():
    # pick a round-2 first-action throw point; for 50 random keys check, by hand with canonical_picture and
    # draw_order, that the three (position, column) pairs are the draw keys of hole − d + discarded, column
    # 1 + draw_order(...).index(discarded) ; and that the multiset has 3 entries always, 2 distinct for some key

def test_current_strategy_is_regret_matching_row_by_row():
    table = random_table(1); sigma = current_strategy(table)
    for key in some_keys(): np.testing.assert_array_equal(sigma[start:start+width], table[key].strategy())

@pytest.mark.parametrize("seat", [0, 1])
def test_own_reach_equals_the_brute_force_walk(seat):
    table = random_table(2); reach = own_reach(table, seat=seat)
    # keys: 2 per point of `seat` (first and middle private key), plus one post-draw throw key with 3
    # distinct predecessors and one with 2 distinct (found by scanning predecessors()), ~130 keys
    for key in sampled: assert reach[table.row_of(key)] == pytest.approx(brute_force_own_reach(table, key), rel=1e-12)
    assert (reach[rows of the other seat] == 0).all()
    assert reach[rows of seat's first-action point] == 1.0 everywhere

def test_own_reach_refuses_a_listed_table_and_a_bad_seat(): ValueError match "whole game" / "seat"
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/deepcfr/test_average.py -v` → `ModuleNotFoundError: drawmaha_solver.deepcfr.average`.

- [ ] **Step 3: Implement `average.py`, sections "The predecessor table" and "Own reach"**

Module docstring: what own reach is and why it is what the average weights by; why a post-draw key has three predecessors (the key forgets which card was drawn) and why chance cancels; the six cases as a table; the pass is 141 gathers in walk order; whole game only. `predecessors()` finds a point's predecessor point by its public signature `(board_cards, draws, betting)` through a dict built from `public_decision_points()`: round-1 later → `(1, (), (line[:-2],))`, column `actions.index(line[-2])`; draw → `(1, (), (line[:i],))` with `i` the last index of the actor's own action in round 1, column `actions.index(line[i])`; round-2 first action → `(1, draws[:player], (line1,))` the actor's draw point; round-2 later → `(2, draws, (line1, line2[:-2]))`. Two shared private maps, built once: `(2, 0) → (1, 0)` positions `(10170, 1)` via `canonical_picture(hole, (), board[:1])`, columns all `DRAW_ACTIONS.index(Action.THROW_NONE)`; `(2, 1) → (1, 0)` positions `(100400, 3)` and columns `1 + draw_order(hole=pre, discarded=(), board=board[:1]).index(discarded[0])` with `pre = sorted(hole − d + discarded)` for d in hole. Same-shape predecessors use `np.arange(n)[:, None]` and `np.broadcast_to(column, (n, 1))`. `own_reach` zeros a row vector, then for each point of `seat` in walk order: first action → 1.0; else `(reach[row_offset[pred.point] + pred.positions] * sigma[column_offset[pred.point] + pred.positions * width[pred.point] + pred.columns]).sum(axis=1)`.

- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/deepcfr/test_average.py -v` → PASS; note the predecessor build time and the pass time (print once, for the docstring).

- [ ] **Step 5: Commit** — `git commit -m "feat(rung4): the predecessor table and the own-reach pass — three pre-draw holes per post-draw key, pinned against a brute-force walk (gate 4)"`

---

### Task 2: the running exact average, its freeze and its file

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/average.py` (sections "The running sum", "The file")
- Test: `tests/deepcfr/test_average.py`

**Interfaces:**
- Consumes: Task 1; `strategy.FrozenStrategy`, `strategy.StrategyInfo`, `strategy.average_rows`; `train.step`, `train.new_run`, `train.OracleLearner`, `train.DeepConfig`, `collect.Collector` (tests only).
- Produces:
  - `class ExactAverage`: `name: str = "exact"`; `sums: np.ndarray` float64 `(columns,)`; `last: list[int]` the last t accumulated per seat, `[0, 0]` when new. `__init__(self)` allocates from `layout()`. `accumulate(self, table: PackedTable, t: int, *, seat: int) -> None`: refuses `t != last[seat] + 1` by name (seat, expected, got) and a non-whole-game table; `sums += t · repeat(own_reach(table, seat=seat), widths) · current_strategy(table)`; `last[seat] = t`. `__call__(self, table, t, seat) -> None` — the `Observer` signature — is `accumulate(table, t, seat=seat)`. `frozen(self, *, iteration: int, seed: int) -> FrozenStrategy`: `FrozenStrategy(average_rows(sums, widths), widths, StrategyInfo(rule="deep", column="exact", iteration=iteration, seed=seed, workers=0))`. `save(self, path: Path) -> None`: `.npz` with `sums`, `last` (int64 (2,)), written through an open file, temp + fsync + `os.replace`. `load(self, path: Path) -> None`: refuses by name a file without both keys, `sums` of the wrong length or dtype, `last` not two non-negative ints; pours in place.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_new_average_is_empty_and_plays_uniform():
    acc = ExactAverage(); assert acc.last == [0, 0] and not acc.sums.any()
    frozen = acc.frozen(iteration=0, seed=0); assert frozen.info == StrategyInfo("deep", "exact", 0, 0, 0)
    np.testing.assert_array_equal(frozen.probabilities, uniform_strategy().probabilities)

def test_accumulate_adds_t_times_reach_times_sigma_for_one_seat_only():
    table = random_table(3); acc = ExactAverage(); acc(table, 1, 1)
    reach = own_reach(table, seat=1); sigma = current_strategy(table)
    np.testing.assert_array_equal(acc.sums, 1 * np.repeat(reach, widths) * sigma)
    assert not acc.sums[columns of seat 0].any() and acc.last == [0, 1]
    acc(table, 1, 0); assert acc.last == [1, 1]; both seats' columns now non-zero
    # a second table at t = 2 for seat 1: sums[seat 1 cols] == first + 2 · reach₂ · σ₂ (bitwise, one add)

def test_the_frozen_average_is_the_normalised_sums_uniform_where_unreached():
    # after two accumulations: frozen[key] == (sums row / row sum).astype(float32) at reached keys; at a key
    # whose row sum is 0 (the other seat's… no: pick a seat-0 key before seat 0 was accumulated) → uniform

def test_accumulate_refuses_a_t_out_of_order():
    acc = ExactAverage(); table = random_table(4)
    acc(table, 1, 0)
    for bad in (1, 3, 0): with pytest.raises(ValueError, match=r"seat 0.*expected 2.*got"): acc(table, bad, 0)
    acc(table, 1, 1)   # seat 1 is independent

def test_the_file_round_trips_atomically(tmp_path):
    acc = ExactAverage(); acc(random_table(5), 1, 1); acc.save(tmp_path / "exact-t00001.npz")
    back = ExactAverage(); back.load(tmp_path / "exact-t00001.npz")
    np.testing.assert_array_equal(back.sums, acc.sums); assert back.last == acc.last; no *.tmp left; file name unchanged (no .npz appended)
    for broken in (missing "last", sums of wrong length, last = [-1, 0]): ValueError naming the file and the field

def test_the_sums_accumulated_in_an_oracle_run_equal_a_replay_from_its_models():
    # DeepConfig(seed=5, walks=8, workers=0, capacity=100_000); OracleLearner; three steps with observers=(acc,),
    # keeping a copy of run.nets after each step (nets[t] = tuple(run.nets)). Replay on a fresh table:
    #   for t in 1..3: write(table, (nets[t-1][0], nets[t-1][1])); replay(table, t, 1); write(table, (nets[t][0], nets[t-1][1])); replay(table, t, 0)
    # np.testing.assert_array_equal(replay.sums, acc.sums); assert acc.sums.any()

def test_the_exact_average_of_the_oracle_run_is_a_strategy_the_grader_reads():
    # frozen = acc.frozen(iteration=3, seed=5); _snapshot-free check: best_response_value(frozen, responder=0) is finite
    # (a Leduc-sized grade is not possible here; the whole-game grade is gate 5)
```

- [ ] **Step 2: Run to verify they fail** — `ImportError: ExactAverage`.

- [ ] **Step 3: Implement `ExactAverage`** as specified. Docstring: why `__call__` is the observer and what "iteration t's σ for seat q" means at the hook; why the denominator is not stored; why float64; why the t check exists.

- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/deepcfr/test_average.py -v` → PASS.

- [ ] **Step 5: Measure, for the PR body** — a scratch script (not committed) over the oracle run of the replay test extended to 5 iterations: the pass's wall time per call on the whole game, resident memory of `sums`, and `max |frozen(float32 sums) − frozen(float64 sums)|` after 5 iterations.

- [ ] **Step 6: Commit** — `git commit -m "feat(rung4): the running exact average — t · own reach · σ summed per column at the observer hook, frozen through average_rows, and its file"`

---

### Task 3: the sums survive a resume and sit in every mark (train.py's one seam)

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/train.py` (`save_run`, `load_run`, `_Saves`, `_open`, `run_deep`'s docstring, the module docstring's resume paragraph; new `StatefulObserver` Protocol beside `Observer`)
- Modify: `scripts/deepcfr/run_deep_cfr.py` (`main` passes `observers=(ExactAverage(),)`)
- Test: `tests/deepcfr/test_average.py` (the resume pins), `tests/deepcfr/test_train.py` (one assertion that a save without stateful observers writes the same files as before)

**Interfaces:**
- Consumes: Task 2's `ExactAverage` (`name`, `save`, `load`, `__call__`).
- Produces:
  - `class StatefulObserver(Protocol)`: `name: str`; `__call__(self, table: PackedTable, t: int, seat: int) -> None`; `save(self, path: Path) -> None` (atomic, as `Reservoir.save` is); `load(self, path: Path) -> None`.
  - `save_run(run, directory, *, learner, observers: Sequence[Observer] = ())`: for each observer with `save`, `load` and `name`, writes `{name}-t{t:05d}.npz` and records it in `state["files"]["observers"][name]`; refuses two stateful observers with one name. `load_run(directory, *, learner, observers=())`: for each stateful observer, requires its file to be named and present (`ValueError` naming the observer and the directory) and calls `observer.load`.
  - `run_deep(..., observers=...)` unchanged in signature; `_open` passes observers to `load_run` and refuses, by name, a resume state naming an observer not among those given. `_Saves.observers` is passed to `save_run` at marks and resume saves.

- [ ] **Step 1: Write the failing tests** (in `test_average.py`, with `OracleLearner`, `DeepConfig(seed=5, walks=5, workers=0, capacity=100_000, grid=(4, 8))`, `log=lambda _: None`):

```python
def test_a_run_stopped_and_restarted_keeps_the_same_sums_bitwise(tmp_path):
    first = ExactAverage(); run_deep(tmp_path, CFG, target=6, learner=OracleLearner(), observers=(first,), log=...)
    second = ExactAverage(); run_deep(tmp_path, CFG, target=10, learner=OracleLearner(), observers=(second,), log=...)
    straight = ExactAverage(); run_deep(tmp_path / "straight", CFG, target=10, learner=OracleLearner(), observers=(straight,), log=...)
    np.testing.assert_array_equal(second.sums, straight.sums); assert second.last == straight.last == [10, 10]
    assert (tmp_path / RESUME / "exact-t00010.npz").exists() and not (tmp_path / RESUME / "exact-t00006.npz").exists()
    mark = ExactAverage(); mark.load(tmp_path / MARKS / mark_name(8) / "exact-t00008.npz")  # a mark carries the sums at its t
    # … and equals a straight run to 8
    assert json.loads((tmp_path / RESUME / STATE).read_text())["files"]["observers"] == {"exact": "exact-t00010.npz"}

def test_a_resume_refuses_a_state_without_the_observers_file(tmp_path):
    run_deep(tmp_path, CFG, target=2, learner=OracleLearner(), log=...)             # saved with no observer
    with pytest.raises(ValueError, match="exact"): run_deep(tmp_path, CFG, target=3, learner=OracleLearner(), observers=(ExactAverage(),), log=...)

def test_a_resume_refuses_to_drop_a_saved_observer(tmp_path):
    run_deep(tmp_path, CFG, target=2, learner=OracleLearner(), observers=(ExactAverage(),), log=...)
    with pytest.raises(ValueError, match="exact"): run_deep(tmp_path, CFG, target=3, learner=OracleLearner(), log=...)
    assert load_run(tmp_path / RESUME, learner=OracleLearner()).t == 2   # a plain load (PR 6's grader) still works

def test_two_stateful_observers_with_one_name_are_refused(tmp_path): ValueError match "exact"

def test_the_cli_passes_the_exact_average(monkeypatch): main() calls run_deep with observers whose [0] is an ExactAverage
```

In `test_train.py`, extend `test_save_and_load_round_trip`'s file list assertion with a comment that no observer means no observer file.

- [ ] **Step 2: Run to verify they fail** — `TypeError: save_run() got an unexpected keyword 'observers'` / the refusal tests pass vacuously → make them fail by asserting the sums are non-empty first.

- [ ] **Step 3: Implement the seam** — `StatefulObserver`, `_stateful(observers) -> list[StatefulObserver]` (duck-typed on the three attributes; raises on duplicate names), `save_run`/`load_run` `observers=`, `_Saves.observers`, `_open`'s refusal, the docstring paragraphs. Order inside `save_run`: memories, nets, observer files, then `state.json`; the cleanup keeps the observer files the state names.

- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/deepcfr/test_average.py tests/deepcfr/test_train.py -v` → PASS, gate 1 untouched.

- [ ] **Step 5: Commit** — `git commit -m "feat(rung4): an observer that carries state joins the resume set and every mark, so the exact average survives a stop and restart bit for bit"`

---

### Task 4: Π — the policy net fit and its freeze

**Files:**
- Create: `src/drawmaha_solver/deepcfr/policy.py`
- Test: `tests/deepcfr/test_policy.py` (`pytest.importorskip("torch")`)

**Interfaces:**
- Consumes: `fit.fit_from_scratch`, `fit.FitConfig`, `fit._fold_seed` (tests), `net.Net`, `net.predict`, `features.layout`, `features.legal_gather`, `features.encode` (tests), `strategy.average_rows`, `strategy.FrozenStrategy`, `strategy.StrategyInfo`, `reservoir.Reservoir.contents()`.
- Produces:
  - `POLICY_TAG = 0x5049`.
  - `policy_seed(seed: int, t: int) -> tuple[int, int, int]` = `(seed, t, POLICY_TAG)`.
  - `fit_policy(rows: np.ndarray, t: np.ndarray, targets: np.ndarray, *, seed: int, iteration: int, config: FitConfig | None = None) -> Net`: `fit_from_scratch(rows, t, targets, seed=policy_seed(seed, iteration), config=config or FitConfig())`; refuses a targets row that is not a distribution at its legal columns (a regret memory handed by mistake), by name. torch and `fit` imported inside.
  - `policy_strategy(net: Net, *, iteration: int, seed: int) -> FrozenStrategy`: 141 batched `predict` calls in `layout()`'s order (as `materialise` does, restated), outputs gathered at `legal_gather()[p, :width]`, `np.maximum(·, 0)` into a flat float64 array, then `FrozenStrategy(average_rows(flat, widths), widths, StrategyInfo(rule="deep", column="pi", iteration=iteration, seed=seed, workers=0))`.

- [ ] **Step 1: Write the failing tests**

```python
TINY = FitConfig(width=16, depth=2, steps=40, batch=64, threads=2)

def policy_memory(seed, walks) -> tuple[rows, t, targets]:  # Collector(workers=0) over a random-regret table, t = 1
def test_policy_seed_is_distinct_from_the_advantage_fits(): policy_seed(9, 1) not in {(9,1,0), (9,1,1)}; fit_policy(..., seed=9, iteration=1) weights ≠ fit_from_scratch(same rows, seed=(9,1,0)) weights; two fit_policy calls equal
def test_fit_policy_refuses_a_regret_memory(): targets with a row summing to 3 → ValueError match "distribution"
def test_policy_strategy_is_a_distribution_and_matches_the_net_key_by_key():
    net = fit_policy(*policy_memory(9, 10), seed=9, iteration=1, config=TINY); frozen = policy_strategy(net, iteration=1, seed=9)
    assert frozen.info == StrategyInfo("deep", "pi", 1, 9, 0) and len(frozen) == 6_220_050
    for key in some_keys(): out = predict(net, encode(key)[None])[0][list(key.legal_actions())]; expected = clip/normalise/uniform; assert_allclose(frozen[key], expected.astype(np.float32), atol=1e-6)
    # force the head's bias to −10 for every output: every row is uniform (FrozenStrategy would raise on NaN)
def test_the_fit_lands_on_the_t_weighted_mean_at_a_repeated_key():
    # rows: one key at t=1 with σ=(1,0,…), t=2 with (0,1,…), t=3 with (0.6,0.4,…) each repeated 50×; FitConfig(steps=400, batch=64)
    # expected = (0.47, 0.53); assert_allclose(frozen[key], expected, atol=0.05)  # the net is deterministic in its seed; 0.05 is what 400 steps of a 16-wide net reach on a 3-row memory
```

- [ ] **Step 2: Run to verify they fail** — `ModuleNotFoundError: drawmaha_solver.deepcfr.policy`.

- [ ] **Step 3: Implement `policy.py`** — docstring: what Π is (the time average learned by regression: the minimiser of t-weighted squared error over σ rows IS the t-weighted mean; the reach weight rides in the sample counts), why a fresh fit per mark, the seed tuple, why the freeze clips and renormalises (cleanup of fit error, not regret matching), whole game only, torch imported lazily.

- [ ] **Step 4: Run to verify they pass** — `uv run --extra deep pytest tests/deepcfr/test_policy.py -v` → PASS; `uv run python -c "import sys, drawmaha_solver.deepcfr.policy; assert 'torch' not in sys.modules"`.

- [ ] **Step 5: Commit** — `git commit -m "feat(rung4): Π — the policy net fit on the policy memory under its own seed, and frozen as the average strategy the grader reads"`

---

### Task 5: gate 5's exact half, the docs, and the numbers

**Files:**
- Modify: `tests/deepcfr/test_train_torch.py` (extend `test_gate_five_the_current_iterate_beats_uniform` → `test_gate_five_the_exact_average_beats_the_current_iterate_and_uniform`)
- Modify: `src/drawmaha_solver/deepcfr/__init__.py` (eight modules), `src/drawmaha_solver/deepcfr/AGENTS.md` (PR 5 landed; the observer rule), `docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md` (PR 5 row and gate 5: dated notes on the denominator, the stateful observer, "exact = tabular" in expectation only)
- Modify: `src/drawmaha_solver/deepcfr/average.py` docstring (measured pass time, memory, build time)

**Interfaces:**
- Consumes: everything above; `net.load_net`, `train.NETS`, `train.net_name`, `exploitability.exploitability`, `strategy.uniform_strategy`.

- [ ] **Step 1: Write the test** (behind `MINIDRAWMAHA_FULL_DEEP=1`):

```python
def test_gate_five_the_exact_average_beats_the_current_iterate_and_uniform(tmp_path):
    config = DeepConfig(seed=0, walks=1_000, workers=8, capacity=10_000_000, steps=1_000, threads=8)
    exact = ExactAverage(); run = run_deep(tmp_path, config, target=5, learner=learner, observers=(exact,), log=print)
    current = exploitability(current_iterate(table after write, ...)); average = exploitability(exact.frozen(iteration=5, seed=0))
    pi = exploitability(policy_strategy(fit_policy(*run.policy.contents(), seed=0, iteration=5, config=learner._fit()), iteration=5, seed=0))
    uniform = exploitability(uniform_strategy()); print all four and the wall clocks
    assert uniform == approx(4.842, abs=0.001); assert average < current < uniform; assert pi < uniform   # Π's bar is the floor; its number is reported
    # replay from out/nets: for t in 1..5: materialise(table, (load p0 t−1 or None, load p1 t−1 or None)); replay(table, t, 1); materialise(table, (p0 t, p1 t−1)); replay(table, t, 0)
    np.testing.assert_array_equal(replay.sums, exact.sums)
    # the resume state and the t=… marks hold the same sums: ExactAverage().load(tmp_path / RESUME / "exact-t00005.npz").sums == exact.sums
```

- [ ] **Step 2: Run it** — `MINIDRAWMAHA_FULL_DEEP=1 uv run --extra deep pytest tests/deepcfr/test_train_torch.py -k gate_five -s` → PASS; record the four grades and the seconds for the PR body and the walkthrough.

- [ ] **Step 3: Docs** — `__init__.py` lists `average` and `policy`; `deepcfr/AGENTS.md` says PRs 1 to 5 are merged and adds the rule "an observer that carries state has `name`, `save`, `load`; its file joins the resume set; a resume refuses to drop one"; the spec's PR 5 row and gate 5 get a dated note; `average.py`'s docstring carries Task 2's measurements.

- [ ] **Step 4: Run the suites** — `uv run pytest -q` (default, no torch) and `uv run --extra deep pytest tests/deepcfr -q` → all PASS; `uv run pytest tests/test_agent_docs.py -q` for the AGENTS.md paths.

- [ ] **Step 5: Commit** — `git commit -m "feat(rung4): gate 5's exact half — the exact average grades below the current iterate and uniform, and replays from the kept nets bit for bit"`
