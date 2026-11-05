# Rung 4 distillation (representation floor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit the rung-4 net to the 10M-iteration LCFR table's average strategy by reach-weighted cross-entropy over every infoset, grade each copy exactly, and record the three gaps (representation, training, both) with figures and a preprint.

**Architecture:** One new module `deepcfr/distill.py` (dataset from a `FrozenStrategy`, the fit on the GPU, the softmax freeze, the reach-weighted error), torch imported inside functions like `policy.py`. One script `scripts/deepcfr/distill_teacher.py` runs the arms and writes `figures/rung4/distill/results.json`; one torch-free script `scripts/deepcfr/distill_figures.py` draws the committed PNGs from it. Reach comes from `average.py` by lifting its σ-array core out of `own_reach`; the per-point prediction loop shared by `materialise`, `policy_strategy` and the new freeze is lifted into `net.predict_by_point`.

**Tech Stack:** Python 3.12, NumPy, torch (`--extra deep`, `mps`), matplotlib, pytest; tectonic for the report.

**Spec:** `docs/superpowers/specs/2026-11-05-rung4-distillation-design.md`

## Global Constraints

- The grader (`minidrawmaha/exploitability.py`) is not imported by the student and imports nothing from `deepcfr/`.
- Default `uv run pytest` passes without torch: every torch test opens with `pytest.importorskip("torch")`; `distill.py` imports torch inside functions.
- Teacher checkpoint is read only; nothing under its run directory is written; the trainer is not started.
- No Modal. Training runs on this Mac with `device="mps"`.
- Results and figures under `figures/rung4/distill/`, each file naming the command that wrote it; `figures/AGENTS.md`, `scripts/AGENTS.md`, `deepcfr/AGENTS.md`, the wiki and `docs/notes/rung4-run-log.md` updated in the same change.
- Commit subjects `feat(rung4): <the finding>`; never `git stash`.

## Review Focus

1. A teacher row whose sum is zero plays uniform; its target must be a distribution (1/w each), not NaN. Pinned in Task 2 (`test_targets_are_distributions_everywhere`).
2. Rows with weight 0 must never be drawn, and a sampler handed all-zero weights must refuse by name rather than divide by zero. Pinned in Task 3.
3. The deal multiplicity must sum, per shape, to the count of physical pictures, derived independently. Pinned in Task 2.
4. The frozen student reloaded from its `.npz` must equal the in-memory freeze, and must be a distribution at every key (the `FrozenStrategy` constructor refuses otherwise). Pinned in Task 4.
5. The results file must label hands per seat correctly: teacher = iteration × workers (W = 10 from the checkpoint), Deep CFR = t × 10,000 from its grades file. Pinned in Task 5.

---

### Task 1: Lift the shared per-point prediction loop into `net.predict_by_point`

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/net.py` (add after `predict`)
- Modify: `src/drawmaha_solver/deepcfr/materialise.py:680-691`, `src/drawmaha_solver/deepcfr/policy.py:552-565`
- Test: `tests/deepcfr/test_net.py`

**Interfaces:**
- Produces: `predict_by_point(net: Net) -> Iterator[tuple[int, np.ndarray]]` yielding, per public point p in walk order, the (rows_p, width_p) float32 outputs gathered at the point's legal columns.

- [ ] Step 1: test `test_predict_by_point_matches_encode_key_by_key` in `tests/deepcfr/test_net.py`: for the two sample keys per point (restate `some_keys` from `test_policy.py`), `predict_by_point` row equals `predict(net, encode(key)[None])[0][[int(a) for a in key.legal_actions()]]`; the yielded widths equal `layout().width[p]` and the row counts `np.diff(layout().row_offset)`.
- [ ] Step 2: run it, expect `ImportError`.
- [ ] Step 3: implement; rewrite the loops in `materialise` and `policy_strategy` over it. Docstrings say why it exists (three callers had the same loop).
- [ ] Step 4: `uv run --extra deep pytest tests/deepcfr/test_net.py tests/deepcfr/test_materialise.py tests/deepcfr/test_policy.py -q` passes.
- [ ] Step 5: commit `refactor(rung4): one per-point forward pass shared by materialise, Π and the distillation freeze`.

### Task 2: The dataset: targets, deal multiplicity, reach weights

**Files:**
- Create: `src/drawmaha_solver/deepcfr/distill.py`
- Modify: `src/drawmaha_solver/deepcfr/average.py:298-328` (split `own_reach`)
- Test: `tests/deepcfr/test_distill.py` (no torch for this task), `tests/deepcfr/test_average.py` (one pin)

**Interfaces:**
- Produces in `average.py`: `reach_of_strategy(sigma: np.ndarray, *, seat: int) -> np.ndarray` — (rows,) own reach at the seat's rows from a flat probability array in column order; `own_reach(table, seat=)` becomes `reach_of_strategy(current_strategy(table), seat=seat)`.
- Produces in `distill.py`:
  - `teacher_targets(teacher: FrozenStrategy) -> np.ndarray` (rows, MAX_WIDTH) float32, padding 0.
  - `deal_multiplicity() -> np.ndarray` (rows,) int64: the number of physical pictures each row's canonical private key stands for, cached.
  - `row_weights(teacher: FrozenStrategy) -> np.ndarray` (rows,) float64 = reach (its seat's) × multiplicity, normalised to sum 1.

- [ ] Step 1: tests (torch-free):
  - `test_reach_of_strategy_equals_own_reach_on_a_random_table` in `test_average.py`.
  - `test_targets_are_distributions_everywhere`: on `uniform_strategy()` every row's first `width` entries are 1/width and the rest 0; on a random-sum strategy the legal entries equal `teacher[key]` for the sample keys.
  - `test_multiplicity_sums_to_the_physical_pictures_per_shape`: Σ multiplicity over the rows of one point of each shape equals C(15,3)·12 = 5,460 for (1 board, 0 discards), C(15,3)·12·11 = 60,060 for (2, 0), C(15,3)·12·11·10 = 600,600 for (2, 1); every value in {1, 2, 3, 6}.
  - `test_row_weights_are_zero_exactly_where_the_teacher_never_arrives`: build a strategy that folds every round-1 first action for seat 0 with probability 1; every later seat-0 row has weight 0; first-own-action rows have weight ∝ multiplicity; sum is 1.
- [ ] Step 2: run, expect import errors.
- [ ] Step 3: implement. Multiplicity: for each private key of each shape, count distinct images of `(frozenset(hole), frozenset(discarded), board)` under `SUIT_RELABELLINGS` via `cards.relabel`; expand per point by `np.repeat` through the layout.
- [ ] Step 4: `uv run pytest tests/deepcfr/test_distill.py tests/deepcfr/test_average.py -q` passes.
- [ ] Step 5: commit `feat(rung4): the distillation dataset — teacher distributions, deal multiplicity and own-reach weights`.

### Task 3: The fit

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/distill.py`
- Test: `tests/deepcfr/test_distill.py` (torch section)

**Interfaces:**
- `DistillConfig(frozen dataclass)`: `width=256, depth=4, steps=100_000, batch=16_384, lr=1e-3, clip=1.0, seed=0, device="mps", train_fraction=1.0, eval_every=5_000`; `__post_init__` refuses steps/batch < 1, lr/clip not > 0, fraction outside (0, 1], unknown device.
- `ReachSampler(weights: np.ndarray, *, seed: int)`: `draw(n) -> np.ndarray int64` via a float64 cumulative sum and `searchsorted`; refuses all-zero weights by name; rows of weight 0 never drawn.
- `training_rows(config, rows: int, *, seed) -> np.ndarray | None`: the random subset for `train_fraction < 1`, else None.
- `Curve = list[dict]` entries `{"step", "loss", "rw_error", "rw_error_unseen"?}`.
- `distill(targets, weights, config, *, log: Callable[[str], None] = print) -> tuple[Net, Curve]`: cosine lr from `config.lr`, Adam, clip; CE over legal columns with illegal logits at −1e9; eval every `eval_every` steps computes `reach_weighted_error` on device over all rows (and over unseen rows for a fraction arm).
- `reach_weighted_error(student_probs: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float` = Σ w · mean_legal|p − q| / Σ w (the `rw_mad` of `pi_reach_weighted.py`, restated over the padded layout).

- [ ] Step 1: tests (`torch = pytest.importorskip("torch")`):
  - `test_sampler_never_draws_a_zero_weight_row_and_refuses_all_zero`.
  - `test_cross_entropy_is_minimised_at_the_teacher`: logits = log(target) at legal columns (−1e9 elsewhere) gives loss = the targets' entropy and zero gradient.
  - `test_a_tiny_fit_memorises_a_slice`: targets for all rows, weights nonzero on 64 rows of three widths only, `DistillConfig(width=32, depth=2, steps=300, batch=64, device="cpu", eval_every=100)`; the final `rw_error` under 0.05 (300 Adam steps on 64 rows; justified in a comment) and the curve has 3 eval entries with `step` 100, 200, 300.
  - `test_fraction_arm_reports_unseen_error_and_is_seeded`: `train_fraction=0.5` returns a subset of half the rows, the same for the same seed, and the curve carries `rw_error_unseen`.
- [ ] Step 2: run, expect failures.
- [ ] Step 3: implement, on the `_fit_on_device` pattern (layout halves to the device once, features gathered per batch). `device="cpu"` runs the same loop on the cpu (so the tests run without a GPU).
- [ ] Step 4: `uv run --extra deep pytest tests/deepcfr/test_distill.py -q` passes; `uv run pytest tests/deepcfr/test_distill.py -q` (no torch) skips the torch tests and passes the rest.
- [ ] Step 5: commit `feat(rung4): the distillation fit — reach-proportional minibatches, masked cross-entropy, cosine Adam`.

### Task 4: The freeze and the file

**Files:**
- Modify: `src/drawmaha_solver/deepcfr/distill.py`
- Test: `tests/deepcfr/test_distill.py`

**Interfaces:**
- `student_strategy(net: Net, *, teacher: StrategyInfo) -> FrozenStrategy`: softmax of the outputs at each key's legal columns via `predict_by_point`; `info = StrategyInfo(rule="distill", column=teacher.column, iteration=teacher.iteration, seed=teacher.seed, workers=teacher.workers)`.
- `student_probabilities(net, device) -> torch.Tensor` (rows, MAX_WIDTH) on device, used by the eval inside `distill` (Task 3 may stub it; this task makes it the one used by both).

- [ ] Step 1: tests: `test_student_strategy_is_the_softmax_at_every_key` (sample keys vs `encode` + softmax, atol 1e-6; `save_strategy` then `load_strategy` gives equal `probabilities`); `test_student_strategy_and_on_device_probabilities_agree` (cpu device, atol 1e-6).
- [ ] Step 2: run, expect failures. Step 3: implement. Step 4: pass.
- [ ] Step 5: commit `feat(rung4): the distilled student frozen as a strategy the exact grader reads`.

### Task 5: The script that runs the arms

**Files:**
- Create: `scripts/deepcfr/distill_teacher.py`
- Modify: `tests/test_rung3_page_data.py` (add the script to `SCRIPTS` compile check only), `scripts/AGENTS.md`, `figures/AGENTS.md`, `src/drawmaha_solver/deepcfr/AGENTS.md`
- Test: `tests/deepcfr/test_distill_script.py` (torch-free: `importlib` the script; test `arm_record` and `references`)

**Interfaces:**
- CLI: `uv run --extra deep python scripts/deepcfr/distill_teacher.py TEACHER.npz --out runs/distill --results figures/rung4/distill/results.json [--arms a,b,c]`.
- `ARMS: dict[str, DistillConfig]` = `3x64`, `3x128`, `4x256`, `4x512`, `4x1024`, `4x256-25k` (steps 25,000), `4x256-unweighted`, `4x256-f0.1`, `4x256-f0.5`; a name already in the results file is skipped.
- `references(teacher_grade: Grade, checkpoint: Path, deep_grades: Path) -> dict`: `_teacher` (grade, iteration, workers, hands_per_seat = iteration × workers, checkpoint name), `_deep_cfr` (exact / pi / current at t = 1000 and 2000 from `figures/rung4/pot_1-256x4-s16000-lr5e-4.json`, with hands_per_seat).
- Per arm record: config fields, `parameters`, `fit_s`, `grade_s`, `br0`, `br1`, `exploitability`, `value_p0`, `rw_error`, `rw_error_unseen` (fraction arms), `curve`, `strategy_file`, `strategy_sha256`.
- The teacher is graded once into `_teacher` (2 min) unless already present; each student is saved under `runs/distill/<arm>.npz`, reloaded, and graded from the file.

- [ ] Step 1: tests: `test_references_label_hands_per_seat_from_the_files` (a fake lockstep header via `read_lockstep` monkeypatch is overkill: pass `workers`/`iteration` explicitly and assert 10,000,000 × 10 = 100,000,000 and Deep CFR 2,000 × 10,000 = 20,000,000 read from the committed file); `test_arm_names_parse_to_their_sizes`.
- [ ] Step 2–4: implement, tests pass, `uv run pytest tests/test_rung3_page_data.py tests/test_agent_docs.py -q` pass after the AGENTS edits.
- [ ] Step 5: commit `feat(rung4): distill_teacher.py runs the size, ablation and fraction arms and grades each copy exactly`.

### Task 6: Run the arms

- [ ] Smoke: `--arms 3x64` with `--steps-override 2000` is not a flag; instead run `3x64` fully first (~5 min) and check the record and the grade against the teacher.
- [ ] Run the rest in the background with `caffeinate -i`, headline first: `4x256`, `4x1024`, `4x512`, `3x128`, `4x256-25k`, `4x256-unweighted`, `4x256-f0.5`, `4x256-f0.1`. Log to `.context/distill-run.log`.
- [ ] Sanity on the first result: a copy below the teacher by more than the grader's noise (1e-4) is investigated, not reported.

### Task 7: Figures

**Files:**
- Create: `scripts/deepcfr/distill_figures.py` (torch-free; added to `SCRIPTS` import-without-torch check)
- Outputs: `figures/rung4/distill/by_size.png`, `gaps.png`, `curves.png`, `fraction.png`

- [ ] Read the `dataviz` skill before the first line of chart code.
- [ ] by_size: exploitability (log y) against parameter count (log x) for each distilled copy; horizontal lines for the teacher, Deep CFR exact (t = 2000) and Π. gaps: a three-bar/arrow figure for 4 × 256 (teacher → copy → Deep CFR). curves: per arm, loss and rw_error against step, the grade annotated at the end. fraction: grade against training fraction with seen/unseen error.
- [ ] Commit `feat(rung4): the distillation figures`.

### Task 8: Docs, wiki, run log, README

- [ ] `docs/notes/rung4-run-log.md`: finding 9 with the table of arms; `README.md` rung-4 bullets: one bullet with the three gaps; wiki: a `research/distillation.md` page (via the `drawmaha-wiki` skill) and a row in `deepcfr-training.md`'s studies table; `nav.json`; regenerate static pages; `tests/deepcfr/test_rung4_quotes.py` extended if the README table gains numbers.
- [ ] `uv run pytest` (whole suite) and `uv run --extra deep pytest tests/deepcfr -q`.
- [ ] Commit `feat(rung4): …<the finding in one line>`.

### Task 9: PR and the preprint

- [ ] `/open-pr` from this branch to `main`; drive to merge-ready (checks, claude[bot] findings).
- [ ] Preprint in `/Users/alec/Desktop/Claude/poker/rung4-distill/`: `distill.tex` (arXiv article style, two columns not required), figures copied from `figures/rung4/distill/`, compiled with tectonic; print absolute paths.
