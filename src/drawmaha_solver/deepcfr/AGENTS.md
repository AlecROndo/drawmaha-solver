# `deepcfr/` — rung 4: Deep CFR on mini-drawmaha

Replaces the regret half of rung 3's ledgers with a net. The walk is rung 3's
`lockstep.walk`, unchanged: `materialise` writes each net's outputs into the
packed table's regret slots, and the walk plays them through the ledger's own
`strategy()`. The spec, the gates and the PR ladder are in
`docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`. PRs 1 to 5 (the
encoder, the net with materialise and the refit, the reservoir with the
collector, the train loop, the exact running average and the policy net Π
as the `average` and `policy` modules) are merged, and so is PR 6: the
Modal launcher `scripts/deepcfr/modal_rung4.py` (train, grade, smoke, sweep,
status), the price figure in `minidrawmaha/answer/analysis.py`
(`deep-analysis`, `figures/rung4/`) and the README section. The Deep CFR
loop itself is complete. On top of it sits the targeted-sampling plan's
boost: `targeting` decides which rows to boost, and `boost` is the stage
`train.step` runs between each seat's walks and the merge — one sampled
boosted row per target, at the weight the walk's samples would have
carried. What remains of that plan is the race (its PR 5).

## Rules

- **No module here imports the grader.** `targeting` ranks rows by how
  much the strategy leaks there, and the leak is measured by
  `minidrawmaha/exploitability.py`, the meter that also grades the run. So
  `targeting` takes the leak and the frequency as arrays and the draw weight
  as a callable, and the one file that runs the meter for it is
  `scripts/deepcfr/targeting_refresh.py`. `tests/deepcfr/test_targeting.py`
  reads every import statement in this package for the grader's name or
  `answer.analysis`, and imports `targeting` in a fresh interpreter to check
  that neither loads. The name check is not transitive: `average`, `distill`,
  `policy` and `train` reach `exploitability` through `answer.strategy`, which borrows
  the grader's row-sum tolerance — a constant, not a walk.
- **A target set never hands back a row twice in one draw**: a repeated row
  would get two boosted rows and double its expected weight. Draws are
  exponential keys on their own stream (`targeting.target_stream`, fifth
  word "TARG"), never the walk's, so they move neither the walk nor the
  reservoir; `boost.inject` refuses a repeat a second time.
- **The boost never computes the exact sum.** A boosted row is a sampled
  play-out from its row (`boost_target.Sampling`; by default the walk's own
  sampling against the whole opponent range), offered at
  (1 − α) · K · arrival — the sampled weight it stands in for — AFTER the
  walk's rows, which are turned down to α and still offered, so within a
  phase the walk's rows keep their stream positions; a run with
  `target=None` is today's run bit for bit, a boosted run's memory equals a
  plain one's in expectation, not row for row. Unbiased always; quieter
  only where one play-out's variance is below a walk sample's over K · p,
  which the race measures. `DeepConfig` refuses an exact `Sampling` and
  `inject` refuses it again; `EXACT` is the tests' oracle. The rule:
  boosting is extra sampling at high-variance nodes, never brute-forcing
  the tree from a node, which costs a hundredfold and sets aside what the
  net is for. A drawn row whose float32 weight would
  equal 1 or α, or whose arrival is 0, is left untreated (natural rows at
  weight 1, no boosted row) and counted as skipped: `targeting.noise` tells
  walk rows from boosted rows by weight alone, and half a treatment would
  bias the row. σ for the boost is read from the collector's table right
  after the phase's `learner.write`, the same array the walks read.
- **torch is optional and stays optional.** The loop never names torch;
  `NetLearner` imports `fit`, `materialise` and `net` inside its methods.
  Every test that needs torch opens with `pytest.importorskip("torch")`, and
  the default suite passes on `uv sync` alone. Real iterations on the whole
  game sit behind `MINIDRAWMAHA_FULL_DEEP=1`.
- **The oracle pin is the correctness gate.** With `OracleLearner` the loop
  must reproduce rung 3's LCFR table bit for bit over three iterations. A
  change to the loop's order, to the float32 cast in `collect.rows_of`, to
  `regret_rules.bank_rows`, or to where the row weight multiplies (it scales
  each sample in float64 before LCFR's per-t′ sum, so weight one is exact)
  breaks that pin. If it breaks, the loop is wrong, not the test.
- **A learner's `write` assigns into `table.cumulative_regret` and never
  rebinds it.** The collector's workers read that array through a shared
  block; a rebound array leaves them walking stale regrets.
- **Materialise runs before every phase, twice per iteration**, because seat
  1's walks sample seat 0's freshly refit net. Observers are told which seat's
  σ is about to be sampled as the opponent.
- **The resume state** is the three memories, the two latest nets, the
  current target set when the run has an aim (`targets-t{t:05d}.npz`, so a
  resume draws from the set the run drew from rather than grading again),
  and a `state.json` written last and atomically as the commit point, every
  file named by its iteration. A resume may change the worker count, which
  changes nothing, and the refit's device (cpu or mps), which changes the
  nets from the move on and is let through on purpose so a run can move to
  the GPU; a moved run is no longer reproducible from its seed and config.
  The seed, the walks, the capacity, the net's size, the other refit fields
  (the pot power among them), the thread count, the aim (`target`) and the
  boost's sampling (`boost_sampling`) are refused by name, because each
  changes the sample or the fit.
- **A field added to `DeepConfig` after runs were saved goes in
  `ADDED_LATER`** with the value those runs had, so a `state.json` that lacks
  it reads as the run it was and never as today's default. `pot_power` is
  the case that made the rule: its default is ½, every run saved before it
  existed fitted raw chips, and `DeepConfig(**saved)` alone would have
  resumed them under ½ with nothing in the log. `target` (None) and
  `boost_sampling` (the default sampling) followed the same rule; both are
  dataclasses that travel as nested dicts in `state.json`, rebuilt by
  `config_from_saved`.
- **The refit's targets are each regret sample over its spot's pot^k**
  (`pot_scaled`, k = `pot_power`, ½ by default). The divisor is the spot's,
  never the sampled outcome's, so regret matching plays the same σ from the
  same regrets; what it changes is where the squared-error fit spends its
  accuracy, and so what the fitted net gets wrong and the σ actually played.
  A divisor that varied within a spot would move the spot's mean regret and
  be a different algorithm.
- **An observer that carries state has `name`, `save` and `load`**
  (`StatefulObserver` in `train.py`; `ExactAverage` in `average.py` is the one). Its file
  joins the resume set and every mark, named in `state.json`. A resume
  refuses to drop one the state names, and refuses an observer the state
  has no file for; a plain `load_run` of a mark needs no observer. The
  exact average is accumulated at the hook and cannot be rebuilt later
  except by replaying every kept net, so `scripts/deepcfr/run_deep_cfr.py` always passes it.
- **Own reach is in key space, with multiplicity.** A post-draw key has
  three predecessor draw keys (the key forgets which card was drawn); two
  that coincide are listed twice. The brute-force walk in
  `tests/deepcfr/test_average.py` is the gate; a one-row `previous_own_row`
  would pass most keys and be wrong at every throw.
- **Anything handed to the collector reaches a spawned worker by pickle**, so
  it must be importable by module name in a fresh process: a module-level
  function, never a lambda or a closure. The tests prepend the repo root to
  `sys.path` for exactly this reason.
- **The distillation never touches the walk, the memories or the grader.**
  Its weight is own reach × deal multiplicity under the teacher
  (`distill.row_weights`), applied by drawing rows in proportion to it;
  its loss is the softmax over the legal columns only, so the net keeps
  `Net`'s seven-output linear head and the freeze is the softmax at each
  key. `net.predict_by_point` is the one whole-game read `materialise`,
  Π's freeze and the distillation freeze share.
- A memory row is 28 bytes (`row: int32`, `t: int32`, `target: float32 × 4`,
  `weight: float32`); the run's reservoirs are 50 million rows, 1.4 GB each. A
  capacity change is a different run, not a resume. The weight is how many
  sampled rows the row stands for: 1 for every row a walk offers (what `merge`
  stamps when given none), and the targeted-sampling plan's boosted rows carry
  `(1 − α) · K · arrival`. The refit and the oracle weigh a row by t × weight,
  so a memory of weight-one rows is today's run bit for bit, and a memory file
  saved before the column existed loads with ones.

## Running it

`scripts/deepcfr/run_deep_cfr.py` is argparse over `DeepConfig`; its
docstring opens with the smoke command, and re-running the same `--out`
resumes. `--boost <arm>` turns the targeted-sampling boost on; the script
then builds the grader-backed refresher around the run's exact average and
hands the loop a `boost.Boost`, which the loop requires whenever
`config.target` is set and refuses when it is not. Ctrl-C or SIGTERM
finishes the iteration and saves. Measured per-iteration timings are in
`train.py`'s docstring, not here. On Modal the
same loop is `scripts/deepcfr/modal_rung4.py`; a grid mark is graded three
ways by its `grade_mark_dir` — the exact average's sums, Π fit fresh on the
mark's policy memory, and the current iterate over the mark's nets — into
`figures/rung4/grades.json`, which `deep-analysis` reads. The grader needs
`NetLearner.fit_config` (Π's fit shares the run's refit config) and nothing
else of this package that the loop does not already expose.
