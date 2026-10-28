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
(`deep-analysis`, `figures/rung4/`) and the README section. The package
itself is complete; what remains is the paid run and its grades.

## Rules

- **torch is optional and stays optional.** The loop never names torch;
  `NetLearner` imports `fit`, `materialise` and `net` inside its methods.
  Every test that needs torch opens with `pytest.importorskip("torch")`, and
  the default suite passes on `uv sync` alone. Real iterations on the whole
  game sit behind `MINIDRAWMAHA_FULL_DEEP=1`.
- **The oracle pin is the correctness gate.** With `OracleLearner` the loop
  must reproduce rung 3's LCFR table bit for bit over three iterations. A
  change to the loop's order, to the float32 cast in `collect.rows_of`, or to
  `regret_rules.bank_rows` breaks that pin. If it breaks, the loop is wrong,
  not the test.
- **A learner's `write` assigns into `table.cumulative_regret` and never
  rebinds it.** The collector's workers read that array through a shared
  block; a rebound array leaves them walking stale regrets.
- **Materialise runs before every phase, twice per iteration**, because seat
  1's walks sample seat 0's freshly refit net. Observers are told which seat's
  σ is about to be sampled as the opponent.
- **The resume state** is the three memories, the two latest nets, and a
  `state.json` written last and atomically as the commit point, every file
  named by its iteration. A resume may change the worker count, which
  changes nothing, and the refit's device (cpu or mps), which changes the
  nets from the move on and is let through on purpose so a run can move to
  the GPU; a moved run is no longer reproducible from its seed and config.
  The seed, the walks, the capacity, the net's size, the other refit fields
  (the pot power among them) and the thread count are refused by name,
  because each changes the sample or the fit.
- **A field added to `DeepConfig` after runs were saved goes in
  `ADDED_LATER`** with the value those runs had, so a `state.json` that lacks
  it reads as the run it was and never as today's default. `pot_power` is
  the case that made the rule: its default is ½, every run saved before it
  existed fitted raw chips, and `DeepConfig(**saved)` alone would have
  resumed them under ½ with nothing in the log.
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
- A memory row is 24 bytes (`row: int32`, `t: int32`, `target: float32 × 4`);
  the run's reservoirs are 50 million rows, 1.2 GB each. A capacity change is
  a different run, not a resume.

## Running it

`scripts/deepcfr/run_deep_cfr.py` is argparse over `DeepConfig`; its
docstring opens with the smoke command, and re-running the same `--out`
resumes. Ctrl-C or SIGTERM finishes the iteration and saves. Measured
per-iteration timings are in `train.py`'s docstring, not here. On Modal the
same loop is `scripts/deepcfr/modal_rung4.py`; a grid mark is graded three
ways by its `grade_mark_dir` — the exact average's sums, Π fit fresh on the
mark's policy memory, and the current iterate over the mark's nets — into
`figures/rung4/grades.json`, which `deep-analysis` reads. The grader needs
`NetLearner.fit_config` (Π's fit shares the run's refit config) and nothing
else of this package that the loop does not already expose.
