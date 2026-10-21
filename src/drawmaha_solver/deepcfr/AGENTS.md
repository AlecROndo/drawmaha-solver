# `deepcfr/` — rung 4: Deep CFR on mini-drawmaha

Replaces the regret half of rung 3's ledgers with a net. The walk is rung 3's
`lockstep.walk`, unchanged: `materialise` writes each net's outputs into the
packed table's regret slots, and the walk plays them through the ledger's own
`strategy()`. The spec, the gates and the PR ladder are in
`docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`. PRs 1 to 4 (the
encoder, the net with materialise and the refit, the reservoir with the
collector, the train loop) are merged; PR 5 (the exact running average and the policy net Π, as the `average`
and `policy` modules) and PR 6 (the Modal launcher,
the net-size sweep, the figures and the README section) are next.

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
  named by its iteration. A resume may change the worker count; the seed, the
  walks, the capacity, the net's size, the refit fields and the thread count
  are refused by name, because each changes the sample or the fit.
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
per-iteration timings are in `train.py`'s docstring, not here.
