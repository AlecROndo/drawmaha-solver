# `scripts/` — hand-run entry points

Grouped by what they serve. Every script is run from the repo root, and its
docstring opens with the exact command to run it.

- `site/` — `vercel_build.sh`, the build command `vercel.json` names, and
  `serve_site.py`, which serves `public/` with the same rewrites, stdlib only.
- `leduc/` — `generate_referee.py` regenerates `tests/leduc/referee.json`
  from OpenSpiel.
- `minidrawmaha/` — rung 3's race and its outputs: `generate_census.py`
  (regenerates `tests/minidrawmaha/census.json`), `run_lockstep.py` (one
  rule on this machine), `modal_rung3.py` (the four-rule race on Modal),
  `grade_checkpoint.py` (the exact meter on a checkpoint),
  `freeze_strategy.py` (one averaging column into the release file).
- `deepcfr/` — `run_deep_cfr.py`, Deep CFR on this machine, resumable;
  `modal_rung4.py`, the same run on Modal (`train`), one grader per grid
  mark (`grade`, writes `figures/rung4/grades.json`), the `smoke`, the
  net-size `sweep` and `status`; `rule_sign_flips.py`, how much a
  per-sample regret rule bends a run's advantage memory (writes
  `figures/rung4/rule_sign_flips.json`).

## Rules

- **Tests find scripts by path.** `tests/minidrawmaha/test_modal_rung3.py`,
  `tests/deepcfr/test_modal_rung4.py` and `tests/deepcfr/test_train.py` load
  a script with `importlib`, and
  `tests/test_vercel_config.py` reads the `rungs=(…)` list out of
  `site/vercel_build.sh`. Moving or renaming a script means updating them,
  `vercel.json`, `README.md`, and the `_source` string the two generators
  write into their fixtures, in the same change.
- **`open_spiel` is never a dependency.** `leduc/generate_referee.py` is the
  only file that imports it, nothing imports that script, and it runs under
  `uv run --with open_spiel --with cvxpy --with ecos --no-project`. The Vercel
  build must never pull its wheels.
- **Modal is deployed before it is run.** `modal deploy` first, then
  `modal run …::main`: `main` spawns on the deployed app, so a run outlives
  the terminal and can spawn its own continuation across Modal's 24-hour cap.
  One container trains one rule, writing to the volume `drawmaha-rung3`
  under `<rule>/`, and a heartbeat in `<rule>/owner.json` keeps a second
  trainer off a rule that is live. Rung 4 is the same shape on the volume
  `drawmaha-rung4`: one trainer per run (the main run under deep/, the sweep
  under sweep/<size>/) with the claim helpers lifted rather than imported,
  and graders fanned out one per mark. It costs money; only a person
  launches it.
- **The `rungs=(…)` list in `site/vercel_build.sh` and the rewrites in
  `vercel.json` move together**; a surface added to one without the other is
  the silent break `tests/test_vercel_config.py` exists to catch.
- Trainers take `--out` for the run directory and resume when re-run with the
  same one; Ctrl-C or SIGTERM finishes the iteration and saves.
