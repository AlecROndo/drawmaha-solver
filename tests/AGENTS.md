# `tests/` — the suite is the CI

No GitHub workflow runs tests. `uv run pytest` is the gate, and you run it
before every PR. One directory per rung, plus the checks on the static site
and on these instruction files at the top level. The full default suite takes
about seven minutes on a laptop; `-x` with one rung's directory is the fast
loop.

## Rules

- **`--import-mode=importlib`** is set in `pyproject.toml` because every rung
  has its own `test_game.py`. So test modules never import one another
  (shared helpers are restated, and say so), and there are no packages, and no
  `__init__` files, under `tests/`.
- **Spawned workers unpickle by module name.** A test that hands a deal or a
  job to the parallel collector defines it at module level and prepends the
  repo root to `sys.path` first (`minidrawmaha/test_parallel.py`,
  `deepcfr/test_collect.py`); a lambda or a closure cannot reach a worker.
- **Fixtures are regenerated, never edited.** `leduc/referee.json` comes from
  OpenSpiel through `scripts/leduc/generate_referee.py`;
  `minidrawmaha/census.json` from `scripts/minidrawmaha/generate_census.py`.
  The tests re-derive every number in them a second way, so a fixture that
  merely asserted itself would pass while the solver rotted.
- **Convergence thresholds are seeded and justified** in a comment beside
  them. Loosening one to make a test pass is a finding to report, not a fix.
- **Slow and heavy work is gated by an environment variable named in the
  skip reason**: `MINIDRAWMAHA_FULL_TABLE`, `MINIDRAWMAHA_FULL_GRADE`,
  `MINIDRAWMAHA_FULL_CALIBRATION`, `MINIDRAWMAHA_FULL_DEEP`. Tests that need
  torch `importorskip` it. The default suite must keep passing without torch,
  Modal, OpenSpiel or a network download.
- **Committed numbers are recomputed, not transcribed.** `minidrawmaha/test_play.py`
  reads `figures/rung3/grades.json`; `minidrawmaha/test_range_export.py` reads
  the export under `web/solver-viz/public/mini/`; `kuhn/test_exploit_server.py`
  recomputes the README's exploit table from the committed solve. A re-export
  changes a test's inputs, never its assertions.
- The static-site tests (`test_ui_standard.py`, `test_shared_chrome.py`,
  `test_cover_page.py`, `test_vercel_config.py`) enforce `web/DESIGN.md` and
  the deploy contract. A deliberate motion exception is allowlisted by
  selector, with its reason, in `EXPLANATORY` inside `test_ui_standard.py`.
- `test_agent_docs.py` checks every `AGENTS.md`: each has a `CLAUDE.md`
  shim beside it, and every repo path it names in backticks exists.
