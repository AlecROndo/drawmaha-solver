# Rung 3 kickoff refactor — implementation plan

> **For agentic workers:** five tasks, disjoint files, run in parallel (superpowers:dispatching-parallel-agents). Each task: audit against the standards, then fix test-first.

**Goal:** PR #45 (`rung3-lockstep-parallel`) and PR #46 (`rung3-analysis-play`) both meet the engineering-kickoff standards in `~/.claude/skills/engineering-kickoff/references/house-style.md`, with no change in behaviour except bugs found along the way, each pinned by a test that failed first.

**Done means:**
1. Every file the two PRs add or change has been read top to bottom against house-style §A–§J, and every violation is fixed or recorded as deliberate (with the reason) in the agent's report.
2. `uv run pytest -q` passes on both branches; the full suite ran after the final merge.
3. Every Modal checkpoint already on disk (`runs/modal/<rule>/iter-*.npz`) still loads, and the release strategy (`rung3-lcfr-2m`, `STRATEGY_SHA256`) still loads and grades 0.061346.
4. `uv run minidraw-analysis` regenerates `figures/rung3/` with identical numbers.
5. Both PRs pushed and driven back to merge-ready; #46 still stacked on #45.

**Architecture:** five agents in isolated worktrees, each on its own branch cut from the PR branch it touches, each owning a disjoint file set. I merge their branches into the PR branches (merge, never rebase), merge #45 into #46, run the full suite, push.

## Global constraints

- **The repo's own conventions win where they conflict with the house style** (house-style preamble): long explanatory prose module docstrings, one blank line between top-level defs, 75-hyphen section banners as already used in `mccfr.py` / `exploitability.py`.
- **Formats are frozen:** checkpoint `.npz` keys (`regret`, `strategy_sum`, `extra_sums`, `stamps`, `widths`, `iteration`, `seed`, `workers`, `rule`, `averages`), the strategy file's keys, `STRATEGY_SHA256`, the release asset, `grades.json` keys, and every CLI flag and entry point name.
- **Numbers are frozen:** lockstep runs stay bit-for-bit identical (`test_the_packed_apply_equals_the_per_ledger_apply`, resume-equals-uninterrupted tests).
- **Fixes go test-first:** a behaviour change starts with a failing test, which the agent watches fail.
- No new dependencies, no Modal launches, no pushes by the agents, never `git stash`, never rebase.

## Review focus

1. A refactor that reorders the operations in `apply_arrays` / `bank_rows` changes float rounding, so the bit-for-bit tests are the gate; never relax them.
2. Moving `Scoreboard` / `QuitGame` / `verdict` out of `leduc/play.py` must leave `leduc-play` and `tests/leduc/test_play.py` working unchanged.
3. `parallel.py` shared-memory lifetime: a refactor must not leak `SharedMemory` segments when a worker raises (a test should kill a worker and check cleanup).
4. `modal_rung3.py` claim protocol: `_claim`, `_settle` and `_continues` keep their tested semantics (`tests/minidrawmaha/test_modal_rung3.py`).
5. `analysis.py` figure titles stay derived from the data, not hard-coded (review round on 5c3c425).

---

### Task 1: lockstep core (#45)
**Files:** `src/drawmaha_solver/minidrawmaha/lockstep.py`, the `bank_rows` section of `regret_rules.py`, the `packed_table.py` additions, `tests/minidrawmaha/test_lockstep.py`, `tests/minidrawmaha/test_regret_rules.py`.
**Interfaces it keeps:** `new_lockstep`, `train_lockstep`, `save_lockstep`, `load_lockstep`, `read_lockstep`, `is_lockstep`, `walk`, `apply`, `apply_arrays`, `flatten`, `stream`, `Recorded`, `Flat`, `LockstepSolve`, `bank_rows` (same signatures, or every caller updated in the same task).

### Task 2: process driver (#45)
**Files:** `src/drawmaha_solver/minidrawmaha/parallel.py`, `src/drawmaha_solver/minidrawmaha/lockstep_run.py`, `scripts/run_lockstep.py`, `tests/minidrawmaha/test_parallel.py`, `tests/minidrawmaha/test_lockstep_run.py`.
**Interfaces it keeps:** everything `scripts/modal_rung3.py` imports from these modules.

### Task 3: Modal launcher and grader script (#45)
**Files:** `scripts/modal_rung3.py`, `tests/minidrawmaha/test_modal_rung3.py`, `scripts/grade_minidrawmaha_checkpoint.py`.

### Task 4: frozen strategy and play (#46)
**Files:** `src/drawmaha_solver/minidrawmaha/strategy.py`, `scripts/freeze_minidrawmaha_strategy.py`, `src/drawmaha_solver/minidrawmaha/play.py`, `tests/minidrawmaha/test_strategy.py`, `tests/minidrawmaha/test_play.py`, plus the terminal-table pieces rung 3 borrows from `src/drawmaha_solver/leduc/play.py` (`QuitGame`, `Scoreboard`, `verdict`): move them to a shared module named for its job, and update `leduc/play.py` and its tests.

### Task 5: analysis (#46)
**Files:** `src/drawmaha_solver/minidrawmaha/analysis.py`, `tests/minidrawmaha/test_analysis.py`, `figures/rung3/*`. Regenerates the figures with `uv run minidraw-analysis` and confirms `answer_sheet.json`'s numbers are unchanged.

**Every task, the same steps:**
- [ ] Read house-style.md, then the owned files top to bottom; write down every violation (§A comments, §B layout and banners, §C value types, §D keyword-only and function shape, §E orchestrators, §F fail loud, §G naming, §I slop, §J depth).
- [ ] For each behaviour change: write the failing test, run it to see it fail, fix, run it to see it pass.
- [ ] Apply the structural fixes (order, banners, docstrings, frozen/slots dataclasses, keyword-only parameters, dead code, duplication).
- [ ] `uv run pytest tests/minidrawmaha -q` (plus `tests/leduc` for Task 4) green.
- [ ] Commit in the repo's `refactor(rung3): ...` / `fix(rung3): ...` style.
- [ ] Report: each finding with a disposition (fixed in SHA / kept deliberately because ...), the test count, and the branch name.
