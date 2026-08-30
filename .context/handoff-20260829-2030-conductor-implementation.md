# Handoff breadcrumb — implementation/PR jobs moving to Conductor (2026-08-29 ~20:30)

Facts from the live session that are NOT rediscoverable from the repo. Everything
else (game spec, plan, research, paper) IS in the repo — verify there.

## Grill ledger delta (vs .context/handoff-20260829-grill-state.md)
- Q9 evaluation story: LOCKED. Per rung: exact exploitability on toys (checked
  against OpenSpiel reference values); mini-drawmaha exact best-response walk as
  ground truth + measure the Deep-CFR-vs-tabular gap there; full game =
  duplicate-dealt head-to-head checkpoint ladder (+AIVAT later) + a trained RL
  exploiter (PPO/DQN vs frozen net) reported as an exploitability LOWER BOUND
  (vacuousness caveat stated). Units: total chips across BOTH pot halves, per
  hand, in bb; per-half EV + scoop rate diagnostics; duplicate dealing in the
  match runner from day one; bootstrap CIs on every claimed difference.
  (Replaces scripted LBR with the trained exploiter — LBR draw-game adaptation
  is an open problem, see paper Section 14F.)
- Working mode: Claude proposed Alec hand-writes algorithmic cores with
  spec+failing-tests handed to him (TA model). NOT confirmed — Alec instead had
  Claude build all of rung 0. Still-open grill branches: working mode
  confirmation, mini-drawmaha exact design (deck/hand size), timeline/pacing.

## PR state
- PR #1 https://github.com/AlecROndo/drawmaha-solver/pull/1 (branch
  alec/rung0-rps): complete v1 rung-0 implementation, verified merge-ready
  (repo has NO CI and NO review bots — gate checked empty twice). Alec asked
  for a redo via the engineering-kickoff workflow, so PR #1 is to be SUPERSEDED:
  once the v2 PR is open, close #1 with a comment pointing at it. Its branch
  also carries README additions (eval-protocol + rung-0 sections with measured
  numbers) — reuse that text for the v2 README task.
- v1 measured numbers (for README/report comparison): self-play 100k iters avg
  (0.3327, 0.3371, 0.3302), exploitability 0.00443 chips/round; vs (0.5,0.25,
  0.25) → 99.9% paper, +0.2366/round.

## v2 redo state (the job being handed to Conductor)
- Committed plan: docs/superpowers/plans/2026-08-29-rung0-rps-redo.md on branch
  alec/rung0-rps-v2 (pushed). It is fully self-contained: complete file
  contents per task, TDD steps, palette hexes, commit messages.
- Execution protocol chosen: engineering-kickoff -> superpowers:writing-plans
  (done) -> superpowers:subagent-driven-development (tasks 1-3 sequential,
  4+5 parallel implementers with git forbidden/coordinator commits, task 6
  coordinator) -> open-pr skill with full merge-readiness gate.
- SDD scratch (LOCAL-ONLY, git-ignored) at
  /Users/alec/Claude Code/poker/drawmaha-solver/.superpowers/sdd/2026-08-29-rung0-rps-redo/
  progress.md holds the pre-flight conflict-scan table + rulings (no worktree;
  model tiers: implementers haiku, task reviews sonnet, final review most
  capable). Briefs task-{1..5}-brief.md are mechanical extracts of the plan —
  regenerate with the skill's scripts/task-brief if working from a fresh
  worktree.
- Task 1 was dispatched to a haiku implementer in the old session; it ran ~1h
  producing NO commits, NO report, NO tree changes, and was killed at handoff.
  Task 1 restarts from scratch. No other task started.

## Standing constraints
- Commits: conventional format `type(scope): imperative` (regex in
  ~/.claude/skills/open-pr/references/commit-conventions.md), NO Claude
  co-author attribution (user settings enforce).
- Figure titles must state the finding; light-mode reference palette hexes are
  in the plan's Global Constraints.
- Do not touch branch alec/rung0-rps (v1 evidence) or merge anything without
  Alec's say-so.

## Unrelated open items (do not pick up unless asked)
- Survey-paper two-critic review pass never completed; PDF not yet copied to
  ~/Desktop/Claude/poker/.
