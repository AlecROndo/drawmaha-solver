# Handoff breadcrumb — drawmaha-solver state + where-to-next (2026-08-30)

Conversation-only facts a fresh agent cannot rediscover from the repo.
Supersedes .context/handoff-20260829-grill-state.md where they conflict; that
file still holds Q1–Q8, purpose, compute estimates, and git-attribution rules.
(A third breadcrumb, handoff-20260829-2030-conductor-implementation.md, exists
only on branch alec/rung0-rps-v2 and describes a now-dead thread.)

## Grill ledger final state
- Q9 evaluation story: LOCKED. Per rung: exact exploitability on toys (checked
  vs OpenSpiel); mini-drawmaha exact best-response walk + Deep-CFR-vs-tabular
  gap; full game = duplicate-dealt head-to-head checkpoint ladder (+AIVAT
  later) + trained RL exploiter (PPO/DQN vs frozen net) reported as an
  exploitability lower bound. Units: total chips across both pot halves per
  hand in bb; per-half EV + scoop-rate diagnostics; bootstrap CIs. (Replaces
  scripted LBR — its draw-game adaptation is an open problem, survey §14F.)
- STILL OPEN grill branches, in priority order: (a) working mode — Claude
  proposed Alec hand-writes algorithmic cores TA-style; never confirmed; de
  facto this session Claude wrote all code and Alec learned via generated
  papers/explanations instead — worth re-asking explicitly before rung 1;
  (b) mini-drawmaha exact design (shrink the DECK, not just stacks — survey
  §14C); (c) timeline/pacing. End state: restate the whole plan with zero TBDs.

## Rung 0 / PR #1 state (verify, don't trust)
- PR #1 (branch alec/rung0-rps, 4 commits to e6233fb) is the LIVE rung 0:
  expected-utility ledger (u − ⟨σ,u⟩, deliberately CFR's counterfactual form),
  house-style pass, 38 tests, figures, in-repo writeup under writeups/rung0-rps/.
  Verified MERGEABLE/CLEAN; repo has NO CI and NO review bots (checked
  repeatedly). Merge decision is Alec's — not yet given.
- Known accepted gap: update() banks strategy_sum before numpy would raise on
  a wrong-length utilities vector (half-updated ledger); harden at rung 1.
  Top follow-up per the writeup: add a GitHub Actions pytest workflow before
  rung 1 lands.
- Branch alec/rung0-rps-v2 is a DEAD thread: it holds only a docs commit (an
  SDD implementation plan + a Conductor handoff breadcrumb) for a redo that
  was instead performed directly on PR #1. Cleanup (delete branch or keep the
  plan doc as reference) is an open housekeeping decision.

## Skills changed this session (user-level, affect future work)
- ~/.claude/skills/engineering-kickoff/references/house-style.md — laurence
  conventions ported (WHY-comments, banners, value types, orchestrators,
  fail-loud, naming, seam rules, anti-slop, deletion test). Both
  engineering-kickoff and research-kickoff step 4 now point at it. "No pandas"
  deliberately excluded per Alec.

## Teaching artifacts delivered (Alec is learning CFR/PyTorch — keep teaching)
- ~/Desktop/Claude/poker/rps-regret-ledger/rps-regret-ledger.pdf — 7pp paper:
  payoff matrix, hand-traced ledger, the floor, current-vs-average simplex,
  1/√T decay, CFR bridge. Reviewed inline (subagent critics unavailable).
- writeups/rung0-rps/rung0-rps-writeup.pdf — post-PR writeup, in-repo.

## Environment warnings
- Pane subagents (Agent tool teammates) STALL in this environment — five hung
  this session and had to be TaskStop'd; a feedback draft is queued. Do work
  inline or warn Alec before spawning agents.
- Desktop paths may deny `ls` in the sandbox while reads/writes succeed
  (test -f works).
- python3 is a uv shim; use `uv run` / the repo .venv for anything needing
  numpy/matplotlib. tectonic (not pdflatex) for LaTeX.

## Still pending from the 2026-08-29 breadcrumb (unrelated to rung 1)
- Survey paper (paper/drawmaha-ml-survey.pdf): the two-critic review pass was
  never completed; PDF never copied to ~/Desktop/Claude/poker/.

## Next-step candidates for "where to go from here" (none decided)
1. Merge PR #1 (Alec's call), then optionally delete/clean the v2 branch.
2. CI: GitHub Actions `uv run pytest` workflow (the writeup's top risk).
3. Settle the remaining grill branches (working mode first — it shapes rung 1).
4. Rung 1: tabular vanilla CFR on Kuhn poker vs the known exact equilibrium
   (README ladder; reading order in paper/sections/s14-checklist.tex mnote;
   Kuhn equilibrium is one-parameter — good exact test target).
5. Optional: survey critic pass + Desktop delivery of the survey PDF.
