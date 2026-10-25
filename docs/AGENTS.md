# `docs/` — dated designs and plans, and the notes on what was found

`superpowers/specs/` are design documents (what is being built and why); `superpowers/plans/` are the
step-by-step implementation plans the PRs were executed from. They are dated
records. Do not rewrite one to match what shipped; a change of direction gets
a new document, and the old one stays as the reason the code looks the way it
does. The live document for current work is
`superpowers/specs/2026-10-07-deep-cfr-m1-design.md`, rung 4's PR ladder,
whose table says which PRs have landed.

Paths inside documents dated before 2026-10-20 predate two moves: `scripts/`
was grouped into `scripts/site/`, `scripts/leduc/`, `scripts/minidrawmaha/`
and `scripts/deepcfr/`, and `src/drawmaha_solver/minidrawmaha/` was grouped
into `src/drawmaha_solver/minidrawmaha/rules/`,
`src/drawmaha_solver/minidrawmaha/solver/` and
`src/drawmaha_solver/minidrawmaha/answer/`. Read an older path with that mapping in mind rather than treating
it as a missing file.

## `notes/` — findings and walkthroughs, kept current

Unlike the specs and plans, the files under `notes/` are living documents.
Update one when a new run or PR changes what it says, and correct a wrong
claim in place. Every number in them is read from a committed file under
`figures/`, and the file wins when they disagree.

- `notes/rung4-run-log.md` — every Deep CFR arm trained so far (where, which
  net, which refit), its graded marks, what the arms say, the running arm and
  its pre-registered stop rule. Add a row when an arm is graded.
- `notes/regret-rules-dcfr-cfrplus.md` — why DCFR and CFR+ lost to linear
  weighting at both rung 3 (the discount's clock) and rung 4 (no running total
  to take a sign from), and the rule that follows for sampled trainers.
- `notes/walkthrough-pr61-mps-refit.md`, `notes/walkthrough-pr62-regret-rule-screen.md`
  — code walkthroughs for merged PRs, in Markdown. From #61 on, walkthroughs
  live here rather than as PDFs outside the repo.
