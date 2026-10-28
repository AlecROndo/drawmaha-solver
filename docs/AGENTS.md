# `docs/` — dated designs and plans, and the notes on what was found

`superpowers/specs/` are design documents (what is being built and why); `superpowers/plans/` are the
step-by-step implementation plans the PRs were executed from. Those two are dated
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
it as a missing file. Documents dated before 2026-10-28 also predate a third:
the top-level `research/` and `writeups/` folders now live at `docs/research/`
and `docs/writeups/`.

## `research/` and `writeups/` — finished reference material

`research/` holds the nine literature notes the survey in `paper/` was written
from, and `research/range-ui-studies/`, the ten design studies Alec placed
`/solver`'s panes from (open it over HTTP from the repo root; its README says
how). `writeups/rung0-rps/` is the rung-0 writeup, Markdown, LaTeX and PDF,
with the scripts that drew its figures. Neither is updated as the code moves.

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
