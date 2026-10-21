# `docs/superpowers/` — dated designs and plans

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
