# `figures/` — committed outputs, kept as regression fingerprints

Each of `rung0/` to `rung3/` is written by that rung's analysis command (`rps-analysis`,
`kuhn-analysis`, `leduc-analysis`, `minidraw-analysis`) and committed. Rungs
1 and 2 run unseeded and enumerate the whole tree, so a regenerated figure
that differs means a number moved: a dirty `figures/` after an unrelated
change is a failing test, not noise. Nothing here is edited by hand.

`rung3/grades.json` is the race's graded record: every grid checkpoint of
every rule, in every averaging column it banked. `minidraw-analysis
--grade-runs DIR` appends the checkpoints the file lacks (about two minutes
and 2 GB each; the checkpoints live on the Modal volume, not here); without
the flag the figures are drawn from the committed file. `rung3/answer_sheet.json`
is what the README and `/rung3` quote, and `web/rung3-viz/src/data/` holds
copies of both files that are refreshed by hand after a regeneration.
