# `figures/` — committed outputs, kept as regression fingerprints

Each of `rung0/` to `rung4/` is written by that rung's analysis command (`rps-analysis`,
`kuhn-analysis`, `leduc-analysis`, `minidraw-analysis`, `deep-analysis`) and committed. Rungs
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

`rung4/grades.json` is rung 4's graded record: every grid mark of the Deep
CFR run, three objects each (`deep/<t>/<exact|pi|current>`), written by
`scripts/deepcfr/modal_rung4.py`'s `grade` from the marks on the Modal
volume `drawmaha-rung4`, with `hands_per_seat`, the training clock and the
run's config beside each grade. `deep-analysis` reads it with `rung3/grades.json`
and writes `rung4/price.png` and `rung4/answer_sheet.json`; the net-size
sweep's verdict, written by `sweep --grade`, lands beside them as `sweep.json`.

The other `rung4/sweep_*.json` files grade the size and refit arms trained on
this machine after the sweep (`sweep_128x3_t200.json`, `sweep_128x3-s8000.json`,
`sweep_128x4.json`, `sweep_256x4.json`, `sweep_256x4-s8000.json`,
`sweep_256x4-s16000-lr5e-4.json`, `sweep_256x4-s16000.json`), each written by
`modal_rung4.py::grade --local runs/<arm> --grades figures/rung4/sweep_<arm>.json`
and keyed as in `grades.json`. The file name is width x depth plus any refit
setting that differs from the spec's; each grade carries the full config.
`rung4/pot_0.5-256x4-s16000-lr5e-4.json` and `rung4/pot_1-256x4-s16000-lr5e-4.json`
grade the pot-normalised arms the same way: the 16,000-step, lr 5e-4 run with
its regret targets divided by √pot and by pot at each spot (`--pot-power`,
the number after `pot_`; `train.pot_scaled`), the setting that became the
default. Which arm is which, and what they showed, is `docs/notes/rung4-run-log.md`.

Three of these files are also what the site quotes. `rung3/answer_sheet.json`'s
`frozen_strategy` object and the Π and exact objects of mark 500 in
`rung4/sweep_256x4-s16000-lr5e-4.json` are the grades of the three released
strategies the range exporter ships to `/solver`, and the exact object of mark
200 in `rung4/pot_0.5-256x4-s16000-lr5e-4.json` is the grade of the one the
deal pack ships to `/rung3`'s Play tab (all five releases are `RELEASES` in
`src/drawmaha_solver/minidrawmaha/answer/releases.py`); each export's
`index.json` — under `web/solver-viz/public/` and at
`web/rung3-viz/public/deals/` — carries a copy of its grade (`exploitability`
and `value_p0`), written at export time and held to the file by
`tests/minidrawmaha/test_range_export.py` and `tests/minidrawmaha/test_deal_pack.py`. A re-grade
of one of those marks therefore means re-exporting its strategy
(`uv run minidraw-ranges --release <name>`, `uv run minidraw-deals`) in the
same change.

`rung3/exploit_by_reach.json` (and its `.png`) is where the frozen table's
exploitability lives: every spot's one-step best-response gain from the exact
walk, binned by how often the spot comes up when the strategy plays itself,
written by `uv run minidraw-exploit-by-reach --release tabular`;
`rung4/exploit_by_reach_deep-exact.json` and `exploit_by_reach_deep-pi.json` are
the same for the two released Deep CFR averages (`--release deep-exact`,
`--release deep-pi`). Each carries the check that the gains sum to each seat's
best-response edge.

`rung4/pi_sweep/results.json` is the policy-net loss sweep: 22 refits of Π on
the stored policy memory of the 4 × 256, 16,000-step arm (loss, head, steps,
schedule, width, depth, mark), each graded exactly, written by
`scripts/deepcfr/pi_loss_sweep.py`; `rung4/pi_sweep/reach_weighted.json` is
each refit's reach-weighted error against the exact average, written by
`scripts/deepcfr/pi_reach_weighted.py`. `/rung3` quotes both.

`web/rung3-viz/src/data/` holds a copy of every file `/rung3` reads — rung 3's
grades and answer sheet, these two, every `rung4/*.json` grade file —
refreshed by hand after a regeneration and held byte for byte to the originals
by `tests/test_rung3_page_data.py`.

`rung4/rule_dcfr-256x4.json` and `rung4/rule_cfrplus-256x4.json` grade the
regret-rule screen: the 256 × 4 net trained to 50 iterations on this machine
by `run_deep_cfr.py --rule dcfr` and `--rule cfr+`, its marks graded by
`modal_rung4.py::grade --local <run dir> --grades <file>`, keyed as in
`grades.json`. `rung4/rule_sign_flips.json` is the measurement of why they
lost, written by `scripts/deepcfr/rule_sign_flips.py` from the same runs'
advantage memories.
