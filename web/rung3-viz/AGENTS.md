# `rung3-viz/` — `/rung3`: the research page, the trainer, and Learn

- **The Learn tab is `/solver`'s mini-drawmaha range interface, imported, not
  copied.** `src/ui/LearnTab.tsx` renders `MiniRange` from
  `../solver-viz/src/ui/MiniRange.tsx`, and `src/App.tsx` imports
  `../solver-viz/src/index.css` beside this app's own stylesheet (the two
  overlap only on `.block` and `:root`, with no conflicting declarations; the
  solver sheet's one page-wide rule hangs off `body.wide`, which this page
  sets only while Learn shows; `tests/test_ui_standard.py` pins both). `vite.config.ts`
  dedupes `react` so the build resolves it from this app's `node_modules`
  whether or not the sibling is installed — the Vercel build installs and
  builds the apps one after another, and this one builds before `solver-viz`,
  so a `solver-viz` module reachable from here may import `react` and
  relative paths and nothing else (`tests/test_vercel_config.py` walks the
  import graph and fails on any other package: it would resolve nowhere on
  Vercel while passing every local check run after both apps were
  installed). The exports are fetched from `/solver/mini*/` (`EXPORTS_BASE`),
  one copy on the site, so `npm run dev` here cannot load them — drive the tab
  through `bash scripts/site/vercel_build.sh` + `uv run python scripts/site/serve_site.py`.
  A change to the range interface is made in `solver-viz` and shows up here;
  do not fork it. The tab, its view and its strategy are the hash
  (`src/learn.ts`: `#learn/versus/deep-pi`).
- `src/data/grades.json` and `src/data/answer_sheet.json` are copies of
  `figures/rung3/`; `minidraw-analysis` regenerates those, and the copies are
  refreshed by hand in the same change.
- `public/deals/` is the pack written by `uv run minidraw-deals`: 600 deals
  in six chunks, each with the shipped strategy's mix at every node either
  player can reach, and `public/deals/index.json` recording the seed and the
  strategy's release, digest and grade. The strategy is the release
  `deal_pack.PACK_RELEASE` names in the catalogue
  (`src/drawmaha_solver/minidrawmaha/answer/releases.py`): rung 4's Deep CFR
  exact average from the ÷ √pot arm at iteration 200, the best graded one.
  The manifest's `strategy` and `grade` blocks are the ones `/solver`'s
  `index.json` opens with, and `PlayPanel` names the bot and quotes its grade
  through `solver-viz`'s `describeStrategy`, never from a number typed here;
  the Python tests hold the manifest's grade to the committed grades file. The
  page refuses a chunk whose deal count disagrees with the manifest, or a deal
  missing a node its grammar can reach. A re-export keeps the seed; the digest
  changes only when the strategy does.
- `src/minidraw.ts` transcribes only the pot-limit betting grammar, and
  `src/minidraw.test.ts` pins it against the same round-1 lines the Python
  suite pins. Everything else on the page is read from the pack or the data
  copies; nothing looks the strategy up.
- The pack holds both holes, so the bot's cards are readable in the network
  tab. That is accepted for a trainer and is not a bug to fix.
