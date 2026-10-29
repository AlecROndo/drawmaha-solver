# `rung3-viz/` — `/research`, `/learn`, `/play`: the research publication, Learn, and the trainer

- **One app, three tabs, picked by path or hash.** `/research`, `/learn`
  and `/play` (and `/rung3`, which picks by hash) all serve this app;
  `parseLocation` in `src/learn.ts` reads the tab from a hash that names
  one, else from the path's first segment, and `addressOf` writes the
  address back in the form the page was opened at. The top bar marks Play
  on the Play tab and Research on the other two.
- **The Research tab is a publication, not a dashboard.** One title block
  for the page (`ui/Masthead.tsx`: eyebrow, serif title stating the
  finding, dek, meta row, and a results strip of three numbers on a top
  rule, each computed from the data and linked to its chapter), the theme's
  `.views` under it, then `ui/ResearchTab.tsx`: the contents rail (the
  theme's `.rail`, with scroll-spy), nine chapters and Methods. Figures
  render inside `ui/Figure.tsx`, a wrapper over the shared `Figure` in
  `site.tsx`: number, bold title that states the finding, a sub-title that
  names the measure and the run, the drawing, a caption of at most two
  sentences, and a source line that names the file. The numbers come from
  three modules and nothing else: `src/research.ts` (rung 3), `src/deep.ts`
  (every rung-4 grade file and the policy-net sweep) and `src/reach.ts`
  (where the exploitability lives, by reach); each has a vitest file that
  pins what the prose quotes. The page speaks in chips per hundred hands (`perHundred`); the
  rung-4 files are in chips per hand. A new chapter number or figure number
  is page order, because the contents rail and the captions are read top to
  bottom.

- **The Learn tab is `/solver`'s mini-drawmaha range interface, imported, not
  copied.** `src/ui/LearnTab.tsx` renders `MiniRange` from
  `../solver-viz/src/ui/MiniRange.tsx`, and `src/App.tsx` imports
  `../solver-viz/src/index.css` beside this app's own stylesheet (the two
  share no selector but `:root`, and no token between them; the
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
  do not fork it. Learn's view and strategy are the hash after the tab
  (`#learn/versus/deep-pi`, or `/learn#learn/versus/deep-pi` at the product
  path). The solver sheet styles a set of unscoped classes (`.hand`, `.lbl`,
  `.cell`, `.mixbar`, `.mixkey` and others) on every tab of this page; a
  class added here is checked against it, and `tests/test_ui_standard.py`
  fails on a shared selector.
- Everything under `src/data/` is a copy of a committed file: `grades.json`
  and `answer_sheet.json` from `figures/rung3/` (`minidraw-analysis`
  regenerates them) and every rung-4 grade file plus the two sweep files
  under `rung4/`. The copies are refreshed by hand in the same change as a
  regeneration, and `tests/test_rung3_page_data.py` holds each byte for byte
  to its original.
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
