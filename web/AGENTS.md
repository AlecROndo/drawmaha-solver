# `web/` — the site

`DESIGN.md` in this directory is the UI standard and it is enforced. Read it
before touching a stylesheet, a `<style>` block or a component's classes, and
afterwards run
`uv run pytest tests/test_ui_standard.py tests/test_shared_chrome.py tests/test_cover_page.py`.

## What is here

- `cover/` — three self-contained static pages: the homepage at `/`, Rules at
  `/rules`, the Cover at `/cover`. `cover/README.md` says which two are
  exports that must not be hand-edited, which hand edits are pinned anyway,
  and the font rule.
- `rung0-viz/`, `rung1-viz/`, `rung2-viz/`, `rung3-viz/`, `solver-viz/` —
  Vite + React + TypeScript apps served at `/rung0` … `/rung3` and `/solver`.
  In each: `npm install`, `npm test` (vitest), `npm run dev`, `npm run build`.
- `hero-studies/` — a static share page and its clips at `/hero`, kept as it
  was.

## Rules

- **Five files are byte-identical across the five apps**:
  `rung0-viz/src/theme.css`, `rung0-viz/src/main.tsx`,
  `rung0-viz/src/ui/site.tsx`, `rung0-viz/src/ui/mark.ts`,
  `rung0-viz/src/ui/ascii.ts` and their copies. Edit rung 0's, `cp` it to the
  other four, `cmp` them; `tests/test_shared_chrome.py` fails otherwise.
  Per-app styling lives in that app's `src/index.css`, on top of the theme.
- **The three cover pages each inline their own copy of the chrome CSS.** A
  chrome change therefore lands in the theme (copied five times) and in all
  three pages, and the motion tokens must agree across every `:root`.
- **No CFR in TypeScript.** The apps render solves exported from Python
  (`kuhn-analysis --json`, `leduc-analysis --json`, `minidraw-deals`,
  `minidraw-ranges`) or POST locked rows to a local Python exploit server.
  Only rules arithmetic is transcribed, and each app's `*.test.ts` pins it
  against the same tables the Python suite pins.
- **Adding a surface** touches `vercel.json` (a rewrite),
  `scripts/site/vercel_build.sh` (the `rungs` list), the `APPS` tuples in
  `tests/test_shared_chrome.py` and `tests/test_ui_standard.py`, and the
  Rungs menu in `site.tsx`.
- **Identity colour lives inside figures only.** Yellow is the thing pointed
  at (the average, the call); pink is the aggressive act (the bet, the current
  iterate). The chrome never uses either for decoration, and oxblood is the
  brand, never ink. One exception, on purpose: the fold / call / pot *mix*
  on `/solver` (and on `/rung3`'s Learn tab, which imports it) is drawn with
  its own fill pair, `--call-fill` and `--pot-fill` in
  `solver-viz/src/index.css`, because stacked bands of yellow and pink were
  too close in weight to read. The ribbon's slices are mix bands and take
  the pair; the hand-so-far chips, the pinned cell and the ribbon's sort
  tabs there still use yellow and pink. `/rung3`'s trainer draws the coach's
  mix in the same pair under its own names (`--mix-call` / `--mix-pot` in
  `rung3-viz/src/index.css`, because the two sheets share one page and may
  not share a token); a change to the pair is made in both.
- **Where a number is illustrative rather than a solver's output, the page
  says so** in its status line. Keep that sentence true when changing what is
  drawn.

To see the whole site wired as Vercel wires it:
`bash scripts/site/vercel_build.sh` then `uv run python scripts/site/serve_site.py`.
