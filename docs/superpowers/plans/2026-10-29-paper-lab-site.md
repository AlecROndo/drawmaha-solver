# The site on paper — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** rebuild every surface under `web/` in the paper-lab design: warm paper, Newsreader and Geist, one oxblood accent, the turning ASCII chip, a product information architecture, declarative copy, one chart system.

**Architecture:** the five Vite apps keep their routes and simulations; one rewritten `theme.css` and `site.tsx` (edited in rung 0, copied byte-for-byte to the other four) carry the chrome; the static pages are hand-written HTML in `web/cover/` that link the same theme from `/theme.css`. Task 1 lays the foundation while every app still builds (old component names kept as thin aliases); tasks 2–7 restyle one surface each and can run in parallel; task 8 rewrites the docs; task 9 is the screenshot review loop, run at least twice; task 10 opens the PR.

**Tech Stack:** Vite + React + TypeScript (vitest), plain HTML/CSS/JS for the static pages, `@fontsource/newsreader`, `@fontsource/geist-sans`, `@fontsource/ibm-plex-mono`, pytest for the site tests, puppeteer-core for screenshots (`.context/revamp/tools/shot.mjs`).

**Spec:** `docs/superpowers/specs/2026-10-29-paper-lab-site-design.md`

## Global Constraints

- Palette tokens exactly as the spec's table: `--paper #f7f5f0`, `--paper-2 #efebe3`, `--paper-3 #e6e1d6`, `--rule #d8d2c6`, `--rule-2 #c2bbac`, `--ink #16140f`, `--ink-2 #4b4740`, `--ink-3 #857f74`, `--accent #8e2038`, `--accent-tint #f0dfe3`, `--act-call #2f7d4a`, `--act-check #2f66a3`.
- Type: Newsreader 400 / 500 / 400-italic for display, headings and research body; Geist 400 / 500 / 600 for UI, tables, charts, captions, with `font-variant-numeric: tabular-nums` on numbers; IBM Plex Mono only inside the chip canvas.
- Scale: 64 / 52 / 32 (30 in the publication) / 22 / lede 21 / body 18 / UI 15 / captions 13, 12, 11 / eyebrow 12 uppercase 0.08 em.
- Grid: 1160 px content, 40 px page padding, 24 px gutters; home sections 112 px apart as 4 + 8 splits; publication 220 + 64 + 860; solver 248 + fluid + 272. Hairlines, never cards; radii only 6 px on buttons and controls.
- Motion: `web/DESIGN.md` unchanged; the tokens `--ease-out`, `--ease-in-out`, `--t-press: 160ms` in every `:root`; press `scale(0.97)`; nothing in the chrome over 300 ms; nothing moves on load.
- Copy: declarative. No "rung", "ladder", "climb" in nav, headings or page copy (the Benchmarks page and the README may describe the ladder as the method). No banned words (Supercharge, Unleash, Seamless, Revolutionize, Effortless, Next-generation, Powered by AI).
- Real numbers only. A state without a committed value says so in place.
- Shared files (`theme.css`, `main.tsx`, `ui/site.tsx`, `ui/mark.ts`) are edited in `web/rung0-viz` and copied to the other four apps; `tests/test_shared_chrome.py` holds them byte-identical.
- Per-surface tasks touch only their own directory. A needed chrome change is reported, not made.
- Never `git stash`. Commit subjects `fix(web): <the finding>` or `feat(web): <the finding>`.

## Review Focus

1. A page opened at 1024 px wide: nothing overlaps or overflows horizontally (task 9 shoots every page at 1024 as well as 1440).
2. The nav's current page is marked on every surface, including `/rung0`–`/rung2`, which belong to Benchmarks (task 1 pins `aria-current` by a `here` prop test in `site.test.tsx`).
3. `/play` and `/learn` open the right tab and `/rung3#play` still works (task 5 pins both in `learn.test.ts`).
4. A static page that loads a font the build does not copy (task 7 keeps `test_the_fonts_it_loads_are_the_ones_the_build_serves` and points it at `fonts.css`).
5. A retired token (`--yellow`, `--pink`, `--field`, `--ivory`) left in any stylesheet, which would silently resolve to nothing (task 1 adds `test_retired_tokens_are_gone` to `tests/test_ui_standard.py`; it fails until tasks 2–7 land, so it is marked `xfail(strict=True)` in task 1 and the marker is removed in task 9).

---

### Task 1: Foundation — the paper theme, the shared chrome, fonts, routes, tests

**Files:**
- Modify: `web/rung0-viz/src/theme.css` (rewrite), `web/rung0-viz/src/ui/site.tsx` (rewrite), `web/rung0-viz/src/ui/mark.ts` (defaults only), `web/rung0-viz/src/main.tsx`, `web/rung0-viz/package.json` (and the same in the other four apps' `package.json`)
- Delete: `web/*/src/ui/ascii.ts`
- Create: `web/rung0-viz/src/ui/site.test.tsx`, `web/cover/fonts.css`, `web/cover/chip.js`
- Modify: `scripts/site/vercel_build.sh`, `vercel.json`, `tests/test_vercel_config.py`, `tests/test_shared_chrome.py`, `tests/test_ui_standard.py`
- Copy across: the four shared files to `web/{rung1,rung2,rung3,solver}-viz/src/...`

**Interfaces:**
- Produces, in `theme.css`: the `:root` tokens above plus the chart tokens `--series-1: var(--ink)`, `--series-2: var(--accent)`, `--series-3: var(--ink-3)`, `--grid: var(--rule)`, `--act-bet: var(--accent)`; the transition aliases the old figures read (`--mark`, `--mark-dim`, `--hair`, `--hair-strong`, `--panel-mark`, `--panel-dim`, `--panel-hair`, `--panel-border`, `--panel`, `--well`) mapped onto paper tokens; `--yellow` → `var(--accent)` and `--pink` → `var(--ink-3)` kept **only until task 9**, each marked `/* retired: task 9 removes */`.
- Produces, in `theme.css`, these chrome classes: `.wrap` (1160 + 40), `.top` / `.chip-home` / `.word` / `.nav` / `.navbtn` / `.navbtn[aria-current=page]`, `.btn` / `.btn.primary` / `.btn.quiet`, `.page-head` / `.eyebrow` / `.page-head h1` / `.lede` / `.page-head .controls`, `.split` / `.split-aside` / `.split-main` (4 + 8), `figure.fig` / `.fig-n` / `.fig-title` / `.fig-sub` / `.fig-body` / `.fig-cap`, `table.hair` and `.rows` (hairline rows), `.views` / `.view` / `.view[aria-selected=true]`, `.seg` / `.seg button[aria-pressed=true]`, `.radios` / `.radios label`, `.stats` / `.stat`, `.rail` (sticky contents rail, `.rail a[aria-current=true]`), `footer` / `.foot-cols` / `.foot-line`, the reduced-motion block, and `@media (max-width: 1240px)`, `(max-width: 1024px)`, `(max-width: 768px)` breakpoints.
- Produces, in `site.tsx`: `TopBar({ here }: { here: 'research' | 'solver' | 'benchmarks' | 'play' | 'rules' | 'home' })`, `Footer()`, `PageHead({ eyebrow, title, lede, controls, children })`, `Figure({ n, id, title, sub, caption, wide, className, label, children })`, `Mark({ font, speed })`. Kept as aliases so every app still compiles: `Panel(props)` renders `Figure` (mapping `k` → nothing, `say` → `sub`, `live` → nothing), `Hero(props)` renders `PageHead` (ignoring `icon`, `iconLabel`, `done`), `Squiggle()` renders nothing. `RungObject` is removed; its two import sites are fixed in this task (`grep -rn RungObject web/*/src`).
- Produces: `/theme.css`, `/fonts.css`, `/chip.js` under `public/` from the build; `window.mountChip(canvas, opts)` from `chip.js` with the same options as `mark.ts`.
- Produces rewrites in `vercel.json`: `/benchmarks` → `/benchmarks.html`, `/releases` → `/releases.html`, `/research` → `/rung3/index.html`, `/learn` → `/rung3/index.html`, `/play` → `/rung3/index.html`.

- [ ] **Step 1: Install the fonts in every app**

In each of the five `web/*-viz/package.json`: add `"@fontsource/newsreader": "^5.3.0"` and `"@fontsource/geist-sans": "^5.3.0"`; remove `@fontsource/plus-jakarta-sans` and `@fontsource/instrument-sans`; keep `@fontsource/ibm-plex-mono` and (rung 0 only) `@fontsource/instrument-serif`. Run `npm install` in each so the lockfiles update. Rewrite `main.tsx` to import `@fontsource/newsreader/400.css`, `/500.css`, `/400-italic.css`, `@fontsource/geist-sans/400.css`, `/500.css`, `/600.css`, `@fontsource/ibm-plex-mono/400.css`, then `./theme.css`, `./index.css`.

- [ ] **Step 2: Write the failing chrome test**

`web/rung0-viz/src/ui/site.test.tsx` (vitest + `@testing-library/react` is not installed; use `react-dom/server`'s `renderToStaticMarkup`):

```ts
test('the top bar names the five surfaces and marks the current one', () => {
  const html = renderToStaticMarkup(<TopBar here="benchmarks" />)
  for (const w of ['Research', 'Solver', 'Benchmarks', 'Play', 'Rules', 'Open the solver']) expect(html).toContain(w)
  expect(html).toMatch(/href="https:\/\/drawmaha.app\/benchmarks"[^>]*aria-current="page"/)
  expect(html).not.toMatch(/Rungs|rung 0|ladder/i)
})
test('the footer has three columns and the chip', () => {
  const html = renderToStaticMarkup(<Footer />)
  for (const w of ['Range explorer', 'Releases', 'Learn', 'ML survey', 'GitHub']) expect(html).toContain(w)
  expect(html).toContain('class="chip-end"')
})
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd web/rung0-viz && npx vitest run src/ui/site.test.tsx`. Expected: FAIL (no `here="benchmarks"` string type, no `Open the solver`).

- [ ] **Step 4: Rewrite `site.tsx`** to the Interfaces above. `HOME` stays `https://drawmaha.app`. Nav hrefs: `/research`, `/solver`, `/benchmarks`, `/play`, `/rules`; the call to action `/solver`. Footer columns per the spec's IA. The chip: `mountChip(canvas, { font, speed, mark: '#16140f', ink: '#8e2038' })`.

- [ ] **Step 5: Rewrite `theme.css`** to the Interfaces above, with the file header rewritten to describe the paper system (the five numbered points become: paper and ink and one accent; two voices and the chip's mono; the chip as the mark; figures as titled blocks; hairline rows). Keep the motion section's tokens and the structure the standard test reads (every pressable has `:active { transform: scale(0.97) }` and `transform var(--t-press) var(--ease-out)` in its transition; the reduced-motion block names each).

- [ ] **Step 6: `mark.ts`** default `mark` → `'#16140f'`, default `ink` unchanged; header comment says it is the mark on paper now. Delete `ui/ascii.ts` in all five apps; fix the imports that referenced `mountObject` / `RungObject`.

- [ ] **Step 7: Copy the shared files across and run the app tests**

Run: `for a in rung1 rung2 rung3 solver; do for f in src/theme.css src/main.tsx src/ui/site.tsx src/ui/mark.ts src/ui/site.test.tsx; do cp web/rung0-viz/$f web/$a-viz/$f; done; done`, then in each app `npx vitest run` and `npm run build`. Expected: all green, five builds succeed (figures look wrong on paper; that is tasks 2–6).

- [ ] **Step 8: The static-page assets.** `web/cover/fonts.css`: `@font-face` for the seven files (`newsreader-latin-400-normal`, `-500-normal`, `-400-italic`, `geist-sans-latin-400-normal`, `-500-`, `-600-`, `ibm-plex-mono-latin-400-normal`) at `/fonts/<file>.woff2`, `font-display: swap`. `web/cover/chip.js`: extract the inline chip script from the current `web/cover/index.html` (`function mountChips` and what it needs) into one plain-JS file exposing `window.mountChip`, defaults recoloured to `mark '#16140f'`, `ink '#8e2038'`. Delete `web/cover/fonts/*.woff2` (Instrument Sans, Plus Jakarta italic).

- [ ] **Step 9: The build and the routes.** `scripts/site/vercel_build.sh`: `copy_font` the seven files above (newsreader and geist-sans from rung 0's `node_modules/@fontsource`), keep `instrument-serif 400` for `/hero`, drop the rest; copy `web/rung0-viz/src/theme.css` → `public/theme.css`, `web/cover/fonts.css` → `public/fonts.css`, `web/cover/chip.js` → `public/chip.js`, and `web/cover/{index,rules,benchmarks,releases,cover}.html` to `public/`. `vercel.json`: add the five rewrites from Interfaces. `tests/test_vercel_config.py`: a test that each of `/research`, `/learn`, `/play` rewrites to `/rung3/index.html` and `/benchmarks`, `/releases` to their files.

- [ ] **Step 10: The site tests.** `tests/test_shared_chrome.py`: `SHARED` drops `src/ui/ascii.ts`, adds `src/ui/site.test.tsx`. `tests/test_ui_standard.py`: `PAGES` becomes `("index.html", "rules.html", "benchmarks.html", "releases.html", "cover.html")`; `_sheets()` reads a page's inline `<style>` blocks **plus** the theme when the page contains `href="/theme.css"`; `PRESSABLES["theme"]` becomes `(".btn", ".navbtn", ".view", ".seg button", ".radios label", ".rail a")`, `.rmain` / `.racts a` are removed everywhere, and `test_retired_tokens_are_gone` is added (asserts no `--yellow`, `--pink`, `--field`, `--ivory`, `--ox`, `--ox-lit` in any sheet except `cover.html`), marked `@pytest.mark.xfail(strict=True, reason="until the per-surface restyles land")`. Note: until task 7 writes the new pages, `benchmarks.html` and `releases.html` do not exist, so this step also creates both as minimal valid pages (nav, one heading, footer, linked theme) that task 7 replaces.

- [ ] **Step 11: Run the site tests**

Run: `uv run pytest tests/test_shared_chrome.py tests/test_ui_standard.py tests/test_vercel_config.py tests/test_cover_page.py -q`. Expected: shared chrome, ui standard (with the xfail) and vercel green; `test_cover_page.py` may fail on `index.html` / `rules.html` fonts and hand-edit pins until task 7 — record which, do not silence.

- [ ] **Step 12: Commit**

```bash
git add -A web scripts/site/vercel_build.sh vercel.json tests docs/superpowers
git commit -m "feat(web): the paper theme and the product chrome, shared by five apps and linked by the static pages"
```

### Task 2: Rock-paper-scissors app (`/rung0`) on paper

**Files:** `web/rung0-viz/src/index.css`, `web/rung0-viz/src/App.tsx`, `web/rung0-viz/src/ui/*.tsx` (not `site.tsx`, `mark.ts`).

**Interfaces:** consumes Task 1's `TopBar here="benchmarks"`, `PageHead`, `Figure`, the chart tokens.

- [ ] **Step 1:** `PageHead` with eyebrow "Benchmarks · rock-paper-scissors", title "Regret matching, graded against the uniform Nash." (declarative), lede naming the grade (0.0009 chips/round after 100k rounds) and a link back to `/benchmarks`. Remove the old `Hero` icon props and the `Panel` title bars (`k`, `live`); use `Figure` with finding titles.
- [ ] **Step 2:** `index.css`: replace every `--yellow` / `--pink` use with `--series-2` (the average, the thing pointed at) and `--series-3` (the current iterate); charts on `--paper` with `--grid` lines; the mode tabs use `.seg`; the scoreboard uses `table.hair`.
- [ ] **Step 3:** Copy: every heading and paragraph declarative; grep `-i "rung\|ladder\|climb"` in `src/` returns nothing but code identifiers.
- [ ] **Step 4:** Run `npx vitest run && npm run build`; shoot `http://localhost:4173/rung0/` after `npm run preview` with `node .context/revamp/tools/shot.mjs` and look at it; fix what is wrong.
- [ ] **Step 5:** Commit `fix(web): /rung0 reads on paper — the average in accent, the iterate in grey, one figure style`.

### Task 3: Kuhn app (`/rung1`) on paper

Same shape as Task 2 for `web/rung1-viz` (`index.css`, `App.tsx`, `ui/*.tsx`): eyebrow "Benchmarks · Kuhn poker", the Solve / Exploit tabs as `.views`, the answer sheet as `table.hair`, the lock buttons as `.btn.quiet`, the convergence chart with the average in `--series-2` and the current iterate in `--series-3`. Commit `fix(web): /rung1 reads on paper`.

### Task 4: Leduc app (`/rung2`) on paper

Same shape for `web/rung2-viz`: eyebrow "Benchmarks · Leduc poker", the two tabs as `.views`, the timeline stations as `.seg`-style buttons (keep their `:active` rule; `PRESSABLES["rung2-viz"]` stays), the felt and chips redrawn on `--paper-2` with the sweep kept (it is allowlisted), the fold / call / pot mix in `--act-check` / `--act-call` / `--act-bet`. Commit `fix(web): /rung2 reads on paper`.

### Task 5: Research, Learn and Play (`/rung3`, `/research`, `/learn`, `/play`) on paper

**Files:** `web/rung3-viz/src/index.css`, `App.tsx`, `learn.ts` + `learn.test.ts`, `ui/*.tsx` (not shared).

- [ ] **Step 1 (test first):** in `learn.test.ts`, `parseLocation(pathname, hash)` returns `{ tab: 'play' }` for `('/play', '')`, `{ tab: 'learn' }` for `('/learn', '')`, `{ tab: 'research' }` for `('/research', '')` and `('/rung3', '')`, and the hash still wins for `('/rung3', '#play')`. Run, see it fail, implement in `learn.ts`, `App.tsx` reads `location.pathname` too.
- [ ] **Step 2:** `Masthead.tsx` becomes the spec's title block (eyebrow, serif title, dek, meta row, results strip 0.061 / 4× / 0.086 on a top rule) and the three tabs as `.views` under it. `TopBar here` is `'research'`, `'play'` or `'research'` for learn.
- [ ] **Step 3:** `Figure.tsx` adopts the theme's `figure.fig` block (no title bar); every chart component swaps `--yellow` / `--pink` / `--mark*` for the series and action tokens; the contents rail uses `.rail`.
- [ ] **Step 4:** `ResearchTab.tsx` prose: declarative register, facts unchanged; `chapters.test.ts` still passes (it guards the chapter order and figure numbering).
- [ ] **Step 5:** `LearnTab.tsx` and `PlayPanel.tsx`: the Learn dashboard and the trainer in the Solver's register (the trainer: hand and board at left, actions, coach bands at right in the action colours, the ledger as `table.hair`); `--mix-call` / `--mix-pot` become `--act-call` / `--act-bet`.
- [ ] **Step 6:** `npx vitest run && npm run build`; preview and shoot `/rung3/`, `/rung3/#learn`, `/rung3/#play`; fix; commit `fix(web): the research page is a publication on paper; Learn and Play share its register`.

### Task 6: The range explorer (`/solver`) on paper

**Files:** `web/solver-viz/src/index.css`, `App.tsx`, `ui/*.tsx`, `ui/sofar/*` (not shared).

- [ ] **Step 1:** `App.tsx`: `PageHead` with title "Range explorer", the one-line sub-title, and the Game `.seg` and Strategy `<select>` in the `controls` slot; the two tabs as `.views`. Remove the hero headline, its paragraphs, the numbered `Panel` frames and the mono caption blocks.
- [ ] **Step 2:** The three-column tool: `.tool { grid-template-columns: 248px 1fr 272px }` in `index.css`; left = hand so far (the existing `sofar/` path layout), inner / outer filters as `.radios`, the deck; centre = `SpotHead`, the mix bar, `Grid`; right = `Score` (this spot), `Ribbon` as the six-band stack, `Ledger` as `table.hair`, the strategy `.radios`. `FullRange`, `Versus`, `DrawBoard`, `Deck`, `Plane` keep their function and take the tokens.
- [ ] **Step 3:** Colours: `--call-fill` → `--act-call`, `--pot-fill` → `--act-bet`, check → `--act-check`; cell tint = the dominant action's colour at low alpha, deeper for a larger region; no `--yellow` / `--pink` left. The `Grid.test.ts`, `MiniScore.test.ts`, `hash.test.ts` keep passing.
- [ ] **Step 4:** `npx vitest run && npm run build`; preview and shoot `/solver/` in three states (opening spot; a cell selected; the path advanced); fix; commit `fix(web): /solver is a three-column tool on paper, GTO Wizard's action colours shifted to the chip's palette`.

### Task 7: The static pages — home, Rules, Benchmarks, Releases

**Files:** rewrite `web/cover/index.html`, `web/cover/rules.html`; replace the Task 1 placeholders `web/cover/benchmarks.html`, `web/cover/releases.html`; `web/cover/README.md`; `tests/test_cover_page.py`.

**Interfaces:** each page is `<link rel="stylesheet" href="/fonts.css">`, `<link rel="stylesheet" href="/theme.css">`, one inline `<style>` of page layout, the same nav and footer markup `site.tsx` renders (copied as HTML, `aria-current` set), `<script src="/chip.js">` and two `mountChip` calls. Data for charts: inline SVG generated once from `figures/rung3/grades.json`, `figures/rung4/grades.json`, `figures/rung3/answer_sheet.json` and the README tables; the prototype's `build.py` under `.context/revamp/ideas/01-paper-lab/` is the reference for the chart helpers and may be adapted into `scripts/site/make_static_pages.py` **only if** the pages are then committed as its output and the script is named in each page's head comment (the repo rule: regenerated, never hand-edited, and the generator named). Otherwise author by hand.

- [ ] **Step 1 (test first):** `tests/test_cover_page.py`: `PAGES` becomes the five pages; `HAND_EDITS` and `test_the_hand_edits_survived_the_last_export` are deleted; `test_the_fonts_it_loads_are_the_ones_the_build_serves` reads `fonts.css` and asserts its seven files are what `vercel_build.sh` copies; `test_the_checked_in_fonts_are_not_ones_the_build_already_copies` is deleted with `web/cover/fonts/`; a new `test_no_page_says_rung_or_ladder_in_its_copy` strips tags and scripts and asserts the words are absent from `index.html`, `rules.html`, `benchmarks.html`, `releases.html` (`cover.html` is exempt). Run; expected FAIL until the pages exist.
- [ ] **Step 2:** `index.html` per the spec's Home section (hero chart: Deep CFR vs tabular at equal hands; results table; the static matrix; research index; a dealt hand; method columns).
- [ ] **Step 3:** `rules.html` per the spec (streets ruler; the hand read twice with two hairline tables; three stat tiles). The raymarched table is removed.
- [ ] **Step 4:** `benchmarks.html` per the spec (results table, "on this page" `.rail`, six sections with their charts and links to `/rung0`, `/rung1`, `/rung2`, `/research`).
- [ ] **Step 5:** `releases.html`: the table from `src/drawmaha_solver/minidrawmaha/answer/releases.py` (release, date, grade, command), authored by hand with the source named.
- [ ] **Step 6:** `web/cover/README.md` rewritten: five pages authored here, the linked theme, `chip.js`, `fonts.css`, the local-check recipe (serve `public/` after the build script).
- [ ] **Step 7:** Run `uv run pytest tests/test_cover_page.py tests/test_ui_standard.py -q` → green except the xfail; `bash scripts/site/vercel_build.sh && uv run python scripts/site/serve_site.py` and shoot `/`, `/rules.html`, `/benchmarks.html`, `/releases.html`; fix; commit `feat(web): the homepage, Rules, Benchmarks and Releases are written in the repo, on paper`.

### Task 8: The documents that describe the system

**Files:** `web/DESIGN.md`, `web/AGENTS.md`, `AGENTS.md` (root: the resolver rows naming the cover README and the site), `README.md` (the front page's site links and any "rung page" wording), `docs/superpowers/specs/2026-10-29-paper-lab-site-design.md` (unchanged), the header comment of `theme.css` (done in Task 1).

- [ ] **Step 1:** `web/DESIGN.md`: the "where the CSS lives" table becomes theme (five copies) + `/theme.css` for the pages; the before/after rows that name the Rungs menu, the rope, the sweep and the reveal are updated or removed to match what exists; the review table stays.
- [ ] **Step 2:** `web/AGENTS.md`: the shared-files rule lists the four files plus `site.test.tsx`; "Identity colour lives inside figures only" becomes the series and action token rule; the route table gains `/benchmarks`, `/releases`, `/research`, `/learn`, `/play`.
- [ ] **Step 3:** Run `uv run pytest tests/test_agent_docs.py -q` → green. Commit `docs(web): the design standard and the agent notes describe the paper system`.

### Task 9: The review loop (run at least twice)

- [ ] **Step 1:** Build the site as Vercel does: `bash scripts/site/vercel_build.sh`, then `uv run python scripts/site/serve_site.py` (note its port).
- [ ] **Step 2:** Shoot every page at 1440 and 1024 into `.context/revamp/after/<loop>/`: `/`, `/rules`, `/benchmarks`, `/releases`, `/research`, `/learn`, `/play`, `/solver` (three states), `/rung0`, `/rung1`, `/rung2`.
- [ ] **Step 3:** Read every screenshot and write the findings table (`Before | After | Why`, the DESIGN.md form) into `.context/revamp/after/<loop>/findings.md`: overflow, misaligned columns, type sizes off the scale, inconsistent spacing rhythm, a chart whose colours break the rule, copy that is not declarative, any leftover window frame or ASCII object.
- [ ] **Step 4:** Fix each finding in the surface that owns it (chrome fixes in rung 0, copied across). Run that app's `npx vitest run` and the site tests.
- [ ] **Step 5:** Commit `fix(web): review loop <n> — <the three biggest findings>`. Repeat from Step 1 until a loop finds nothing that is not cosmetic, and never fewer than two loops.
- [ ] **Step 6:** Remove the `xfail` from `test_retired_tokens_are_gone` and the retired aliases from `theme.css`; copy across; `uv run pytest -q` → green.

### Task 10: Ship

- [ ] **Step 1:** `uv run pytest -q` and `npm test` in all five apps → green. `bash scripts/site/vercel_build.sh` → succeeds.
- [ ] **Step 2:** Open the PR against `main` with the `open-pr` skill; title `feat(web): the site on paper — one theme, a product nav, declarative copy, the chip kept turning`; body links the spec and the plan and carries the before / after home screenshots.
- [ ] **Step 3:** Drive it to merge-ready per the repo rule (CI is the automated review only; address its findings).
