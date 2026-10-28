# `/rung3` as a research publication page — implementation plan

> **As built (2026-10-28):** nine chapters, not ten. The clock chapter ("Why DCFR became CFR+") was dropped in `3cbeb7b` for the reach decomposition of chapter 04, and the √pot regret targets of PR #68 joined chapter 06 (`9cdea14`, Figure 12). The dated plan below is kept as written.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Research tab of `/rung3` reads like a research announcement page, not a trainer: a masthead in place of the ASCII cigar hero (one masthead for all three tabs), a summary, four key numbers, a contents rail, ten chapters plus Methods, every figure at column width with a caption that states the finding and a source line that names its file, and the Deep CFR work on the same game (rung 4) as chapters 06 to 09 beside the tabular race. Every number is read from a committed file. The outside data the page quotes (the policy-net loss sweep, the DCFR clock forensics, the visit-clock DCFR grades) is committed first with the scripts that wrote it.

**Architecture:** The page keeps the shared chrome untouched (`site.tsx`, `theme.css` and the other three shared files stay byte-identical across the five apps); everything new lives in `web/rung3-viz/src/`. `research.ts` keeps rung 3's numbers; a sibling `deep.ts` reads copies of every rung-4 grade file plus the two sweep files under `src/data/rung4/`, and `clock.ts` reads the forensics and visit-clock copies. `ResearchTab.tsx` becomes an `<article>` of chapters; `Figure` replaces the window `Panel` on this tab; the five existing charts are retrofitted to render inside `Figure`; six new charts draw the Deep CFR chapters. Styles land in `index.css` on the theme's tokens, per `web/DESIGN.md`. A Python test holds every `src/data` copy byte-identical to its `figures/` original.

**Tech stack:** Python 3.12 + numpy (ported scripts, tests), Vite + React 19 + TypeScript + vitest (the app), puppeteer-core for screenshots.

**Spec:** Alec's redirect of 2026-10-26/27 recorded in the mockup artifact (https://claude.ai/artifact/4cC8RGFvfAfw7WiaMqQv79): announcement-page register, the cigar removed, papers shelved, Deep CFR as chapters of this page, the dark field first, one PR for data plus page. Units: the page speaks in chips per hundred hands everywhere; the rung-4 files are in chips per hand and the page multiplies by 100 in one place (`perHundred`), which Methods says.

## Global constraints

- `AGENTS.md`: a grader never shares code with what it grades; figures and `src/data` copies are regenerated or copied, never hand-edited; `uv run pytest` and `npm test` before the PR; commit subjects `feat(web): <the finding>`; never `git stash`; no Modal launch.
- `web/DESIGN.md`: a new pressable gets `:active` and is listed in `PRESSABLES`; chrome motion under 300 ms; identity colour only inside figures; no new curves.
- Every comparison on one seed under about 10% is labelled suggestive on the page.
- Deferred to a follow-up, said so in the PR: `deep-analysis` redrawing `price.png` over every arm; the cross-entropy fit inside `policy.py`; the cover ladder's rung-3 object.

## Review focus

1. No number is typed: every figure and every sentence quotes `research.ts`, `deep.ts` or `clock.ts`, and the vitest files pin the quoted values against the data copies.
2. The data copies cannot drift: `tests/test_rung3_page_data.py` compares each copy under `web/rung3-viz/src/data/` to its `figures/` original byte for byte.
3. The ported scripts import the modules that exist on main (`minidrawmaha.rules.features`, `minidrawmaha.answer.analysis`, `deepcfr.*`), say the command that produced each committed file, and are exercised by an import test without torch.
4. The shared chrome is untouched (`tests/test_shared_chrome.py`), and every new stylesheet rule passes `tests/test_ui_standard.py`.
5. Nothing gamified remains on the Research tab: no window title bars, no dots, no ASCII object, no stat readouts; the Learn and Play tabs are unchanged apart from the masthead above them.

---

### Task 1: commit the outside data with its scripts

- [ ] `figures/rung4/pi_sweep/results.json` and `reach_weighted.json` copied from the main checkout's `runs/sweep-256x4-s16000-lr5e-4/pi-sweep/`
- [ ] `scripts/deepcfr/pi_loss_sweep.py` and `scripts/deepcfr/pi_reach_weighted.py`, ported from the 2026-10-26 scratchpad, with the commands that wrote the two files
- [ ] `figures/rung3/visit_forensics.json` (the Desktop paper's `forensics.json`) with `scripts/minidrawmaha/checkpoint_forensics.py`
- [ ] `figures/rung3/grades_visit_clock.json` (the Desktop paper's `grades_visit.json`) with `scripts/minidrawmaha/grade_visit_clock.py`
- [ ] `figures/AGENTS.md` names the four files and their writers
- [ ] `tests/test_rung3_page_data.py`: the copies equal the originals; the ported scripts import

### Task 2: the page's numbers modules

- [ ] `web/rung3-viz/src/data/rung4/*.json` copies of every rung-4 grade file and the two sweep files; `src/data/visit_forensics.json`, `src/data/grades_visit_clock.json`
- [ ] `src/deep.ts`: typed arms, marks, three objects per mark, the size sweep, the refit arms, the rule screens, the sign flips, the sweep's arms, the key numbers; `src/deep.test.ts` written first
- [ ] `src/clock.ts`: the forensics the clock chapter quotes and the visit-clock grades; `src/clock.test.ts`
- [ ] `research.ts` loses its typed run-facts (`UNREACHED_ROWS`, `MEDIAN_REVISIT`, `HALVINGS_TO_ZERO`) to the forensics file; the typed uniform-random and Leduc literals in the prose go

### Task 3: the page

- [ ] `ui/Masthead.tsx` replaces `Hero` in `App.tsx` on every tab: eyebrow, title, date, byline, summary; the `views` strip under it
- [ ] `ui/Figure.tsx`: number, children, caption, source; the five existing charts render inside it (no `Panel` on this tab)
- [ ] `ui/ResearchTab.tsx`: `KeyNumbers`, `Contents`, ten chapters, Methods, authors; prose in running paragraphs with the top-down structure
- [ ] New charts: `DeepRaceChart` (Fig. 7), `SizeChart` (8), `RefitChart` (9), `AveragesChart` (10), `PiLossChart` (11), `LossShapes` (12), `RuleScreenChart` (13)
- [ ] `index.css`: `.masthead`, `.article`, `.contents`, `.chapter`, `.figure`, `.keynums`, `.callout`, responsive rules, reduced motion; `PRESSABLES["rung3-viz"]` gains `.contents a`

### Task 4: verify, screenshot, serve, ship

- [ ] `uv run pytest`, `npm test`, `npm run build`; `bash scripts/site/vercel_build.sh` and `uv run python scripts/site/serve_site.py` for Alec's look
- [ ] full-page screenshots under `.context/`
- [ ] README's "Seeing it" paragraph for rung 3 updated; `web/rung3-viz/AGENTS.md` updated
- [ ] commit and open the PR; drive it to merge-ready

## Revisions after the first PR commit (2026-10-28)

- The clock chapter (DCFR = CFR+ on the global clock) is dropped at Alec's request, with its two data files and
  scripts; the finding stays as one callout in chapter 03 pointing at `docs/notes/regret-rules-dcfr-cfrplus.md`.
- New: where the exploitability lives. `exploitability.gains_by_infoset` decomposes a seat's best-response edge
  into one-step deviations per table infoset (gains sum to BR − EV, tested on the Leduc oracle), with each
  infoset's frequency under the strategy's own play; `analysis.reach_profile` bins the spots by frequency and
  `minidraw-exploit-by-reach --release tabular | deep-exact | deep-pi` writes `figures/rung3/exploit_by_reach.json`
  and `figures/rung4/exploit_by_reach_<release>.json`. The page draws the table's profile in chapter 04 and the
  three side by side in chapter 05 (`ui/ReachChart.tsx`, `src/reach.ts`).
