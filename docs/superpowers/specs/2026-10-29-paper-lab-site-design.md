# The site on paper: design for the drawmaha.app revamp

**Date:** 2026-10-29. **Decided by:** Alec, from five mockup concepts built on
2026-10-28; concept 1, "Paper lab", chosen with one change (the spinning
ASCII chip stays). The interactive prototype this design was judged on lives
outside git at `.context/revamp/ideas/01-paper-lab/` in the palenque
workspace; its `IDEA.md` is the long form of the visual decisions below.

## Intent

Friends and peers read the current site as "a demo, a Claude-coded project, a
whitepaper", and named the black field as the reason. The site should read as
**a combination of an AI lab website and GTO Wizard**: it presents honestly as
one person's research project, built to the standard of a proper company
website (Anthropic, OpenAI, Notion, Modal as the references for standard, not
for branding).

## Goals

- Every surface is rebuilt: the homepage, Rules, the four rung visualizers and
  the range explorer. The repo becomes the source of every page; the outside
  studies workflow that exported the homepage and Rules is retired.
- Warm paper, near-black ink, one accent, a serif for display and body, a
  grotesk for UI and numbers. Hairlines, not cards. Nothing moves on load.
- The information architecture is the product's, not the method's: the words
  "rung" and "ladder" leave the navigation, headings and copy. The ladder
  stays the method, told on the Benchmarks page and in the README.
- Copy is declarative. Short sentences that state what is on the page.
- Charts read as one system: ink and one accent on paper, direct labels,
  a bold title that states the finding.
- The ASCII chip stays, turning, as the mark in the top bar and the footer.

## Non-goals

- Mobile. Desktop at 1440 first; narrow screens must not break but are not
  designed in this round.
- Merging the five Vite apps into one. The apps keep their routes; the new
  pages sit over them (see "Routes").
- Rewriting the research prose from scratch. Sentences are made declarative;
  facts, figures and chapter structure stay.

## Anti-goals (what would make this a failure even if it works)

- Anything from the "looks AI-generated" list in the brief: gradients,
  glow, glassmorphism, bento feature grids, icon rows, emoji, pill tags as
  decoration, monospace as the UI voice, title-barred windows around figures,
  dense captions under every element, rainbow chart palettes.
- Fake company furniture: pricing, sign-in, "trusted by", a team page.
- Inventing a number. Where the committed data has no value for a state the
  page reaches, the page says so in place.

## The visual system

### Palette

| Token | Hex | Use |
|---|---|---|
| `--paper` | `#f7f5f0` | the page |
| `--paper-2` / `--paper-3` | `#efebe3` / `#e6e1d6` | recessed cells, bar tracks, code |
| `--rule` / `--rule-2` | `#d8d2c6` / `#c2bbac` | hairlines, grid lines, control borders |
| `--ink` | `#16140f` | text, primary chart series, the primary button |
| `--ink-2` / `--ink-3` | `#4b4740` / `#857f74` | secondary text, captions, muted series |
| `--accent` | `#8e2038` | the chip's oxblood: the winning series in every chart, bet / raise, links, the focus ring |
| `--accent-tint` | `#f0dfe3` | the accent at its lightest: the homepage matrix's bet tint (mixed into the paper by bet share), text selection, a link's underline |
| `--act-call` | `#2f7d4a` | call |
| `--act-check` | `#2f66a3` | check and fold |

The accent is the chip's own colour, so the logo and "this is the finding"
are one colour. The action triplet is GTO Wizard's convention shifted to this
palette (bet oxblood, call green, check / fold blue) and passes the dataviz
validator on paper. Comparison charts use ink, accent and ink-3 with a dash
pattern for a fourth series; every line ends in a direct label. The old
`--yellow` / `--pink` tokens are retired; each app renames its uses to the
semantic tokens above.

### Type

- **Newsreader** (serif) for display, headings and the research body:
  400 for titles, 500 for small headings, 400 italic where a title needs it.
  Research body at 18 / 1.6.
- **Geist** (sans) for UI, tables, charts, captions: 400 / 500 / 600 with
  tabular numerals wherever a number appears.
- **IBM Plex Mono** is kept for two jobs: the glyph grid of the ASCII chip
  (600), and inline code and commands (400); never a prose or UI voice.

Scale: display 64 (home hero), title 52, h2 32 (30 inside the publication),
h3 22, lede 21, body 18 serif, UI 15, captions 13 / 12 / 11, eyebrow 12
uppercase at 0.08 em tracking.

### Grid, spacing, containers

12 columns, 24 px gutter, 1160 px content width inside 40 px page padding.
Home sections 112 px apart, each a 4 + 8 split (eyebrow, heading, two
sentences and a link at left; the thing itself at right). The publication is
a 220 px rail, 64 px gap, 860 px document with prose at 640 px and figures at
860. The solver is 248 px controls, fluid matrix, 272 px stats. Every
container is a top or bottom hairline, never a box with a shadow; the only
radii are 6 px on buttons and controls.

### Motion

The existing motion standard (`web/DESIGN.md`) holds unchanged: the three
tokens, press at `scale(0.97)` for 160 ms, hover colour 150 ms, nothing in
the chrome over 300 ms, reduced motion by name. Nothing moves on load and
nothing moves for decoration. The only things that animate are the chip
turning, the rock-paper-scissors matcher iterating when run, and the
figures that already draw themselves once on scroll.

### The chip

The turning ASCII chip from `mark.ts` / `chip.js`, unchanged in geometry,
recoloured for paper as a typical red chip with black indents: the face in
poker red, the edge sectors and the inner ring as solid black blocks
(`mountChip(canvas, { mark: '#c8102e', ink: '#141414' })`; Alec's call on
2026-10-29 after the oxblood face read muddy on the light field). The site's
oxblood accent is the chip's deeper shade. It is the mark in
the top bar, 1.5 px cells, and closes every page in the footer. The per-rung
ASCII hero objects (`ascii.ts`) are removed.

## Information architecture

Nav: **Research · Solver · Benchmarks · Play · Rules**, and one call to
action, **Open the solver**. The Rungs menu is gone.

Footer: the chip, a one-paragraph project line, then three columns:
Product (Range explorer, Play the solver, Rules, Releases), Research
(Mini-drawmaha solved, Learn, Benchmarks, ML survey PDF), Resources (GitHub,
figure data, strategy packs). The archived old cover stays at `/cover`,
linked from nowhere.

### Routes

The five apps keep their build and their paths. New paths are added as
rewrites so the nav speaks the new names and old links keep working.

| Path | Serves | Notes |
|---|---|---|
| `/` | `web/cover/index.html` | the homepage, hand-authored in the repo |
| `/rules` | `web/cover/rules.html` | hand-authored |
| `/benchmarks` | `web/cover/benchmarks.html` | new; the four graded games on one page |
| `/releases` | `web/cover/releases.html` | new; the real release table from `releases.py` |
| `/research` | `/rung3/index.html` | rung3-viz, Research tab; `/rung3` still works |
| `/learn` | `/rung3/index.html` | rung3-viz opens the Learn tab when `location.pathname` is `/learn` |
| `/play` | `/rung3/index.html` | rung3-viz opens the Play tab when the pathname is `/play` |
| `/solver` | solver-viz | unchanged |
| `/rung0` `/rung1` `/rung2` | the three benchmark apps | restyled; their chrome names them "Benchmarks · Rock-paper-scissors" and so on |
| `/cover` | `web/cover/cover.html` | the old front page, kept as an archive, unlinked from the nav |
| `/hero` | `web/hero-studies` | kept as it was |

The Benchmarks page carries the argument (results table, each game's
convergence chart and answer, the mini-drawmaha and Deep CFR summary, Full
Drawmaha not started) and links into each live demo. The three apps are not
merged into it: each is a React app with its own simulation and tests, and
the page is where a visitor reads, while the app is where they run it.

### Pages

**Home.** Nav; hero 6 + 6 (eyebrow, "A solver for a poker game nobody has
solved.", lede, primary button "Open the range explorer", text link "Read the
research"; at right the one chart that states the current result: Deep CFR
against tabular LCFR at equal hands per seat); Results (the real six-row
table, the unstarted row dimmed); Range explorer (the opening spot's mix bar
and the 6 × 5 matrix, static, drawn from the committed answer sheet); Research
index (dated list: the write-up, the √pot chapter, the survey PDF, the
benchmarks); Play (a real dealt hand and the coach's bands); Method (three
text columns on hairlines); footer. The embedded rock-paper-scissors console
and the Leduc explorer leave the homepage.

**Solver.** Page header with title, one-line sub-title, the Game segmented
control and the Strategy select; tabs Your range / Range vs range; the tool
in three columns (hand so far, filters and deck at left; the spot title, mix
bar and matrix in the centre; this spot, strength ribbon, ledger and the
strategy radio at right); one paragraph on what the numbers are; footer.
Everything the tool does today is kept. The numbered window frames, the hero
headline and its paragraphs, and the mono caption blocks go.

**Research** (`/rung3`, Research tab). Title block left-aligned to 900 px
(eyebrow, serif title stating the finding, dek, meta row, results strip on a
top rule); the sticky contents rail with scroll-spy; the chapters and their
seventeen figures restyled: bold finding title, one-line sub-title naming the
measure and run, caption of at most two sentences, ink and accent, no window
frames. The Learn and Play tabs stay as tabs of the same app, reached by
`/learn` and `/play`, restyled in the same register.

**Benchmarks.** Title, lede, the results table; an "on this page" rail and
six sections: rock-paper-scissors, Kuhn, Leduc (each with its static chart,
its grade and a link to the live app), mini-drawmaha and Deep CFR (the race
and equal-hands charts, linking to the chapter), Full Drawmaha (not started,
the plan stated plainly).

**The three benchmark apps.** Same chrome as every page; a page header that
names the game under "Benchmarks"; their figures restyled to ink and accent;
their copy made declarative. Their simulations and tabs are unchanged.

**Rules.** Title, lede, three figures in the publication style: the five
streets as a ruler; one worked hand as five cards read twice with the two
scorings as two hairline tables; "why nobody has solved it" as three stat
tiles. The raymarched table goes.

**Play** (`/rung3`, Play tab). The trainer in the Solver's register: hand and
board at left, the action buttons, the coach's bands at right, the settled-hand
ledger as a hairline table, the pack named in the sub-title.

**Releases.** A dated hairline table of the releases `releases.py` knows,
each with its grade and the command that produced it.

## How it is built (the builder's constraints)

- **One theme, one place.** `web/rung0-viz/src/theme.css` is rewritten as the
  paper theme and copied byte-for-byte to the other four apps, as today. The
  static pages no longer inline a copy: the build copies the same file to
  `public/theme.css` and each page links it, with a hand-written
  `web/cover/fonts.css` of `@font-face` rules for the files under `/fonts/`.
  Per-page layout CSS stays inline in each page.
- **Shared chrome.** `site.tsx` is rewritten: `TopBar` (chip, wordmark, the
  five nav items, the call to action), `Footer` (chip, project line, three
  columns), `PageHead` (eyebrow, title, lede, optional controls slot),
  `Figure` (number, finding title, sub-title, the figure, caption) replacing
  `Panel`, `Hero` and `RungObject`. `ascii.ts` is deleted from every app and
  from the shared set in `tests/test_shared_chrome.py`. `main.tsx` imports
  the new fonts. `mark.ts` is unchanged except for its default colours.
- **Fonts.** `@fontsource/newsreader` (400, 500, 400-italic) and
  `@fontsource/geist-sans` (400, 500, 600) are added to every app;
  `@fontsource/ibm-plex-mono` stays, 600 for the chip and 400 for inline
  code and commands. Plus Jakarta Sans and
  Instrument Sans are removed, from the packages and from `web/cover/fonts/`.
  `scripts/site/vercel_build.sh` copies the new files to `/fonts/`;
  Instrument Serif stays copied for `/hero`.
- **The static pages** are plain HTML written in the repo. The chip script
  is one file, `web/cover/chip.js`, served at `/chip.js` and loaded by every
  static page; it is the plain-JS twin of `mark.ts`.
- **Tests change with the design, in the same change.** `test_ui_standard.py`
  reads the linked theme for the pages and its `PRESSABLES` and `PAGES` are
  updated; `test_cover_page.py` drops the export-era hand-edit pins and checks
  the new font files and the new pages; `test_vercel_config.py` learns the new
  rewrites; `test_shared_chrome.py` drops `ascii.ts`; `test_agent_docs.py`
  keeps passing as `web/DESIGN.md`, `web/AGENTS.md` and `web/cover/README.md`
  are rewritten to describe the new system.
- **Per-app work touches only that app** (`src/index.css`, its components,
  its copy). Shared files are edited once, in rung 0, and copied.
- **Review loop.** After every page is restyled, the site is built the way
  Vercel builds it, every page is screenshotted at 1440 and read, and the
  formatting, sizing and visual faults found are fixed; the loop runs at
  least twice before the PR opens.

## Left to the builder

- Exact copy on every page, within the declarative register.
- Which chart form each of the seventeen research figures takes, within the
  two forms the concept uses (lines on log axes; magnitude or stacked bars).
- The narrow-screen behaviour, as long as nothing overlaps or overflows at
  1024 and the pages remain readable at 768.

## Risks

1. Cream paper, a serif and one red is a much-copied editorial look. The
   defence is specificity: the red is the chip, the serif carries real body
   copy, every chart follows one rule, every number is real.
2. The matrix is quieter on paper than on GTO Wizard's dark field. The tints
   and the row mix bars have to carry it; if they cannot, the matrix alone
   may need a recessed surface, decided in the review loop.
3. Eleven chapters of declarative prose is a writing job. The chapter
   structure and facts stay; only the register changes.
