---
title: The site at a glance
description: Six static pages and five Vite apps, served from one public/ folder by Vercel, with every number read from a committed file.
---

The site at [drawmaha-solver.vercel.app](/) is how the project shows its work. It has two kinds of surface. **Static pages** are plain HTML written by a Python generator. **Apps** are Vite + React + TypeScript bundles that draw solves exported from Python. One shell script builds both into `public/`, and Vercel serves that folder with a handful of rewrites from [`vercel.json`](repo:vercel.json).

The rule that shapes the whole site is that **the browser never solves anything**. Every strategy, grade and curve on a page comes from a file the Python side wrote and committed, so a number on the site always has a source in the repository and a test that holds it to that source.

## Every route and what serves it

| Route | Serves | Kind |
| --- | --- | --- |
| `/` | `web/cover/index.html`: the homepage, with the current result, the results table, the opening spot's range matrix and the research index | static, generated |
| `/rules` | `web/cover/rules.html`: the rules of Drawmaha | static, generated |
| `/benchmarks` | `web/cover/benchmarks.html`: the four graded games on one page | static, generated |
| `/releases` | `web/cover/releases.html`: every released strategy, with its tag and grade | static, generated |
| `/wiki` | `web/cover/wiki.html`: this wiki, rendered from `docs/wiki/` | static, generated |
| `/research`, `/learn`, `/play` | `web/rung3-viz`, opening on its Research, Learn or Play tab | app |
| `/rung3` | the same app, choosing the tab by hash | app |
| `/solver` | `web/solver-viz`, the range explorer | app |
| `/rung0`, `/rung1`, `/rung2` | the rock-paper-scissors, Kuhn and Leduc apps | app |
| `/cover` | `web/cover/cover.html`, the old front page, kept as an archive and unlinked | static, hand-kept |
| `/hero` | `web/hero-studies/`, a share page with video clips | static, hand-kept |

`web/AGENTS.md` keeps the same table. When the two disagree, `vercel.json` is the truth: [`scripts/site/serve_site.py`](repo:scripts/site/serve_site.py) reads its rewrites rather than mirroring them, and [`tests/test_vercel_config.py`](repo:tests/test_vercel_config.py) checks that every app the build emits has a rewrite.

## How a number gets from Python to a page

```eq
solver run  →  committed file (figures/, an export, a release asset)  →  generator or app  →  page
```

- **Solver run.** An analysis command (`kuhn-analysis --json`, `minidraw-ranges`, `minidraw-deals`, `deep-analysis`) or a trainer writes its output.
- **Committed file.** Grades live under `figures/`. The apps' exports live under `web/*/public/`. Copies the apps import live under `web/*/src/data/`. Large frozen strategies are GitHub release assets, pinned by SHA-256. None of these is edited by hand.
- **Generator or app.** The static pages are rewritten by [`make_static_pages.py`](repo:scripts/site/make_static_pages.py). An app imports its data copy or fetches its export at runtime.
- **Tests at each join.** `tests/test_cover_page.py` fails when a committed page differs from what the generator writes now. `tests/test_rung3_page_data.py` holds each `src/data/` copy byte for byte to its `figures/` original. The Python exporter tests hold each manifest's grade to its grades file.

So a re-run that moves a number has to show up as a diff in the committed file, the page, or a red test. It cannot drift silently.

## One look across every surface

Everything the reader sees is one visual system, defined in a single theme file:

- **The theme is one file in five copies.** `web/rung0-viz/src/theme.css` holds the tokens (paper, ink, one oxblood accent), the top bar, the footer, figures, hairline tables and the contents rail. Each of the other four apps carries a byte-identical copy, and the static pages link the build's copy at `/theme.css`. See [the design standard](wiki:ops/design-standard).
- **The chrome is written twice.** `site.tsx` renders the top bar and footer in the apps, and the generator writes the same markup out as HTML for the static pages. `test_the_pages_chrome_says_what_site_tsx_says` checks that the two copies say the same words.
- **The fonts are self-hosted.** Newsreader, Geist and IBM Plex Mono are copied out of rung 0's `@fontsource` packages into the site's `fonts` folder and declared once in `/fonts.css`. Nothing is fetched from another host.

:::cards
- [The static pages](wiki:ops/static-pages) The generator, what each page reads, and how the wiki is rendered.
- [The apps](wiki:ops/apps) Five Vite apps: what each draws and where its data comes from.
- [The design standard](wiki:ops/design-standard) Tokens, motion and the tests that enforce them.
- [Deploying](wiki:ops/deploy) The build script, Vercel's settings and checking it all locally.
:::
