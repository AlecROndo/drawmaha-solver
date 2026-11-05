---
title: The static pages and their generator
description: Five pages written by make_static_pages.py from committed data, one archived page kept by hand, and the tests that keep the generated ones regenerated.
---

The homepage, Rules, Benchmarks, Releases and this wiki are plain HTML files under `web/cover/`. They are not written by hand. Each one is the output of [`scripts/site/make_static_pages.py`](repo:scripts/site/make_static_pages.py), a standard-library-only Python script that reads committed files and writes the pages. Each page names the generator in a comment above its `<head>`.

Generating them, rather than editing HTML, solves one problem. The pages quote numbers (exploitabilities, game values, release grades), and a hand-typed number goes stale the moment a run is regraded. The generator reads every number from the file that holds it, so regenerating the pages is how they stay true.

## Changing a page means changing the generator

```bash
uv run python scripts/site/make_static_pages.py
```

Edit the generator (or the data it reads), run it, and commit the regenerated pages in the same change. [`tests/test_cover_page.py`](repo:tests/test_cover_page.py) renders every page in memory and fails if a committed page differs (`test_the_pages_are_what_the_generator_writes`). A hand edit to `index.html` therefore shows up as a test failure, not as a quiet fork.

:::warning
`web/cover/cover.html` is the one exception. It is the archived old front page, served at `/cover` and linked from nowhere. It is edited by hand if at all and never regenerated. It loads fonts the build no longer ships, so it renders in fallback faces by design.
:::

## Each page's data: README tables, figures, the release catalogue

| Page | Reads |
| --- | --- |
| `index.html` (`/`) | the README's results table; `figures/rung3/` and `figures/rung4/` grades for the headline chart; the frozen table's export under `web/solver-viz/public/mini/` (the opening matrix is rebuilt from it and checked against the answer sheet); the first chunk of the Play tab's deal pack, for a dealt hand |
| `rules.html` (`/rules`) | the rules, written in the generator |
| `benchmarks.html` (`/benchmarks`) | the README's results and Kuhn tables; the Kuhn and Leduc solves the live demos draw; the rung 3 and rung 4 grade files |
| `releases.html` (`/releases`) | the release catalogue in `src/drawmaha_solver/minidrawmaha/answer/releases.py`, read with `ast` rather than imported, plus each release's grade file |
| `wiki.html` (`/wiki`) | every page under `docs/wiki/`, in the order `docs/wiki/nav.json` gives |

Two kinds of fact have no committed home: the publication dates of the release tags, and the dates in the homepage's research index. They are set in the generator beside a comment naming their source (`RESEARCH_DATES`, `PUBLISHED`).

## The wiki is a generated page too

This wiki is written as markdown under `docs/wiki/`. [`scripts/site/wiki.py`](repo:scripts/site/wiki.py) renders that markdown, and the generator wraps the result in the site's chrome as `web/cover/wiki.html`. The source layout, the markdown dialect (callouts, cards, `eq` blocks, `wiki:` and `repo:` links) and the writing rules are in [`docs/wiki/README.md`](repo:docs/wiki/README.md). An edit to a wiki page follows the same loop as any other page: edit the markdown, regenerate, commit both.

## Every page loads the theme, the fonts, the chip and the favicon

- `/theme.css` is rung 0's theme, copied by the build. The chrome, tokens and type come from there, and each page adds one inline `<style>` block of its own layout and chart classes (the generator's `BASE_CSS` plus a per-page sheet).
- `/fonts.css` declares the `@font-face` rules for the files in the site's `fonts` folder. A page never names a font file itself.
- `/chip.js` is the plain-JS twin of the apps' `src/ui/mark.ts`. Each page mounts the turning chip twice, once in the top bar and once in the footer.
- `/favicon.svg` is the chip, flat.

## What `test_cover_page.py` holds

| Check | Why it exists |
| --- | --- |
| every `href` reaches a route `vercel.json` serves, or the repository | a link to a page that was never deployed is the commonest silent break |
| every `#anchor` into a page lands on an id that page has | anchors break invisibly when a heading is renamed |
| nothing loads from another host | no CDN; the site works from its own origin |
| `fonts.css` declares exactly the files the build copies, and the faces the apps bundle | a declared font that is never served falls back silently |
| each page links the theme, the fonts and the chip, and mounts the chip twice | the chrome comes from one place |
| `chip.js` draws the same chip as `mark.ts` | two hand-kept copies of one drawing |
| every inline script parses | a syntax error kills the page's scripts with no build step to catch it |
| the product pages never say "rung", "ladder" or "climb" | the method's words belong in the README and this wiki; the product pages speak the product's words |
| the pages are byte for byte what the generator writes | no hand edits |
| the top bar and footer say what `site.tsx` says | the chrome is written twice |

The design rules (motion, tokens) for the pages' inline styles are checked separately by `tests/test_ui_standard.py`. See [the design standard](wiki:ops/design-standard).

## Looking at the pages locally

The real check is the full build (see [Deploying](wiki:ops/deploy)), but that runs five `npm ci` builds first. To look at the static pages alone, copy them into a temp folder shaped like `public/` and serve it with `python3 -m http.server`. [`web/cover/README.md`](repo:web/cover/README.md) has the exact commands under "Checking them locally". Extensionless routes (`/rules`) are Vercel rewrites and will not resolve there, so open each page by file name. Look for an empty console, a 200 for every font request, and no request leaving the host.

:::cards
- [The apps](wiki:ops/apps) The other half of the site.
- [Deploying](wiki:ops/deploy) How public/ is built and served.
:::
