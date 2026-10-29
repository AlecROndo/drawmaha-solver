# The static pages

Five plain HTML files served at the root of the site, beside the five Vite
apps. Four are the product's pages, written by a generator in this repo; the
fifth is the old front page, kept as an archive:

| File | Route | What it is |
|---|---|---|
| `index.html` | `/` | the homepage: the current result as a chart, the results table, the opening spot's range matrix, the research index, a dealt hand, the method |
| `rules.html` | `/rules` | the rules of Drawmaha: the streets as a ruler, one hand read two ways, why the game is unsolved |
| `benchmarks.html` | `/benchmarks` | the four graded games on one page: the results table, then each game's chart, grade and link to its live demo |
| `releases.html` | `/releases` | every released strategy with its tag, grade, the pages that ship it and the command that exports it |
| `cover.html` | `/cover` | the original front page, archived with its own inlined chrome; unlinked from the nav and not regenerated |

## Written by a generator, never by hand

`index.html`, `rules.html`, `benchmarks.html` and `releases.html` are the
output of `scripts/site/make_static_pages.py`, and each names it in a comment
above its `<head>`. To change a page, change the generator and run

```bash
uv run python scripts/site/make_static_pages.py
```

The generator needs only the standard library. Every number on the pages is
read from a committed file: the README's results and Kuhn tables, the grades
under `figures/rung3/` and `figures/rung4/`, the Kuhn and Leduc solves the
live demos draw, the frozen table's export under
`web/solver-viz/public/mini/` (the homepage matrix is rebuilt from it and
checked against the answer sheet), the first chunk of the Play tab's deal
pack, and the release catalogue in
`src/drawmaha_solver/minidrawmaha/answer/releases.py`, read with `ast`. The
release tags' publication dates and the research index's dates have no
committed home and are set, with their source, in the generator. Rerun it
whenever any of those files changes; `tests/test_cover_page.py` renders the
pages in memory and fails if a committed page differs.

`cover.html` is the exception: it is an archive, edited by hand if at all.

## What a page loads

- `/fonts.css` — `fonts.css` in this directory: the `@font-face` rules for the
  seven woff2 files `scripts/site/vercel_build.sh` copies to `/fonts/` out of
  rung 0's `@fontsource` packages (Newsreader 400, 500 and 400 italic; Geist
  Sans 400, 500, 600; IBM Plex Mono 400 for the chip). A page never names a
  font file itself.
- `/theme.css` — the apps' theme, `web/rung0-viz/src/theme.css`, copied by the
  build. The top bar, the footer, buttons, figures, hairline tables and the
  contents rail are styled there; each page adds one inline `<style>` of page
  layout and chart classes on top.
- `/chip.js` — `chip.js` in this directory, the plain-JS twin of the apps'
  `src/ui/mark.ts`. Each page mounts the turning chip twice with
  `mountChip(canvas, { font, speed, mark, ink })`: in the top bar and where the
  footer closes the page.

The top bar and the footer are the markup `site.tsx` renders in the apps,
written out as HTML with the same classes, with `aria-current="page"` on the
current nav item. Nothing is fetched from another host; the only outside
links are to the repository on GitHub.

## What the tests hold

`tests/test_cover_page.py`: every link reaches a route `vercel.json` serves
(or the repository), every anchor into a page lands on an id it has, nothing
loads from another host, `fonts.css` declares exactly what the build copies,
each page loads the theme, the fonts and the chip and mounts it twice, every
inline script parses, no page says "rung", "ladder" or "climb" in its copy
(the method's words stay in the README), and the four pages are byte for byte
what the generator writes. `tests/test_ui_standard.py` reads each page's
inline style together with the linked theme and holds it to `web/DESIGN.md`.

## Checking them locally

The real thing is the build (`bash scripts/site/vercel_build.sh`, then
`uv run python scripts/site/serve_site.py`, at `http://localhost:4321`), but
it runs five `npm ci` builds first. To look at just these pages, mimic
`public/` in a temp dir and serve it:

```bash
tmp=$(mktemp -d); mkdir "$tmp/fonts"
cp web/cover/{index,rules,benchmarks,releases,cover}.html web/cover/fonts.css web/cover/chip.js "$tmp/"
cp web/rung0-viz/src/theme.css "$tmp/theme.css"
f=web/rung0-viz/node_modules/@fontsource   # after `npm install` in web/rung0-viz
for w in 400 500; do cp $f/newsreader/files/newsreader-latin-$w-normal.woff2 "$tmp/fonts/"; done
cp $f/newsreader/files/newsreader-latin-400-italic.woff2 "$tmp/fonts/"
for w in 400 500 600; do cp $f/geist-sans/files/geist-sans-latin-$w-normal.woff2 "$tmp/fonts/"; done
cp $f/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2 "$tmp/fonts/"
(cd "$tmp" && python3 -m http.server 8790)
```

Then open `http://127.0.0.1:8790/index.html` and the other three by file
name. The extensionless routes (`/rules`, `/benchmarks`, `/releases`) and the
apps they link to are Vercel's rewrites and will not resolve under
`http.server`. What to look for: an empty console, every `/fonts/` request a
200, and no request leaving the host.
