# The cover page

Two self-contained HTML files, served as the site's front door:

- `index.html` — the homepage at `/`. A product fold (the ASCII-rendered chip
  stack beside rung 0's live regret-matching console), the rung-2 Leduc
  strategy explorer, the three verbs of the product (Solve, Verify, Play, each
  with a running widget), and the validation ladder as a walk through the five
  rungs with their real numbers and graphs.
- `rules.html` — the Rules tab at `/rules`. What Drawmaha is and how a hand is
  played, in the same chrome: a scroll-driven raymarched table that deals,
  flops, draws, turns, rivers and splits the pot, the worked hand read two
  ways, and why nobody has solved the game.

Each page inlines its own CSS, its data (the committed rung-1 and rung-2
solves) and its scripts. Nothing here is built; there is no bundler and no
third-party CDN. The only outside requests either page makes are for the
fonts under `/fonts/`.

## Designed outside the repo, then exported

The pages are not authored in this directory. They come out of a studies
workflow that lives beside the repo: rounds of competing homepage studies,
one chosen, then tightened with the Rules tab and built from a source file
plus shared parts (the raymarching kernel, `chip.js`, the inlined solves).
That workflow exports the finished pages with their font URLs rewritten to
`/fonts/<file>`, and the export is what gets checked in here, byte for byte.

So: do not hand-edit `index.html` or `rules.html`. A fix goes into the
studies source, gets re-exported, and replaces both files. The tests in
`tests/test_cover_page.py` check what the export can get wrong silently —
a link to a route that does not exist, a font the build never copies, a
script that does not parse — not the wording.

## The font rule

The pages load eight woff2 files from `/fonts/`, all named the @fontsource
way (`<family>-latin-<weight>-<style>.woff2`), and they reach `public/fonts/`
by two routes:

- `scripts/vercel_build.sh` already copies IBM Plex Mono 400/500/600 and
  Instrument Serif 400 regular out of rung 0's `@fontsource` packages with
  `copy_font`. Those are **not** duplicated here.
- The four files no visualizer bundles — Instrument Sans 400/500/600 (the
  prose voice) and Instrument Serif 400 italic — live in `fonts/` in this
  directory, and the build copies them beside the rest.

A page may only reference a font that one of those two routes produces; the
test enforces it. Adding a fifth voice means adding its file here (or a
`copy_font` line in the build script) in the same change.

## Routes

`vercel.json` serves `public/` filesystem-first, so:

| URL            | Served                             |
|----------------|------------------------------------|
| `/`            | `public/index.html` (no rewrite)   |
| `/rules`       | rewritten to `/rules.html`         |
| `/rules.html`  | the file itself                    |
| `/index.html`  | the file itself                    |
| `/fonts/*`     | the woff2 files                    |

The pages link to each other relatively (`rules.html`, `index.html#ladder`),
which resolves at the root either way, and to the visualizers absolutely
(`https://drawmaha.app/rung0` and so on); every such link must match a
rewrite source in `vercel.json` or the test fails.

## Checking them locally

The build script is the real thing (`bash scripts/vercel_build.sh`, then
`uv run python scripts/serve_site.py`), but it runs three `npm ci` builds
first. To look at just these two pages, mimic `public/` in a temp dir and
serve it:

```bash
tmp=$(mktemp -d)
cp web/cover/index.html web/cover/rules.html "$tmp/"
mkdir "$tmp/fonts"
cp web/cover/fonts/*.woff2 "$tmp/fonts/"
# the four the build copies from @fontsource (after an npm ci in web/rung0-viz):
f=web/rung0-viz/node_modules/@fontsource
cp $f/instrument-serif/files/instrument-serif-latin-400-normal.woff2 "$tmp/fonts/"
for w in 400 500 600; do cp $f/ibm-plex-mono/files/ibm-plex-mono-latin-$w-normal.woff2 "$tmp/fonts/"; done
(cd "$tmp" && python3 -m http.server 8790)
```

Then open `http://127.0.0.1:8790/` and `http://127.0.0.1:8790/rules.html`.
The `/rules` rewrite is Vercel's and will not resolve under `http.server`;
the in-page links use `rules.html`, so the Rules tab and its back link work.
What to look for: an empty devtools console, every `/fonts/` request a 200,
and no request leaving the host.
