# The cover pages

Three self-contained HTML files, served as the site's front door. Two are
exports of a design workflow that lives beside the repo and are never edited
here; the third is authored in this directory by hand:

- `index.html` — the homepage at `/`. A product fold (the ASCII-rendered chip
  stack beside rung 0's live regret-matching console), the rung-2 Leduc
  strategy explorer, the three verbs of the product (Solve, Verify, Play, each
  with a running widget), and the validation ladder as a walk through the five
  rungs with their real numbers and graphs.
- `rules.html` — the Rules tab at `/rules`. What Drawmaha is and how a hand is
  played, in the same chrome: a scroll-driven raymarched table that deals,
  flops, draws, turns, rivers and splits the pot, the worked hand read two
  ways, and why nobody has solved the game.
- `cover.html` — the Cover at `/cover`: the site's original front page, kept
  and rebuilt in the same theme. The homepage's nav and `chip.js` verbatim,
  the two ASCII windows from `web/hero-studies` (the hand from the sidelines,
  five giants at the bar) on an oxblood field, one window of numbers per
  solved rung, and the ladder as hairline rows. Hand-authored here, not
  exported.

Each page inlines its own CSS, its data (the committed rung-1 and rung-2
solves) and its scripts. Nothing here is built; there is no bundler and no
third-party CDN. The only outside requests any page makes are for the fonts
under `/fonts/`.

## Two exported, one authored

The homepage and the Rules tab are not authored in this directory. They come
out of a studies workflow that lives beside the repo: rounds of competing
homepage studies, one chosen, then tightened with the Rules tab and built
from a source file plus shared parts (the raymarching kernel, `chip.js`, the
inlined solves). That workflow exports the finished pages with their font
URLs rewritten to `/fonts/<file>`, and the export is what gets checked in
here, byte for byte.

So: do not hand-edit `index.html` or `rules.html`. A fix goes into the
studies source, gets re-exported, and replaces both files. `cover.html` is
the exception: it is written here, so a fix to it is an ordinary edit. The
tests in `tests/test_cover_page.py` check all three for what a static page
can get wrong silently — a link to a route that does not exist, a font the
build never copies, a script that does not parse — not the wording.

One standing exception. The studies source predates #42, so the bowl
settling in the Solve widget (#42), the rung-0 triangle drawing that same
bowl and the Rules tab's beat anchor (#43) were made here, in the exported
files, because the source has no bowl code to fix. Until those edits are
ported into the source, a re-export would silently put the old code back;
`test_the_hand_edits_survived_the_last_export` pins each one by a line only
the new code has, so the revert fails the suite instead. Port the edits and
re-export to retire that test.

The UI standard (`web/DESIGN.md`) was likewise applied to the exported pages
by hand: the motion tokens in `:root`, the `:active` press on every
pressable, the tightened Rungs menu, the `--ease-out` scroll reveal and the
hover gate on the ladder's arrow. `tests/test_ui_standard.py` checks all of
that on the inline `<style>` blocks, so a re-export from the stale source
fails there too.

## The font rule

The pages load eight woff2 files from `/fonts/`, all named the @fontsource
way (`<family>-latin-<weight>-<style>.woff2`), and they reach `public/fonts/`
by two routes:

- `scripts/vercel_build.sh` copies IBM Plex Mono 400/500/600 and Plus
  Jakarta Sans 600 regular (the display voice) out of rung 0's `@fontsource`
  packages with `copy_font`. Those are **not** duplicated here.
- The other four — Instrument Sans 400/500/600 (the prose voice) and Plus
  Jakarta Sans 600 italic — live in `fonts/` in this directory, and the
  build copies them beside the rest. (The visualizers bundle their own copies
  of all eight into their `assets/`; `/fonts/` serves only these pages.)

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
| `/cover`       | rewritten to `/cover.html`         |
| `/cover.html`  | the file itself                    |
| `/index.html`  | the file itself                    |
| `/fonts/*`     | the woff2 files                    |

The pages link to each other relatively (`rules.html`, `cover.html`,
`index.html#rung3`), which resolves at the root either way, and to the
visualizers absolutely (`https://drawmaha.app/rung0` and so on); every such
link must match a rewrite source in `vercel.json` or the test fails.

## Checking them locally

The build script is the real thing (`bash scripts/vercel_build.sh`, then
`uv run python scripts/serve_site.py`), but it runs three `npm ci` builds
first. To look at just these three pages, mimic `public/` in a temp dir and
serve it:

```bash
tmp=$(mktemp -d)
cp web/cover/index.html web/cover/rules.html web/cover/cover.html "$tmp/"
mkdir "$tmp/fonts"
cp web/cover/fonts/*.woff2 "$tmp/fonts/"
# the four the build copies from @fontsource (after an npm ci in web/rung0-viz):
f=web/rung0-viz/node_modules/@fontsource
cp $f/plus-jakarta-sans/files/plus-jakarta-sans-latin-600-normal.woff2 "$tmp/fonts/"
for w in 400 500 600; do cp $f/ibm-plex-mono/files/ibm-plex-mono-latin-$w-normal.woff2 "$tmp/fonts/"; done
(cd "$tmp" && python3 -m http.server 8790)
```

Then open `http://127.0.0.1:8790/`, `http://127.0.0.1:8790/rules.html` and
`http://127.0.0.1:8790/cover.html`. The `/rules` and `/cover` rewrites are
Vercel's and will not resolve under `http.server`; the in-page links use the
file names, so the tabs and their back links work.
What to look for: an empty devtools console, every `/fonts/` request a 200,
and no request leaving the host.
