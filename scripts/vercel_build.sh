#!/usr/bin/env bash
# Vercel buildCommand entrypoint (vercel.json) — kept in a script because
# Vercel caps buildCommand at 256 characters.
# Builds each rung's visualizer into public/<rung>, copies the woff2 files the
# site self-hosts at /fonts, and lays the static cover page (web/cover) at the
# root of public/.
#
# None of that output is reachable unless vercel.json also pins
# `"framework": null`. The repo root has a pyproject.toml, so Vercel otherwise
# auto-detects the "python" preset, and a backend-framework project appends a
# `/(.*) -> /python` catch-all that hands every path to a serverless function
# that no longer exists — the visualizers, the fonts and the cover page build
# fine and then never get served.
set -euo pipefail

rungs=(rung0 rung1 rung2 rung3)

for rung in "${rungs[@]}"; do
  (cd "web/$rung-viz" && npm ci && npm run build)
done

rm -rf public
mkdir -p public/fonts
for rung in "${rungs[@]}"; do
  cp -r "web/$rung-viz/dist" "public/$rung"
done

# The hero studies: a static page and its clips, served at /hero.
cp -r web/hero-studies public/hero

# Both visualizers bundle the same @fontsource files, so either copy serves the
# site; rung 0's is the one that has always been here.
#
# Two of the site's three voices come from here: the display face (Plus
# Jakarta Sans) ships at one upright weight, mono is the UI and every number so
# it needs three. (The visualizers bundle their own copies of these plus
# Instrument Sans, so /fonts/ serves only the static pages.) Instrument Serif
# is no longer a voice of the site; it is copied for the hero studies at /hero,
# which are kept as they were, and rung 0 keeps the package for that reason.
fontsource=web/rung0-viz/node_modules/@fontsource
copy_font() {
  cp "$fontsource/$1/files/$1-latin-$2-normal.woff2" public/fonts/
}
copy_font plus-jakarta-sans 600
copy_font instrument-serif 400
for weight in 400 500 600; do copy_font ibm-plex-mono "$weight"; done

# The homepage, its Rules tab and the Cover are three self-contained HTML
# files (see web/cover/README.md: the first two are exported from the studies
# workflow, the cover is authored here), so there is nothing to build: they
# are copied as-is to the root of public/, where Vercel serves index.html at /
# with no rewrite and /rules and /cover are rewritten to their files.
#
# They add a fourth voice, Instrument Sans for prose, and the display face's italic.
# copy_font above cannot reach those (the visualizers bundle them into their
# own assets rather than /fonts/), so the four files are checked in beside the
# pages and land next to the rest. Their names follow the @fontsource pattern
# so one /fonts/ directory stays uniform.
cp web/cover/index.html public/index.html
cp web/cover/rules.html public/rules.html
cp web/cover/cover.html public/cover.html
cp web/cover/fonts/*.woff2 public/fonts/
