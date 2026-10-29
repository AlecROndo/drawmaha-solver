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

rungs=(rung0 rung1 rung2 rung3 solver)

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

# Every app bundles the same @fontsource files, so any copy serves the site;
# rung 0's is the one that has always been here.
#
# The site's voices: Newsreader (serif; 400, 500 and the 400 italic) for display
# and the research body, Geist (sans; 400, 500, 600) for the UI and every
# number, and IBM Plex Mono 400 for one job, the chip's glyph grid. The static
# pages read them through /fonts.css, so what is copied here and what that file
# declares are the same seven files. Instrument Serif is not a voice of the
# site; it is copied for the hero studies at /hero, which are kept as they were,
# and rung 0 keeps the package for that reason.
fontsource=web/rung0-viz/node_modules/@fontsource
copy_font() {
  cp "$fontsource/$1/files/$1-latin-$2-normal.woff2" public/fonts/
}
copy_font_italic() {
  cp "$fontsource/$1/files/$1-latin-$2-italic.woff2" public/fonts/
}
for weight in 400 500; do copy_font newsreader "$weight"; done
copy_font_italic newsreader 400
for weight in 400 500 600; do copy_font geist-sans "$weight"; done
copy_font ibm-plex-mono 400
copy_font instrument-serif 400

# The static pages are plain HTML written in the repo (web/cover/README.md):
# the homepage, Rules, Benchmarks, Releases and the Cover. There is nothing to
# build; they are copied as-is to the root of public/, where Vercel serves
# index.html at / with no rewrite and the others are rewritten to their files.
# They share the apps' theme rather than inlining a copy: the same theme.css
# the five apps carry is served at /theme.css, the @font-face rules for the
# files above at /fonts.css, and the chip's script at /chip.js.
cp web/rung0-viz/src/theme.css public/theme.css
cp web/cover/fonts.css public/fonts.css
cp web/cover/chip.js public/chip.js
for page in index rules benchmarks releases cover; do
  cp "web/cover/$page.html" "public/$page.html"
done
