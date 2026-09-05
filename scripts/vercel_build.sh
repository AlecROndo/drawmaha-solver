#!/usr/bin/env bash
# Vercel buildCommand entrypoint (vercel.json) — kept in a script because
# Vercel caps buildCommand at 256 characters.
# Builds each rung's visualizer into public/<rung> and copies the IBM Plex
# woff2 files the cover page (api/index.py) self-hosts at /fonts.
#
# None of that output is reachable unless vercel.json also pins
# `"framework": null`. The repo root has a pyproject.toml, so Vercel otherwise
# auto-detects the "python" preset, and a backend-framework project appends a
# `/(.*) -> /python` catch-all that hands every path to api/index.py — the
# visualizers and the fonts build fine and then never get served.
set -euo pipefail

rungs=(rung0 rung1)

for rung in "${rungs[@]}"; do
  (cd "web/$rung-viz" && npm ci && npm run build)
done

rm -rf public
mkdir -p public/fonts
for rung in "${rungs[@]}"; do
  cp -r "web/$rung-viz/dist" "public/$rung"
done

# Both visualizers bundle the same @fontsource files, so either copy serves the
# cover page; rung 0's is the one that has always been here.
#
# The weights differ by role, so each family names its own: sans carries the
# 700 the cover's h1 sets in, serif is reading copy at one weight only, and
# mono stops at 500 because it is only ever field labels and figures.
fontsource=web/rung0-viz/node_modules/@fontsource
copy_font() {
  cp "$fontsource/ibm-plex-$1/files/ibm-plex-$1-latin-$2-normal.woff2" public/fonts/
}
for weight in 400 500 600 700; do copy_font sans "$weight"; done
for weight in 400 500; do copy_font mono "$weight"; done
copy_font serif 400
