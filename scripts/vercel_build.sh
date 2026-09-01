#!/usr/bin/env bash
# Vercel buildCommand entrypoint (vercel.json) — kept in a script because
# Vercel caps buildCommand at 256 characters.
# Builds each rung's visualizer into public/<rung> and copies the IBM Plex
# woff2 files the cover page (api/index.py) self-hosts at /fonts.
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
fontsource=web/rung0-viz/node_modules/@fontsource
for fam in sans mono; do
  for weight in 400 500 600; do
    cp "$fontsource/ibm-plex-$fam/files/ibm-plex-$fam-latin-$weight-normal.woff2" public/fonts/
  done
done
