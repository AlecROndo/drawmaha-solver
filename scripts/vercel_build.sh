#!/usr/bin/env bash
# Vercel buildCommand entrypoint (vercel.json) — kept in a script because
# Vercel caps buildCommand at 256 characters.
# Builds the rung-0 visualizer into public/rung0 and copies the IBM Plex
# woff2 files the cover page (api/index.py) self-hosts at /fonts.
set -euo pipefail

cd web/rung0-viz
npm ci
npm run build
cd ../..

rm -rf public
mkdir -p public/fonts
cp -r web/rung0-viz/dist public/rung0

fontsource=web/rung0-viz/node_modules/@fontsource
for fam in sans mono; do
  for weight in 400 500 600; do
    cp "$fontsource/ibm-plex-$fam/files/ibm-plex-$fam-latin-$weight-normal.woff2" public/fonts/
  done
done
