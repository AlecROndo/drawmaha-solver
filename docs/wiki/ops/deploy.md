---
title: Deploying to Vercel
description: One build script lays every app, font and static page into public/, framework detection is pinned off, and serve_site.py reproduces the rewrites locally.
---

The site deploys on Vercel from [`vercel.json`](repo:vercel.json). There is no server and no framework: Vercel runs one build script, serves the folder it writes, and applies a list of rewrites. Every PR gets a preview deployment through Vercel's GitHub integration, which shows up as a status on the PR.

## What `vercel.json` says

```json
{
  "framework": null,
  "buildCommand": "bash scripts/site/vercel_build.sh",
  "outputDirectory": "public",
  "rewrites": [{ "source": "/rules", "destination": "/rules.html" }, "…"]
}
```

:::warning
**`"framework": null` is load-bearing.** The repo root has a `pyproject.toml`, so Vercel would otherwise auto-detect its Python preset. That preset appends a `/(.*) → /python` catch-all that sends every path to a serverless function that does not exist. The apps, the fonts and the pages would all build fine and then never be served. `test_framework_detection_is_pinned_off` holds it.
:::

The build command lives in a script because Vercel caps `buildCommand` at 256 characters.

## The build script

[`scripts/site/vercel_build.sh`](repo:scripts/site/vercel_build.sh), in order:

1. **Builds each app.** For `rung0 rung1 rung2 rung3 solver`, it runs `npm ci && npm run build` in `web/<app>-viz` and copies `dist/` to `public/<app>`. `rung3` builds before `solver` (`test_rung3_builds_before_solver_in_the_build_script`).
2. **Copies the hero studies** to `public/hero`.
3. **Copies the fonts.** Every app bundles the same `@fontsource` packages, so the script copies rung 0's woff2 files into the build's `fonts` folder: Newsreader 400, 500 and 400 italic; Geist 400, 500 and 600; IBM Plex Mono 400 and 600; and Instrument Serif, kept only for `/hero`.
4. **Copies the shared assets**: rung 0's `theme.css` to `/theme.css`, plus `fonts.css`, `chip.js` and `favicon.svg` from `web/cover/`.
5. **Copies the static pages** from `web/cover/` to the root of `public/`. Vercel serves `index.html` at `/` without a rewrite, and the rewrites map `/rules`, `/benchmarks`, `/releases`, `/wiki` and `/cover` to their files.

The static pages are not built here. They are generated ahead of time and committed ([The static pages](wiki:ops/static-pages)). So a deploy can never produce a page that differs from the one reviewed in the PR.

## The deploy contract, as tests

| Test in `tests/test_vercel_config.py` | Holds |
| --- | --- |
| `test_framework_detection_is_pinned_off` | `"framework": null` |
| `test_static_output_is_served_from_public` | the output folder is `public/` |
| `test_the_cover_page_is_served_from_the_filesystem` | `/` needs no rewrite |
| `test_the_product_paths_reach_their_pages` | `/research`, `/learn`, `/play` and the other product routes land on their files |
| `test_every_rung_the_build_script_emits_has_a_rewrite` | the `rungs=(…)` list and the rewrites move together |
| `test_what_rung3_imports_from_solver_viz_imports_only_react` | a sibling-app import that would resolve on a laptop but not on Vercel |

## Checking the whole site locally

```bash
bash scripts/site/vercel_build.sh          # build public/ (five npm ci builds first)
uv run python scripts/site/serve_site.py   # http://localhost:4321
```

[`serve_site.py`](repo:scripts/site/serve_site.py) serves `public/` filesystem-first and applies the rewrites it reads from `vercel.json`. It does not mirror them, because a hand copy silently missed each new route's rewrite. It is standard library only, so it can never break on the solver's numeric stack. This is the only local way to drive `/learn`, which fetches its exports from `/solver/mini*/`.

## Debugging a deploy

| Symptom | Usual cause |
| --- | --- |
| a route 404s on the preview but works locally in `npm run dev` | a missing rewrite in `vercel.json`, or the app missing from `rungs=(…)` |
| every path returns a function error | framework detection is back on |
| an app's build fails on Vercel only | an import from a sibling app that only resolves when both are installed |
| text renders in a fallback face | a font that `fonts.css` declares but the build does not copy |
| a static page shows an old number | the generator was not re-run after a data change; `test_the_pages_are_what_the_generator_writes` fails locally |
