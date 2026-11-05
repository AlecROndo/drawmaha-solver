---
title: The five apps
description: Vite + React + TypeScript visualizers that draw solves exported from Python, transcribe only rules arithmetic, and share one copied chrome.
---

Five apps live under `web/`, one per interactive surface. Each is a Vite + React + TypeScript project with the same four commands:

```bash
cd web/<app>
npm install
npm test        # vitest
npm run dev     # local dev server
npm run build   # tsc -b && vite build, into dist/
```

## The apps never run CFR

No app reimplements CFR or anything else that touches the solver. Each one draws a solve that Python exported (`kuhn-analysis --json`, `leduc-analysis --json`, `minidraw-deals`, `minidraw-ranges`), or it POSTs a locked strategy to a local Python exploit server and draws the trace that comes back. The reason is the same one that keeps graders apart from solvers ([Exploitability](wiki:theory/exploitability)). A second copy of the reach-weighted tree walk is exactly the routine where a swapped weight converges, silently, to a wrong answer.

What an app may transcribe is **rules arithmetic**: payoffs, the betting grammar, the hand-ranking key. Each transcription has a `*.test.ts` that pins it against the same tables the Python suite pins, so a port that disagrees with Python fails a test.

## What each app is

| App | Route | Draws | Data |
| --- | --- | --- | --- |
| `rung0-viz` | `/rung0` | regret matching learning rock-paper-scissors live | none needed; the 63-line ledger is small enough to port safely |
| `rung1-viz` | `/rung1` | Kuhn's twelve information sets converging, plus the Exploit tab | `kuhn-analysis --json`, copied into `src/data/`; Exploit posts to `kuhn-exploit-server` |
| `rung2-viz` | `/rung2` | Leduc's solve on a timeline, a playable table, an exploit tab | `leduc-analysis --json`; `leduc-exploit-server` |
| `rung3-viz` | `/research`, `/learn`, `/play`, `/rung3` | the research publication, the Learn range interface and the trainer | copies of `figures/rung3/` and `figures/rung4/` under `src/data/`; the deal pack under `public/deals/` |
| `solver-viz` | `/solver` | the range explorer: mini-drawmaha exactly, full Drawmaha with an illustrative policy | three exports under `public/mini*/` written by `minidraw-ranges` |

### `rung3-viz`: one app, three tabs

The tab comes from the path (`/research`, `/learn`, `/play`) or, at `/rung3`, from the hash. `parseLocation` in `src/learn.ts` reads it.

- **Research** is a publication, not a dashboard: a masthead whose results strip computes its three numbers from the data, a contents rail with scroll-spy, nine chapters and Methods. Its numbers come from exactly three modules, `src/research.ts`, `src/deep.ts` and `src/reach.ts`, each with a vitest file that pins what the prose quotes.
- **Learn** is `/solver`'s mini-drawmaha range interface, *imported* from `../solver-viz/src/ui/MiniRange.tsx`, not copied. A change to the range interface is made in `solver-viz` and shows up here.
- **Play** deals from the pack written by `uv run minidraw-deals`: 600 deals in six chunks, each carrying the shipped strategy's mix at every node either player can reach. The bot is the release `deal_pack.PACK_RELEASE` names, which is the Deep CFR exact average from the ÷ √pot arm at iteration 200.

:::warning
`rung3-viz` builds *before* `solver-viz` on Vercel, and the two install separately. So a `solver-viz` module reachable from `rung3-viz` may import only `react` and relative paths. `tests/test_vercel_config.py` walks the import graph and fails on anything else. Such an import would pass every local check (both apps installed) and then resolve to nothing on Vercel.
:::

Learn fetches its exports from `/solver/mini*/`, the one copy on the site, so `npm run dev` inside `rung3-viz` cannot load them. To drive Learn, build the whole site and serve it ([Deploying](wiki:ops/deploy)).

### `solver-viz`: two games, not equally real

- **Mini-drawmaha is exact.** Three strategies sit behind one switch: the rung 3 tabular LCFR, and rung 4's Deep CFR at iteration 500 as Π (the policy net) and as the exact average. The three exports share their key tables and point walk byte for byte and differ only in the chunks, so switching changes the strategy under a spot while the line and the filters stay. The browser rebuilds the actor's range from the export, weighted by the actor's own action probabilities along the line.
- **Full Drawmaha is the interface built ahead of the network.** The range's composition (`src/engine/sample.ts`) is sampled and real. The **action mix is illustrative**, a smooth function of each hand's two equities in `src/engine/policy.ts`, and every view drawing it says so in its status line. When the network exists, that is the file that changes.

The ports of the Python key, hand ranking, betting grammar and public tree (`src/mini/canonical.ts`, `ranking.ts`, `grammar.ts`, `points.ts`) are pinned to `src/mini/fixtures.json`. The Python exporter writes that file only when told (`--write-fixtures`). When the Python key or grammar changes, re-pin the fixtures on purpose and re-export every strategy. Never patch a port to agree with stale data.

## Five files are one file

Five files are copied, byte for byte, into every app: `src/theme.css`, `src/main.tsx`, `src/ui/site.tsx` (the top bar, page head, figures and footer), `src/ui/site.test.tsx` and `src/ui/mark.ts` (the turning chip). They are copied rather than extracted into a shared package on purpose. The apps deploy independently, and a workspace would buy about 300 lines at the cost of a build step.

The price is drift, and [`tests/test_shared_chrome.py`](repo:tests/test_shared_chrome.py) is what pays it. Edit rung 0's copy, `cp` it to the other four, and `cmp` them. Per-app styling goes in that app's own `src/index.css`, on top of the theme.

## Adding a surface

A new app or page touches, in the same change:

1. `vercel.json`: a rewrite for its route.
2. `scripts/site/vercel_build.sh`: the `rungs` list for an app, or the page loop for a static page.
3. The `APPS` tuples in `tests/test_shared_chrome.py` and `tests/test_ui_standard.py`, or `PAGES` there and in `tests/test_cover_page.py` for a page.
4. If the nav names it: `NAV` and the `Surface` type in `site.tsx` (copied to all five apps), and `NAV` in the generator, then regenerate the pages.

`test_every_rung_the_build_script_emits_has_a_rewrite` exists because a surface added to the build but not to the rewrites builds fine and is then never served.

:::cards
- [The design standard](wiki:ops/design-standard) The tokens and motion rules every app's stylesheet is held to.
- [Releases](wiki:ops/releases) Where the strategies the apps ship come from.
:::
