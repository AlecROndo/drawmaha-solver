---
title: Quickstart
description: Install the package, run the suite, solve a game, play against it, and see the site locally.
---

Everything below runs from the repository root with uv. Python 3.12 or later; NumPy and matplotlib are the only heavy dependencies of the default install, and torch comes in only with the `deep` extra.

## Install and run the suite

```bash
uv sync                  # Python 3.12+; add --extra deep for torch (rung 4 training only)
uv run pytest            # the whole suite; needs no torch, Modal, OpenSpiel or network
```

The full default suite takes about seven minutes on a laptop (`tests/AGENTS.md`). The fast loop is one rung's directory with `-x`:

```bash
uv run pytest -x tests/kuhn
```

## Solve a game and redraw its figures

Each rung has an analysis command that solves it and rewrites its figures under `figures/<rung>/`. Rungs 0 to 2 run unseeded and deterministically, so a clean run leaves the working tree clean; a changed number shows up in `git diff`.

| Command | What it does | Time |
| --- | --- | --- |
| `uv run rps-analysis` | rock-paper-scissors self-play and the biased-opponent run → `figures/rung0/` | seconds |
| `uv run kuhn-analysis` | Kuhn to 100k iterations, 60 checkpoints → `figures/rung1/`, prints the answer sheet | seconds |
| `uv run leduc-analysis` | Leduc to 2,000 iterations → `figures/rung2/` | about two minutes |
| `uv run minidraw-analysis` | reads the committed grades and the frozen release → `figures/rung3/` and `answer_sheet.json` | about two minutes |
| `uv run minidraw-exploit-by-reach --release tabular` | where mini-drawmaha's exploitability lives, binned by reach | minutes; downloads the release |
| `uv run deep-analysis` | the rung 4 price figure → `figures/rung4/price.png` and `answer_sheet.json` | quick; reads committed grades |

`minidraw-analysis` downloads the frozen strategy (`rung3-lcfr-2m`, 18 MB compressed) once and checks it against a SHA-256 pinned in `src/drawmaha_solver/minidrawmaha/answer/strategy.py`.

## Play against the solver in the terminal

```bash
uv run rps-play          # against the regret-matching learner
uv run kuhn-play         # against Kuhn's equilibrium; seats alternate each hand
uv run leduc-play        # against Leduc's solved strategy
uv run minidraw-play     # against the frozen mini-drawmaha strategy
```

Every bot plays the **average** strategy, reads only its own infoset, and reveals both hands at the end, folds included.

## Lock a strategy and watch CFR exploit it

The Kuhn and Leduc pages have an Exploit tab that posts your locked strategy to a local Python server and draws the counter-strategy CFR finds.

```bash
uv run kuhn-exploit-server            # the page POSTs locked spots to this
cd web/rung1-viz && npm install && npm run dev   # then http://localhost:5173/rung1/#exploit
```

`leduc-exploit-server` with `web/rung2-viz` works the same way.

## Serve the whole site at localhost:4321

An app alone:

```bash
cd web/rung3-viz && npm install && npm run dev
```

The whole site, wired the way Vercel wires it:

```bash
bash scripts/site/vercel_build.sh          # builds public/: five npm ci builds, then the static pages
uv run python scripts/site/serve_site.py   # http://localhost:4321
```

After changing anything a static page reads — including this wiki — regenerate the pages:

```bash
uv run python scripts/site/make_static_pages.py
```

:::warning
Do not run the Modal launchers to try things out. `scripts/minidrawmaha/modal_rung3.py` and `scripts/deepcfr/modal_rung4.py` start paid containers, and they are launched by a person. See [Modal](wiki:ops/modal).
:::

:::cards
- [Onboarding](wiki:overview/onboarding) The rules every change follows.
- [Testing](wiki:ops/testing) The gated slow tests and what each env var unlocks.
- [The site](wiki:ops/site) How the pages and apps are built.
:::
