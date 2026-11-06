---
title: Package map
description: Where every piece of the solver lives, the five layers each game repeats, and the commands that run them.
---

Everything the solver does lives in one Python package, [`src/drawmaha_solver/`](repo:src/drawmaha_solver), with one subpackage per graded game and three small modules they all share. The games grow from rock-paper-scissors to mini-drawmaha, but each subpackage is built from the same five layers, so once you can read one you can read them all.

## One subpackage per graded game

| Subpackage | Game | Solver | Ground truth it is graded against |
| --- | --- | --- | --- |
| [`rps/`](repo:src/drawmaha_solver/rps) | rock-paper-scissors | regret matching, self-play | the uniform Nash, closed form |
| [`kuhn/`](repo:src/drawmaha_solver/kuhn) | Kuhn poker, 12 infosets | vanilla CFR | the closed-form equilibrium family; all 64 pure strategies |
| [`leduc/`](repo:src/drawmaha_solver/leduc) | Leduc poker, 288 infosets | vanilla CFR | OpenSpiel, frozen in `tests/leduc/referee.json` |
| [`minidrawmaha/`](repo:src/drawmaha_solver/minidrawmaha) | mini-drawmaha, 6,220,050 infosets | lockstep external-sampling MCCFR | the exact best-response walk (no outside engine exists) |
| [`deepcfr/`](repo:src/drawmaha_solver/deepcfr) | mini-drawmaha again | Deep CFR | the same exact walk, and the tabular answer as the yardstick |

`minidrawmaha/` is the only one split further, into `rules/`, `solver/` and `answer/`, because it is the first game too large for one file per layer. See [Mini-drawmaha rules](wiki:implementation/minidrawmaha-rules).

## The three shared modules

| Module | What it is for |
| --- | --- |
| [`regret_matching.py`](repo:src/drawmaha_solver/regret_matching.py) | **`RegretMatcher`**, the ledger every game trains. It stores how much better each action would have done than the mix actually played, and turns that into the next strategy. The learning rule never changes as the games grow; only what feeds it does. See [The regret ledger](wiki:implementation/ledger). |
| [`play_session.py`](repo:src/drawmaha_solver/play_session.py) | The bookkeeping around a hand that the terminal tables share. **`QuitGame`** is raised from any prompt where you type `q`, so the hand in progress is dropped rather than scored. **`Scoreboard`** tallies chips from the human's side, per seat as well as in total, because no poker game here is symmetric. **`verdict`** is the one line you read at the end of each hand. |
| [`plotting.py`](repo:src/drawmaha_solver/plotting.py) | The house figure style. **`new_axes`**, **`legend`** and **`save`** give every game's figures the same axes, three categorical inks and neutral everything else, so you can compare a rock-paper-scissors curve with a Kuhn one without re-reading the axes. **`log_indices`** picks log-spaced checkpoints from a long trajectory. |

## Every game repeats five layers

The package's own [`AGENTS.md`](repo:src/drawmaha_solver/AGENTS.md) names the layering. The README section for each game explains the same layers in the same order:

| Layer | File | Job |
| --- | --- | --- |
| 1. The rules | `game.py` | What the game is: the deck, legal actions, chance, payoffs and the **infoset** (what the player to act can see). No solver code at all. |
| 2. The memory | `infoset_table.py` | One ledger per infoset, all allocated up front so that no code path can invent a new one. |
| 3. The solver | `cfr.py` (Kuhn, Leduc), `mccfr.py` + `lockstep.py` (mini-drawmaha) | The walk that computes each action's utility and banks it into the ledgers. |
| 4. The meter | `exploitability.py` | How much a perfect adversary wins against the strategy. **It imports `game.py` and nothing from the solver.** |
| 5. Entry points | `analysis.py`, `play.py`, `exploit_server.py` | Solve and draw the figures; play against the answer in a terminal; serve a live exploit to the site. |

:::warning
The meter never imports the solver. If the two shared a routine, a bug in that routine would show up in both, cancel, and the meter would certify the very thing that broke. A review of any `exploitability.py` therefore starts at its import block. [Exploitability](wiki:theory/exploitability) explains why the grader has to be independent.
:::

## The commands, from `pyproject.toml`

Every entry point is registered under `[project.scripts]` in [`pyproject.toml`](repo:pyproject.toml) and runs with `uv run <name>`:

| Command | Module | What it does |
| --- | --- | --- |
| `rps-analysis`, `rps-play` | `rps.analysis`, `rps.play` | Run both self-play experiments and draw `figures/rung0/`; play the learner yourself |
| `kuhn-analysis`, `kuhn-play`, `kuhn-exploit-server` | `kuhn.*` | Solve Kuhn and draw `figures/rung1/` (`--json` for the site); play the equilibrium; serve the Exploit tab |
| `leduc-analysis`, `leduc-play`, `leduc-exploit-server` | `leduc.*` | The same three for Leduc and `figures/rung2/` |
| `minidraw-analysis` | `minidrawmaha.answer.analysis:main` | Grade new race checkpoints (`--grade-runs`) and draw `figures/rung3/` |
| `minidraw-play` | `minidrawmaha.answer.play` | Play the frozen tabular strategy in the terminal |
| `minidraw-deals` | `minidrawmaha.answer.deal_pack` | Write the pre-dealt hand pack the `/play` trainer reads |
| `minidraw-ranges` | `minidrawmaha.answer.range_export` | Write the per-spot range export `/solver` reads |
| `minidraw-exploit-by-reach` | `minidrawmaha.answer.analysis:reach_main` | Bin a released strategy's exploitability by how often each spot is reached |
| `deep-analysis` | `minidrawmaha.answer.analysis:deep_main` | Draw `figures/rung4/price.png` and write the Deep CFR answer sheet |

The scripts under [`scripts/`](repo:scripts) are run by hand and are not registered: the OpenSpiel referee (`scripts/leduc/`), the census, freeze and grade tools and the Modal race (`scripts/minidrawmaha/`), the Deep CFR trainers and studies (`scripts/deepcfr/`), and the site generator (`scripts/site/`).

## Tests mirror the package

[`tests/`](repo:tests) has one directory per subpackage (`rps/`, `kuhn/`, `leduc/`, `minidrawmaha/`, `deepcfr/`), with roughly one test file per module, plus top-level checks on the shared ledger, the static site and the agent instruction files. The default `uv run pytest` needs no torch, Modal, OpenSpiel or network. The expensive whole-game checks sit behind environment flags (`MINIDRAWMAHA_FULL_TABLE`, `MINIDRAWMAHA_FULL_GRADE`, `MINIDRAWMAHA_FULL_CALIBRATION`, `MINIDRAWMAHA_FULL_DEEP`). See [Testing](wiki:ops/testing).

:::cards
- [The regret ledger](wiki:implementation/ledger) The one class every game trains, and its two keyword-only weights.
- [Kuhn poker](wiki:implementation/kuhn) The first tree: the five layers at their smallest.
- [Mini-drawmaha solver](wiki:implementation/minidrawmaha-solver) The packed table, external sampling and lockstep.
- [Deep CFR package](wiki:implementation/deepcfr) Nets in place of the regret half of the ledgers.
:::
