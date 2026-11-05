---
title: Mini-drawmaha answer
description: The frozen strategy and its release, the race's analysis and figures, the terminal table, and the two exports — the deal pack and the range export — that let a browser play a 6.2-million-row strategy.
---

[`answer/`](repo:src/drawmaha_solver/minidrawmaha/answer) is everything built on the solved game. A training checkpoint is the wrong thing to hand anyone: 555 MB, almost all of it the learner's working state. What the game needs is one probability per legal action. This package freezes that, publishes it as a release asset pinned by SHA-256, grades it, and cuts it into the two shapes the site reads, because the browser cannot look the strategy up itself.

## Files

| File | Command | What it is for |
| --- | --- | --- |
| [`strategy.py`](repo:src/drawmaha_solver/minidrawmaha/answer/strategy.py) | — | The frozen strategy: two flat arrays, a pinned digest, download-and-verify |
| [`releases.py`](repo:src/drawmaha_solver/minidrawmaha/answer/releases.py) | — | The catalogue of every released strategy and where its grade is recorded |
| [`analysis.py`](repo:src/drawmaha_solver/minidrawmaha/answer/analysis.py) | `minidraw-analysis`, `minidraw-exploit-by-reach`, `deep-analysis` | The race's figures and answer sheet, the reach profile, Deep CFR's price figure |
| [`play.py`](repo:src/drawmaha_solver/minidrawmaha/answer/play.py) | `minidraw-play` | The terminal table against the frozen strategy |
| [`deal_pack.py`](repo:src/drawmaha_solver/minidrawmaha/answer/deal_pack.py) | `minidraw-deals` | Hands dealt in advance, with the strategy at every node, for `/play` |
| [`range_export.py`](repo:src/drawmaha_solver/minidrawmaha/answer/range_export.py) | `minidraw-ranges` | The strategy cut by public decision point, for `/solver` and Learn |

## `strategy.py`: 57 MB of probabilities, refused if the layout differs

**`FrozenStrategy`** holds two arrays in the packed table's order: `probabilities`, every infoset's average strategy laid end to end (14,253,840 float32s, about 18 MB compressed), and `widths`, each row's number of legal actions. **`widths`** is the layout's fingerprint. A file written against a different enumeration, such as the old unordered-board key's 3,142,290 rows, is refused on load instead of being read as somebody else's strategy.

- **`from_checkpoint`** freezes one averaging column of a lockstep checkpoint, using **`average_rows`** to normalise each row. A row that sums to zero plays uniformly, exactly as `RegretMatcher.average_strategy` reads it.
- **`uniform_strategy`** is the race's floor line, every action played equally.
- **`save_strategy`** / **`load_strategy`** write and read the compressed `.npz`.
- **`fetch_strategy`** downloads a release asset once into **`default_cache`** (`~/.cache/drawmaha-solver`) and checks **`sha256_of`** against the pinned digest. **`StrategyInfo`** records where a strategy came from.

[`scripts/minidrawmaha/freeze_strategy.py`](repo:scripts/minidrawmaha/freeze_strategy.py) is the hand-run freeze. It prints the size and digest to pin, and `--grade` checks the freeze against the checkpoint's own exploitability to the sixth decimal.

Of the LCFR run's 6,220,050 rows, 2,710,428 were never banked and 3,537,196 read exactly uniform, nearly all in round 2. A human who deviates walks the bot into them. The remedy is more iterations or Deep CFR's nets, not a patch to the lookup.

## `releases.py`: one catalogue both exporters read

**`Release`** names one shipped strategy: its tag and asset on GitHub (**`url`**), its SHA-256, and the committed file and key that hold its exact grade. **`exported`** says whether `/solver` carries it behind its switch. `RELEASES` holds every one:

| Name | Tag | What it is | Grade from |
| --- | --- | --- | --- |
| `tabular` | `rung3-lcfr-2m` | LCFR's linear average at 2M iterations, the tabular answer | `figures/rung3/answer_sheet.json` |
| `deep-pi`, `deep-exact` | `rung4-deep-t500` | the 4 × 256 Deep CFR arm at t = 500: its policy net Π and its exact average | `figures/rung4/sweep_256x4-s16000-lr5e-4.json` |
| `deep-pot-exact`, `deep-pot-pi` | `rung4-deep-pot0.5-t200` | the ÷ √pot arm at t = 200; its exact average is what the Play tab deals against | `figures/rung4/pot_0.5-256x4-s16000-lr5e-4.json` |

**`read_grade`** copies a release's grade from its committed file into an export's manifest at export time, and **`entry_mismatch`** says why an entry does not match. A page therefore quotes a number with a committed source and never one typed into TypeScript. **`provenance_json`** writes the `strategy` block both manifests open with, byte for byte the same shape. **`release_of`** recognises a local file by its digest. See [Releases](wiki:ops/releases).

:::warning
Re-freezing the tabular strategy changes three things in one PR: the digest pinned in `strategy.py`, its entry in `releases.py`, and the copy recorded in `web/solver-viz/public/mini/index.json`.
:::

## `analysis.py`: grade rarely, draw often

Grading a checkpoint takes about two minutes and needs the 555 MB file from the Modal Volume, so the work is split.

- **`--grade-runs DIR`** (**`grade_runs`**) grades every checkpoint and averaging column not yet in `figures/rung3/grades.json`, saving after each. **`grade`** computes both best responses, their mean and P0's self-play value. **`grade_key`** builds keys `rule/iteration/column`, with the iteration zero-padded so keys sort in training order.
- **Without it**, `main` reads the committed grades and the released strategy and draws the four figures: **`fig_race`** (exploitability against hands per rule, built from **`series`**), **`fig_columns`** (every averaging column at the last iteration), **`fig_round_one`** and **`fig_draw`** (what the answer does by inner category, from **`round_one_frequencies`** and **`draw_frequencies`**). **`answer_sheet`** writes every number the figures and README quote; **`print_summary`** prints them.
- **`reach_main`** (`minidraw-exploit-by-reach`) splits a strategy's exploitability over spots with **`reach_profile`**. It bins the spots by how often play reaches them, computes **`per_visit`**, what a spot leaks each time it is reached, and draws **`fig_reach`**. **`stage_of`** classes a node as round 1, the draw or round 2.
- **`deep_main`** (`deep-analysis`) reads the two committed grade files. **`price_table`** pairs each Deep CFR object with LCFR at equal hands, **`fig_price`** draws the result, and **`deep_answer_sheet`** writes `figures/rung4/answer_sheet.json`. **`deep_key`** and **`deep_series`** read the `deep/<t>/<column>` keys.

The tabular answer graded: **0.061346 chips a hand** (BR₀ −0.034421, BR₁ +0.157114). In self-play P0 is worth −0.0906 a hand, because P1 acts last in both rounds and in the draw.

![Round 1 betting by inner category](fig:rung3/round_one_betting.png)

## `play.py`: the terminal table

**`main`** loads the frozen strategy, **`introduce`** says what is at the table, and **`play_session`** runs hands until you quit, seats alternating. **`play_hand`** deals, plays both rounds and the draw, and scores. **`deal_chance`** turns cards at the deck's own odds, and **`bot_action`** samples the strategy at the bot's own infoset. **`narrate`** reports a bot decision only as the table would see it: whether it drew, never what. **`thrown_card`** maps a throw's canonical position to the physical card, so "throw low" is never a guess. **`showdown_line`** and **`category`** name both halves' winners. **`ask_action`**, **`prompt_for`** and **`parse_action`** handle input. **`report`** prints the session's net beside each seat's self-play value.

## The two exports: the lookup done ahead of time

The table's key is the joint canonical suit relabelling, the one routine the repo most wants to exist exactly once, and the site runs no Python. So the lookup happens in Python, ahead of time, and the browser only follows paths.

### `deal_pack.py`: every node of 600 pre-dealt hands

A deal is a fixed deck order: P0's three cards, P1's three, board card 1, then the stub in the order it will be consumed. For one deck order the hand's decision tree is finite (8 nodes in round 1, 21 at the draw, 28 per throw combination in round 2). **`export_deal`** walks every node for both players and records the strategy's mix there, quantised by **`per_mille`** to integers summing to exactly 1000. Beside the vectors it records each player's cards in canonical low/mid/top order and the **`showdown`** of all 16 post-draw worlds, so the browser transcribes neither the key nor the hand ranking. **`random_decks`** draws the deck orders, **`deal_from_deck`** sets up the hand, and **`write_pack`** writes chunks of 100 plus a manifest under `web/rung3-viz/public/deals/`. **`card_index`** is the shared card numbering, `rank * 3 + suit`.

### `range_export.py`: every hand at every spot

`/solver` asks the opposite question: what does the strategy do with *every* hand at *this* spot? The export keeps the table's factorisation on disk. Each of the 141 public points is one chunk, `point-<index>.bin.gz` (**`point_meta`**, **`PointMeta`**). Each shape's sorted key list is written once (**`shape_name`** gives `b1d0`, `b2d0`, `b2d1`; **`ShapeMeta`**; **`encode_keys`**). Every point is a plain array in its shape's key order (**`point_rows`**), quantised to bytes by **`quantise`** with each row summing exactly to the scale. **`export_ranges`** writes it all with an `index.json` **`Manifest`**.

The browser must still relabel suits to find a key. **`write_fixtures`** therefore writes seeded random **`Picture`**s (**`random_picture`**, **`fixture_case`**) with Python's four answers beside each, and **`settle_fixtures`** refuses to overwrite fixtures that differ. The TypeScript side is tested against them.

:::note
Every committed export (`web/solver-viz/public/mini*/`, `web/rung3-viz/public/deals/`) is regenerated by its command, never hand-edited, and records the digest and grade of the strategy it was cut from.
:::

:::cards
- [Releases](wiki:ops/releases) Tags, assets and how the site ships each strategy.
- [The apps](wiki:ops/apps) The `/play`, `/solver` and `/research` apps that read these exports.
- [Deep CFR package](wiki:implementation/deepcfr) Where the `deep-*` releases came from.
:::
