# Rung-3 visualizer: the trainer and the research page for mini-drawmaha

Date: 2026-10-07 · Status: designed autonomously from the request "create the
UI/page dedicated to the minidrawmaha (non-DeepCFR version): the trainer /
play-against-solver page, and the analysis page — more research, what did we
discover, convergence results, what types of CFR, graphs". The session was
non-interactive, so the decisions below were made rather than asked; each one
says why.

## What this is

`/rung3` — one Vite + React app, `web/rung3-viz`, in the site's shared chrome,
with two views in the bar under the hero exactly as `/rung2` has:

| hash | view | what it is |
| --- | --- | --- |
| `#research` (default) | **The research** | the race, the averaging columns, the two seats' best responses, what the frozen strategy does in round 1 and at the draw, how big the game is, and the prose: the game, the four regret rules, lockstep MCCFR, what was found |
| `#play` | **Play the solver** | heads-up against the frozen LCFR average on the oval table, with the draw, the two-way showdown, a coach that shows the solver's mix at your spots and grades your choice by a roll |

Rungs menu: `3 · Mini-Drawmaha · done`, acts **Analysis** → `/rung3`,
**Trainer** → `/rung3#play`. The top bar's "Play the solver →" button points at
`#play` on this page as it does on rung 2.

## The one architectural decision: a pack of pre-dealt hands

Rung 2 shipped its whole strategy (288 rows) to the browser and transcribed the
infoset key. Rung 3's strategy is 6,220,050 rows and 14.25M probabilities, and
its key is a joint canonical suit relabelling of hole, discards and ordered
board, which is exactly the routine the repo warns against duplicating. No
Python runs on Vercel either (`framework: null`, the function is gone).

So the browser never looks a strategy up. Python deals N hands as fixed deck
orders and, for each deal, walks every reachable decision node of **both**
players — 8 in round 1, 21 at the draw, 28 per throw combination in round 2
(16 combinations), about 477 nodes — and records the frozen strategy's mix at
each. It also records the physical card each throw discards (the canonical
low/mid/top order, which depends on the board's suit) and the showdown of each
of the 16 post-draw worlds. The browser follows the path it is on and reads the
vector. There is nothing to canonicalise and no table.

Both players' spots are recorded so the coach can show the solver's mix at
**your** decisions too, as rung 2's coach does.

Consequences, stated plainly:

- The pack holds both holes, so a reader of the network tab can see the bot's
  cards. The CLI has the same property (the state is in memory). This is a
  trainer, not a casino.
- Hands repeat after the pack is exhausted. 600 deals, each playable from
  either seat, is 1,200 sittings; the session walks a shuffled order and only
  then wraps.
- The pack is regenerable from a seed and the pinned release digest, and its
  manifest records both, so a re-export cannot silently swap the strategy.

### Format

`web/rung3-viz/public/deals/index.json`:

```json
{ "seed": 0, "deals": 600, "chunk": 100, "chunks": ["pack-00.json", …],
  "strategy_sha256": "d49f…", "rule": "lcfr", "iteration": 2000000,
  "exploitability": 0.0613465, "value_p0": -0.090645 }
```

Each `pack-NN.json` is a list of deals:

```json
{ "deck": [15 card indices, rank*3+suit; P0 = 0..2, P1 = 3..5, board1 = 6, then the stub in order],
  "order": [[P0's three cards low,mid,top], [P1's]],
  "r1":  { "": [per-mille…], "x": […], "xp": […], … },
  "d0":  { "xx": […], "xpc": […], … },
  "d1":  { "xx": { "0": […], "1": […] }, … },
  "r2":  { "xx": { "nn": { "": […], "x": […], … }, "nl": {…}, … }, … },
  "show": { "nn": { "holes": [[…],[…]], "board": [a, b], "inner": 0|1|2, "outer": 0|1|2,
                    "cats": [["pair","straight"],["high card","pair"]] }, … } }
```

Vectors are per-mille integers summing to 1000 (largest remainder), in
`legal_actions()` order: fold, check/call, pot at a betting node; stand pat,
throw low, throw mid, throw top at a draw node. Throw codes are `n l m t`; `d1`
is keyed by P0's draw **count** because that is all P1's infoset knows. The stub
is consumed in order: P0's replacement is `deck[7]` if P0 drew; P1's is `deck[7]`
if P0 stood pat else `deck[8]`; board 2 is `deck[7 + count0 + count1]`. `show`
has 16 entries, one per throw combination, with `2` meaning a chop of that half.

Produced by `src/drawmaha_solver/minidrawmaha/deal_pack.py` and the
`minidraw-deals` entry point; tested with the uniform stand-in profile so the
suite never needs the 18 MB file. Chunks of 100 deals are about 700 KB raw
each; the app fetches chunk 0 at start and the next when the current one runs
low.

## What TypeScript transcribes, and what pins it

`src/minidraw.ts`: the 15-card deck and its symbols, the pot-limit betting
grammar (`_legal_betting`, `_pot_size`, `_replay`, `_is_closed`), the throw
codes, and the settlement (a fold hands over what the folder had matched; a
showdown pays each half on the exported winner). `minidraw.test.ts` pins the
13 round-1 lines and their chip states against a fixture the Python suite
also pins (`tests/minidrawmaha/test_game.py`'s table), and the settlement
against hand-worked cases including the quarter split. Hand ranking is **not**
transcribed: the pack carries the showdown.

## The Play view

The oval table from rung 2, redrawn for three cards: your three face up, the
solver's three face down, two board frames mid-table (the second dashed until
it lands), the pot and each seat's chips this round, the swept chips when a
round closes. Under it, the legal actions as buttons: `check / bet`, `fold /
call / raise` with the amount, and at the draw `stand pat / throw 2c / throw
4d / throw 6h` naming the physical card. The solver's draw is spoken as the
table sees it: "draws one" or "stands pat". The showdown reveals both holdings
and says who took each half and with what: "inner: you (pair v high card) ·
outer: solver (straight v pair) — you win 3".

The coach sidebar is rung 2's: a roll drawn when your turn arrives, the
solver's mix at your spot, and the grade. Betting segments stack aggressive
first (raise/bet, then call/check, then fold); draw segments stack in the
export's order (stand pat, low, mid, top). The seat tally is the CLI's: hands,
chips, per hand, and per seat beside the seat's self-play value (P0 −0.0906)
so a reader knows that matching the bot averages zero.

Deals come from the pack in a shuffled order, seats alternating.

## The Research view

Windows, numbered, each stating its finding in the title. Every number is read
from `grades.json` and `answer_sheet.json` (copied into `src/data/`), never
typed in.

1. **The race** — exploitability against hands per seat, both axes log, the
   four rules' linear averages, uniform random as a dashed reference, a
   1/√N guide. Hovering or arrowing a checkpoint fills the readout below with
   the four numbers at that column.
2. **Which average** — bars at 20M hands: vanilla linear/uniform, LCFR linear,
   CFR+ linear/uniform, DCFR linear/quadratic.
3. **Two seats, two best responses** — BR₀, BR₁ and P0's self-play value for
   LCFR over the run. BR₀ ends below zero: a perfect P0 still loses, because
   P1's seat is worth about a tenth of an ante.
4. **Round 1 by category** — three stacked bars per spot (P0 opens, P1 checked
   to, P1 facing a bet), six inner categories in rarity order.
5. **The draw by category** — the same for P0, P1 after a pat, P1 after a draw.
6. **How big it is** — information sets per rung on a log axis (12, 288,
   6.22M), the 85.1 billion decision nodes, and what one lockstep iteration
   touches (10 hands per seat).

Between the figures, hairline rows carry the prose, with every formula in
unicode: the game in one paragraph; the four regret rules (R ← R + r;
R ← R + t·r; R ← max(R + r, 0); DCFR's t^1.5/(t^1.5+1) on the positive side and
½ on the negative); external sampling and lockstep; why CFR+ and DCFR coincide
under sampling (median row revisited after about 37,000 iterations, and
halving a float64 1,075 times reaches exactly zero); the frozen strategy
(14,253,840 float32, 18 MB, pinned digest, 8,036 rows never reached); the
caveat that it is one seed. Identity colour inside the figures: the four rules
get four inks held across every figure; the categories use the theme's
sequential ramp.

## Chrome and site plumbing

- `site.tsx`: rung 3 is `done` with Analysis and Trainer acts; the top bar's
  local `#play` button on rungs 2 and 3; footer gains `/rung3`. Edited in rung
  0's copy and `cp`'d to rungs 1, 2 and 3. `tests/test_shared_chrome.py` and
  `tests/test_ui_standard.py` add `rung3-viz` to `APPS`, with the new app's
  pressables listed.
- The homepage, Rules and Cover: the Rungs menu's rung-3 cell links to `/rung3`
  with a Trainer act; the homepage's rung-3 article reads as complete (LCFR,
  6.22M information sets, 0.061 chips/hand, 20M hands per seat) and links to
  the page; the footer and CTA band count four live rungs.
  `tests/test_cover_page.py` allows fragments into `/rung3`.
- `vercel.json` rewrites `/rung3`; `scripts/vercel_build.sh` builds `rung3`.
- README: "Seeing it" and "Playing it" under rung 3; the site section counts
  seven surfaces.

## Testing

- Python: `tests/minidrawmaha/test_deal_pack.py` — a deal's deck is a
  permutation of the 15 cards; every reachable node of a hand-walked deal has a
  vector of the right width summing to 1000; the draw order matches
  `draw_order`; the showdown entries match `pot_shares`; chunking and the
  manifest round-trip.
- TypeScript: `minidraw.test.ts` (grammar, chips, settlement), `play.test.ts`
  (a scripted hand against a tiny hand-written deal: the bot follows the pack,
  the draw consumes the stub in the right order, the result banks once), the
  research view's data shaping (`research.test.ts`: the series are ordered by
  hands and the 2M columns are found).
- `uv run pytest` and `npm test` in the new app both green; the production
  build served by `scripts/serve_site.py` screenshotted at both views.

## Build order

1. `deal_pack.py`, its test and entry point; export the pack.
2. Scaffold `web/rung3-viz` from rung 2 (configs, shared chrome, fonts).
3. `minidraw.ts` + test; `play.ts` + test; `PlayPanel.tsx`.
4. `research.ts` + charts + prose; `App.tsx`; `index.css` under `DESIGN.md`.
5. Chrome, static pages, vercel, build script, tests, README.
6. Build, test, screenshot, PR.
