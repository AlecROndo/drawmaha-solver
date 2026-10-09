# `/solver` over the real thing: mini-drawmaha's frozen LCFR strategy — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The range interface built in PR #53 (`web/solver-viz`, `/solver`) gets a second game: mini-drawmaha over the rung-3 frozen LCFR strategy, where every number is the solver's — the actor's exact reach-weighted range at any public decision point (round 1, the draw, round 2), the real mix per region and per hand, and range vs range with exact showdown equities at round 2.

**Architecture:** Python exports the frozen strategy once, per public decision point, as compact binary chunks plus the private-key tables and a manifest (`uv run minidraw-ranges`). The browser ports the one piece of game logic it needs — the joint suit canonicalisation, the draw order, both hand rankings and the betting grammar — pinned against Python-generated fixtures, and computes ranges exactly: enumerate the actor's physical private states at the chosen spot, canonicalise each into its key, look up its mix, and weight it by the product of the actor's own action probabilities along the line (chance factors are constant across states and cancel). The UI is the PR #53 window generalised over a `Game` descriptor and a `RangeTable` so the grid, ribbon, plane, ledger, panes, deck and draw board draw either game.

**Tech Stack:** Python 3 (numpy) for the exporter; Vite 8 + React 19 + TypeScript 6 + vitest for the app. Binary chunks are gzip-compressed and inflated in the browser with `DecompressionStream`.

**Spec:** PR #53's plan `docs/superpowers/plans/2026-10-07-solver-range-interface.md` (the UI), plus `src/drawmaha_solver/minidrawmaha/{game,cards,hands,enumeration,strategy,packed_table}.py` (the rules and the table layout).

## Global Constraints

- Nothing in `src/drawmaha_solver/minidrawmaha/` changes behaviour; the exporter only reads the strategy. Tests for the exporter use a stand-in strategy (as `tests/minidrawmaha/test_deal_pack.py` does) and never need the 18 MB release asset.
- The five shared chrome files in `web/solver-viz/src` stay byte-identical to rung 0's. `web/DESIGN.md` governs the stylesheet; new pressables go into `PRESSABLES["solver-viz"]`.
- Every number shown in mini mode is the solver's. The status line says "rung-3 LCFR, 2M iterations per seat, exact". Where the page approximates (nothing at round 2; equities at round 1 are showdown-now on the inner half only) it says so.
- Card index in the browser is `rank*3 + suit`, ranks `23456`, suits `cdh` — the rung-3 deal pack's convention.
- Fail loud: an unknown public point, a key not found in its shape, a chunk whose rows do not sum to the scale, a fixture mismatch — all throw with the offending value named.

## Review Focus

1. A spot whose line is not a decision (the hand is over, or it is a chance node) must not look up a point; `pointFor` throws and the UI shows the last decision. Task 3 tests `pointFor` on `xf` and on `xx` before the draw.
2. Card removal: when the user holds cards, the other seat's enumeration excludes them, and the pairwise equity sum skips overlapping states. Task 3 tests that a held card never appears in the opponent's range and that equities over disjoint states sum to 1 (win + lose + tie).
3. The throw position: for a post-draw state the discard's position in the pre-draw canonical draw order chooses the column; a hand whose three cards share a rank-pattern under a suit swap must still name the same card. Task 3 tests `drawOrder` against the Python fixtures.
4. Chunk integrity: a truncated or wrong-length chunk throws before any mix is read. Task 3 tests the decoder on a short buffer.
5. Stand-pat players at round 2 have 286 states and throwers 2,860; the grid's share must still sum to 1 within one seat's range. Task 3 tests both.

---

## Export format (the contract between Python and TypeScript)

Output directory: `web/solver-viz/public/mini/`.

`index.json`
```json
{
  "strategy": {"sha256": "…", "rule": "lcfr", "column": "linear", "iteration": 2000000, "workers": 10, "seed": 0},
  "deck": {"ranks": "23456", "suits": "cdh"},
  "scale": 250,
  "shapes": {
    "b1d0": {"file": "keys-b1d0.bin", "n": 970,    "stride": 4},
    "b2d0": {"file": "keys-b2d0.bin", "n": 10170,  "stride": 5},
    "b2d1": {"file": "keys-b2d1.bin", "n": 100400, "stride": 6}
  },
  "points": [
    {"id": 0, "player": 0, "boardCards": 1, "draws": [], "lines": [""], "discards": 0, "draw": false,
     "actions": ["c", "p"], "shape": "b1d0", "file": "point-000.bin.gz", "n": 970}
  ]
}
```
- `points` are `public_decision_points()` in order; `id` is the index. `lines` are `line_symbol` strings per betting round (`x` check, `c` call, `p` pot, `f` fold); `draws` are the `DrawSignal.count`s present; `actions` are the ledger's columns in `legal_actions()` order as symbols: betting `f`/`c`/`p` (`c` is CHECK_CALL whether it checks or calls), draw `n`/`l`/`m`/`t`.
- `keys-<shape>.bin`: `Uint8`, one row per key in `private_keys(board_cards, discards)` sorted order, row = hole (3 cards, canonical order) + discarded (d cards) + board (bc cards, dealt order), each card `rank*3+suit`.
- `point-<id>.bin.gz`: gzip of `Uint8[n * width]`, row-major, value = probability × `scale` rounded by largest remainder so each row sums to exactly `scale`; columns in `actions` order.
- `fixtures.json` (written to `web/solver-viz/src/mini/fixtures.json`): `{"cases": [{"hole": [c,c,c], "thrown": [], "board": [c] | [c,c], "key": {"hole": [...], "thrown": [...], "board": [...]}, "relabelling": [s0,s1,s2], "drawOrder": [c,c,c], "inner": [cat, r, r, r], "outer": [cat, r, r, r, r] | null}]}` — 400 seeded random pictures across the three shapes plus every picture from `tests/minidrawmaha` that names a flush. `drawOrder` is the physical hole sorted by `(rank, relabelling[suit])`; `inner`/`outer` are `Score` as `[category, *ranks]`.

## File structure

```
src/drawmaha_solver/minidrawmaha/range_export.py     the exporter + fixtures writer; CLI main
tests/minidrawmaha/test_range_export.py              stand-in strategy; shapes, sums, order, fixtures
pyproject.toml                                        minidraw-ranges entry point
web/solver-viz/public/mini/                           the export (committed)
web/solver-viz/src/mini/cards.ts                     15-card deck
web/solver-viz/src/mini/canonical.ts                 canonicalPicture, drawOrder, keyString
web/solver-viz/src/mini/ranking.ts                   innerScore, outerScore (ints, comparable)
web/solver-viz/src/mini/grammar.ts                   betting grammar (ported from rung3-viz/src/minidraw.ts)
web/solver-viz/src/mini/data.ts                      manifest + key tables + lazy point chunks
web/solver-viz/src/mini/points.ts                    pointFor(spot)
web/solver-viz/src/mini/regions.ts                   6 inner rows; 5 round-1 columns; 7 round-2 columns
web/solver-viz/src/mini/range.ts                     rangeAt(): exact reach-weighted range; equities()
web/solver-viz/src/mini/*.test.ts, fixtures.json
web/solver-viz/src/engine/table.ts                   RangeTable: the game-agnostic shape aggregate.ts reads
web/solver-viz/src/engine/aggregate.ts               functions take a RangeTable (full game adapts its sample)
web/solver-viz/src/ui/game.ts                        Game descriptor: rows/cols, deck layout, hand size, labels
web/solver-viz/src/ui/MiniScore.tsx                  the two-lane score for mini: board 1 · round 1 · draw · board 2 · round 2
web/solver-viz/src/ui/MiniRange.tsx                  the mini window: spot head, grid/ribbon/plane/ledger/deck/draw board/versus
web/solver-viz/src/App.tsx                           the game switch
```

## TypeScript mini contract

```ts
// cards.ts
export type MiniCard = number                 // rank*3 + suit; ranks 0..4 = 2..6; suits 0..2 = c d h
export const MINI_DECK: readonly MiniCard[]   // 0..14
export const miniLabel: (c: MiniCard) => string   // '4h'
export const miniGlyph: (c: MiniCard) => string   // '4♥'
export function parseMini(s: string): MiniCard    // throws naming the token

// canonical.ts
export interface Picture { hole: MiniCard[]; thrown: MiniCard[]; board: MiniCard[] }
export function canonicalPicture(p: Picture): { key: Picture; relabelling: [number, number, number] }
export function drawOrder(p: Picture): MiniCard[]              // physical hole by (rank, relabelling[suit])
export function keyString(key: Picture): string                // the Map key: cards joined, groups separated

// ranking.ts
export function innerScore(hole: readonly MiniCard[]): number   // category*10**4 + ranks base-5, comparable
export function outerScore(hole: readonly MiniCard[], board: readonly MiniCard[]): number  // board.length must be 2
export const INNER_CATEGORY_NAMES, OUTER_CATEGORY_NAMES: readonly string[]
export function innerCategory(score: number): number           // 1..6
export function outerCategory(score: number): number           // 1..7

// grammar.ts — the rung-3 grammar: Bet 'f'|'c'|'p', Throw 'n'|'l'|'m'|'t', replay(lines), legalBets(lines), actorOf(line), isClosed(line, behind), isFold(line), betWord(bet, line), throwWord

// data.ts
export interface PointMeta { id; player: 0|1; boardCards: 1|2; draws: number[]; lines: string[]; discards: 0|1; draw: boolean; actions: string[]; shape: 'b1d0'|'b2d0'|'b2d1'; file: string; n: number }
export interface MiniIndex { strategy: {...}; scale: number; shapes: Record<string, {file; n; stride}>; points: PointMeta[] }
export class MiniData {
  static async load(base: string): Promise<MiniData>        // fetches index.json and the three key tables
  readonly index: MiniIndex
  keyIndex(shape: string, key: Picture): number              // throws if absent
  async point(id: number): Promise<Uint8Array>               // lazy, cached, inflated, length-checked
}

// points.ts
export interface MiniSpot { board: MiniCard[]; lines: string[]; draws: number[]; player: 0|1 }
export function pointFor(index: MiniIndex, spot: MiniSpot): PointMeta   // throws when no public decision point matches

// regions.ts
export const MINI_INNER_ROWS: Region[]        // high card, pair, straight, flush, straight flush, trips
export const MINI_R1_COLS: Region[]           // nothing, two to the straight, two to the flush, pairs the board, trips with the board
export const MINI_R2_COLS: Region[]           // high card, pair, straight, two pair, trips, flush, straight flush
export function r1Col(hole, board1): number; export function r2Col(outerCat): number

// range.ts
export interface MiniRange {
  spot: MiniSpot; point: PointMeta
  n: number
  cards: Uint8Array        // n*4: hole(3) then discard or 255
  weight: Float32Array     // reach-weighted, sums to 1 over the range
  row: Uint8Array; col: Uint8Array
  mix: Float32Array        // n*width, probabilities, columns = point.actions
  innerScore: Int32Array; outerScore: Int32Array   // outerScore -1 when boardCards = 1
}
export async function rangeAt(data: MiniData, spot: MiniSpot, exclude: readonly MiniCard[]): Promise<MiniRange>
export function equities(mine: MiniRange, theirs: MiniRange): { inner: Float32Array; outer: Float32Array | null; scoop: Float32Array | null }
```

`rangeAt` weights: for each of the actor's states, P(dealt) is uniform over states, times the product over the actor's OWN earlier decisions in the spot of the mix probability of the action actually taken (looked up at the earlier point with the state's earlier key), including at round 2 the draw probability of the state's own throw (stand pat, or the position of the discard in the pre-draw `drawOrder`). Chance factors are equal across states and are not applied. `exclude` removes states holding any of those cards.

## Tasks

1. **Python exporter** (agent): `range_export.py` with `export_ranges(strategy, out)`, `write_fixtures(path, seed)`, `main()`; tests with a stand-in strategy; `pyproject.toml` entry; run it for real and commit `public/mini/` + `fixtures.json`.
2. **TS mini core** (agent): `cards`, `canonical`, `ranking`, `grammar`, `regions` with fixture-pinned tests.
3. **TS data + range** (agent, after 2): `data.ts`, `points.ts`, `range.ts` + tests using a small synthetic manifest built in the test (two points, a dozen keys) plus the real export when present.
4. **UI generalisation** (me): `RangeTable`, `Game`, components take `game`; full game unchanged in behaviour.
5. **Mini UI** (me): `MiniScore`, `MiniRange`, the game switch in `App.tsx`; status line; draw board over `n/l/m/t`; versus with exact equities.
6. **Wiring + tests + screenshots**: `PRESSABLES`, build, headless run through round 1 → draw → round 2, PR.
