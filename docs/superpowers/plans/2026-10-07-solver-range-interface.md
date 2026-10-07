# The solver's range interface (`/solver`) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fifth visualizer, `web/solver-viz`, served at `/solver`, that answers "here is the spot, here is everything I could hold, what does the solver do with each" for full Drawmaha on any flop — with the layout Alec chose from the ten studies in `research/range-ui-studies/`.

**Architecture:** A pure-TypeScript engine samples N five-card holdings from the cards unseen on the chosen flop, classifies each twice (inner: the five hole cards as a poker hand; outer: exactly two hole + three board, Omaha), flags draws, assigns each hand two showdown-now equities (percentile of its score in the sample) and an **illustrative** action mix (rung 4 is not trained; this is labelled on the page). Every view is an aggregate over that sample under a filter mask, recomputed on demand (N = 60,000 → every aggregate is a few ms). The React UI is one window: the two-lane score as the lead-up, a spot header whose controls open the secondary views, the two-half grid with the strength ribbon and the equity plane in a right column, the ledger under the grid, the draw board when the cursor is on the draw node, and a second tab for range vs range with the filmstrip at its foot.

**Tech Stack:** Vite 8 + React 19 + TypeScript 6 + vitest, copied from `web/rung3-viz`. Shared chrome files byte-identical (`tests/test_shared_chrome.py`). CSS under `web/DESIGN.md`, enforced by `tests/test_ui_standard.py`.

**Spec:** `research/range-ui-studies/index.html` (the ten studies, with Alec's placement decisions in the 2026-10-07 conversation: grid + ribbon together on the main page; plane in the right column; ledger opens from the action buttons; three panes open from the mix bar; deck opens from a cards button; draw board at the draw node; range vs range is its own tab; filmstrip at the bottom of that tab; the funnel is folded into the ledger's breadcrumb).

## Global Constraints

- Anything under `web/` follows `web/DESIGN.md`; run `uv run pytest tests/test_ui_standard.py tests/test_shared_chrome.py` after any stylesheet change.
- The five shared files (`src/theme.css`, `src/main.tsx`, `src/ui/{site.tsx,mark.ts,ascii.ts}`) are copied from `web/rung0-viz` byte for byte and never edited here.
- Every pressable in `src/index.css` has `:active { transform: scale(0.97) }`, `transform var(--t-press) var(--ease-out)` in its transition, and `transform: none` in the reduced-motion block; its selector is listed in `PRESSABLES["solver-viz"]` in `tests/test_ui_standard.py`.
- No chrome motion over 300 ms; only `transform`/`opacity` move; no `transition: all`; no `ease-in`.
- The policy is illustrative. Every place a percentage appears, the window's status line says so. No fake "solver" claims.
- Fail loud: an unparseable board is an error shown in the field, never a silent fallback to the last good board.

## Review Focus

1. A board with a duplicate card (`Ks Ks 4d`) or a card spelled wrongly (`Kx`) must show an error and keep the previous range on screen; test in Task 1 (`parseBoard`).
2. Holding cards placed in the deck that are on the board must be impossible (board cards are not pressable); a filter that empties the range (five cards placed that no sampled hand has) must show "no sampled hand holds this; the exact hand's mix is the policy evaluated directly" — Task 4 computes the single-hand mix directly rather than from the empty mask.
3. On a monotone or paired flop the composite outer columns can be empty across the board; cells with no hands render as dashes and marginals divide by zero nowhere; Task 4 tests a paired flop.
4. The ribbon's percentile marker for a held hand with fewer than five cards must not be drawn (a region has no percentile); Task 7.
5. Reduced motion: the popover must still appear (opacity only); the press must be off; Task 9 checks the stylesheet with the standard's test.

---

## File structure

```
web/solver-viz/
  index.html, package.json, package-lock.json, vite.config.ts, tsconfig*.json
  public/favicon.svg, public/icons.svg           (copied from rung3-viz)
  src/theme.css, main.tsx, ui/site.tsx, ui/mark.ts, ui/ascii.ts   (shared, byte-identical)
  src/index.css                                   this app's own figures
  src/App.tsx                                     tabs, the hand state, the window
  src/engine/cards.ts        Card = 0..51, parse/print, deck minus a board
  src/engine/evaluate.ts     score5(): the 5-card ranking as one integer
  src/engine/classify.ts     inner/outer classes with draw flags
  src/engine/sample.ts       sampleRange(): N holdings, struct-of-arrays, seeded
  src/engine/policy.ts       the illustrative mix (fold/call/pot, check/pot) and throw probabilities
  src/engine/regions.ts      the composite axes (9 inner rows × 8 outer columns)
  src/engine/aggregate.ts    grid, marginals, ribbon, density, deck stats, draw table, examples, under a mask
  src/engine/filter.ts       Filter → mask
  src/engine/equity.ts       range vs range: weighted equities of a region against the other seat's range
  src/engine/*.test.ts
  src/ui/Score.tsx           lead-up A (+ one-line collapsed form)
  src/ui/SpotHead.tsx        the fixed header: who/what, mix bar (→ Panes), action buttons (→ Ledger), cards button (→ Deck), count
  src/ui/Grid.tsx            01 the two-half grid, filters, pinning
  src/ui/Ribbon.tsx          02 vertical strip, percentile marker
  src/ui/Plane.tsx           03 the two equities
  src/ui/Ledger.tsx          04 hairline rows that open, breadcrumb
  src/ui/Panes.tsx           05 one pane per action (popover)
  src/ui/Deck.tsx            06 the deck as the filter (popover) + the holding strip
  src/ui/DrawBoard.tsx       07 throw-count board + per-card throw bars
  src/ui/Versus.tsx          09 both seats + equity readout, 10 filmstrip at the foot
  src/ui/Popover.tsx         anchored popover, Escape/outside closes
  src/ui/bars.tsx            MixBar, MixKey, CardFace — the small shared marks
tests/test_shared_chrome.py, tests/test_ui_standard.py   add "solver-viz"
scripts/vercel_build.sh, vercel.json                     build + rewrite /solver
```

## Engine interfaces (the contract both halves build against)

```ts
// cards.ts
export type Card = number                         // rank*4 + suit; rank 0..12 = 2..A; suit 0..3 = ♠ ♥ ♦ ♣
export const rankOf: (c: Card) => number; export const suitOf: (c: Card) => number
export function parseCard(s: string): Card        // 'Ks' | 'K♠' → 47; throws Error naming the token
export function parseBoard(s: string): Card[]     // 3 cards, distinct; throws Error with the reason
export function cardLabel(c: Card): string        // 'K♠'
export function isRed(c: Card): boolean
export function unseen(board: readonly Card[]): Card[]

// evaluate.ts
export const CATEGORY_NAMES = ['high card','pair','two pair','trips','straight','flush','full house','quads','straight flush'] as const
export function score5(c: readonly Card[]): number   // exactly five cards; higher wins; category = Math.floor(score / 13**5)
export function categoryOf(score: number): number

// classify.ts
export interface Inner { cat: number; score: number; fourFlush: boolean; fourStraight: 0|1|2 }   // 0 none, 1 gutshot, 2 open
export interface Outer { cat: number; score: number; flushDraw: boolean; straightDraw: 0|1|2|3 } // 0 none, 1 gut, 2 open, 3 wrap
export function classifyInner(hole: readonly Card[]): Inner
export function classifyOuter(hole: readonly Card[], board: readonly Card[]): Outer

// sample.ts  — struct of arrays, n hands
export interface RangeSample {
  n: number; board: Card[]; seed: number
  cards: Uint8Array            // n*5
  innerCat: Uint8Array; outerCat: Uint8Array
  innerScore: Uint32Array; outerScore: Uint32Array
  innerFourFlush: Uint8Array; innerFourStraight: Uint8Array
  outerFlushDraw: Uint8Array; outerStraightDraw: Uint8Array
  innerEquity: Float32Array; outerEquity: Float32Array   // percentile of score in the sample (ties at half)
  row: Uint8Array; col: Uint8Array                        // composite region indices (regions.ts)
}
export function sampleRange(board: readonly Card[], n: number, seed: number): RangeSample

// policy.ts  — ILLUSTRATIVE
export type Node = 'facing' | 'open'                       // facing a pot bet: fold/call/pot; first to act: check/pot (fold = 0)
export interface Policy { f: Float32Array; c: Float32Array; p: Float32Array; eq: Float32Array }
export function policyFor(s: RangeSample, node: Node): Policy
export function mixOfHand(hole: readonly Card[], board: readonly Card[], s: RangeSample, node: Node): { f: number; c: number; p: number; eq: number }
export function throwCounts(s: RangeSample): Float32Array  // n*6: P(throw k) per hand
export function throwOfHand(hole: readonly Card[], board: readonly Card[], s: RangeSample): { counts: number[]; perCard: number[] }

// regions.ts
export interface Region { label: string; long: string }
export const INNER_ROWS: readonly Region[]   // 9: nothing, nothing·4-straight, nothing·4-flush, pair, pair·4-straight, pair·4-flush, two pair, trips, straight or better
export const OUTER_COLS: readonly Region[]   // 8: nothing, nothing·straight draw, nothing·flush draw, pair, pair·straight draw, pair·flush draw, two pair, trips or better
export function innerRow(cat: number, fourFlush: boolean, fourStraight: number): number
export function outerCol(cat: number, flushDraw: boolean, straightDraw: number): number

// filter.ts
export interface Filter { rows: ReadonlySet<number> | null; cols: ReadonlySet<number> | null; held: readonly Card[] }
export const EMPTY_FILTER: Filter
export function maskFor(s: RangeSample, f: Filter): Uint8Array | null     // null = everything

// aggregate.ts
export interface Mix { f: number; c: number; p: number }
export interface Cell extends Mix { share: number; n: number }            // share of the WHOLE sample
export interface GridView { cells: Cell[][]; rows: Cell[]; cols: Cell[]; all: Cell }
export function gridView(s: RangeSample, pol: Policy, mask: Uint8Array | null): GridView
export type RibbonKey = 'total' | 'inner' | 'outer' | 'scoop'
export function ribbon(s: RangeSample, pol: Policy, mask: Uint8Array | null, key: RibbonKey, bins: number): Mix[]
export function percentileOf(s: RangeSample, pol: Policy, key: RibbonKey, hole: readonly Card[], board: readonly Card[]): number
export function density(s: RangeSample, pol: Policy, mask: Uint8Array | null, bins: number): Cell[]   // bins*bins cells, x = outer, y = inner
export function deckStats(s: RangeSample, pol: Policy, mask: Uint8Array | null): { inRange: Float32Array; potLift: Float32Array }  // 52 each
export function drawTable(s: RangeSample, throws: Float32Array, mask: Uint8Array | null): { share: number; counts: number[] }[]  // per inner row
export function examples(s: RangeSample, pol: Policy, mask: Uint8Array | null, row: number, col: number, k: number): { cards: Card[]; mix: Mix; eqI: number; eqO: number }[]

// equity.ts
export function weightsForBet(pol: Policy): Float32Array                 // the other seat's betting range: weight = P(pot)
export function equityAgainst(s: RangeSample, mask: Uint8Array, other: RangeSample, weights: Float32Array): { inner: number; outer: number; total: number; scoop: number; scooped: number }
```

The two seats sample from the same board; the other seat's `RangeSample` is a second seed. Card removal between the seats is ignored (documented on the page).

## Tasks

### Task 1: scaffold + cards + evaluate (engine agent)
Create `web/solver-viz` from `web/rung3-viz` (configs, public icons, shared files copied). `cards.ts` and `evaluate.ts` with tests: `parseBoard('Ks 9s 4d')` → `[47, 31, 10]`; `parseBoard('Ks Ks 4d')` throws mentioning "twice"; `parseBoard('Kx 9s 4d')` throws naming `Kx`; `score5` orders quads > full house > flush > straight > trips > two pair > pair > high card; wheel A-2-3-4-5 is a 5-high straight; `categoryOf(score5(['2s','3s','4s','5s','6s']))` is 8.

### Task 2: classify (engine agent)
`classifyInner`: `A♠ A♦ Q♠ 7♠ 3♠` → cat 1, fourFlush true. `9♥ 8♣ 7♦ 6♠ 2♣` → fourStraight 2 (open). `9♥ 8♣ 7♦ 5♠ 2♣` → 1 (gut). `classifyOuter` on K♠ 9♠ 4♦: `A♠ A♦ Q♠ 7♠ 3♠` → cat 1 (pair of aces), flushDraw true; `K♥ K♣ …` → cat 3 (trips); `Q♥ J♣ T♦ …` with board K 9 4 → straightDraw 1 (Q J T need K-high: Q J T + K + … needs 9? K Q J T 9: hole QJ + board K9 → 4 of K Q J 9 … compute: outs = {T} gut) — write the expected value from the rule, not from running the code.

### Task 3: sample + regions + policy + filter (engine agent)
Determinism (same seed → identical arrays), no board card in any hand, five distinct cards per hand, equities in [0,1] and monotone in score, `policyFor` rows sum to 1 ± 1e-6, `open` node has f = 0 everywhere, `throwCounts` rows sum to 1, `maskFor` with `held=[A♠]` keeps only hands containing A♠, `maskFor(EMPTY_FILTER)` is null.

### Task 4: aggregate + equity (engine agent)
`gridView(null mask).all.share === 1`; sum of cells' shares equals `all.share`; marginals equal the row/col sums; a cell with n = 0 has share 0 and f = c = p = 0 (not NaN); on a paired flop `K♠ K♦ 4♣` the grid still sums to 1; `ribbon(..., bins=10)` returns 10 mixes each summing to 1; `density` shares sum to 1; `deckStats.inRange` sums to 5 (each hand holds five cards); `drawTable` shares sum to 1; `equityAgainst` of a mask against the same sample with uniform weights ≈ 0.5 inner and outer (± 0.02).

### Task 5: App shell, Score, SpotHead, Popover, bars (UI)
Hand state: `{ board: Card[]; villainAct: 'pot' | 'check'; cursor: 'you' | 'draw' }`. Score renders flop / draw / turn / river / showdown columns, three lanes + pot; clicking seat 0's flop chip toggles pot/check (a `button`), clicking your flop chip sets cursor 'you', clicking your draw chip sets cursor 'draw'. SpotHead: title from the node, mix bar (a `button` → Panes), three action buttons (→ Ledger sorted by that action), cards button (→ Deck), count. Board field: text input, parse on Enter/blur, error shown in red mono beneath.

### Task 6: Grid + Ledger (UI)
Grid with filters (row/col sets) and a pinned cell; cell click pins and opens the Ledger at that row. Ledger: level 1 inner rows, level 2 outer columns within an open row, level 3 the example hands (from `examples`); breadcrumb shows the filter; sort control by action.

### Task 7: Ribbon + Plane (UI, right column)
Ribbon vertical (y = percentile, weakest at the bottom), key tabs (total / inner / outer / scoop), the held hand's marker when five cards are placed. Plane: 18×18 density, quadrant labels, the held hand's dot.

### Task 8: Panes, Deck, DrawBoard, Versus (UI)
Panes popover: three mini grids shaded by one action + top-5 regions. Deck popover: 13×4 cards, board cards inert, held cards outlined, bar = `potLift`; tapping toggles a held card (max five); the holding strip in the SpotHead shows them. DrawBoard replaces Grid when cursor = 'draw': rows = inner classes, columns = throw 0..5, plus the held hand's per-card bars. Versus tab: the other seat's betting range (weights) beside yours, hover a cell → `equityAgainst`; filmstrip under it: deal, their action, your action (thumbnails from `gridView` with weights), draw frames as throw-count bars.

### Task 9: stylesheet, tests, build wiring
`index.css` with the pressables listed; add `"solver-viz"` to `APPS` in both Python tests and `PRESSABLES["solver-viz"]`; `vercel.json` rewrite `/solver`; `scripts/vercel_build.sh` rungs list gets `solver`; README gets a paragraph. Run `uv run pytest tests/test_ui_standard.py tests/test_shared_chrome.py tests/test_cover_page.py`, `npm test`, `npm run build` in the app; screenshot with puppeteer at 1360 px and review every view.
