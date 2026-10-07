/**
 * Every view of the range as an aggregate over the sample under a mask: the
 * grid and its marginals, the strength ribbon, the equity plane, the deck
 * statistics, the draw table and the example hands of a cell.
 *
 * Terms: a *holding* is a sampled hand of five hole cards; its *inner* hand
 * is the holding as a poker hand and its *outer* hand the best two-hole-plus-
 * three-board Omaha hand; each has an equity, the share of the sample it
 * beats (`sample.ts`). A *region* is an inner row or outer column
 * (`regions.ts`); a *cell* here is any bucket of hands — a grid cell, a
 * ribbon bin, a plane pixel — carrying its share of the WHOLE sample (so a
 * filtered view shows how much of the full range is left) and the mean
 * *illustrative policy* mix (`policy.ts`) of the hands in it. A mask of 0/1
 * (`filter.ts`) or null, meaning every hand, says which hands count.
 */

import type { Card } from './cards'
import { INNER_ROWS, OUTER_COLS } from './regions'
import { type Policy, equitiesOfHand, mixOfHand } from './policy'
import type { RangeSample } from './sample'

export interface Mix {
  f: number
  c: number
  p: number
}

/** A bucket of hands: its share of the whole sample, its count, its mean mix. */
export interface Cell extends Mix {
  share: number
  n: number
}

export interface GridView {
  /** [inner row][outer column]. */
  cells: Cell[][]
  rows: Cell[]
  cols: Cell[]
  all: Cell
}

export type RibbonKey = 'total' | 'inner' | 'outer' | 'scoop'

const ROWS = INNER_ROWS.length
const COLS = OUTER_COLS.length

/** Running sums for a set of cells: count, Σf, Σc, Σp at [4k, 4k + 4). */
class Sums {
  readonly a: Float64Array
  constructor(cells: number) {
    this.a = new Float64Array(cells * 4)
  }
  add(k: number, pol: Policy, i: number): void {
    const b = k * 4
    this.a[b] += 1
    this.a[b + 1] += pol.f[i]
    this.a[b + 2] += pol.c[i]
    this.a[b + 3] += pol.p[i]
  }
  /** The cell for bucket `k`; an empty bucket has share 0 and a zero mix, never NaN. */
  cell(k: number, total: number): Cell {
    const b = k * 4
    const n = this.a[b]
    if (n === 0) return { share: 0, n: 0, f: 0, c: 0, p: 0 }
    return { share: n / total, n, f: this.a[b + 1] / n, c: this.a[b + 2] / n, p: this.a[b + 3] / n }
  }
}

function assertMask(s: RangeSample, mask: Uint8Array | null): void {
  if (mask !== null && mask.length !== s.n) throw new Error(`mask has ${mask.length} entries for a sample of ${s.n}`)
}

function assertPolicy(s: RangeSample, pol: Policy): void {
  if (pol.p.length !== s.n) throw new Error(`policy has ${pol.p.length} entries for a sample of ${s.n}`)
}

/** The grid: a cell per region pair, the row and column marginals, and the whole. */
export function gridView(s: RangeSample, pol: Policy, mask: Uint8Array | null): GridView {
  assertMask(s, mask)
  assertPolicy(s, pol)
  const sums = new Sums(ROWS * COLS)
  for (let i = 0; i < s.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    sums.add(s.row[i] * COLS + s.col[i], pol, i)
  }
  const cells: Cell[][] = []
  const rowSums = new Sums(ROWS)
  const colSums = new Sums(COLS)
  const allSums = new Sums(1)
  for (let r = 0; r < ROWS; r++) {
    const line: Cell[] = []
    for (let c = 0; c < COLS; c++) {
      const b = (r * COLS + c) * 4
      line.push(sums.cell(r * COLS + c, s.n))
      for (let t = 0; t < 4; t++) {
        rowSums.a[r * 4 + t] += sums.a[b + t]
        colSums.a[c * 4 + t] += sums.a[b + t]
        allSums.a[t] += sums.a[b + t]
      }
    }
    cells.push(line)
  }
  return {
    cells,
    rows: Array.from({ length: ROWS }, (_, r) => rowSums.cell(r, s.n)),
    cols: Array.from({ length: COLS }, (_, c) => colSums.cell(c, s.n)),
    all: allSums.cell(0, s.n),
  }
}

function keyValue(s: RangeSample, pol: Policy, key: RibbonKey, i: number): number {
  switch (key) {
    case 'total': return pol.eq[i]
    case 'inner': return s.innerEquity[i]
    case 'outer': return s.outerEquity[i]
    case 'scoop': return s.innerEquity[i] * s.outerEquity[i]
  }
}

/** Keys in [0, 1] are quantised to this many steps before packing: finer than a Float32 can tell apart. */
const KEY_STEPS = 2 ** 30

/**
 * The indices of the masked hands in ascending order of a key in [0, 1].
 * The key and the index are packed into one double and sorted natively,
 * which is an order of magnitude faster than a comparator sort.
 */
function orderedBy(s: RangeSample, mask: Uint8Array | null, key: (i: number) => number): Uint32Array {
  let m = 0
  for (let i = 0; i < s.n; i++) if (mask === null || mask[i] !== 0) m++
  const packed = new Float64Array(m)
  let k = 0
  for (let i = 0; i < s.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    const v = key(i)
    if (!(v >= 0 && v <= 1)) throw new Error(`sort key ${v} of hand ${i} is outside [0, 1]`)
    packed[k++] = Math.round(v * KEY_STEPS) * s.n + i
  }
  packed.sort()
  const order = new Uint32Array(m)
  for (let j = 0; j < m; j++) order[j] = packed[j] % s.n
  return order
}

/**
 * The strength ribbon: the masked hands sorted weakest to strongest by `key`
 * and cut into `bins` equal-count slices, each the mean mix of its slice.
 * Fewer masked hands than bins gives one bin per hand; none gives [].
 */
export function ribbon(s: RangeSample, pol: Policy, mask: Uint8Array | null, key: RibbonKey, bins: number): Mix[] {
  assertMask(s, mask)
  assertPolicy(s, pol)
  if (!Number.isInteger(bins) || bins < 1) throw new Error(`bins must be a positive integer, got ${bins}`)
  const order = orderedBy(s, mask, (i) => keyValue(s, pol, key, i))
  const m = order.length
  const count = Math.min(bins, m)
  const out: Mix[] = []
  for (let b = 0; b < count; b++) {
    const from = Math.floor((b * m) / count)
    const to = Math.floor(((b + 1) * m) / count)
    let f = 0
    let c = 0
    let p = 0
    for (let k = from; k < to; k++) {
      f += pol.f[order[k]]
      c += pol.c[order[k]]
      p += pol.p[order[k]]
    }
    const len = to - from
    out.push({ f: f / len, c: c / len, p: p / len })
  }
  return out
}

/**
 * Where one hand sits on the ribbon: the fraction of the whole (unmasked)
 * sample whose key is strictly below the hand's. The hand need not be in
 * the sample; its key is computed the way `mixOfHand` computes its equities.
 */
export function percentileOf(s: RangeSample, pol: Policy, key: RibbonKey, hole: readonly Card[], board: readonly Card[]): number {
  assertPolicy(s, pol)
  const { eqI, eqO } = equitiesOfHand(hole, board, s)
  const v =
    key === 'total' ? mixOfHand(hole, board, s, 'facing').eq
    : key === 'inner' ? eqI
    : key === 'outer' ? eqO
    : eqI * eqO
  let below = 0
  for (let i = 0; i < s.n; i++) if (keyValue(s, pol, key, i) < v) below++
  return below / s.n
}

/**
 * The equity plane: bins × bins cells over (outer equity → x, inner equity →
 * y), cell (x, y) at index y × bins + x, each a `Cell` of the hands landing
 * there. An equity of exactly 1 lands in the last bin.
 */
export function density(s: RangeSample, pol: Policy, mask: Uint8Array | null, bins: number): Cell[] {
  assertMask(s, mask)
  assertPolicy(s, pol)
  if (!Number.isInteger(bins) || bins < 1) throw new Error(`bins must be a positive integer, got ${bins}`)
  const sums = new Sums(bins * bins)
  for (let i = 0; i < s.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    const x = Math.min(bins - 1, Math.floor(s.outerEquity[i] * bins))
    const y = Math.min(bins - 1, Math.floor(s.innerEquity[i] * bins))
    sums.add(y * bins + x, pol, i)
  }
  return Array.from({ length: bins * bins }, (_, k) => sums.cell(k, s.n))
}

/**
 * The deck under the mask. `inRange[c]`: the fraction of masked hands
 * holding card c (Σ = 5, every hand holds five). `potLift[c]`: the card's
 * share of the masked hands' total pot probability divided by its plain
 * share of the masked hands — 1 is neutral, above 1 the card leans the
 * range towards potting; 0 when the card is never held (or nothing pots).
 */
export function deckStats(s: RangeSample, pol: Policy, mask: Uint8Array | null): { inRange: Float32Array; potLift: Float32Array } {
  assertMask(s, mask)
  assertPolicy(s, pol)
  const count = new Float64Array(52)
  const potSum = new Float64Array(52)
  let m = 0
  let potTotal = 0
  for (let i = 0; i < s.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    m++
    potTotal += pol.p[i]
    for (let k = 0; k < 5; k++) {
      const c = s.cards[i * 5 + k]
      count[c] += 1
      potSum[c] += pol.p[i]
    }
  }
  const inRange = new Float32Array(52)
  const potLift = new Float32Array(52)
  if (m === 0) return { inRange, potLift }
  for (let c = 0; c < 52; c++) {
    inRange[c] = count[c] / m
    if (count[c] > 0 && potTotal > 0) potLift[c] = potSum[c] / potTotal / (count[c] / m)
  }
  return { inRange, potLift }
}

/**
 * The draw table: per inner row, its share of the whole sample and the mean
 * throw-count distribution (`throwCounts`) of its masked hands; an empty row
 * has share 0 and zero counts.
 */
export function drawTable(s: RangeSample, throws: Float32Array, mask: Uint8Array | null): { share: number; counts: number[] }[] {
  assertMask(s, mask)
  if (throws.length !== s.n * 6) throw new Error(`throws has ${throws.length} entries for a sample of ${s.n} (wanted ${s.n * 6})`)
  const n = new Float64Array(ROWS)
  const sums = new Float64Array(ROWS * 6)
  for (let i = 0; i < s.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    const r = s.row[i]
    n[r] += 1
    for (let k = 0; k < 6; k++) sums[r * 6 + k] += throws[i * 6 + k]
  }
  return Array.from({ length: ROWS }, (_, r) => ({
    share: n[r] / s.n,
    counts: Array.from({ length: 6 }, (_, k) => (n[r] === 0 ? 0 : sums[r * 6 + k] / n[r])),
  }))
}

/**
 * Up to `k` hands of cell (row, col) under the mask, the ones that pot most
 * first; each with its cards high to low, its mix and its two equities.
 */
export function examples(
  s: RangeSample, pol: Policy, mask: Uint8Array | null, row: number, col: number, k: number,
): { cards: Card[]; mix: Mix; eqI: number; eqO: number }[] {
  assertMask(s, mask)
  assertPolicy(s, pol)
  if (!Number.isInteger(row) || row < 0 || row >= ROWS) throw new Error(`${row} is not an inner row (0..${ROWS - 1})`)
  if (!Number.isInteger(col) || col < 0 || col >= COLS) throw new Error(`${col} is not an outer column (0..${COLS - 1})`)
  if (!Number.isInteger(k) || k < 0) throw new Error(`k must be a non-negative integer, got ${k}`)
  const inCell = new Uint8Array(s.n)
  for (let i = 0; i < s.n; i++) if ((mask === null || mask[i] !== 0) && s.row[i] === row && s.col[i] === col) inCell[i] = 1
  // Ascending in 1 − p is descending in p.
  const order = orderedBy(s, inCell, (i) => 1 - pol.p[i])
  const out = []
  for (let j = 0; j < Math.min(k, order.length); j++) {
    const i = order[j]
    out.push({
      cards: Array.from(s.cards.subarray(i * 5, i * 5 + 5)).sort((a, b) => b - a),
      mix: { f: pol.f[i], c: pol.c[i], p: pol.p[i] },
      eqI: s.innerEquity[i],
      eqO: s.outerEquity[i],
    })
  }
  return out
}
