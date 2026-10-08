/**
 * Every view of a range as an aggregate over a `RangeTable` under a mask:
 * the grid and its marginals, the strength ribbon, the equity plane, the
 * deck statistics, the draw table and the example hands of a cell.
 *
 * Terms: a *hand* is one row of the table (`table.ts`) with its weight, its
 * region (inner row × outer column), its mix and its equities. A *cell*
 * here is any bucket of hands — a grid cell, a ribbon bin, a plane pixel —
 * carrying its *share* (the bucket's weight over the WHOLE table's weight,
 * so a filtered view shows how much of the full range is left), its count,
 * and the weighted mean mix of the hands in it. A *mask* of 0/1
 * (`filter.ts`) or null, meaning every hand, says which hands count. The
 * table may be the full game's uniform sample or mini-drawmaha's exact
 * reach-weighted enumeration; nothing here knows which.
 */

import { type RangeTable, assertTable, weightOf } from './table'

export interface Mix {
  f: number
  c: number
  p: number
}

/** A bucket of hands: its share of the whole table's weight, its count, its weighted mean mix. */
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

/** Running sums for a set of cells: count, Σw, Σw·f, Σw·c, Σw·p at [5k, 5k + 5). */
class Sums {
  readonly a: Float64Array
  constructor(cells: number) {
    this.a = new Float64Array(cells * 5)
  }
  add(k: number, t: RangeTable, i: number): void {
    const b = k * 5
    const w = weightOf(t, i)
    this.a[b] += 1
    this.a[b + 1] += w
    this.a[b + 2] += w * t.f[i]
    this.a[b + 3] += w * t.c[i]
    this.a[b + 4] += w * t.p[i]
  }
  /** The cell for bucket `k`; an empty bucket has share 0 and a zero mix, never NaN. */
  cell(k: number, totalWeight: number): Cell {
    const b = k * 5
    const n = this.a[b]
    const w = this.a[b + 1]
    if (n === 0 || w === 0) return { share: 0, n, f: 0, c: 0, p: 0 }
    return { share: w / totalWeight, n, f: this.a[b + 2] / w, c: this.a[b + 3] / w, p: this.a[b + 4] / w }
  }
}

function assertMask(t: RangeTable, mask: Uint8Array | null): void {
  if (mask !== null && mask.length !== t.n) throw new Error(`mask has ${mask.length} entries for a table of ${t.n}`)
}

/** The whole table's weight: the denominator every share is taken over. */
function totalWeight(t: RangeTable): number {
  if (t.weight === null) return 1
  let s = 0
  for (let i = 0; i < t.n; i++) s += t.weight[i]
  if (!(s > 0)) throw new Error(`the table's weights sum to ${s}; a range has to weigh something`)
  return s
}

/** The grid: a cell per region pair, the row and column marginals, and the whole. */
export function gridView(t: RangeTable, mask: Uint8Array | null): GridView {
  assertTable(t)
  assertMask(t, mask)
  const ROWS = t.rows
  const COLS = t.cols
  const total = totalWeight(t)
  const sums = new Sums(ROWS * COLS)
  for (let i = 0; i < t.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    if (t.row[i] >= ROWS || t.col[i] >= COLS) throw new Error(`hand ${i} sits at (${t.row[i]}, ${t.col[i]}) outside a ${ROWS}×${COLS} grid`)
    sums.add(t.row[i] * COLS + t.col[i], t, i)
  }
  const cells: Cell[][] = []
  const rowSums = new Sums(ROWS)
  const colSums = new Sums(COLS)
  const allSums = new Sums(1)
  for (let r = 0; r < ROWS; r++) {
    const line: Cell[] = []
    for (let c = 0; c < COLS; c++) {
      const b = (r * COLS + c) * 5
      line.push(sums.cell(r * COLS + c, total))
      for (let k = 0; k < 5; k++) {
        rowSums.a[r * 5 + k] += sums.a[b + k]
        colSums.a[c * 5 + k] += sums.a[b + k]
        allSums.a[k] += sums.a[b + k]
      }
    }
    cells.push(line)
  }
  return {
    cells,
    rows: Array.from({ length: ROWS }, (_, r) => rowSums.cell(r, total)),
    cols: Array.from({ length: COLS }, (_, c) => colSums.cell(c, total)),
    all: allSums.cell(0, total),
  }
}

/** The ribbon's sort value of hand `i` under `key`; outer-based keys need an outer equity. */
export function keyValue(t: RangeTable, key: RibbonKey, i: number): number {
  switch (key) {
    case 'total':
      return t.eq[i]
    case 'inner':
      return t.eqI[i]
    case 'outer':
      if (t.eqO === null) throw new Error('this table has no outer equity to sort by')
      return t.eqO[i]
    case 'scoop':
      if (t.eqO === null) throw new Error('this table has no outer equity to sort by')
      return t.eqI[i] * t.eqO[i]
  }
}

/** Keys in [0, 1] are quantised to this many steps before packing: finer than a Float32 can tell apart. */
const KEY_STEPS = 2 ** 30

/**
 * The indices of the masked hands in ascending order of a key in [0, 1].
 * The key and the index are packed into one double and sorted natively,
 * which is an order of magnitude faster than a comparator sort.
 */
function orderedBy(t: RangeTable, mask: Uint8Array | null, key: (i: number) => number): Uint32Array {
  let m = 0
  for (let i = 0; i < t.n; i++) if (mask === null || mask[i] !== 0) m++
  const packed = new Float64Array(m)
  let k = 0
  for (let i = 0; i < t.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    const v = key(i)
    if (!(v >= 0 && v <= 1)) throw new Error(`sort key ${v} of hand ${i} is outside [0, 1]`)
    packed[k++] = Math.round(v * KEY_STEPS) * t.n + i
  }
  packed.sort()
  const order = new Uint32Array(m)
  for (let j = 0; j < m; j++) order[j] = packed[j] % t.n
  return order
}

/**
 * The strength ribbon: the masked hands sorted weakest to strongest by `key`
 * and cut into `bins` slices of equal weight, each the weighted mean mix of
 * its slice. A slice boundary falls inside a hand only when the hand weighs
 * more than a slice, in which case the hand fills the slice on its own.
 * Fewer masked hands than bins gives one bin per hand; none gives [].
 */
export function ribbon(t: RangeTable, mask: Uint8Array | null, key: RibbonKey, bins: number): Mix[] {
  assertTable(t)
  assertMask(t, mask)
  if (!Number.isInteger(bins) || bins < 1) throw new Error(`bins must be a positive integer, got ${bins}`)
  const order = orderedBy(t, mask, (i) => keyValue(t, key, i))
  const m = order.length
  if (m === 0) return []
  const count = Math.min(bins, m)
  let masked = 0
  for (let j = 0; j < m; j++) masked += weightOf(t, order[j])
  if (!(masked > 0)) return []
  const perBin = masked / count
  const out: Mix[] = []
  let j = 0
  for (let b = 0; b < count; b++) {
    // The last bin takes whatever rounding left over, so every hand lands somewhere.
    const target = b === count - 1 ? Number.POSITIVE_INFINITY : perBin
    let w = 0
    let f = 0
    let c = 0
    let p = 0
    while (j < m && (w < target || w === 0)) {
      const i = order[j++]
      const wi = weightOf(t, i)
      w += wi
      f += wi * t.f[i]
      c += wi * t.c[i]
      p += wi * t.p[i]
    }
    if (w === 0) break
    out.push({ f: f / w, c: c / w, p: p / w })
  }
  return out
}

/**
 * Where a value sits on the ribbon: the weight of the whole (unmasked) table
 * whose key is strictly below `v`, as a fraction of the table's weight.
 */
export function percentileOf(t: RangeTable, key: RibbonKey, v: number): number {
  assertTable(t)
  const total = totalWeight(t)
  let below = 0
  for (let i = 0; i < t.n; i++) if (keyValue(t, key, i) < v) below += weightOf(t, i)
  return below / total
}

/**
 * The equity plane: bins × bins cells over (outer equity → x, inner equity →
 * y), cell (x, y) at index y × bins + x, each a `Cell` of the hands landing
 * there. An equity of exactly 1 lands in the last bin. A table without an
 * outer equity has no plane.
 */
export function density(t: RangeTable, mask: Uint8Array | null, bins: number): Cell[] {
  assertTable(t)
  assertMask(t, mask)
  if (t.eqO === null) throw new Error('this table has no outer equity; there is no plane to draw')
  if (!Number.isInteger(bins) || bins < 1) throw new Error(`bins must be a positive integer, got ${bins}`)
  const total = totalWeight(t)
  const sums = new Sums(bins * bins)
  for (let i = 0; i < t.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    const x = Math.min(bins - 1, Math.floor(t.eqO[i] * bins))
    const y = Math.min(bins - 1, Math.floor(t.eqI[i] * bins))
    sums.add(y * bins + x, t, i)
  }
  return Array.from({ length: bins * bins }, (_, k) => sums.cell(k, total))
}

/**
 * The deck under the mask. `inRange[c]`: the weighted fraction of masked
 * hands holding card c (Σ = handSize). `potLift[c]`: the card's share of
 * the masked hands' total pot probability divided by its plain share — 1 is
 * neutral, above 1 the card leans the range towards potting; 0 when the
 * card is never held (or nothing pots). An empty slot (255) is skipped.
 */
export function deckStats(t: RangeTable, mask: Uint8Array | null): { inRange: Float32Array; potLift: Float32Array } {
  assertTable(t)
  assertMask(t, mask)
  const count = new Float64Array(t.deckSize)
  const potSum = new Float64Array(t.deckSize)
  let m = 0
  let potTotal = 0
  for (let i = 0; i < t.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    const w = weightOf(t, i)
    m += w
    potTotal += w * t.p[i]
    for (let k = 0; k < t.handSize; k++) {
      const c = t.cards[i * t.handSize + k]
      if (c === 255) continue
      if (c >= t.deckSize) throw new Error(`hand ${i} holds card ${c}, outside a deck of ${t.deckSize}`)
      count[c] += w
      potSum[c] += w * t.p[i]
    }
  }
  const inRange = new Float32Array(t.deckSize)
  const potLift = new Float32Array(t.deckSize)
  if (m === 0) return { inRange, potLift }
  for (let c = 0; c < t.deckSize; c++) {
    inRange[c] = count[c] / m
    if (count[c] > 0 && potTotal > 0) potLift[c] = potSum[c] / potTotal / (count[c] / m)
  }
  return { inRange, potLift }
}

/**
 * The draw table: per inner row, its share of the whole table and the
 * weighted mean distribution over the `k` throw options (`throws` is
 * n × k, row-major) of its masked hands; an empty row has share 0 and zero
 * counts.
 */
export function drawTable(t: RangeTable, throws: Float32Array, k: number, mask: Uint8Array | null): { share: number; counts: number[] }[] {
  assertTable(t)
  assertMask(t, mask)
  if (!Number.isInteger(k) || k < 1) throw new Error(`k must be a positive integer, got ${k}`)
  if (throws.length !== t.n * k) throw new Error(`throws has ${throws.length} entries for a table of ${t.n} (wanted ${t.n * k})`)
  const total = totalWeight(t)
  const w = new Float64Array(t.rows)
  const sums = new Float64Array(t.rows * k)
  for (let i = 0; i < t.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    const r = t.row[i]
    const wi = weightOf(t, i)
    w[r] += wi
    for (let j = 0; j < k; j++) sums[r * k + j] += wi * throws[i * k + j]
  }
  return Array.from({ length: t.rows }, (_, r) => ({
    share: w[r] / total,
    counts: Array.from({ length: k }, (_, j) => (w[r] === 0 ? 0 : sums[r * k + j] / w[r])),
  }))
}

export interface Example {
  cards: number[]
  mix: Mix
  eqI: number
  /** NaN when the table has no outer equity */
  eqO: number
  weight: number
}

/**
 * Up to `k` hands of cell (row, col) under the mask, the ones that pot most
 * first; each with its cards high to low, its mix, its equities and its
 * weight.
 */
export function examples(t: RangeTable, mask: Uint8Array | null, row: number, col: number, k: number): Example[] {
  assertTable(t)
  assertMask(t, mask)
  if (!Number.isInteger(row) || row < 0 || row >= t.rows) throw new Error(`${row} is not an inner row (0..${t.rows - 1})`)
  if (!Number.isInteger(col) || col < 0 || col >= t.cols) throw new Error(`${col} is not an outer column (0..${t.cols - 1})`)
  if (!Number.isInteger(k) || k < 0) throw new Error(`k must be a non-negative integer, got ${k}`)
  const inCell = new Uint8Array(t.n)
  for (let i = 0; i < t.n; i++) if ((mask === null || mask[i] !== 0) && t.row[i] === row && t.col[i] === col) inCell[i] = 1
  // Ascending in 1 − p is descending in p.
  const order = orderedBy(t, inCell, (i) => 1 - t.p[i])
  const out: Example[] = []
  for (let j = 0; j < Math.min(k, order.length); j++) {
    const i = order[j]
    out.push({
      cards: Array.from(t.cards.subarray(i * t.handSize, (i + 1) * t.handSize))
        .filter((c) => c !== 255)
        .sort((a, b) => b - a),
      mix: { f: t.f[i], c: t.c[i], p: t.p[i] },
      eqI: t.eqI[i],
      eqO: t.eqO === null ? Number.NaN : t.eqO[i],
      weight: weightOf(t, i),
    })
  }
  return out
}
