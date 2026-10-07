/**
 * The sampled range: N holdings dealt from the cards unseen on the board,
 * classified twice, stored as a struct of typed arrays.
 *
 * Terms: a *holding* is five hole cards; the *inner* hand is the holding as a
 * poker hand and the *outer* hand its best two-hole-plus-three-board Omaha
 * hand (`classify.ts`). Each hand's *equity* on a half is the share of the
 * sample its score beats, ties counted at half — a showdown-now percentile,
 * not a run-out. The *region* indices (`row`, `col`) place each hand in the
 * grid (`regions.ts`). Everything downstream — the *illustrative policy*, the
 * aggregates, the range-vs-range equities — is a pass over these arrays, so
 * one sample is drawn per board and seed and reused. When the user pins held
 * cards the sample is too thin to narrow (three held cards leave ~27 of 50k),
 * so `exactRange` enumerates every holding containing them instead and
 * measures their equities against the sample, so a percentile means the same
 * thing in both views.
 */

import { type Card, assertDistinct, cardLabel, unseen } from './cards'
import { classifyInner, classifyOuter } from './classify'
import { innerRow, outerCol } from './regions'

export interface RangeSample {
  n: number
  board: Card[]
  seed: number
  /** n × 5, hand i at [5i, 5i + 5). */
  cards: Uint8Array
  innerCat: Uint8Array
  outerCat: Uint8Array
  innerScore: Uint32Array
  outerScore: Uint32Array
  innerFourFlush: Uint8Array
  innerFourStraight: Uint8Array
  outerFlushDraw: Uint8Array
  outerStraightDraw: Uint8Array
  /** Percentile of the score in the sample, ties at half. */
  innerEquity: Float32Array
  outerEquity: Float32Array
  /** Composite region indices (`regions.ts`). */
  row: Uint8Array
  col: Uint8Array
}

/**
 * Sorting keys are packed as `value × n + index` into a Float64Array so the
 * engine can use the typed array's native numeric sort; this is the largest
 * n for which score × n + index stays an exact double.
 */
export const MAX_N = 2 ** 23

/** mulberry32: a small, fast, seedable PRNG returning doubles in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed | 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * For every score, the fraction of the others it beats plus half the
 * fraction it ties: `(count below + ½ count equal) / n`.
 */
export function percentileRanks(scores: Uint32Array): Float32Array {
  const n = scores.length
  const keys = new Float64Array(n)
  for (let i = 0; i < n; i++) keys[i] = scores[i] * n + i
  keys.sort()
  const out = new Float32Array(n)
  let i = 0
  while (i < n) {
    const score = Math.floor(keys[i] / n)
    let j = i
    while (j + 1 < n && Math.floor(keys[j + 1] / n) === score) j++
    const p = (i + 0.5 * (j - i + 1)) / n
    for (let k = i; k <= j; k++) out[keys[k] - score * n] = p
    i = j + 1
  }
  return out
}

/**
 * Deal `n` holdings of five distinct cards from the cards unseen on `board`
 * (three to five distinct cards), with the PRNG seeded by `seed`, and fill
 * every array. The same board, n and seed give byte-identical arrays.
 */
export function sampleRange(board: readonly Card[], n: number, seed: number): RangeSample {
  if (board.length < 3 || board.length > 5) throw new Error(`a board is three to five cards, got ${board.length}`)
  assertDistinct(board, 'board')
  if (!Number.isInteger(n) || n < 1 || n > MAX_N) throw new Error(`sample size must be an integer in 1..${MAX_N}, got ${n}`)
  if (!Number.isInteger(seed)) throw new Error(`seed must be an integer, got ${seed}`)

  const rand = mulberry32(seed)
  const deck = Uint8Array.from(unseen(board))
  const m = deck.length
  const cards = new Uint8Array(n * 5)
  for (let i = 0; i < n; i++) {
    // Partial Fisher–Yates: the deck stays a permutation of the unseen
    // cards, so there is nothing to reset between hands.
    for (let k = 0; k < 5; k++) {
      const j = k + Math.floor(rand() * (m - k))
      const t = deck[k]
      deck[k] = deck[j]
      deck[j] = t
      cards[i * 5 + k] = deck[k]
    }
  }
  const classes = classifyAll(board, cards)
  return {
    n, board: [...board], seed, cards, ...classes,
    innerEquity: percentileRanks(classes.innerScore), outerEquity: percentileRanks(classes.outerScore),
  }
}

/** The number of k-subsets of an m-set. */
function choose(m: number, k: number): number {
  let v = 1
  for (let i = 0; i < k; i++) v = (v * (m - i)) / (i + 1)
  return Math.round(v)
}

/**
 * Every holding that contains all of `held` (one to five distinct cards, none
 * on the board, which must be `reference`'s board), as a RangeSample: the
 * held cards first, then the others in deck order, hands in lexicographic
 * order of the others. `n` is C(49 − k, 5 − k); `seed` is the reference's.
 * The equities are measured against the REFERENCE sample's scores — the
 * fraction of its hands beaten, ties at half — not within the enumeration,
 * so a hand's percentile means the same thing here as in the sample.
 */
export function exactRange(board: readonly Card[], held: readonly Card[], reference: RangeSample): RangeSample {
  assertSameBoard(reference, board)
  if (held.length < 1 || held.length > 5) throw new Error(`exactRange wants one to five held cards, got ${held.length}`)
  assertDistinct(held, 'held cards')
  for (const c of held) if (board.includes(c)) throw new Error(`held card ${cardLabel(c)} is on the board`)

  const k = held.length
  const pool = unseen(board).filter((c) => !held.includes(c))
  const rest = 5 - k
  const n = choose(pool.length, rest)
  const cards = new Uint8Array(n * 5)
  // Lexicographic combinations of `rest` positions into the pool.
  const pick = Array.from({ length: rest }, (_, i) => i)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) cards[i * 5 + j] = held[j]
    for (let j = 0; j < rest; j++) cards[i * 5 + k + j] = pool[pick[j]]
    let j = rest - 1
    while (j >= 0 && pick[j] === pool.length - rest + j) j--
    if (j < 0) break
    pick[j]++
    for (let t = j + 1; t < rest; t++) pick[t] = pick[t - 1] + 1
  }

  const classes = classifyAll(board, cards)
  const sorted = sortedScores(reference)
  const innerEquity = new Float32Array(n)
  const outerEquity = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    innerEquity[i] = beatFraction(sorted.inner, classes.innerScore[i])
    outerEquity[i] = beatFraction(sorted.outer, classes.outerScore[i])
  }
  return { n, board: [...board], seed: reference.seed, cards, ...classes, innerEquity, outerEquity }
}

type Classes = Omit<RangeSample, 'n' | 'board' | 'seed' | 'cards' | 'innerEquity' | 'outerEquity'>

/** Classify every hand of a packed n × 5 card array, both ways, with its region. */
function classifyAll(board: readonly Card[], cards: Uint8Array): Classes {
  const n = cards.length / 5
  const innerCat = new Uint8Array(n)
  const outerCat = new Uint8Array(n)
  const innerScore = new Uint32Array(n)
  const outerScore = new Uint32Array(n)
  const innerFourFlush = new Uint8Array(n)
  const innerFourStraight = new Uint8Array(n)
  const outerFlushDraw = new Uint8Array(n)
  const outerStraightDraw = new Uint8Array(n)
  const row = new Uint8Array(n)
  const col = new Uint8Array(n)
  const hole: Card[] = [0, 0, 0, 0, 0]
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 5; k++) hole[k] = cards[i * 5 + k]
    const inner = classifyInner(hole)
    const outer = classifyOuter(hole, board)
    innerCat[i] = inner.cat
    innerScore[i] = inner.score
    innerFourFlush[i] = inner.fourFlush ? 1 : 0
    innerFourStraight[i] = inner.fourStraight
    outerCat[i] = outer.cat
    outerScore[i] = outer.score
    outerFlushDraw[i] = outer.flushDraw ? 1 : 0
    outerStraightDraw[i] = outer.straightDraw
    row[i] = innerRow(inner.cat, inner.fourFlush, inner.fourStraight)
    col[i] = outerCol(outer.cat, outer.flushDraw, outer.straightDraw)
  }
  return { innerCat, outerCat, innerScore, outerScore, innerFourFlush, innerFourStraight, outerFlushDraw, outerStraightDraw, row, col }
}

const sortedCache = new WeakMap<RangeSample, { inner: Uint32Array; outer: Uint32Array }>()

/**
 * The sample's inner and outer scores sorted ascending, computed once per
 * sample: the binary-search base for placing a hand that is not in the sample.
 */
export function sortedScores(s: RangeSample): { inner: Uint32Array; outer: Uint32Array } {
  let sorted = sortedCache.get(s)
  if (!sorted) {
    sorted = { inner: s.innerScore.slice().sort(), outer: s.outerScore.slice().sort() }
    sortedCache.set(s, sorted)
  }
  return sorted
}

/**
 * The share of a sorted score array that `score` beats, ties at half: the
 * same quantity `percentileRanks` gives a member of the sample.
 */
export function beatFraction(sorted: Uint32Array, score: number): number {
  const below = lowerBound(sorted, score)
  const upTo = lowerBound(sorted, score + 1)
  return (below + 0.5 * (upTo - below)) / sorted.length
}

/** The first index whose value is ≥ `v` in an ascending array (its length if none). */
export function lowerBound(sorted: ArrayLike<number>, v: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (sorted[mid] < v) lo = mid + 1
    else hi = mid
  }
  return lo
}

/**
 * Throws unless `board` is the sample's board (same cards, any order): every
 * number read off a sample is meaningless on another board.
 */
export function assertSameBoard(s: RangeSample, board: readonly Card[]): void {
  const same = board.length === s.board.length && board.every((c) => s.board.includes(c))
  if (!same) throw new Error(`the hand's board [${board.join(' ')}] is not the sample's board [${s.board.join(' ')}]`)
}
