/**
 * The five-card poker ranking as one integer, higher wins.
 *
 * Terms: a *card* is 0..51 = rank × 4 + suit (see `cards.ts`). A *holding* is
 * five hole cards; the *inner* hand is that holding scored here directly, the
 * *outer* hand is the best two-hole-plus-three-board selection scored here
 * (`classify.ts`). A score is `category × 13⁵ + tiebreak`, the tiebreak being
 * the ranks that decide the category, most important first, written base 13
 * and right-padded to five digits — the same encoding as the Python study's
 * `score_int`, so numbers agree across the two.
 */

import { type Card, assertDistinct, rankOf, suitOf } from './cards'

export const CATEGORY_NAMES = [
  'high card', 'pair', 'two pair', 'trips', 'straight', 'flush', 'full house', 'quads', 'straight flush',
] as const

/** 13⁵: the width of one category in the score. */
export const CATEGORY_BASE = 13 ** 5

/** Ranks 2..6 up to T..A, then the wheel A 2 3 4 5, as 13-bit rank masks. */
const STRAIGHT_WINDOWS: readonly number[] = [...Array.from({ length: 9 }, (_, lo) => 0b11111 << lo), (1 << 12) | 0b1111]
const WHEEL = STRAIGHT_WINDOWS[9]

/** Number of set bits in a 13-bit rank mask. */
export function popcount(mask: number): number {
  let n = 0
  for (let m = mask; m !== 0; m &= m - 1) n++
  return n
}

/** The 13-bit mask of the ranks present among `cards`. */
export function rankMaskOf(cards: readonly Card[]): number {
  let mask = 0
  for (const c of cards) mask |= 1 << rankOf(c)
  return mask
}

/**
 * The high rank of the straight these five distinct ranks make, or −1. The
 * wheel is five-high (rank index 3).
 */
export function straightHigh(rankMask: number): number {
  if (popcount(rankMask) !== 5) return -1
  const hi = 31 - Math.clz32(rankMask)
  const lo = 31 - Math.clz32(rankMask & -rankMask)
  if (hi - lo === 4) return hi
  return rankMask === WHEEL ? 3 : -1
}

// One scratch array for the rank counts: score5 runs ~700k times per sample
// and must not allocate.
const counts = new Uint8Array(13)

/**
 * Score exactly five distinct cards. Higher wins; equal scores tie;
 * `categoryOf(score)` recovers the category 0..8.
 */
export function score5(c: readonly Card[]): number {
  if (c.length !== 5) throw new Error(`score5 wants exactly five cards, got ${c.length}`)
  assertDistinct(c, 'hand')
  counts.fill(0)
  let rankMask = 0
  let flush = true
  const suit = suitOf(c[0])
  for (let i = 0; i < 5; i++) {
    const r = rankOf(c[i])
    counts[r]++
    rankMask |= 1 << r
    if (suitOf(c[i]) !== suit) flush = false
  }
  const high = straightHigh(rankMask)
  if (high >= 0) return (flush ? 8 : 4) * CATEGORY_BASE + high * 13 ** 4

  // Groups ordered by size then rank, descending, give both the category
  // and the tiebreak digits in one pass.
  let tiebreak = 0
  let digits = 0
  let g0 = 0
  let g1 = 0
  for (let size = 4; size >= 1; size--) {
    for (let r = 12; r >= 0; r--) {
      if (counts[r] !== size) continue
      if (digits === 0) g0 = size
      else if (digits === 1) g1 = size
      tiebreak = tiebreak * 13 + r
      digits++
    }
  }
  const category =
    g0 === 4 ? 7
    : g0 === 3 && g1 === 2 ? 6
    : flush ? 5
    : g0 === 3 ? 3
    : g0 === 2 && g1 === 2 ? 2
    : g0 === 2 ? 1
    : 0
  return category * CATEGORY_BASE + tiebreak * 13 ** (5 - digits)
}

/** The category 0..8 a `score5` score belongs to. */
export function categoryOf(score: number): number {
  return Math.floor(score / CATEGORY_BASE)
}
