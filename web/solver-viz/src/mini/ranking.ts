/**
 * Both halves of mini-drawmaha's split pot, scored as plain integers that
 * compare the way `hands.Score` tuples compare in Python. A transcript of
 * `src/drawmaha_solver/minidrawmaha/hands.py`, pinned against Python-generated
 * fixtures in `ranking.test.ts`.
 *
 * Terms. The *inner* half goes to the best 3-card hand made of the three
 * private cards alone — nothing public touches it. The *outer* half goes to
 * the best 4-card hand made of EXACTLY two private cards plus BOTH board
 * cards (Omaha's exactly-two rule at this scale), the best of the three
 * pairings. A *category* is what the hand is; its *ranks* are the hand's rank
 * indices ordered by count and then by rank, descending, so a category's own
 * rank beats its kicker. Suits appear nowhere in a score, so two hands tie
 * exactly when nothing but their suits differs.
 *
 * Both rankings are measured rarity in THIS deck, not 52-card convention, with
 * high card pinned to the floor by definition. So the outer order puts a
 * four-card straight BELOW two pair and trips (a run of four out of five ranks
 * is easy), and the inner order puts trips at the top. A straight is distinct
 * consecutive ranks with no wrap: rank 4 does not sit next to rank 0, so
 * `4 0 1 2` is not a run. Trips is never a flush — three of a rank is one
 * card per suit.
 *
 * Encoding. A score is `category × 10_000 + R`, where R is the ranks tuple
 * read as a base-5 numeral, most significant first:
 *
 *   inner (3 ranks):  R = r₀·25 + r₁·5 + r₂            (R ≤ 124)
 *   outer (4 ranks):  R = r₀·125 + r₁·25 + r₂·5 + r₃   (R ≤ 624)
 *
 * The category term dominates because R < 10_000, and a fixed-length
 * most-significant-first numeral preserves lexicographic order on the tuple,
 * so `a < b` between two scores of the same half is exactly Python's
 * `Score(a) < Score(b)`, and `a === b` iff the tuples are equal. Inner and
 * outer scores are never compared with each other.
 */

import { assertMiniCard, rankOf, suitOf, type MiniCard } from './cards'

export const HOLE_CARDS = 3
export const BOARD_CARDS = 2
const OUTER_HOLE_CARDS = 2
const N_RANKS = 5
const CATEGORY_WEIGHT = 10_000

/** `hands.InnerCategory` values; index `category - 1` into `INNER_CATEGORY_NAMES`. */
export const INNER_CATEGORY_NAMES: readonly string[] = [
  'high card',
  'pair',
  'straight',
  'flush',
  'straight flush',
  'trips',
]

/** `hands.OuterCategory` values; index `category - 1` into `OUTER_CATEGORY_NAMES`. */
export const OUTER_CATEGORY_NAMES: readonly string[] = [
  'high card',
  'pair',
  'straight',
  'two pair',
  'trips',
  'flush',
  'straight flush',
]

const INNER = { HIGH_CARD: 1, PAIR: 2, STRAIGHT: 3, FLUSH: 4, STRAIGHT_FLUSH: 5, TRIPS: 6 } as const
const OUTER = { HIGH_CARD: 1, PAIR: 2, STRAIGHT: 3, TWO_PAIR: 4, TRIPS: 5, FLUSH: 6, STRAIGHT_FLUSH: 7 } as const

interface Shape {
  /** rank counts, descending by (count, rank) — the first entry is the hand's dominant rank */
  byCount: { count: number; rank: number }[]
  flush: boolean
  straight: boolean
  /** every rank repeated by its count, in `byCount` order */
  ranks: number[]
}

/**
 * The four facts both rankings are built from. Written once because the two
 * rankings differ only in how they ORDER the categories these facts pick out.
 */
function shape(cards: readonly MiniCard[]): Shape {
  const counts = new Map<number, number>()
  const suits = new Set<number>()
  for (const c of cards) {
    assertMiniCard(c)
    counts.set(rankOf(c), (counts.get(rankOf(c)) ?? 0) + 1)
    suits.add(suitOf(c))
  }
  const distinct = [...counts.keys()].sort((a, b) => b - a)
  // No wrap: the top rank is not the bottom rank's neighbour.
  const straight = distinct.length === cards.length && distinct[0] - distinct[distinct.length - 1] === cards.length - 1
  const byCount = [...counts.entries()]
    .map(([rank, count]) => ({ count, rank }))
    .sort((a, b) => b.count - a.count || b.rank - a.rank)
  const ranks: number[] = []
  for (const { count, rank } of byCount) for (let i = 0; i < count; i++) ranks.push(rank)
  return { byCount, flush: suits.size === 1, straight, ranks }
}

function encode(category: number, ranks: readonly number[]): number {
  let r = 0
  for (const rank of ranks) r = r * N_RANKS + rank
  return category * CATEGORY_WEIGHT + r
}

function decode(score: number, nRanks: number): number[] {
  const category = Math.floor(score / CATEGORY_WEIGHT)
  let r = score - category * CATEGORY_WEIGHT
  const ranks: number[] = Array.from({ length: nRanks }, () => 0)
  for (let i = nRanks - 1; i >= 0; i--) {
    ranks[i] = r % N_RANKS
    r = Math.floor(r / N_RANKS)
  }
  return [category, ...ranks]
}

/** Score the inner half: what the three private cards are. Throws unless exactly three. */
export function innerScore(hole: readonly MiniCard[]): number {
  if (hole.length !== HOLE_CARDS) throw new Error(`the inner half scores three private cards, got ${hole.length}`)
  const { byCount, flush, straight, ranks } = shape(hole)
  const top = byCount[0].count
  let category: number
  if (top === 3) category = INNER.TRIPS
  else if (straight && flush) category = INNER.STRAIGHT_FLUSH
  else if (flush) category = INNER.FLUSH
  else if (straight) category = INNER.STRAIGHT
  else if (top === 2) category = INNER.PAIR
  else category = INNER.HIGH_CARD
  return encode(category, ranks)
}

/** The outer ranking applied to one chosen four-card hand: two hole cards and both board cards. */
function fourCardScore(cards: readonly MiniCard[]): number {
  const { byCount, flush, straight, ranks } = shape(cards)
  const top = byCount[0].count
  const second = byCount.length > 1 ? byCount[1].count : 0
  let category: number
  if (top === 3) category = OUTER.TRIPS
  else if (straight && flush) category = OUTER.STRAIGHT_FLUSH
  else if (flush) category = OUTER.FLUSH
  else if (straight) category = OUTER.STRAIGHT
  else if (top === 2 && second === 2) category = OUTER.TWO_PAIR
  else if (top === 2) category = OUTER.PAIR
  else category = OUTER.HIGH_CARD
  return encode(category, ranks)
}

/**
 * Score the outer half: the best of exactly two hole cards plus both board
 * cards, over the three pairings. Throws unless the hole has three cards and
 * the board two — the outer half does not exist on a one-card board.
 */
export function outerScore(hole: readonly MiniCard[], board: readonly MiniCard[]): number {
  if (hole.length !== HOLE_CARDS) throw new Error(`the outer half chooses from three cards, got ${hole.length}`)
  if (board.length !== BOARD_CARDS) throw new Error(`the outer half uses both board cards, got ${board.length}`)
  let best = -1
  for (let i = 0; i < HOLE_CARDS; i++) {
    for (let j = i + 1; j < HOLE_CARDS; j++) {
      const s = fourCardScore([hole[i], hole[j], ...board])
      if (s > best) best = s
    }
  }
  return best
}

/** The inner category 1..6 of an `innerScore`. */
export const innerCategory = (score: number): number => Math.floor(score / CATEGORY_WEIGHT)
/** The outer category 1..7 of an `outerScore`. */
export const outerCategory = (score: number): number => Math.floor(score / CATEGORY_WEIGHT)

/** An `innerScore` back to Python's `[category, r0, r1, r2]`. */
export const decodeInner = (score: number): number[] => decode(score, HOLE_CARDS)
/** An `outerScore` back to Python's `[category, r0, r1, r2, r3]`. */
export const decodeOuter = (score: number): number[] => decode(score, OUTER_HOLE_CARDS + BOARD_CARDS)
