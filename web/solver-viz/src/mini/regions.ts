/**
 * The grid axes for mini-drawmaha: six inner rows, and either five or seven
 * outer columns depending on how much board there is.
 *
 * Terms. A *region* is one row or column of the range grid (`Region` from
 * `../engine/regions`), and a cell is a row crossed with a column. The
 * *inner rows* are the six `InnerCategory` values of `ranking.ts`, weakest
 * first, because the inner half is decided by the hole alone and never needs
 * a board. The *round-1 columns* describe a hole against ONE board card,
 * where the outer half does not exist yet: they name what the hole is
 * reaching for — a run, a flush, a pair or trips with the board — and are a
 * reading of the position, not a ranking. The *round-2 columns* are the
 * seven `OuterCategory` values once both board cards are out, again weakest
 * first, so the grid reads up and right towards strength in both games.
 */

import type { Region } from '../engine/regions'
import { assertMiniCard, rankOf, suitOf, type MiniCard } from './cards'
import { HOLE_CARDS, INNER_CATEGORY_NAMES, OUTER_CATEGORY_NAMES } from './ranking'

export const MINI_INNER_ROWS: readonly Region[] = [
  { label: 'high card', long: 'three cards, no pair, no run, no suit' },
  { label: 'pair', long: 'two of a rank and a kicker' },
  { label: 'straight', long: 'three ranks in a row, mixed suits' },
  { label: 'flush', long: 'three of a suit, not in a row' },
  { label: 'straight flush', long: 'three of a suit in a row' },
  { label: 'trips', long: 'three of a rank — one card per suit' },
]

export const MINI_R1_COLS: readonly Region[] = [
  { label: 'nothing', long: 'no pair with the board, no two to a run or a suit' },
  { label: 'two to the straight', long: 'two hole ranks and the board rank within a span of four' },
  { label: 'two to the flush', long: 'two hole cards in the board card\'s suit' },
  { label: 'pairs the board', long: 'one hole card of the board card\'s rank' },
  { label: 'trips with the board', long: 'two hole cards of the board card\'s rank' },
]

export const MINI_R2_COLS: readonly Region[] = [
  { label: 'high card', long: 'two hole cards and the board make no pair, run or suit' },
  { label: 'pair', long: 'one pair among two hole cards and the board' },
  { label: 'straight', long: 'four ranks in a row — easy in five ranks, so below two pair' },
  { label: 'two pair', long: 'two pairs among two hole cards and the board' },
  { label: 'trips', long: 'three of a rank among two hole cards and the board' },
  { label: 'flush', long: 'two hole cards and both board cards of one suit' },
  { label: 'straight flush', long: 'four of a suit in a row' },
]

// The axes are the category enums; a drift between the two lists would put a
// hand's mass in another category's row.
if (MINI_INNER_ROWS.length !== INNER_CATEGORY_NAMES.length) throw new Error('inner rows must match InnerCategory')
if (MINI_R2_COLS.length !== OUTER_CATEGORY_NAMES.length) throw new Error('round-2 columns must match OuterCategory')

/**
 * The round-1 column of a hole against the single board card, strongest
 * reading first: 4 trips with the board, 3 pairs the board, 2 two to the
 * flush (two hole cards in the board's suit — a flush needs exactly two hole
 * cards plus both board cards, so this is the live draw), 1 two to the
 * straight (two hole ranks and the board rank are three distinct ranks whose
 * span is at most 3, so one more board card can complete a run of four), 0
 * nothing. Throws unless the hole has three cards.
 */
export function r1Col(hole: readonly MiniCard[], board1: MiniCard): number {
  if (hole.length !== HOLE_CARDS) throw new Error(`a round-1 column reads three hole cards, got ${hole.length}`)
  assertMiniCard(board1)
  for (const c of hole) assertMiniCard(c)
  const boardRank = rankOf(board1)
  const boardSuit = suitOf(board1)
  const paired = hole.filter((c) => rankOf(c) === boardRank).length
  if (paired >= 2) return 4
  if (paired === 1) return 3
  if (hole.filter((c) => suitOf(c) === boardSuit).length >= 2) return 2
  for (let i = 0; i < hole.length; i++) {
    for (let j = i + 1; j < hole.length; j++) {
      const ranks = [rankOf(hole[i]), rankOf(hole[j]), boardRank]
      const distinct = new Set(ranks).size === 3
      if (distinct && Math.max(...ranks) - Math.min(...ranks) <= 3) return 1
    }
  }
  return 0
}

/** The round-2 column of an outer category 1..7: its index in `MINI_R2_COLS`. */
export function r2Col(outerCat: number): number {
  if (!Number.isInteger(outerCat) || outerCat < 1 || outerCat > OUTER_CATEGORY_NAMES.length) {
    throw new Error(`${outerCat} is not an outer category (1..${OUTER_CATEGORY_NAMES.length})`)
  }
  return outerCat - 1
}
