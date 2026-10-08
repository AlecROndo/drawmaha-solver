/**
 * The composite axes of the grid: nine inner rows by eight outer columns.
 *
 * Terms: the *inner* hand is the holding (five hole cards) as a poker hand,
 * the *outer* hand the best two-hole-plus-three-board Omaha hand
 * (`classify.ts`). A *region* is one row or column — a made-hand class split,
 * below two pair, by its draw — and a cell is a row crossed with a column.
 * Rows and columns are ordered weakest first so the grid reads up and right
 * towards strength. The *illustrative policy* is aggregated over these
 * regions in `aggregate.ts`.
 */

export interface Region {
  label: string
  long: string
}

export const INNER_ROWS: readonly Region[] = [
  { label: 'nothing', long: 'high card, no draw' },
  { label: 'nothing · 4-straight', long: 'four to a straight' },
  { label: 'nothing · 4-flush', long: 'four to a flush' },
  { label: 'pair', long: 'one pair, no draw' },
  { label: 'pair · 4-straight', long: 'pair with four to a straight' },
  { label: 'pair · 4-flush', long: 'pair with four to a flush' },
  { label: 'two pair', long: 'two pair' },
  { label: 'trips', long: 'three of a kind' },
  { label: 'straight or better', long: 'a made five-card hand' },
]

export const OUTER_COLS: readonly Region[] = [
  { label: 'nothing', long: 'no pair, no draw' },
  { label: 'nothing · str. draw', long: 'straight draw only' },
  { label: 'nothing · flush draw', long: 'flush draw, with or without a straight draw' },
  { label: 'pair', long: 'a pair with the board' },
  { label: 'pair · str. draw', long: 'pair with a straight draw' },
  { label: 'pair · flush draw', long: 'pair with a flush draw' },
  { label: 'two pair', long: 'two pair, any draw' },
  { label: 'trips or better', long: 'a set, trips or better, any draw' },
]

function assertCategory(cat: number): void {
  if (!Number.isInteger(cat) || cat < 0 || cat > 8) throw new Error(`${cat} is not a hand category (0..8)`)
}

/**
 * The inner row of a hand from its category and draw flags. Below two pair a
 * four-flush outranks a four-straight when both are set.
 */
export function innerRow(cat: number, fourFlush: boolean, fourStraight: number): number {
  assertCategory(cat)
  if (cat >= 4) return 8
  if (cat === 3) return 7
  if (cat === 2) return 6
  const base = cat === 1 ? 3 : 0
  return base + (fourFlush ? 2 : fourStraight > 0 ? 1 : 0)
}

/**
 * The outer column of a hand from its category and draw flags. Below two
 * pair a flush draw outranks a straight draw when both are set.
 */
export function outerCol(cat: number, flushDraw: boolean, straightDraw: number): number {
  assertCategory(cat)
  if (cat >= 3) return 7
  if (cat === 2) return 6
  const base = cat === 1 ? 3 : 0
  return base + (flushDraw ? 2 : straightDraw > 0 ? 1 : 0)
}
