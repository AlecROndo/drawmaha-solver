import { describe, expect, it } from 'vitest'
import { parseMini } from './cards'
import { INNER_CATEGORY_NAMES, OUTER_CATEGORY_NAMES, outerCategory, outerScore } from './ranking'
import { MINI_INNER_ROWS, MINI_R1_COLS, MINI_R2_COLS, r1Col, r2Col } from './regions'

const cards = (text: string): number[] => text.split(' ').map(parseMini)
const col = (hole: string, board1: string): number => r1Col(cards(hole), parseMini(board1))

describe('the mini axes', () => {
  it('have six inner rows in InnerCategory order', () => {
    expect(MINI_INNER_ROWS.map((r) => r.label)).toEqual(INNER_CATEGORY_NAMES)
    expect(MINI_INNER_ROWS[0].long).toBe('three cards, no pair, no run, no suit')
  })

  it('have five round-1 columns naming what the hole reaches for', () => {
    expect(MINI_R1_COLS.map((r) => r.label)).toEqual([
      'nothing',
      'two to the straight',
      'two to the flush',
      'pairs the board',
      'trips with the board',
    ])
  })

  it('have seven round-2 columns in OuterCategory order', () => {
    expect(MINI_R2_COLS.map((r) => r.label)).toEqual(OUTER_CATEGORY_NAMES)
  })

  it('give every region a label and a longer reading', () => {
    for (const r of [...MINI_INNER_ROWS, ...MINI_R1_COLS, ...MINI_R2_COLS]) {
      expect(r.label.length).toBeGreaterThan(0)
      expect(r.long.length).toBeGreaterThan(r.label.length)
    }
  })
})

describe('r1Col', () => {
  it('reads trips with the board above a pair with the board', () => {
    expect(col('4c 4d 2h', '4h')).toBe(4)
    expect(col('4c 2d 6h', '4h')).toBe(3)
  })

  it('reads a pair with the board above a flush draw or a straight draw', () => {
    expect(col('4c 5c 2h', '4h')).toBe(3)
    expect(col('4h 5h 2c', '4c')).toBe(3)
  })

  it('reads two to the flush when two hole cards wear the board suit, above a straight draw', () => {
    expect(col('2h 5h 3c', '6h')).toBe(2)
    expect(col('2h 3h 6c', '4h')).toBe(2)
    expect(col('2h 5h 3c', '6c')).toBe(1)
  })

  it('reads two to the straight when two hole ranks and the board rank are distinct within a span of three', () => {
    expect(col('2c 4d 6h', '3h')).toBe(1) // 2 3 4
    expect(col('2c 5d 6h', '3h')).toBe(1) // 2 3 5 spans 3
    expect(col('2c 4d 6h', '5h')).toBe(1) // 4 5 6
    expect(col('2c 2d 5h', '4h')).toBe(1) // 2 4 5 spans 3; the paired deuces do not count twice
  })

  it('reads nothing when no two hole cards reach anything with the board', () => {
    expect(col('2c 2d 2h', '6c')).toBe(0)
    expect(col('2c 2d 6h', '6c')).toBe(3)
    expect(col('2c 2d 3h', '6c')).toBe(0) // 2 3 6 spans 4
    expect(col('2c 2d 6h', '4h')).toBe(0) // 2 4 6 spans 4; 2 2 4 is not three distinct ranks
    expect(col('2c 6d 6h', '3c')).toBe(0) // 2 3 6 spans 4; 6 6 3 is not three distinct ranks
  })

  it('refuses a hole that is not three cards or a card outside the deck', () => {
    expect(() => r1Col([0, 1], 5)).toThrow(/2/)
    expect(() => r1Col([0, 1, 2], 15)).toThrow(/15/)
  })
})

describe('r2Col', () => {
  it('is the outer category’s index, so an outer score lands in its own column', () => {
    for (let cat = 1; cat <= 7; cat++) expect(r2Col(cat)).toBe(cat - 1)
    expect(MINI_R2_COLS[r2Col(outerCategory(outerScore(cards('2c 4d 6h'), cards('3h 5c'))))].label).toBe('straight')
  })

  it('refuses a category outside 1..7', () => {
    expect(() => r2Col(0)).toThrow(/0/)
    expect(() => r2Col(8)).toThrow(/8/)
  })
})
