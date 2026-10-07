import { describe, expect, it } from 'vitest'
import { INNER_ROWS, OUTER_COLS, innerRow, outerCol } from './regions'

describe('regions', () => {
  it('has nine inner rows and eight outer columns', () => {
    expect(INNER_ROWS).toHaveLength(9)
    expect(OUTER_COLS).toHaveLength(8)
    expect(INNER_ROWS.map((r) => r.label)).toEqual([
      'nothing', 'nothing · 4-straight', 'nothing · 4-flush',
      'pair', 'pair · 4-straight', 'pair · 4-flush',
      'two pair', 'trips', 'straight or better',
    ])
    expect(OUTER_COLS.map((r) => r.label)).toEqual([
      'nothing', 'nothing · str. draw', 'nothing · flush draw',
      'pair', 'pair · str. draw', 'pair · flush draw',
      'two pair', 'trips or better',
    ])
  })
  it('made hands land on the last three rows', () => {
    expect(innerRow(4, false, 0)).toBe(8)
    expect(innerRow(8, false, 0)).toBe(8)
    expect(innerRow(3, false, 0)).toBe(7)
    expect(innerRow(2, false, 0)).toBe(6)
  })
  it('a pair splits by its draw', () => {
    expect(innerRow(1, false, 0)).toBe(3)
    expect(innerRow(1, false, 1)).toBe(4)
    expect(innerRow(1, false, 2)).toBe(4)
    expect(innerRow(1, true, 0)).toBe(5)
  })
  it('nothing splits by its draw, and a four-flush beats a four-straight', () => {
    expect(innerRow(0, false, 0)).toBe(0)
    expect(innerRow(0, false, 2)).toBe(1)
    expect(innerRow(0, true, 0)).toBe(2)
    expect(innerRow(0, true, 2)).toBe(2)
    expect(innerRow(1, true, 2)).toBe(5)
  })
  it('outer: trips or better, two pair, then pair and nothing split by draw', () => {
    expect(outerCol(7, false, 0)).toBe(7)
    expect(outerCol(3, true, 3)).toBe(7)
    expect(outerCol(2, true, 0)).toBe(6)
    expect(outerCol(1, false, 0)).toBe(3)
    expect(outerCol(1, false, 2)).toBe(4)
    expect(outerCol(1, true, 2)).toBe(5)
    expect(outerCol(0, false, 0)).toBe(0)
    expect(outerCol(0, false, 1)).toBe(1)
    expect(outerCol(0, true, 3)).toBe(2)
  })
  it('refuses a category outside 0..8', () => {
    expect(() => innerRow(9, false, 0)).toThrow(/9/)
    expect(() => outerCol(-1, false, 0)).toThrow(/-1/)
  })
})
