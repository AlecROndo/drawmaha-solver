import { describe, expect, it } from 'vitest'
import { type GridView, gridView } from '../engine/aggregate'
import { parseBoard } from '../engine/cards'
import { maskFor } from '../engine/filter'
import { policyFor } from '../engine/policy'
import { sampleRange } from '../engine/sample'
import { tableOf } from '../engine/table'
import { MARK, sideOf } from './Grid'

describe('sideOf: the square of a region, sized against the largest region in view', () => {
  it('gives the largest region the whole mark', () => {
    expect(sideOf(0.2, 0.2)).toBe(MARK.max)
    expect(sideOf(0.3, 0.2)).toBe(MARK.max)
  })

  it('gives nothing the minimum chip', () => {
    expect(sideOf(0, 0.2)).toBe(MARK.min)
    expect(sideOf(-1, 0.2)).toBe(MARK.min)
    expect(sideOf(Number.NaN, 0.2)).toBe(MARK.min)
  })

  it('scales by area: a quarter of the share is half the side', () => {
    expect(sideOf(0.05, 0.2)).toBeCloseTo(MARK.max / 2, 10)
    expect(sideOf(0.05, 0.2, 100, 1)).toBeCloseTo(50, 10)
  })

  it('never draws below the chip, and the chip is under a tenth of the mark', () => {
    expect(sideOf(1e-6, 0.2)).toBe(MARK.min)
    expect(MARK.min / MARK.max).toBeLessThan(0.1)
  })

  it('is the whole mark when nothing in view has weight, rather than a division by zero', () => {
    expect(sideOf(0.1, 0)).toBe(MARK.max)
    expect(Number.isFinite(sideOf(0.1, 0))).toBe(true)
  })
})

describe('the mark is sized against the view, so a filter is a zoom', () => {
  // The same maximum Grid takes: over the cells gridView returns under the mask.
  const maxOf = (v: GridView) => Math.max(0, ...v.cells.flat().map((c) => c.share))

  it('masking off the largest region drops it to share 0 and hands the whole mark to the largest survivor', () => {
    const s = sampleRange(parseBoard('Ks 9s 4d'), 2000, 11)
    const t = tableOf(s, policyFor(s, 'facing'))
    const whole = gridView(t, null)

    // The row the largest region sits in, and every other row that has anything in it.
    let big = { r: 0, c: 0, share: -1 }
    whole.cells.forEach((row, r) => row.forEach((cell, c) => { if (cell.share > big.share) big = { r, c, share: cell.share } }))
    const kept = new Set(whole.rows.map((row, r) => (row.n > 0 && r !== big.r ? r : -1)).filter((r) => r >= 0))
    expect(kept.size).toBeGreaterThan(0)

    const view = gridView(t, maskFor(s, { rows: kept, cols: null, held: [] }))
    // The masked row is empty in the view: nothing there can set the maximum.
    for (const cell of view.cells[big.r]) {
      expect(cell.n).toBe(0)
      expect(cell.share).toBe(0)
    }
    // A survivor's share is unchanged (it is still a share of the whole sample), so the
    // view's maximum is the largest kept region of the unfiltered grid ...
    const keptMax = Math.max(...whole.cells.flatMap((row, r) => (kept.has(r) ? row.map((c) => c.share) : [])))
    expect(maxOf(view)).toBeCloseTo(keptMax, 12)
    expect(maxOf(view)).toBeLessThan(maxOf(whole))
    // ... and that region, less than the whole mark before, fills it now.
    expect(sideOf(keptMax, maxOf(whole))).toBeLessThan(MARK.max)
    expect(sideOf(keptMax, maxOf(view))).toBe(MARK.max)
  })
})
