import { describe, expect, it } from 'vitest'
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

  it('never draws below the chip, and the chip is well under a fifth of the mark', () => {
    expect(sideOf(1e-6, 0.2)).toBe(MARK.min)
    expect(MARK.min / MARK.max).toBeLessThan(0.1)
  })

  it('is the whole mark when nothing in view has weight, rather than a division by zero', () => {
    expect(sideOf(0.1, 0)).toBe(MARK.max)
    expect(Number.isFinite(sideOf(0.1, 0))).toBe(true)
  })
})
