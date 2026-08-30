import { describe, expect, test } from 'vitest'
import { mulberry32, sampleIndex } from './rng'

describe('sampleIndex', () => {
  test('degenerate distributions always return the certain action', () => {
    const rng = mulberry32(5)
    for (let i = 0; i < 100; i++) {
      expect(sampleIndex([1, 0, 0], rng)).toBe(0)
      expect(sampleIndex([0, 0, 1], rng)).toBe(2)
    }
  })

  test('samples match the distribution frequencies', () => {
    const rng = mulberry32(11)
    const counts = [0, 0, 0]
    for (let i = 0; i < 30000; i++) counts[sampleIndex([0.5, 0.25, 0.25], rng)]++
    expect(counts[0] / 30000).toBeCloseTo(0.5, 1)
    expect(counts[1] / 30000).toBeCloseTo(0.25, 1)
    expect(counts[2] / 30000).toBeCloseTo(0.25, 1)
  })
})

describe('mulberry32', () => {
  test('same seed produces the identical sequence', () => {
    const a = mulberry32(42)
    const b = mulberry32(42)
    for (let i = 0; i < 1000; i++) expect(a()).toBe(b())
  })

  test('different seeds diverge', () => {
    const a = mulberry32(1)
    const b = mulberry32(2)
    const seqA = Array.from({ length: 10 }, () => a())
    const seqB = Array.from({ length: 10 }, () => b())
    expect(seqA).not.toEqual(seqB)
  })

  test('outputs stay in [0, 1) and are roughly uniform', () => {
    const rng = mulberry32(7)
    let sum = 0
    const n = 10000
    for (let i = 0; i < n; i++) {
      const x = rng()
      expect(x).toBeGreaterThanOrEqual(0)
      expect(x).toBeLessThan(1)
      sum += x
    }
    expect(sum / n).toBeCloseTo(0.5, 1)
  })
})
