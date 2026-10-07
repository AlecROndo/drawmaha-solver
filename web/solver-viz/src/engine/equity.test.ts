import { describe, expect, it } from 'vitest'
import { parseBoard } from './cards'
import { equityAgainst, weightsForBet } from './equity'
import { maskFor } from './filter'
import { policyFor } from './policy'
import { sampleRange } from './sample'

const FLOP = parseBoard('Ks 9s 4d')
const s = sampleRange(FLOP, 4000, 21)
const pol = policyFor(s, 'facing')

describe('weightsForBet', () => {
  it('is the pot probability, copied', () => {
    const w = weightsForBet(pol)
    expect(w).toEqual(pol.p)
    w[0] = 0.123
    expect(pol.p[0]).not.toBe(0.123)
  })
})

describe('equityAgainst', () => {
  const all = new Uint8Array(s.n).fill(1)
  const uniform = new Float32Array(s.n).fill(1)

  it('a range against itself with uniform weights is a coin flip on both halves', () => {
    const e = equityAgainst(s, all, s, uniform)
    expect(e.inner).toBeCloseTo(0.5, 2)
    expect(e.outer).toBeCloseTo(0.5, 2)
    expect(e.total).toBeCloseTo(0.5, 2)
    expect(e.scoop + e.scooped).toBeLessThanOrEqual(1)
    expect(e.scoop).toBeGreaterThan(0)
  })
  it('the made row beats the whole range on the inner half', () => {
    const made = maskFor(s, { rows: new Set([8]), cols: null, held: [] })!
    const e = equityAgainst(s, made, s, uniform)
    expect(e.inner).toBeGreaterThan(0.9)
  })
  it('weights matter: against a betting range the whole range does worse than a coin flip', () => {
    const e = equityAgainst(s, all, s, weightsForBet(pol))
    expect(e.total).toBeLessThan(0.5)
  })
  it('works against a second seed of the same board', () => {
    const other = sampleRange(FLOP, 3000, 22)
    const e = equityAgainst(s, all, other, new Float32Array(other.n).fill(1))
    expect(e.inner).toBeCloseTo(0.5, 1)
    expect(e.total).toBe((e.inner + e.outer) / 2)
  })
  it('refuses weights of the wrong length, zero weights and an empty mask', () => {
    expect(() => equityAgainst(s, all, s, new Float32Array(10).fill(1))).toThrow(/10/)
    expect(() => equityAgainst(s, all, s, new Float32Array(s.n))).toThrow(/weight/)
    expect(() => equityAgainst(s, new Uint8Array(s.n), s, uniform)).toThrow(/mask|hand/)
  })
})
