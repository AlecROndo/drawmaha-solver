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

describe('ignoring card removal between the seats', () => {
  // Brute force, per hand of `mine`: the weighted total equity over every hand
  // of `theirs`, and over only the hands that share no card with it. The
  // module note's "the error is small" is this difference. 2,000 × 2,000
  // pairs × 25 card comparisons is ~100M per weighting, about 80 ms each on
  // a laptop; the module note's numbers were taken at 3,000 a side.
  const mine = sampleRange(FLOP, 2000, 21)
  const theirs = sampleRange(FLOP, 2000, 22)
  const theirPolicy = policyFor(theirs, 'open')
  const sharesACard = (i: number, j: number): boolean => {
    for (let x = 0; x < 5; x++) for (let y = 0; y < 5; y++) if (mine.cards[i * 5 + x] === theirs.cards[j * 5 + y]) return true
    return false
  }
  const beat = (a: number, b: number): number => (a > b ? 1 : a === b ? 0.5 : 0)
  const totals = (w: Float32Array): { all: Float64Array; disjoint: Float64Array } => {
    const all = new Float64Array(mine.n)
    const disjoint = new Float64Array(mine.n)
    for (let i = 0; i < mine.n; i++) {
      let wA = 0
      let wD = 0
      let winA = 0
      let winD = 0
      for (let j = 0; j < theirs.n; j++) {
        const win = w[j] * (beat(mine.innerScore[i], theirs.innerScore[j]) + beat(mine.outerScore[i], theirs.outerScore[j])) / 2
        wA += w[j]
        winA += win
        if (!sharesACard(i, j)) {
          wD += w[j]
          winD += win
        }
      }
      all[i] = winA / wA
      disjoint[i] = winD / wD
    }
    return { all, disjoint }
  }
  // Fails loudly on an empty row (a NaN mean is silent), so the seed is pinned by name.
  const mean = (v: Float64Array, keep: (i: number) => boolean): number => {
    let sum = 0
    let m = 0
    for (let i = 0; i < v.length; i++) if (keep(i)) {
      sum += v[i]
      m++
    }
    expect(m, 'the row must hold enough hands for a mean to say anything').toBeGreaterThan(20)
    return sum / m
  }

  for (const [name, weights] of [
    ['their betting range', theirPolicy.p],
    ['their checking range', theirPolicy.c],
    ['their whole range', new Float32Array(theirs.n).fill(1)],
  ] as const) {
    it(`moves the whole range's total by under 1 point and each inner row's by under 2 against ${name}`, () => {
      const { all, disjoint } = totals(weights)
      const everything = () => true
      expect(Math.abs(mean(all, everything) - mean(disjoint, everything))).toBeLessThan(0.01)
      expect(mean(all, everything)).toBeCloseTo(equityAgainst(mine, null, theirs, weights).total, 6)
      for (let cat = 0; cat <= 3; cat++) {
        const inRow = (i: number) => mine.innerCat[i] === cat
        expect(Math.abs(mean(all, inRow) - mean(disjoint, inRow))).toBeLessThan(0.02)
      }
    })
  }
})
