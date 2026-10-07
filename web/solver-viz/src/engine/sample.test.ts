import { describe, expect, it } from 'vitest'
import { parseBoard, parseCard } from './cards'
import { classifyInner, classifyOuter } from './classify'
import { innerRow, outerCol } from './regions'
import { gridView } from './aggregate'
import { equitiesOfHand, policyFor } from './policy'
import { CATEGORY_BASE, score5 } from './evaluate'
import { MAX_N, exactRange, mulberry32, percentileRanks, sampleRange } from './sample'

const FLOP = parseBoard('Ks 9s 4d')

describe('MAX_N', () => {
  it('keeps score × n + index an exact double: every score is below 9 × 13⁵ < 2²², so the key stays under 2⁴⁶', () => {
    const scoreBound = 9 * CATEGORY_BASE
    expect(scoreBound).toBeLessThan(2 ** 22)
    // the highest hand there is: a royal flush
    expect(score5('As Ks Qs Js Ts'.split(' ').map(parseCard))).toBeLessThan(scoreBound)
    const largestKey = (scoreBound - 1) * MAX_N + (MAX_N - 1)
    expect(largestKey).toBeLessThan(2 ** 46)
    expect(Number.isSafeInteger(largestKey)).toBe(true)
    expect(Number.isSafeInteger(largestKey + 1)).toBe(true)
  })
})

describe('mulberry32', () => {
  it('is deterministic and lands in [0, 1)', () => {
    const a = mulberry32(7)
    const b = mulberry32(7)
    for (let i = 0; i < 100; i++) {
      const x = a()
      expect(x).toBe(b())
      expect(x).toBeGreaterThanOrEqual(0)
      expect(x).toBeLessThan(1)
    }
    expect(mulberry32(8)()).not.toBe(mulberry32(7)())
  })
})

describe('percentileRanks', () => {
  it('is the fraction beaten, ties counted at half', () => {
    const p = percentileRanks(Uint32Array.from([10, 20, 20, 30]))
    expect(Array.from(p)).toEqual([0.125, 0.5, 0.5, 0.875])
  })
})

describe('sampleRange', () => {
  const s = sampleRange(FLOP, 3000, 7)

  it('is deterministic in the seed', () => {
    const t = sampleRange(FLOP, 3000, 7)
    expect(t.cards).toEqual(s.cards)
    expect(t.innerEquity).toEqual(s.innerEquity)
    const u = sampleRange(FLOP, 3000, 8)
    expect(u.cards).not.toEqual(s.cards)
  })
  it('carries its board, size and seed', () => {
    expect(s.n).toBe(3000)
    expect(s.board).toEqual(FLOP)
    expect(s.seed).toBe(7)
    expect(s.cards).toHaveLength(15000)
  })
  it('never deals a board card', () => {
    for (let i = 0; i < s.cards.length; i++) expect(FLOP).not.toContain(s.cards[i])
  })
  it('deals five distinct cards per hand', () => {
    for (let i = 0; i < s.n; i++) {
      const five = new Set(s.cards.subarray(i * 5, i * 5 + 5))
      expect(five.size).toBe(5)
    }
  })
  it('classifies each hand the way classifyInner and classifyOuter do', () => {
    for (let i = 0; i < 200; i++) {
      const hole = Array.from(s.cards.subarray(i * 5, i * 5 + 5))
      const inner = classifyInner(hole)
      const outer = classifyOuter(hole, FLOP)
      expect(s.innerCat[i]).toBe(inner.cat)
      expect(s.innerScore[i]).toBe(inner.score)
      expect(s.innerFourFlush[i]).toBe(inner.fourFlush ? 1 : 0)
      expect(s.innerFourStraight[i]).toBe(inner.fourStraight)
      expect(s.outerCat[i]).toBe(outer.cat)
      expect(s.outerScore[i]).toBe(outer.score)
      expect(s.outerFlushDraw[i]).toBe(outer.flushDraw ? 1 : 0)
      expect(s.outerStraightDraw[i]).toBe(outer.straightDraw)
      expect(s.row[i]).toBe(innerRow(inner.cat, inner.fourFlush, inner.fourStraight))
      expect(s.col[i]).toBe(outerCol(outer.cat, outer.flushDraw, outer.straightDraw))
    }
  })
  it('equities lie in [0, 1], average one half, and rise with the score', () => {
    let sumI = 0
    let sumO = 0
    for (let i = 0; i < s.n; i++) {
      expect(s.innerEquity[i]).toBeGreaterThanOrEqual(0)
      expect(s.innerEquity[i]).toBeLessThanOrEqual(1)
      sumI += s.innerEquity[i]
      sumO += s.outerEquity[i]
    }
    expect(sumI / s.n).toBeCloseTo(0.5, 3)
    expect(sumO / s.n).toBeCloseTo(0.5, 3)
    const order = Array.from({ length: s.n }, (_, i) => i).sort((a, b) => s.innerScore[a] - s.innerScore[b])
    for (let k = 1; k < order.length; k++) {
      const [a, b] = [order[k - 1], order[k]]
      if (s.innerScore[a] === s.innerScore[b]) expect(s.innerEquity[a]).toBe(s.innerEquity[b])
      else expect(s.innerEquity[b]).toBeGreaterThan(s.innerEquity[a])
    }
  })
  it('refuses a bad board or a bad size', () => {
    expect(() => sampleRange([44, 44, 10], 10, 1)).toThrow(/twice/)
    expect(() => sampleRange(FLOP, 0, 1)).toThrow(/0/)
    expect(() => sampleRange(FLOP, 2.5, 1)).toThrow(/2\.5/)
  })
  it('samples 60,000 holdings in well under two seconds', () => {
    const t0 = performance.now()
    sampleRange(FLOP, 60000, 1)
    expect(performance.now() - t0).toBeLessThan(2000)
  })
})

describe('exactRange', () => {
  const ref = sampleRange(FLOP, 3000, 7)
  const AS = parseCard('As')
  const AD = parseCard('Ad')
  const two = exactRange(FLOP, [AS, AD], ref)

  it('enumerates every holding with the held cards: C(47, 3) = 16,215 for two', () => {
    expect(two.n).toBe(16215)
    expect(two.cards).toHaveLength(16215 * 5)
    expect(two.board).toEqual(FLOP)
    expect(two.seed).toBe(ref.seed)
  })
  it('every hand holds the held cards, five distinct, none from the board', () => {
    const seen = new Set<string>()
    for (let i = 0; i < two.n; i++) {
      const five = Array.from(two.cards.subarray(i * 5, i * 5 + 5))
      expect(five).toContain(AS)
      expect(five).toContain(AD)
      expect(new Set(five).size).toBe(5)
      for (const c of FLOP) expect(five).not.toContain(c)
      seen.add([...five].sort((a, b) => a - b).join(','))
    }
    expect(seen.size).toBe(16215)
  })
  it('is the whole of its own range: gridView share 1 and n = 16,215', () => {
    const g = gridView(two, policyFor(two, 'facing'), null)
    expect(g.all.share).toBe(1)
    expect(g.all.n).toBe(16215)
  })
  it('five held cards give the one hand, classified like classifyInner and classifyOuter', () => {
    const hole = 'As Ad Qs 7s 3s'.split(' ').map(parseCard)
    const one = exactRange(FLOP, hole, ref)
    expect(one.n).toBe(1)
    const inner = classifyInner(hole)
    const outer = classifyOuter(hole, FLOP)
    expect(one.innerScore[0]).toBe(inner.score)
    expect(one.outerScore[0]).toBe(outer.score)
    expect(one.row[0]).toBe(innerRow(inner.cat, inner.fourFlush, inner.fourStraight))
    expect(one.col[0]).toBe(outerCol(outer.cat, outer.flushDraw, outer.straightDraw))
  })
  it('measures equity against the reference sample, not within itself', () => {
    for (const i of [0, 777, 16214]) {
      const hole = Array.from(two.cards.subarray(i * 5, i * 5 + 5))
      const e = equitiesOfHand(hole, FLOP, ref)
      expect(two.innerEquity[i]).toBeCloseTo(e.eqI, 6)
      expect(two.outerEquity[i]).toBeCloseTo(e.eqO, 6)
    }
    let mean = 0
    for (let i = 0; i < two.n; i++) mean += two.innerEquity[i]
    // Hands holding two aces are well above the middle of the full range.
    expect(mean / two.n).toBeGreaterThan(0.6)
  })
  it('counts C(48, 4) = 194,580 holdings for one held card, in well under a second', () => {
    const t0 = performance.now()
    const one = exactRange(FLOP, [AS], ref)
    const dt = performance.now() - t0
    expect(one.n).toBe(194580)
    expect(dt).toBeLessThan(1000)
  })
  it('refuses no held card, a board card, a repeated card, or another board', () => {
    expect(() => exactRange(FLOP, [], ref)).toThrow(/held/)
    expect(() => exactRange(FLOP, [parseCard('Ks')], ref)).toThrow(/K♠/)
    expect(() => exactRange(FLOP, [AS, AS], ref)).toThrow(/twice/)
    expect(() => exactRange(parseBoard('Kh 9s 4d'), [AS], ref)).toThrow(/board/)
    expect(() => exactRange(FLOP, [AS, AD, parseCard('2c'), parseCard('3c'), parseCard('4c'), parseCard('5c')], ref)).toThrow(/6/)
  })
})
