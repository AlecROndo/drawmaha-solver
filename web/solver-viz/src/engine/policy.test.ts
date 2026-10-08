import { describe, expect, it } from 'vitest'
import { parseBoard, parseCard } from './cards'
import { mixOfHand, policyFor, throwCounts, throwOfHand } from './policy'
import { sampleRange } from './sample'

const FLOP = parseBoard('Ks 9s 4d')
const hand = (s: string) => s.split(' ').map(parseCard)
const s = sampleRange(FLOP, 4000, 3)

describe('policyFor (illustrative)', () => {
  it('facing a bet, fold + call + pot = 1 for every hand', () => {
    const pol = policyFor(s, 'facing')
    for (let i = 0; i < s.n; i++) {
      expect(pol.f[i] + pol.c[i] + pol.p[i]).toBeCloseTo(1, 5)
      expect(pol.f[i]).toBeGreaterThanOrEqual(0)
      expect(pol.c[i]).toBeGreaterThanOrEqual(0)
      expect(pol.p[i]).toBeGreaterThanOrEqual(0)
      expect(pol.eq[i]).toBeGreaterThanOrEqual(0)
      expect(pol.eq[i]).toBeLessThanOrEqual(1)
    }
  })
  it('first to act, nobody folds and check + pot = 1', () => {
    const pol = policyFor(s, 'open')
    for (let i = 0; i < s.n; i++) {
      expect(pol.f[i]).toBe(0)
      expect(pol.c[i] + pol.p[i]).toBeCloseTo(1, 5)
    }
  })
  it('pots more with the strongest hands than the weakest', () => {
    const pol = policyFor(s, 'facing')
    let best = 0
    let worst = 0
    for (let i = 0; i < s.n; i++) {
      if (pol.eq[i] > pol.eq[best]) best = i
      if (pol.eq[i] < pol.eq[worst]) worst = i
    }
    expect(pol.p[best]).toBeGreaterThan(pol.p[worst])
    expect(pol.f[worst]).toBeGreaterThan(pol.f[best])
  })
})

describe('mixOfHand', () => {
  it('agrees with policyFor for a hand that is in the sample', () => {
    const pol = policyFor(s, 'facing')
    for (const i of [0, 17, 1234]) {
      const hole = Array.from(s.cards.subarray(i * 5, i * 5 + 5))
      const m = mixOfHand(hole, FLOP, s, 'facing')
      expect(m.f).toBeCloseTo(pol.f[i], 5)
      expect(m.c).toBeCloseTo(pol.c[i], 5)
      expect(m.p).toBeCloseTo(pol.p[i], 5)
      expect(m.eq).toBeCloseTo(pol.eq[i], 5)
    }
  })
  it('gives a mix to a hand that is not in the sample', () => {
    const m = mixOfHand(hand('As Ad Qs 7s 3s'), FLOP, s, 'facing')
    expect(m.f + m.c + m.p).toBeCloseTo(1, 5)
    const o = mixOfHand(hand('As Ad Qs 7s 3s'), FLOP, s, 'open')
    expect(o.f).toBe(0)
  })
  it('refuses a board other than the sample’s', () => {
    expect(() => mixOfHand(hand('As Ad Qs 7s 3s'), parseBoard('Kh 9s 4d'), s, 'facing')).toThrow(/board/)
  })
})

describe('throwCounts (illustrative)', () => {
  const t = throwCounts(s)
  it('gives every hand a distribution over 0..5 cards thrown', () => {
    expect(t).toHaveLength(s.n * 6)
    for (let i = 0; i < s.n; i++) {
      let sum = 0
      for (let k = 0; k < 6; k++) sum += t[i * 6 + k]
      expect(sum).toBeCloseTo(1, 5)
    }
  })
  it('a made straight or better stands pat', () => {
    for (let i = 0; i < s.n; i++) if (s.innerCat[i] >= 4) expect(t[i * 6]).toBe(1)
  })
  it('a four-flush throws exactly one', () => {
    for (let i = 0; i < s.n; i++) if (s.innerCat[i] < 2 && s.innerFourFlush[i]) expect(t[i * 6 + 1]).toBe(1)
  })
})

describe('throwOfHand', () => {
  it('a pair throws three 60 % of the time and two otherwise', () => {
    const r = throwOfHand(hand('Ah Ad 9c 7d 2h'), FLOP, s)
    expect(r.counts).toHaveLength(6)
    expect(r.counts[3]).toBeCloseTo(0.6)
    expect(r.counts[2]).toBeCloseTo(0.4)
  })
  it('a pair with a four-flush throws its off-suit card, even an ace of the pair', () => {
    const r = throwOfHand(hand('As Ad Qs 7s 3s'), FLOP, s)
    expect(r.counts[1]).toBe(1)
    expect(r.perCard).toEqual([0, 1, 0, 0, 0])
  })
  it('a pair with no draw throws its three kickers 60 % and its two lowest 40 %', () => {
    const r = throwOfHand(hand('Kh Kc 9d 6s 5s'), FLOP, s)
    expect(r.perCard[0]).toBe(0)
    expect(r.perCard[1]).toBe(0)
    expect(r.perCard[2]).toBeCloseTo(0.6, 9)
    expect(r.perCard[3]).toBeCloseTo(1, 9)
    expect(r.perCard[4]).toBeCloseTo(1, 9)
  })
  it('an open four-straight throws the card outside it', () => {
    const r = throwOfHand(hand('9h 8c 7d 6s 2c'), FLOP, s)
    expect(r.perCard).toEqual([0, 0, 0, 0, 1])
    // with a pair, the second card of the paired rank is the odd one
    const paired = throwOfHand(hand('9h 9c 8d 7s 6c'), FLOP, s)
    expect(paired.perCard).toEqual([0, 1, 0, 0, 0])
  })
  it('two pair and trips break for the card outside the groups', () => {
    const r = throwOfHand(hand('Ah Ad 9c 9d 2h'), FLOP, s)
    expect(r.perCard).toEqual([0, 0, 0, 0, 0.55])
    const t = throwOfHand(hand('Ah Ad Ac 9d 2h'), FLOP, s)
    expect(t.perCard).toEqual([0, 0, 0, 0, 0.55])
  })
  it('nothing with an outer flush draw throws two off-suit cards and keeps the spades', () => {
    const r = throwOfHand(hand('As 7s Qd Jc 2h'), FLOP, s)
    expect(r.counts[2]).toBe(1)
    expect(r.perCard[0]).toBe(0)
    expect(r.perCard[1]).toBe(0)
    expect(r.perCard.reduce((a, b) => a + b, 0)).toBeCloseTo(2, 9)
  })
  it('nothing without a draw throws the three outside the best outer hand, or the four lowest', () => {
    // best outer two on K 9 4: A and J (A K J 9 4 high); 8 7 2 go at k = 3, 8 7 2 J at k = 4
    const r = throwOfHand(hand('Ah Jc 8d 7c 2h'), FLOP, s)
    expect(r.counts[3]).toBe(0.5)
    expect(r.counts[4]).toBe(0.5)
    expect(r.perCard).toEqual([0, 0.5, 1, 1, 1])
  })
  it('a made straight throws nothing', () => {
    const r = throwOfHand(hand('9h 8c 7d 6s 5c'), FLOP, s)
    expect(r.counts[0]).toBe(1)
    expect(r.perCard).toEqual([0, 0, 0, 0, 0])
  })
  it('per-card throw probabilities sum to the expected number thrown for 200 sampled hands', () => {
    for (let i = 0; i < 200; i++) {
      const hole = Array.from(s.cards.subarray(i * 5, i * 5 + 5))
      const r = throwOfHand(hole, FLOP, s)
      let expected = 0
      for (let k = 0; k < 6; k++) expected += k * r.counts[k]
      expect(r.perCard).toHaveLength(5)
      expect(Math.abs(r.perCard.reduce((a, b) => a + b, 0) - expected)).toBeLessThan(1e-9)
      for (const p of r.perCard) {
        expect(p).toBeGreaterThanOrEqual(0)
        expect(p).toBeLessThanOrEqual(1)
      }
    }
  })
})
