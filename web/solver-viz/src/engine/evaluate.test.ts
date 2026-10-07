import { describe, expect, it } from 'vitest'
import { parseCard } from './cards'
import { CATEGORY_NAMES, categoryOf, score5 } from './evaluate'

const hand = (s: string) => s.split(' ').map(parseCard)

describe('score5', () => {
  it('orders the nine categories', () => {
    const ladder = [
      'As Kd 9c 7h 3s', // high card
      'As Ad 9c 7h 3s', // pair
      'As Ad 9c 9h 3s', // two pair
      'As Ad Ac 9h 3s', // trips
      '9s 8d 7c 6h 5s', // straight
      'As Ks 9s 7s 3s', // flush
      'As Ad Ac 9h 9s', // full house
      'As Ad Ac Ah 3s', // quads
      '9s 8s 7s 6s 5s', // straight flush
    ]
    const scores = ladder.map((h) => score5(hand(h)))
    for (let i = 1; i < scores.length; i++) expect(scores[i]).toBeGreaterThan(scores[i - 1])
    scores.forEach((s, i) => expect(categoryOf(s)).toBe(i))
  })
  it('a wheel is a five-high straight', () => {
    const wheel = score5(hand('As 2d 3c 4h 5s'))
    expect(categoryOf(wheel)).toBe(4)
    expect(wheel).toBeLessThan(score5(hand('2s 3d 4c 5h 6s')))
    expect(wheel).toBeGreaterThan(score5(hand('As Ad Ac Kh Qs')))
  })
  it('a straight flush is category 8', () => {
    expect(categoryOf(score5(hand('2s 3s 4s 5s 6s')))).toBe(8)
    expect(categoryOf(score5(hand('As 2s 3s 4s 5s')))).toBe(8)
  })
  it('kickers break ties in rank order', () => {
    expect(score5(hand('As Ad Kc 7h 3s'))).toBeGreaterThan(score5(hand('As Ad Qc 7h 3s')))
    expect(score5(hand('As Ad Kc 7h 3s'))).toBeGreaterThan(score5(hand('As Ad Kc 6h 5s')))
    expect(score5(hand('Ks Kd 9c 9h 2s'))).toBeGreaterThan(score5(hand('Ks Kd 8c 8h As')))
  })
  it('the same ranks in other suits tie', () => {
    expect(score5(hand('As Ad Kc 7h 3s'))).toBe(score5(hand('Ah Ac Kd 7s 3d')))
  })
  it('encodes category × 13^5 plus the base-13 tiebreak', () => {
    // quad sevens with a king kicker: category 7, tiebreak (7-rank 5, K-rank 11)
    expect(score5(hand('7s 7d 7c 7h Ks'))).toBe(7 * 13 ** 5 + 5 * 13 ** 4 + 11 * 13 ** 3)
    // 9-high straight: category 4, tiebreak (rank 7)
    expect(score5(hand('9s 8d 7c 6h 5s'))).toBe(4 * 13 ** 5 + 7 * 13 ** 4)
  })
  it('wants exactly five distinct cards', () => {
    expect(() => score5(hand('As Ad Kc 7h'))).toThrow(/5|five/)
    expect(() => score5(hand('As Ad Kc 7h 3s 2s'))).toThrow(/5|five/)
    expect(() => score5(hand('As As Kc 7h 3s'))).toThrow(/twice/)
  })
  it('names the nine categories', () => {
    expect(CATEGORY_NAMES).toHaveLength(9)
    expect(CATEGORY_NAMES[0]).toBe('high card')
    expect(CATEGORY_NAMES[8]).toBe('straight flush')
  })
})
