import { describe, expect, it } from 'vitest'
import { parseMini } from './cards'
import fixtures from './fixtures.json'
import {
  INNER_CATEGORY_NAMES,
  OUTER_CATEGORY_NAMES,
  decodeInner,
  decodeOuter,
  innerCategory,
  innerScore,
  outerCategory,
  outerScore,
} from './ranking'

interface FixtureCase {
  hole: number[]
  thrown: number[]
  board: number[]
  inner: number[]
  outer: number[] | null
}

const cases = (fixtures as { cases: FixtureCase[] }).cases

const cards = (text: string): number[] => text.split(' ').map(parseMini)
const inner = (hole: string): number => innerScore(cards(hole))
const outer = (hole: string, board: string): number => outerScore(cards(hole), cards(board))

const INNER = { HIGH_CARD: 1, PAIR: 2, STRAIGHT: 3, FLUSH: 4, STRAIGHT_FLUSH: 5, TRIPS: 6 }
const OUTER = { HIGH_CARD: 1, PAIR: 2, STRAIGHT: 3, TWO_PAIR: 4, TRIPS: 5, FLUSH: 6, STRAIGHT_FLUSH: 7 }

describe('the Python fixtures', () => {
  it('score the inner half as hands.inner_score does', () => {
    for (const c of cases) {
      expect(decodeInner(innerScore(c.hole)), JSON.stringify(c)).toEqual(c.inner)
    }
  })

  it('score the outer half as hands.outer_score does, and only on a two-card board', () => {
    for (const c of cases) {
      if (c.outer === null) {
        expect(c.board.length, JSON.stringify(c)).toBe(1)
        expect(() => outerScore(c.hole, c.board)).toThrow()
      } else {
        expect(decodeOuter(outerScore(c.hole, c.board)), JSON.stringify(c)).toEqual(c.outer)
      }
    }
  })
})

describe('the inner ranking', () => {
  it('names six categories in InnerCategory order', () => {
    expect(INNER_CATEGORY_NAMES).toEqual(['high card', 'pair', 'straight', 'flush', 'straight flush', 'trips'])
  })

  it('classifies every category', () => {
    expect(innerCategory(inner('2c 4d 6h'))).toBe(INNER.HIGH_CARD)
    expect(innerCategory(inner('2c 2d 6h'))).toBe(INNER.PAIR)
    expect(innerCategory(inner('2c 3d 4h'))).toBe(INNER.STRAIGHT)
    expect(innerCategory(inner('2c 3c 5c'))).toBe(INNER.FLUSH)
    expect(innerCategory(inner('4c 5c 6c'))).toBe(INNER.STRAIGHT_FLUSH)
    expect(innerCategory(inner('4c 4d 4h'))).toBe(INNER.TRIPS)
  })

  it('puts a straight above a pair, a flush above a straight, and trips at the top', () => {
    expect(inner('2c 3d 4h')).toBeGreaterThan(inner('6c 6d 5h'))
    expect(inner('2c 3c 5c')).toBeGreaterThan(inner('4c 5d 6h'))
    expect(inner('2c 3c 4c')).toBeGreaterThan(inner('3c 4c 6c'))
    expect(inner('2c 2d 2h')).toBeGreaterThan(inner('4c 5c 6c'))
  })

  it('never calls trips a flush: three of a rank is one card per suit', () => {
    expect(decodeInner(inner('5c 5d 5h'))).toEqual([INNER.TRIPS, 3, 3, 3])
  })

  it('does not wrap: 6 2 3 is not a run', () => {
    expect(innerCategory(inner('6c 2d 3h'))).toBe(INNER.HIGH_CARD)
    expect(innerCategory(inner('6c 2c 3c'))).toBe(INNER.FLUSH)
  })

  it('orders the ranks by count then rank, so the pair beats the kicker', () => {
    expect(decodeInner(inner('2c 2d 6h'))).toEqual([INNER.PAIR, 0, 0, 4])
    expect(decodeInner(inner('6c 2d 2h'))).toEqual([INNER.PAIR, 0, 0, 4])
    expect(inner('3c 3d 2h')).toBeGreaterThan(inner('2c 2d 6h'))
    expect(inner('2c 2d 6h')).toBeGreaterThan(inner('2c 2d 5h'))
  })

  it('ties exactly when nothing but the suits differ', () => {
    expect(inner('2c 3d 5h')).toBe(inner('2d 3h 5c'))
    expect(inner('4c 4d 6h')).toBe(inner('4d 4h 6c'))
  })

  it('encodes category × 10 000 plus the ranks as a base-5 numeral, most significant first', () => {
    expect(inner('6c 6d 6h')).toBe(6 * 10_000 + 4 * 25 + 4 * 5 + 4)
    expect(inner('2c 2d 6h')).toBe(2 * 10_000 + 4) // ranks 0 0 4: the two leading digits are zero
    expect(inner('2c 4d 6h')).toBe(1 * 10_000 + 4 * 25 + 2 * 5 + 0)
  })

  it('refuses anything but three cards', () => {
    expect(() => innerScore(cards('2c 3d'))).toThrow(/2/)
    expect(() => innerScore(cards('2c 3d 4h 5c'))).toThrow(/4/)
    expect(() => innerScore([0, 1, 15])).toThrow(/15/)
  })
})

describe('the outer ranking', () => {
  it('names seven categories in OuterCategory order', () => {
    expect(OUTER_CATEGORY_NAMES).toEqual([
      'high card',
      'pair',
      'straight',
      'two pair',
      'trips',
      'flush',
      'straight flush',
    ])
  })

  it('classifies every category over the best pairing of two hole cards with both board cards', () => {
    expect(outerCategory(outer('2c 4d 6h', '3h 5c'))).toBe(OUTER.STRAIGHT)
    expect(outerCategory(outer('2c 2d 6h', '4h 5c'))).toBe(OUTER.PAIR)
    expect(outerCategory(outer('2c 5h 6c', '2d 5c'))).toBe(OUTER.TWO_PAIR)
    expect(outerCategory(outer('4c 4d 2h', '4h 6c'))).toBe(OUTER.TRIPS)
    expect(outerCategory(outer('3c 4d 5h', '2c 6d'))).toBe(OUTER.HIGH_CARD) // every pairing leaves out a middle rank
    expect(outerCategory(outer('2c 4c 6h', '5c 3c'))).toBe(OUTER.STRAIGHT_FLUSH)
    expect(outerCategory(outer('2c 4c 6h', '6c 3c'))).toBe(OUTER.FLUSH)
  })

  it('ranks a straight below two pair below trips, as this deck’s rarity has it', () => {
    expect(outer('2c 4d 6h', '3h 5c')).toBeLessThan(outer('2c 5h 6c', '2d 5c'))
    expect(outer('2c 5h 6c', '2d 5c')).toBeLessThan(outer('4c 4d 2h', '4h 6c'))
    expect(outer('4c 4d 2h', '4h 6c')).toBeLessThan(outer('2c 4c 6h', '6c 3c'))
    expect(outer('2c 4c 6h', '6c 3c')).toBeLessThan(outer('2c 4c 6h', '5c 3c'))
  })

  it('does not wrap: 6 2 with 3 4 on the board is not a run', () => {
    // The pairing 6-2 reads 4 0 1 2; if the top wrapped it would be a straight
    // and beat the pair of sixes the other pairing makes.
    expect(outerCategory(outer('6c 2d 6d', '3h 4c'))).toBe(OUTER.PAIR)
    expect(outerCategory(outer('6c 2d 2h', '3h 4c'))).toBe(OUTER.PAIR)
  })

  it('uses exactly two hole cards, so dealt trips are worth a pair outside', () => {
    expect(outerCategory(outer('4c 4d 4h', '2c 6d'))).toBe(OUTER.PAIR)
    expect(decodeOuter(outer('4c 4d 4h', '2c 6d'))).toEqual([OUTER.PAIR, 2, 2, 4, 0])
  })

  it('orders the ranks by count then rank, so trips beat their kicker and two pair reads high pair first', () => {
    expect(decodeOuter(outer('3c 3d 2h', '3h 6c'))).toEqual([OUTER.TRIPS, 1, 1, 1, 4])
    expect(decodeOuter(outer('3c 3d 6h', '3h 2c'))).toEqual([OUTER.TRIPS, 1, 1, 1, 0])
    expect(decodeOuter(outer('2c 5h 6c', '2d 5c'))).toEqual([OUTER.TWO_PAIR, 3, 3, 0, 0])
    expect(outer('3c 3d 2h', '3h 6c')).toBeGreaterThan(outer('3c 3d 6h', '3h 2c'))
  })

  it('ties exactly when nothing but the suits differ', () => {
    expect(outer('2c 4d 6h', '3h 5c')).toBe(outer('2d 4h 6c', '3c 5d'))
  })

  it('refuses a one-card board or a hole that is not three cards', () => {
    expect(() => outerScore(cards('2c 4d 6h'), cards('3h'))).toThrow(/1/)
    expect(() => outerScore(cards('2c 4d'), cards('3h 5c'))).toThrow(/2/)
  })
})

describe('the decoders', () => {
  it('invert the encoding for every inner and outer score', () => {
    expect(decodeInner(1 * 10_000 + 4 * 25 + 2 * 5 + 0)).toEqual([1, 4, 2, 0])
    expect(decodeOuter(5 * 10_000 + 1 * 125 + 1 * 25 + 1 * 5 + 4)).toEqual([5, 1, 1, 1, 4])
  })
})
