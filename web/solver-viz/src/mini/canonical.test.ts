import { describe, expect, it } from 'vitest'
import { parseMini } from './cards'
import { SUIT_RELABELLINGS, canonicalPicture, drawOrder, keyString, type Picture } from './canonical'
import fixtures from './fixtures.json'

interface FixtureCase {
  hole: number[]
  thrown: number[]
  board: number[]
  key: Picture
  relabelling: [number, number, number]
  drawOrder: number[]
  inner: number[]
  outer: number[] | null
}

const cases = (fixtures as { cases: FixtureCase[] }).cases

const cards = (text: string): number[] => text.split(' ').map(parseMini)
const picture = (hole: string, board = '', thrown = ''): Picture => ({
  hole: cards(hole),
  thrown: thrown ? cards(thrown) : [],
  board: board ? cards(board) : [],
})

describe('the Python fixtures', () => {
  it('ship enough pictures to pin the three shapes', () => {
    expect(cases.length).toBeGreaterThan(0)
  })

  it('canonicalise to the same key and the same relabelling as the Python', () => {
    for (const c of cases) {
      const { key, relabelling } = canonicalPicture({ hole: c.hole, thrown: c.thrown, board: c.board })
      expect(key, JSON.stringify(c)).toEqual(c.key)
      expect(relabelling, JSON.stringify(c)).toEqual(c.relabelling)
    }
  })

  it('order the physical hole for the throws the way game.draw_order does', () => {
    for (const c of cases) {
      expect(drawOrder({ hole: c.hole, thrown: c.thrown, board: c.board }), JSON.stringify(c)).toEqual(c.drawOrder)
    }
  })
})

describe('the suit relabellings', () => {
  it('are the six permutations in itertools.permutations order', () => {
    expect(SUIT_RELABELLINGS).toEqual([
      [0, 1, 2],
      [0, 2, 1],
      [1, 0, 2],
      [1, 2, 0],
      [2, 0, 1],
      [2, 1, 0],
    ])
  })
})

describe('canonicalPicture', () => {
  it('takes the 455 dealt holes down to 95 classes', () => {
    const keys = new Set<string>()
    let dealt = 0
    for (let a = 0; a < 15; a++) {
      for (let b = a + 1; b < 15; b++) {
        for (let c = b + 1; c < 15; c++) {
          dealt++
          keys.add(canonicalPicture({ hole: [a, b, c], thrown: [], board: [] }).key.hole.join(','))
        }
      }
    }
    expect(dealt).toBe(455)
    expect(keys.size).toBe(95)
  })

  it('relabels the hole and the board together, so a shared suit survives', () => {
    const suited = canonicalPicture(picture('2h 3h 5c', '6h')).key
    const suitOf = (c: number) => c % 3
    const heartsInKey = new Set([suitOf(suited.hole[0]), suitOf(suited.hole[1]), suitOf(suited.board[0])])
    expect(heartsInKey.size).toBe(1)

    const offsuit = canonicalPicture(picture('2h 3h 5c', '6c')).key
    const pairSuit = suitOf(offsuit.hole[0])
    expect(suitOf(offsuit.hole[1])).toBe(pairSuit)
    expect(suitOf(offsuit.board[0])).not.toBe(pairSuit)
    expect(keyString(suited)).not.toBe(keyString(offsuit))
  })

  it('maps a picture and its suit-swapped twin to one key', () => {
    const a = canonicalPicture(picture('2d 2h 5c', '3h'))
    const b = canonicalPicture(picture('2h 2d 5c', '3d'))
    expect(keyString(a.key)).toBe(keyString(b.key))
    expect(a.relabelling).not.toEqual(b.relabelling)
  })

  it('sorts the hole and the thrown cards and keeps the board in dealt order', () => {
    const { key } = canonicalPicture(picture('6c 2c 4c', '5c 3c'))
    expect(key.hole).toEqual(cards('2c 4c 6c'))
    expect(key.board).toEqual(cards('5c 3c'))
    const swapped = canonicalPicture(picture('6c 2c 4c', '3c 5c'))
    expect(keyString(swapped.key)).not.toBe(keyString(key))
  })

  it('merges the two board orders only when a relabelling swaps the board and fixes the hole', () => {
    const a = canonicalPicture(picture('2c 4c 6c', '3d 3h'))
    const b = canonicalPicture(picture('2c 4c 6c', '3h 3d'))
    expect(keyString(a.key)).toBe(keyString(b.key))
  })

  it('breaks a tie between relabellings towards the first, so a symmetric picture gets the identity', () => {
    expect(canonicalPicture(picture('2c 2d 2h')).relabelling).toEqual([0, 1, 2])
    expect(canonicalPicture(picture('2c 2d 4h')).relabelling).toEqual([0, 1, 2])
    expect(canonicalPicture(picture('2c 2d 4h', '3h 5h')).relabelling).toEqual([0, 1, 2])
  })

  it('refuses a card outside the deck or a card dealt twice', () => {
    expect(() => canonicalPicture({ hole: [0, 1, 15], thrown: [], board: [] })).toThrow(/15/)
    expect(() => canonicalPicture({ hole: [0, 1, 2], thrown: [], board: [1] })).toThrow(/twice/)
    expect(() => canonicalPicture({ hole: [0, 1, 2], thrown: [0], board: [] })).toThrow(/twice/)
  })
})

describe('drawOrder', () => {
  it('orders equal ranks by canonical suit, not physical, so the board-suited card sits in the same position either way', () => {
    // Two deuces and a five against a board trey: the deuce sharing the
    // board's suit is the one the canonical labels put first.
    expect(drawOrder(picture('2d 2h 5c', '3h'))).toEqual(cards('2h 2d 5c'))
    expect(drawOrder(picture('2d 2h 5c', '3d'))).toEqual(cards('2d 2h 5c'))
    expect(drawOrder(picture('2h 2d 5c', '3d'))).toEqual(cards('2d 2h 5c'))
  })

  it('returns the physical cards, sorted by rank first', () => {
    const order = drawOrder(picture('6h 2d 4c', '3c'))
    expect(order).toEqual(cards('2d 4c 6h'))
  })
})

describe('keyString', () => {
  it('separates the three groups so two keys collide only card for card', () => {
    expect(keyString({ hole: [0, 4, 8], thrown: [], board: [7] })).toBe('0,4,8/|7')
    expect(keyString({ hole: [0, 4, 8], thrown: [7], board: [] })).toBe('0,4,8/7|')
    expect(keyString({ hole: [0, 4], thrown: [], board: [8, 7] })).not.toBe(
      keyString({ hole: [0, 4, 8], thrown: [], board: [7] }),
    )
  })
})
