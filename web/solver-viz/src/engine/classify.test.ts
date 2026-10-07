import { describe, expect, it } from 'vitest'
import { parseBoard, parseCard } from './cards'
import { classifyInner, classifyOuter } from './classify'

const hand = (s: string) => s.split(' ').map(parseCard)
const FLOP = parseBoard('Ks 9s 4d')

describe('classifyInner', () => {
  it('a pair with four spades is a pair with a four-flush', () => {
    const r = classifyInner(hand('As Ad Qs 7s 3s'))
    expect(r.cat).toBe(1)
    expect(r.fourFlush).toBe(true)
    expect(r.fourStraight).toBe(0)
  })
  it('9 8 7 6 with a blank is an open-ended four-straight', () => {
    expect(classifyInner(hand('9h 8c 7d 6s 2c')).fourStraight).toBe(2)
  })
  it('9 8 7 5 with a blank is a gutshot', () => {
    expect(classifyInner(hand('9h 8c 7d 5s 2c')).fourStraight).toBe(1)
  })
  it('A 2 3 4 with a blank is a gutshot to the wheel', () => {
    expect(classifyInner(hand('Ah 2c 3d 4s Kc')).fourStraight).toBe(1)
  })
  it('2 3 4 5 with a blank is open-ended (ace or six)', () => {
    expect(classifyInner(hand('2h 3c 4d 5s Kc')).fourStraight).toBe(2)
  })
  it('a pair can carry a four-straight', () => {
    const r = classifyInner(hand('9h 9c 8d 7s 6c'))
    expect(r.cat).toBe(1)
    expect(r.fourStraight).toBe(2)
  })
  it('a made straight or flush carries no draw flag', () => {
    const straight = classifyInner(hand('9h 8c 7d 6s 5c'))
    expect(straight.cat).toBe(4)
    expect(straight.fourStraight).toBe(0)
    expect(straight.fourFlush).toBe(false)
    const flush = classifyInner(hand('As Ks 9s 7s 3s'))
    expect(flush.cat).toBe(5)
    expect(flush.fourFlush).toBe(false)
  })
  it('a blank hand has no draw', () => {
    const r = classifyInner(hand('Ah 9c 7d 4s 2c'))
    expect(r.cat).toBe(0)
    expect(r.fourFlush).toBe(false)
    expect(r.fourStraight).toBe(0)
  })
  it('wants five cards', () => {
    expect(() => classifyInner(hand('Ah 9c 7d 4s'))).toThrow()
  })
})

describe('classifyOuter on K♠ 9♠ 4♦', () => {
  it('aces with four spades make a pair with a flush draw', () => {
    const r = classifyOuter(hand('As Ad Qs 7s 3s'), FLOP)
    expect(r.cat).toBe(1)
    expect(r.flushDraw).toBe(true)
  })
  it('two kings make trips', () => {
    const r = classifyOuter(hand('Kh Kc 7d 3c 2h'), FLOP)
    expect(r.cat).toBe(3)
    expect(r.flushDraw).toBe(false)
    expect(r.straightDraw).toBe(0)
  })
  it('Q J T wraps the board: a ten, a jack or a queen each complete a straight', () => {
    const r = classifyOuter(hand('Qh Jc Td 3c 2h'), FLOP)
    expect(r.cat).toBe(0)
    expect(r.straightDraw).toBe(3)
  })
  it('Q J alone is a gutshot (only a ten)', () => {
    const r = classifyOuter(hand('Qh Jc 7d 3c 2h'), FLOP)
    expect(r.straightDraw).toBe(1)
  })
  it('a single spade in hand is no flush draw', () => {
    expect(classifyOuter(hand('As 8d 7c 3c 2h'), FLOP).flushDraw).toBe(false)
  })
  it('two hearts in hand are no flush draw on a spade-spade-diamond board', () => {
    expect(classifyOuter(hand('Ah 8h 7c 3c 2d'), FLOP).flushDraw).toBe(false)
  })
  it('a blank hand has no pair and no draw', () => {
    const r = classifyOuter(hand('Ah 8d 7c 3c 2h'), FLOP)
    expect(r.cat).toBe(0)
    expect(r.flushDraw).toBe(false)
    expect(r.straightDraw).toBe(0)
  })
  it('picks the best two of the five hole cards', () => {
    // 9 9 would be trips; A A is only a pair of aces; trips wins
    const r = classifyOuter(hand('9h 9c Ad Ac 2h'), FLOP)
    expect(r.cat).toBe(3)
  })
  it('a made straight carries no straight draw', () => {
    // Q J T with K and 9 on board: hole Q J + board K 9 ... needs T from the board; use J T hole: K Q J T 9 impossible,
    // so use a board that makes one: hole Q J on K T 9
    const r = classifyOuter(hand('Qh Jc 7d 3c 2h'), parseBoard('Ks Ts 9d'))
    expect(r.cat).toBe(4)
    expect(r.straightDraw).toBe(0)
  })
  it('a made flush carries no flush draw', () => {
    const r = classifyOuter(hand('As 7s 3c 2h 8d'), parseBoard('Ks 9s 4s'))
    expect(r.cat).toBe(5)
    expect(r.flushDraw).toBe(false)
  })
  it('reads a four- or five-card board by its best three', () => {
    expect(classifyOuter(hand('Kh Kc 7d 3c 2h'), parseCards('Ks 9s 4d 2c')).cat).toBe(3)
    expect(classifyOuter(hand('9d Ah 7c 3c 2h'), parseCards('Ks 9s 4d 9c 9h')).cat).toBe(7)
  })
  it('wants a board of three to five cards', () => {
    expect(() => classifyOuter(hand('Kh Kc 7d 3c 2h'), parseCards('Ks 9s'))).toThrow()
    expect(() => classifyOuter(hand('Kh Kc 7d 3c 2h'), parseCards('Ks 9s 4d 2c 3c 5c'))).toThrow()
  })
  it('refuses a hole card that is also on the board', () => {
    expect(() => classifyOuter(hand('Ks Kc 7d 3c 2h'), FLOP)).toThrow(/twice/)
  })
})

function parseCards(s: string): number[] {
  return s.split(' ').map(parseCard)
}
