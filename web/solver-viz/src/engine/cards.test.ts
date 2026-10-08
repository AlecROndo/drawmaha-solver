import { describe, expect, it } from 'vitest'
import { cardLabel, isRed, parseBoard, parseCard, rankOf, suitOf, unseen } from './cards'

describe('parseCard', () => {
  it('reads a rank letter and a suit letter', () => {
    expect(parseCard('Ks')).toBe(44)
    expect(parseCard('2s')).toBe(0)
    expect(parseCard('Ac')).toBe(51)
    expect(parseCard('Kc')).toBe(47)
  })
  it('reads a suit glyph', () => {
    expect(parseCard('K♠')).toBe(44)
    expect(parseCard('4♦')).toBe(10)
  })
  it('is case-insensitive', () => {
    expect(parseCard('ks')).toBe(44)
    expect(parseCard('kS')).toBe(44)
  })
  it('spells ten as T, t or 10', () => {
    expect(parseCard('Ts')).toBe(32)
    expect(parseCard('ts')).toBe(32)
    expect(parseCard('10s')).toBe(32)
  })
  it('names the token it cannot read', () => {
    expect(() => parseCard('Kx')).toThrow(/Kx/)
    expect(() => parseCard('1s')).toThrow(/1s/)
    expect(() => parseCard('')).toThrow()
    expect(() => parseCard('Kss')).toThrow(/Kss/)
  })
})

describe('parseBoard', () => {
  it('reads three space-separated cards', () => {
    expect(parseBoard('Ks 9s 4d')).toEqual([44, 28, 10])
  })
  it('rejects a card dealt twice', () => {
    expect(() => parseBoard('Ks Ks 4d')).toThrow(/twice/)
  })
  it('names a card it cannot read', () => {
    expect(() => parseBoard('Kx 9s 4d')).toThrow(/Kx/)
  })
  it('wants exactly three cards', () => {
    expect(() => parseBoard('Ks 9s')).toThrow(/3/)
    expect(() => parseBoard('Ks 9s 4d 2c')).toThrow(/3/)
  })
})

describe('cards', () => {
  it('splits a card into rank and suit', () => {
    expect(rankOf(44)).toBe(11)
    expect(suitOf(44)).toBe(0)
    expect(suitOf(47)).toBe(3)
  })
  it('labels a card with its glyph', () => {
    expect(cardLabel(44)).toBe('K♠')
    expect(cardLabel(47)).toBe('K♣')
    expect(cardLabel(32)).toBe('T♠')
    expect(cardLabel(parseCard('4d'))).toBe('4♦')
  })
  it('hearts and diamonds are red', () => {
    expect(isRed(parseCard('4d'))).toBe(true)
    expect(isRed(parseCard('4h'))).toBe(true)
    expect(isRed(parseCard('4s'))).toBe(false)
    expect(isRed(parseCard('4c'))).toBe(false)
  })
  it('the unseen deck is everything but the board', () => {
    const board = parseBoard('Ks 9s 4d')
    const rest = unseen(board)
    expect(rest).toHaveLength(49)
    for (const c of board) expect(rest).not.toContain(c)
    expect(new Set(rest).size).toBe(49)
  })
  it('refuses a board with a duplicate', () => {
    expect(() => unseen([44, 44, 10])).toThrow(/twice/)
  })
})
