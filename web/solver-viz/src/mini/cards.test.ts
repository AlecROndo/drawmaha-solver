import { describe, expect, it } from 'vitest'
import { MINI_DECK, MINI_SUIT_GLYPH, miniGlyph, miniLabel, parseMini, rankOf, suitOf } from './cards'

describe('the 15-card deck', () => {
  it('indexes rank × 3 + suit and runs 0..14', () => {
    expect(MINI_DECK).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    expect(rankOf(7)).toBe(2)
    expect(suitOf(7)).toBe(1)
    expect(MINI_SUIT_GLYPH).toEqual(['♣', '♦', '♥'])
  })

  it('spells 2c … 6h the way the Python suite does, and draws a glyph for a face', () => {
    expect(miniLabel(0)).toBe('2c')
    expect(miniLabel(14)).toBe('6h')
    expect(miniLabel(7)).toBe('4d')
    expect(miniGlyph(7)).toBe('4♦')
    expect(miniGlyph(14)).toBe('6♥')
  })

  it('reads a label or a glyph back to the same card', () => {
    expect(parseMini('4h')).toBe(8)
    expect(parseMini('4♥')).toBe(8)
    expect(parseMini('2c')).toBe(0)
    expect(parseMini('6H')).toBe(14)
    expect(parseMini(' 3d ')).toBe(4)
    for (const c of MINI_DECK) {
      expect(parseMini(miniLabel(c))).toBe(c)
      expect(parseMini(miniGlyph(c))).toBe(c)
    }
  })

  it('names the token it cannot read instead of inventing a card', () => {
    expect(() => parseMini('7h')).toThrow(/7h/)
    expect(() => parseMini('4s')).toThrow(/4s/)
    expect(() => parseMini('Ah')).toThrow(/Ah/)
    expect(() => parseMini('4hh')).toThrow(/4hh/)
    expect(() => parseMini('')).toThrow()
  })

  it('refuses a card outside the deck', () => {
    expect(() => miniLabel(15)).toThrow(/15/)
    expect(() => miniLabel(-1)).toThrow(/-1/)
    expect(() => miniGlyph(2.5)).toThrow(/2.5/)
  })
})
