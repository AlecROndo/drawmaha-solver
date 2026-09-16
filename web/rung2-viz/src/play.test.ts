import { describe, expect, it } from 'vitest'

import { DECK, sampleAction, settle, strength } from './play'
import type { Strategy } from './leduc'

describe('the deck', () => {
  it('is six cards, two per rank', () => {
    expect(DECK).toHaveLength(6)
    for (const rank of ['J', 'Q', 'K'] as const) {
      expect(DECK.filter((c) => c.rank === rank)).toHaveLength(2)
    }
  })
})

describe('strength', () => {
  it('pairing the board beats every unpaired card', () => {
    expect(strength('J', 'J')).toBeGreaterThan(strength('K', 'J'))
    expect(strength('Q', 'K')).toBeGreaterThan(strength('J', 'K'))
    expect(strength('K', 'Q')).toBeGreaterThan(strength('Q', 'K'))
  })
})

describe('settle — chips to P0 at a finished hand', () => {
  it('a fold costs the folder what they had put in, ante included', () => {
    // P1 folds to the open bet: loses the ante alone.
    expect(settle('rf', null, '', 'J', 'K')).toBe(1)
    // P0 folds to a re-raise: ante + the raise already made — the referee’s ±3.
    expect(settle('rrf', null, '', 'K', 'J')).toBe(-3)
  })

  it('a checked-down hand is a showdown for the antes', () => {
    expect(settle('cc', 'Q', 'cc', 'K', 'J')).toBe(1)
    expect(settle('cc', 'Q', 'cc', 'J', 'K')).toBe(-1)
  })

  it('equal unpaired ranks split: nobody wins chips', () => {
    expect(settle('cc', 'Q', 'cc', 'K', 'K')).toBe(0)
  })

  it('pairing the board wins the whole line', () => {
    // Jack pairs the jack board and beats the king.
    expect(settle('cc', 'J', 'rc', 'J', 'K')).toBe(5) // 1 ante + 4 round-2 bet
  })

  it('a called raise war pays every level', () => {
    // r1: 2+2 each on 'rrc' is 4 each; r2 'rc' is 4 each; ante 1 → 9 total.
    expect(settle('rrc', 'Q', 'rc', 'K', 'J')).toBe(9)
  })

  it('a round-2 fold still pays the round-1 chips', () => {
    // P1 called round 1 (2), folded round 2: loses 1 + 2.
    expect(settle('rc', 'Q', 'rf', 'J', 'K')).toBe(3)
  })
})

describe('sampleAction', () => {
  const strategy: Strategy = { 'K:': { c: 0.25, r: 0.75 } }

  it('walks the cumulative distribution with the supplied uniform draw', () => {
    expect(sampleAction(strategy, 'K:', ['c', 'r'], () => 0.1)).toBe('c')
    expect(sampleAction(strategy, 'K:', ['c', 'r'], () => 0.9)).toBe('r')
  })

  it('falls back to uniform over legal actions at an unknown spot', () => {
    expect(sampleAction(strategy, 'J:zz', ['f', 'c'], () => 0.6)).toBe('c')
  })
})
