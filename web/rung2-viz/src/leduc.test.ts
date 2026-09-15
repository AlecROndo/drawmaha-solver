import { describe, expect, it } from 'vitest'

import { closed, keyFor, legal, potAfter, reach, word, type Strategy } from './leduc'

describe('legal / closed', () => {
  it('opens with check or bet, faces a bet with fold/call/raise, caps at two raises', () => {
    expect(legal('')).toEqual(['c', 'r'])
    expect(legal('r')).toEqual(['f', 'c', 'r'])
    expect(legal('rr')).toEqual(['f', 'c'])
  })

  it('a round closes on check-check, call, or fold — not on a lone check or open bet', () => {
    expect(closed('')).toBe(false)
    expect(closed('c')).toBe(false)
    expect(closed('r')).toBe(false)
    expect(closed('cc')).toBe(true)
    expect(closed('rc')).toBe(true)
    expect(closed('rf')).toBe(true)
    expect(closed('crc')).toBe(true)
  })
})

describe('word', () => {
  it('names the same symbol by what it faces', () => {
    expect(word('c', '')).toBe('check')
    expect(word('c', 'r')).toBe('call')
    expect(word('r', '')).toBe('bet')
    expect(word('r', 'cr')).toBe('raise')
    expect(word('f', 'r')).toBe('fold')
  })
})

describe('keyFor matches str(InfoSet) on the Python side', () => {
  it('round 1: rank, colon, line', () => {
    expect(keyFor('K', null, '', '')).toBe('K:')
    expect(keyFor('J', null, 'cr', '')).toBe('J:cr')
  })

  it('round 2: pipe, board rank, colon, line', () => {
    expect(keyFor('J', 'Q', 'cc', 'r')).toBe('J:cc|Q:r')
  })
})

describe('potAfter', () => {
  it('antes only until chips move', () => {
    expect(potAfter('', null, '')).toBe(2)
    expect(potAfter('cc', null, '')).toBe(2)
  })

  it('a called bet is two stacks of it, a raise war escalates', () => {
    expect(potAfter('rc', null, '')).toBe(6) // 2 antes + 2 + 2
    expect(potAfter('rrc', null, '')).toBe(10) // 2 + 4 + 4
  })

  it('an uncalled bet is ONE seat’s chips — the fold puts nothing in', () => {
    expect(potAfter('rf', null, '')).toBe(4) // 2 antes + bettor’s 2 only
  })

  it('round 2 bets are size 4 on top of the round-1 pot', () => {
    expect(potAfter('cc', 'Q', 'rc')).toBe(10) // 2 + (4 + 4)
  })
})

describe('reach', () => {
  // A strategy where every rank checks 90% / bets 10% at the root.
  const flat: Strategy = {
    'J:': { c: 0.9, r: 0.1 },
    'Q:': { c: 0.9, r: 0.1 },
    'K:': { c: 0.9, r: 0.1 },
  }

  it('the empty line is always reached', () => {
    expect(reach('', flat)).toBeCloseTo(1, 12)
  })

  it('one uniform step multiplies by that probability', () => {
    expect(reach('c', flat)).toBeCloseTo(0.9, 10)
    expect(reach('r', flat)).toBeCloseTo(0.1, 10)
  })

  it('weights ranks by the deck, not by a flat third', () => {
    // Only the king ever bets: betting frequency at the root is 1/3 regardless
    // of weighting (the acting seat is dealt each rank uniformly)…
    const kingOnly: Strategy = {
      'J:': { c: 1, r: 0 },
      'Q:': { c: 1, r: 0 },
      'K:': { c: 0, r: 1 },
      'J:r': { f: 1, c: 0 },
      'Q:r': { f: 1, c: 0 },
      'K:r': { f: 0, c: 1 },
    }
    expect(reach('r', kingOnly)).toBeCloseTo(1 / 3, 10)
    // …but the caller behind that bet is NOT a uniform deck: the bettor holds
    // a king, so the second seat holds K with probability 1/5, not 1/3.
    expect(reach('rc', kingOnly)).toBeCloseTo((1 / 3) * (1 / 5), 10)
  })
})
