import { describe, expect, it } from 'vitest'

import { DECK, drive, nextHand, playAct, sampleAction, settle, strength, type Hand } from './play'
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

describe('the hand loop — drive, playAct, nextHand', () => {
  /** Human is P0 with the king, bot P1 with the jack, queen waiting to turn. */
  const fresh = (humanSeat: 0 | 1 = 0): Hand => ({
    cards: [
      { rank: 'K', suit: 0 },
      { rank: 'J', suit: 0 },
    ],
    boardCard: { rank: 'Q', suit: 0 },
    humanSeat,
    l1: '',
    board: null,
    l2: '',
    lines: [],
    result: null,
  })

  /** The bot checks and calls everything, at every key this suite reaches —
   * including 'K:', which the bot opens from when the human sits in seat 1;
   * a missing key falls back to a RANDOM uniform draw and a coin-flip test. */
  const passive: Strategy = {
    'J:': { c: 1 },
    'J:c': { c: 1 },
    'J:cc|Q:c': { c: 1 },
    'K:': { c: 1 },
  }

  it('stops at the human without acting for them', () => {
    const h = drive(fresh(0), passive)
    expect(h.l1).toBe('')
    expect(h.result).toBeNull()
    expect(h.lines).toEqual([])
  })

  it('lets the bot open when it has the first seat', () => {
    const h = drive(fresh(1), passive)
    expect(h.l1).toBe('c')
    expect(h.lines).toEqual(['the solver checks'])
    expect(h.result).toBeNull()
  })

  it('plays a checked-down hand end to end: board turns, showdown, bank', () => {
    const s0 = { hand: fresh(0), chips: 0, hands: 0 }
    const s1 = playAct(s0, 'c', passive)
    expect(s1.hand.l1).toBe('cc')
    expect(s1.hand.board).toBe('Q') // round 1 closed, so drive turned the board
    expect(s1.hand.lines).toEqual(['you check', 'the solver checks', 'the board turns Q'])
    expect(s1.hand.result).toBeNull() // round 2 opens on the human

    const s2 = playAct(s1, 'c', passive)
    expect(s2.hand.l2).toBe('cc')
    expect(s2.hand.result).toBe(1) // king over jack for the antes
    expect(s2.hand.lines.at(-1)).toBe('you held K · the solver held J — you win 1')
    expect(s2.chips).toBe(1)
    expect(s2.hands).toBe(1)
  })

  it('a fold ends the hand before any board and banks the loss', () => {
    const aggro: Strategy = { 'J:c': { r: 1 } }
    const s1 = playAct({ hand: fresh(0), chips: 0, hands: 0 }, 'c', aggro)
    expect(s1.hand.lines).toEqual(['you check', 'the solver bets'])
    const s2 = playAct(s1, 'f', aggro)
    expect(s2.hand.board).toBeNull()
    expect(s2.hand.result).toBe(-1) // the ante, exactly what settle('crf') says
    expect(s2.chips).toBe(-1)
    expect(s2.hands).toBe(1)
  })

  it('leaves a finished hand untouched', () => {
    const done = { hand: { ...fresh(0), result: 1 }, chips: 1, hands: 1 }
    expect(playAct(done, 'c', passive)).toBe(done)
  })

  it('nextHand swaps the seats and carries the bankroll', () => {
    const s = nextHand({ hand: fresh(0), chips: 3, hands: 2 }, passive, () => 0)
    expect(s.hand.humanSeat).toBe(1)
    expect(s.chips).toBe(3)
    expect(s.hands).toBe(2)
    expect(s.hand.result).toBeNull() // a hand can never end without the human
  })
})
