import { describe, expect, it } from 'vitest'

import {
  DECK,
  drive,
  mistakeOf,
  nextHand,
  playAct,
  prescribed,
  sampleAction,
  settle,
  strength,
  type Hand,
} from './play'
import { word, type Act, type Strategy } from './leduc'

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
    rows: [],
    pendingRoll: null,
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

describe('the roll — a mixed strategy made playable', () => {
  // Aggression descending: bet/raise owns the LOW rolls, then check/call,
  // then fold. bet 50% / check 50%: rolls 0-50 bet, 50-100 check.
  const half: Partial<Record<Act, number>> = { c: 0.5, r: 0.5 }

  it('prescribes the action whose segment holds the roll', () => {
    expect(prescribed(half, 40)).toBe('r')
    expect(prescribed(half, 60)).toBe('c')
  })

  it('grades a deviation by its distance into the right region', () => {
    // Supposed to bet on anything under 50; checking on a 40 is 10 points deep.
    expect(mistakeOf(half, 40, 'c')).toBe(10)
    expect(mistakeOf(half, 40, 'r')).toBe(0)
    // Betting on a 70 overshoots the bet region by 20.
    expect(mistakeOf(half, 70, 'r')).toBe(20)
  })

  it('an action the strategy never takes costs the full distance to its empty segment', () => {
    const never: Partial<Record<Act, number>> = { c: 1, r: 0 }
    expect(mistakeOf(never, 40, 'r')).toBe(40)
  })

  it('an absent spot prescribes the always-legal check/call', () => {
    expect(prescribed({}, 40)).toBe('c')
  })

  it('three actions stack fold at the top of the rolls', () => {
    const mix: Partial<Record<Act, number>> = { f: 0.2, c: 0.3, r: 0.5 }
    expect(prescribed(mix, 10)).toBe('r')
    expect(prescribed(mix, 60)).toBe('c')
    expect(prescribed(mix, 90)).toBe('f')
    expect(mistakeOf(mix, 90, 'r')).toBeCloseTo(40, 10) // 90 is 40 past bet's end at 50
  })
})

describe('the transcript rows', () => {
  const fresh0: Hand = {
    cards: [
      { rank: 'K', suit: 0 },
      { rank: 'J', suit: 0 },
    ],
    boardCard: { rank: 'Q', suit: 0 },
    humanSeat: 0,
    l1: '',
    board: null,
    l2: '',
    lines: [],
    rows: [],
    pendingRoll: null,
    result: null,
  }

  const passive: Strategy = {
    'J:': { c: 1 },
    'J:c': { c: 1 },
    'J:cc|Q:c': { c: 1 },
    'K:': { c: 1 },
    'K:cc|Q:': { c: 0.5, r: 0.5 },
  }

  it('draws a roll for the live human decision and grades the action against it', () => {
    const dealt = drive(fresh0, passive, () => 0.4)
    expect(dealt.pendingRoll).toBe(40)

    const s1 = playAct({ hand: dealt, chips: 0, hands: 0 }, 'c', passive, () => 0.4)
    const humanRow = s1.hand.rows[0]
    expect(humanRow.kind).toBe('action')
    if (humanRow.kind === 'action') {
      expect(humanRow.human).toBe(true)
      expect(humanRow.roll).toBe(40)
      expect(humanRow.act).toBe('c')
      // K: checks 100% here, so any roll prescribes the check — no mistake.
      expect(humanRow.mistake).toBe(0)
    }
  })

  it('records bot and board rows without rolls, in order', () => {
    const dealt = drive(fresh0, passive, () => 0.4)
    const s1 = playAct({ hand: dealt, chips: 0, hands: 0 }, 'c', passive, () => 0.4)
    const kinds = s1.hand.rows.map((r) => (r.kind === 'board' ? 'board' : r.human ? 'you' : 'bot'))
    expect(kinds).toEqual(['you', 'bot', 'board'])
    const bot = s1.hand.rows[1]
    if (bot.kind === 'action') expect(bot.roll).toBeNull()
  })

  it('rows carry the round line the actor faced, so a call names itself a call', () => {
    // The sidebar re-words every action from row.line; a hardcoded '' would
    // read "check" at a facing-a-bet spot. Bot bets after the human checks,
    // the human calls: that row must carry 'cr', which words `c` as a call.
    const aggro: Strategy = { 'J:c': { r: 1 }, 'K:cr': { f: 0.5, c: 0.5 } }
    const dealt = drive(fresh0, aggro, () => 0.4)
    const s1 = playAct({ hand: dealt, chips: 0, hands: 0 }, 'c', aggro, () => 0.4)
    const s2 = playAct(s1, 'c', aggro, () => 0.4)
    const call = s2.hand.rows[2]
    expect(call.kind).toBe('action')
    if (call.kind === 'action') {
      expect(call.human).toBe(true)
      expect(call.line).toBe('cr')
      expect(word(call.act, call.line)).toBe('call')
      // roll 40 sits inside call's 0-50 segment, so the prescription IS the call
      expect(word(call.correct as Act, call.line)).toBe('call')
    }
  })

  it('a half-and-half round-2 spot grades a deliberate deviation', () => {
    const dealt = drive(fresh0, passive, () => 0.4)
    const s1 = playAct({ hand: dealt, chips: 0, hands: 0 }, 'c', passive, () => 0.4)
    // Round 2, human holds K at 'K:cc|Q:' = bet 50 / check 50, roll 40 → bet.
    const s2 = playAct(s1, 'c', passive, () => 0.4)
    const row = s2.hand.rows.find((r) => r.kind === 'action' && r.human && r.mistake !== 0)
    expect(row && row.kind === 'action' && row.mistake).toBe(10)
  })
})
