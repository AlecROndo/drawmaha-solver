import { describe, expect, it } from 'vitest'

import { sym } from './minidraw'
import {
  addDeals,
  dealHand,
  freshSession,
  holeOf,
  mistakeOf,
  nextHand,
  phaseOf,
  playAct,
  prescribed,
  replacementOf,
  runningLow,
  spotOf,
  toMix,
  verdictOf,
  type Deal,
  type Showdown,
} from './play'

/** A card by its symbol, for readable fixtures. */
const c = (s: string): number => '23456'.indexOf(s[0]) * 3 + 'cdh'.indexOf(s[1])
const cards = (s: string): number[] => s.trim().split(/\s+/).map(c)

/**
 * A hand-built deal: P0 holds 2c 3d 4h, P1 holds 5c 5d 6h, board 1 is 2d, and
 * the stub is 3c 4c 5h 6c 2h 3h 4d 6d in that order. The strategy is pure
 * wherever the scripted hands go — check/call at every betting node, stand
 * pat at every draw — unless a test overrides a node.
 */
function fixture(over: Partial<Deal> = {}): Deal {
  const deck = cards('2c 3d 4h  5c 5d 6h  2d  3c 4c 5h 6c 2h 3h 4d 6d')
  // ledger order is fold, check/call, pot: [check, bet] with nothing owed,
  // [fold, call, raise] facing a bet, [fold, call] facing the all-in
  const check = [1000, 0]
  const call3 = [0, 1000, 0]
  const call2 = [0, 1000]
  const r1: Record<string, number[]> = {
    '': check, x: check, p: call3, xp: call3, pp: call3, xpp: call3, ppp: call2, xppp: call2,
  }
  const pat = [1000, 0, 0, 0]
  const lines = ['xx', 'xpc', 'xppc', 'xpppc', 'pc', 'ppc', 'pppc']
  const d0 = Object.fromEntries(lines.map((l) => [l, pat]))
  const d1 = Object.fromEntries(lines.map((l) => [l, { '0': pat, '1': pat }]))
  const r2: Deal['r2'] = {}
  for (const l of lines) {
    r2[l] = {}
    for (const a of 'nlmt') for (const b of 'nlmt') r2[l][a + b] = { ...r1 }
  }
  const show: Record<string, Showdown> = {}
  for (const a of 'nlmt')
    for (const b of 'nlmt')
      show[a + b] = {
        holes: [cards('2c 3d 4h'), cards('5c 5d 6h')],
        board: [c('2d'), c('3c')],
        inner: 1,
        outer: 1,
        cats: [['straight', 'pair'], ['pair', 'two pair']],
      }
  // the one world the first scripted hand reaches: P0 threw low (2c), drew 3c; P1 pat; board 2 is 4c
  show.ln = {
    holes: [cards('3c 3d 4h'), cards('5c 5d 6h')],
    board: [c('2d'), c('4c')],
    inner: 1,
    outer: 2,
    cats: [['pair', 'two pair'], ['pair', 'two pair']],
  }
  return { deck, order: [cards('2c 3d 4h'), cards('5c 5d 6h')], r1, d0, d1, r2, show, ...over }
}

const never = () => {
  throw new Error('no randomness expected here')
}
const always = (u: number) => () => u

describe('the cards', () => {
  it('consume the stub in draw order: P0 first, then P1, then board 2', () => {
    const deal = fixture()
    expect(replacementOf(deal, 0, ['l', null])).toBe(c('3c'))
    expect(replacementOf(deal, 1, ['n', 'm'])).toBe(c('3c'))
    expect(replacementOf(deal, 1, ['l', 'm'])).toBe(c('4c'))
    expect(replacementOf(deal, 0, ['n', 'm'])).toBeNull()
    expect(holeOf(deal, 0, [null, null]).map(sym)).toEqual(['2c', '3d', '4h'])
    expect(holeOf(deal, 0, ['l', null]).map(sym)).toEqual(['3c', '3d', '4h'])
    expect(holeOf(deal, 1, ['l', 't']).map(sym)).toEqual(['4c', '5c', '5d'])
  })
})

describe('the mix and the roll', () => {
  it('reads per-mille over the legal actions and falls back to uniform', () => {
    expect(toMix(['c', 'p'], [250, 750])).toEqual({ c: 0.25, p: 0.75 })
    expect(toMix(['f', 'c', 'p'], undefined)).toEqual({ f: 1 / 3, c: 1 / 3, p: 1 / 3 })
  })

  it('stacks aggression first: a low roll bets, and a deviation is graded by depth', () => {
    const mix = { c: 0.6, p: 0.4 }
    expect(prescribed(mix, 0)).toBe('p')
    expect(prescribed(mix, 39)).toBe('p')
    expect(prescribed(mix, 40)).toBe('c')
    expect(mistakeOf(mix, 10, 'c')).toBe(30)
    expect(mistakeOf(mix, 10, 'p')).toBe(0)
    expect(mistakeOf(mix, 90, 'p')).toBe(50)
  })

  it('at the draw, standing pat owns the low rolls and the throws follow in canonical order', () => {
    const mix = { n: 0.1, l: 0.6, m: 0.1, t: 0.2 }
    expect(prescribed(mix, 5)).toBe('n')
    expect(prescribed(mix, 50)).toBe('l')
    expect(prescribed(mix, 75)).toBe('m')
    expect(prescribed(mix, 99)).toBe('t')
  })
})

describe('one hand against the pack', () => {
  it('plays check–check, a throw, a pat, board 2, check–check, and settles the showdown', () => {
    let s = freshSession([fixture()], always(0.5))
    const h0 = s.hand!
    expect(h0.humanSeat).toBe(0)
    expect(phaseOf(h0)).toBe('round1')
    expect(h0.pendingRoll).toBe(50)
    expect(spotOf(h0)).toEqual({ legal: ['c', 'p'], mix: { c: 1, p: 0 } })

    s = playAct(s, 'c', always(0.5)) // you check; the solver checks; now your draw
    let h = s.hand!
    expect(h.l1).toBe('xx')
    expect(phaseOf(h)).toBe('draw')
    expect(spotOf(h).legal).toEqual(['n', 'l', 'm', 't'])

    s = playAct(s, 'l', always(0.5)) // you throw 2c and draw 3c; the solver stands pat; board 2 lands
    h = s.hand!
    expect(h.throws).toEqual(['l', 'n'])
    expect(h.l2).toBe('')
    expect(phaseOf(h)).toBe('round2')
    expect(holeOf(h.deal, 0, h.throws).map(sym)).toEqual(['3c', '3d', '4h'])
    expect(h.rows.filter((r) => r.kind === 'board').map((r) => (r.kind === 'board' ? sym(r.card) : ''))).toEqual([
      '2d',
      '4c',
    ])

    s = playAct(s, 'c', always(0.5)) // you check; the solver checks; showdown
    h = s.hand!
    expect(h.l2).toBe('xx')
    expect(phaseOf(h)).toBe('over')
    // inner to P1 (pair of fives over pair of threes), outer chopped: P0 gets 0.25 of a 2-chip pot, net -0.5
    expect(h.result).toBe(-0.5)
    expect(s.chips).toBe(-0.5)
    expect(s.hands).toBe(1)
    expect(s.seatChips).toEqual([-0.5, 0])
    expect(s.seatHands).toEqual([1, 0])
    expect(verdictOf(h)).toContain('inner: the solver (pair v pair)')
    expect(verdictOf(h)).toContain('outer: chop')
    expect(verdictOf(h)).toContain('the solver wins 0.5')

    // the transcript, in order: board, you, solver, you (with the card drawn), solver, board, you, solver, result
    const labels = h.rows.map((r) => (r.kind === 'action' ? `${r.human ? 'you' : 'bot'}:${r.label}` : r.kind))
    expect(labels).toEqual([
      'board', 'you:check', 'bot:check', 'you:throw 2c', 'bot:stands pat', 'board', 'you:check', 'bot:check', 'result',
    ])
    const throwRow = h.rows[3]
    expect(throwRow.kind === 'action' && throwRow.note).toBe('drew 3c')

    // a finished hand ignores further actions
    expect(playAct(s, 'c', never)).toBe(s)
  })

  it('lets the solver open when you sit in seat 1, and a fold costs the ante', () => {
    const deal = fixture()
    deal.r1[''] = [0, 1000] // the solver in seat 0 opens with a pure bet: [check, bet]
    const h = dealHand(deal, 1, always(0.2))
    expect(h.l1).toBe('p')
    expect(phaseOf(h)).toBe('round1')
    expect(spotOf(h).legal).toEqual(['f', 'c', 'p'])
    expect(h.rows[1]).toMatchObject({ kind: 'action', human: false, label: 'bet', cost: 2 })
    const s = playAct({ ...freshSession([], never), hand: h, pending: [] }, 'f', never)
    expect(s.hand!.result).toBe(-1)
    expect(s.chips).toBe(-1)
    expect(verdictOf(s.hand!)).toBe('you held 5c 5d 6h · the solver held 2c 3d 4h — the solver wins 1')
  })

  it('grades your decision against the roll it showed you', () => {
    const deal = fixture()
    deal.r1[''] = [600, 400] // check 60, bet 40: the bet owns rolls 0–39
    let s = freshSession([deal], always(0.1)) // roll 10
    expect(s.hand!.pendingRoll).toBe(10)
    s = playAct(s, 'c', always(0.1))
    const mine = s.hand!.rows[1]
    expect(mine).toMatchObject({ kind: 'action', human: true, act: 'c', roll: 10, correct: 'p', mistake: 30 })
  })
})

describe('the session', () => {
  it('alternates seats, works through the pack, then reshuffles the used deals back in', () => {
    const deals = [fixture(), fixture(), fixture()]
    let s = freshSession(deals, always(0.5))
    expect(s.pending.length).toBe(2)
    expect(s.hand!.humanSeat).toBe(0)
    s = nextHand(s, always(0.5))
    expect(s.hand!.humanSeat).toBe(1)
    expect(s.pending.length).toBe(1)
    expect(s.used.length).toBe(1)
    s = nextHand(s, always(0.5))
    expect(s.pending.length).toBe(0)
    expect(runningLow(s)).toBe(true)
    s = nextHand(s, always(0.5)) // the queue is empty: the three used deals come back
    expect(s.hand).not.toBeNull()
    expect(s.pending.length).toBe(2)
    expect(s.used.length).toBe(0)
  })

  it('takes more deals as a later chunk arrives, and starts with no hand when none has', () => {
    const empty = freshSession([], never)
    expect(empty.hand).toBeNull()
    const s = addDeals(empty, [fixture(), fixture()], always(0.5))
    expect(s.pending.length).toBe(2)
    expect(nextHand(s, always(0.5)).hand!.humanSeat).toBe(0)
  })
})
