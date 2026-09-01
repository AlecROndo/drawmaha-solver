import { describe, expect, it } from 'vitest'
import {
  CARDS,
  DEALS,
  DECISION_NODES,
  EDGES,
  TERMINAL_NODES,
  actionFrequency,
  actionLabel,
  actor,
  isTerminal,
  returns,
  infosetKey,
  type Act,
  type BetProbability,
  type Card,
} from './kuhn'

const [J, Q, K]: Card[] = [0, 1, 2]

/**
 * The same behaviour table `tests/kuhn/test_game.py` pins in Python, typed out
 * again here rather than derived. These rules are the one thing this app
 * re-implements, so they get the same treatment as the original.
 */
const PAYOFFS: [string, Card, Card, number][] = [
  ['pp', J, Q, -1], ['pp', J, K, -1], ['pp', Q, J, 1],
  ['pp', Q, K, -1], ['pp', K, J, 1], ['pp', K, Q, 1],
  ['pbp', J, Q, -1], ['pbp', J, K, -1], ['pbp', Q, J, -1],
  ['pbp', Q, K, -1], ['pbp', K, J, -1], ['pbp', K, Q, -1],
  ['pbb', J, Q, -2], ['pbb', J, K, -2], ['pbb', Q, J, 2],
  ['pbb', Q, K, -2], ['pbb', K, J, 2], ['pbb', K, Q, 2],
  ['bp', J, Q, 1], ['bp', J, K, 1], ['bp', Q, J, 1],
  ['bp', Q, K, 1], ['bp', K, J, 1], ['bp', K, Q, 1],
  ['bb', J, Q, -2], ['bb', J, K, -2], ['bb', Q, J, 2],
  ['bb', Q, K, -2], ['bb', K, J, 2], ['bb', K, Q, 2],
]

describe('payoffs', () => {
  it.each(PAYOFFS)('%s with (%i, %i) pays P0 %i', (history, p0, p1, chips) => {
    expect(returns(history, [p0, p1])).toEqual([chips, -chips])
  })

  it('covers every terminal history against every deal', () => {
    const covered = new Set(PAYOFFS.map(([h, a, b]) => `${h}:${a}${b}`))
    expect(covered.size).toBe(5 * DEALS.length)
  })

  it('is zero sum', () => {
    for (const [history, p0, p1] of PAYOFFS) {
      const [a, b] = returns(history, [p0, p1])
      expect(a + b).toBe(0)
    }
  })
})

describe('the state machine', () => {
  it('alternates players, P0 first', () => {
    expect(actor('')).toBe(0)
    expect(actor('p')).toBe(1)
    expect(actor('pb')).toBe(0)
  })

  it('ends only on the five terminal histories', () => {
    for (const key of ['', 'p', 'b', 'pb']) expect(isTerminal(key)).toBe(false)
    for (const key of ['pp', 'pbp', 'pbb', 'bp', 'bb']) expect(isTerminal(key)).toBe(true)
  })

  it.each([
    ['p', '', 'check'],
    ['b', '', 'bet'],
    ['p', 'b', 'fold'],
    ['b', 'b', 'call'],
    ['p', 'pb', 'fold'],
    ['b', 'pb', 'call'],
  ])('calls %s at "%s" a %s', (act, history, label) => {
    expect(actionLabel(act as Act, history)).toBe(label)
  })
})

describe('infoset keys', () => {
  it('uses the literature notation the exporter writes', () => {
    expect(infosetKey(K, '')).toBe('K')
    expect(infosetKey(J, 'pb')).toBe('Jpb')
  })

  it('hides the opponent card by construction', () => {
    // The key takes one card, so there is nowhere to put the opponent's — the
    // same property the Python InfoSet has.
    expect(infosetKey(K, 'b')).toBe(infosetKey(K, 'b'))
  })
})

describe('the drawn tree', () => {
  it('has an edge for both actions at every decision node', () => {
    for (const node of DECISION_NODES) {
      const outgoing = EDGES.filter((e) => e.from === node.key).map((e) => e.act)
      expect(outgoing.sort()).toEqual(['b', 'p'])
    }
  })

  it('points every edge at a node that exists', () => {
    const keys = new Set([
      ...DECISION_NODES.map((n) => n.key as string),
      ...TERMINAL_NODES.map((n) => n.key),
    ])
    for (const edge of EDGES) expect(keys.has(edge.to)).toBe(true)
  })

  it('draws all nine nodes of the game', () => {
    expect(DECISION_NODES).toHaveLength(4)
    expect(TERMINAL_NODES).toHaveLength(5)
  })
})

describe('edge frequency', () => {
  /** The equilibrium at alpha = 0.22, the value the committed solve landed on. */
  const ALPHA = 0.22
  const EQUILIBRIUM: Record<string, number> = {
    J: ALPHA, Q: 0, K: 3 * ALPHA,
    Jp: 1 / 3, Qp: 0, Kp: 1,
    Jb: 0, Qb: 1 / 3, Kb: 1,
    Jpb: 0, Qpb: ALPHA + 1 / 3, Kpb: 1,
  }
  const equilibrium: BetProbability = (card, history) => EQUILIBRIUM[infosetKey(card, history)]

  const flatMean = (history: string, bet: BetProbability) =>
    CARDS.reduce<number>((sum, card) => sum + bet(card, history), 0) / CARDS.length

  it('sums to one across the two actions at every node', () => {
    for (const node of DECISION_NODES) {
      const p = actionFrequency(node.key, 'p', equilibrium)
      const b = actionFrequency(node.key, 'b', equilibrium)
      expect(p + b).toBeCloseTo(1, 12)
    }
  })

  it('equals the flat mean at the root, where the deal is still uniform', () => {
    expect(actionFrequency('', 'b', equilibrium)).toBeCloseTo(flatMean('', equilibrium), 12)
  })

  it('reach-weights downstream nodes away from the flat mean', () => {
    // Reaching "p" means the opener checked: 0.78 of the time with a jack, all
    // of it with a queen, 0.34 with a king. So the player facing that check
    // holds J/Q/K in proportion (1 + .34) : (.78 + .34) : (.78 + 1) — nowhere
    // near thirds — and each card's P(bet) is weighted accordingly.
    const w = [1 + 0.34, 0.78 + 0.34, 0.78 + 1]
    const total = w[0] + w[1] + w[2]
    // The queen never bets after a check, so its weight drops out of the top.
    const expected = (w[0] * (1 / 3) + w[2] * 1) / total
    expect(actionFrequency('p', 'b', equilibrium)).toBeCloseTo(expected, 12)
    // The correction is worth ~8pp here, so it is not a rounding detail.
    expect(expected).toBeCloseTo(0.525, 3)
    expect(flatMean('p', equilibrium)).toBeCloseTo(4 / 9, 12)
  })

  it('falls back to the flat mean at a node no hand reaches', () => {
    // Nobody ever bets, so "b" and "pb" are unreachable and the ratio is 0/0.
    const neverBets: BetProbability = (_card, history) => (history === '' ? 0 : 0.25)
    expect(actionFrequency('b', 'b', neverBets)).toBeCloseTo(0.25, 12)
    expect(actionFrequency('pb', 'p', neverBets)).toBeCloseTo(0.75, 12)
  })

  it('never leaves the unit interval', () => {
    for (const node of DECISION_NODES) {
      for (const act of ['p', 'b'] as Act[]) {
        const f = actionFrequency(node.key, act, equilibrium)
        expect(f).toBeGreaterThanOrEqual(0)
        expect(f).toBeLessThanOrEqual(1)
      }
    }
  })
})
