import { describe, expect, it } from 'vitest'
import {
  DEALS,
  DECISION_NODES,
  EDGES,
  TERMINAL_NODES,
  actionLabel,
  actor,
  isTerminal,
  returns,
  infosetKey,
  type Act,
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
