import { describe, expect, it } from 'vitest'

import {
  advance,
  boardPending,
  handOver,
  jump,
  lineOver,
  stateAt,
  type Step,
  type Walk,
} from './timeline'

const CC_J_C: Step[] = [
  { k: 'a', v: 'c' },
  { k: 'a', v: 'c' },
  { k: 'b', v: 'J' },
  { k: 'a', v: 'c' },
]

describe('stateAt replays a prefix into (round-1 line, board, round-2 line)', () => {
  it('splits at the board card', () => {
    expect(stateAt(CC_J_C, 0)).toEqual({ l1: '', board: null, l2: '' })
    expect(stateAt(CC_J_C, 2)).toEqual({ l1: 'cc', board: null, l2: '' })
    expect(stateAt(CC_J_C, 4)).toEqual({ l1: 'cc', board: 'J', l2: 'c' })
  })
})

describe('the rewind contract: the cursor moves, the path survives', () => {
  const atEnd: Walk = { path: CC_J_C, cur: 4 }

  it('jumping back and forward never shortens the path', () => {
    const back = jump(atEnd, 1)
    expect(back.path).toHaveLength(4)
    expect(jump(back, 3).path).toHaveLength(4)
  })

  it('re-taking the action already on the path just walks forward', () => {
    const back = jump(atEnd, 1)
    const on = advance(back, { k: 'a', v: 'c' })
    expect(on.path).toHaveLength(4)
    expect(on.cur).toBe(2)
  })

  it('a DIFFERENT action is the only thing that forks — truncate then append', () => {
    const back = jump(atEnd, 1)
    const fork = advance(back, { k: 'a', v: 'r' })
    expect(fork.path).toEqual([
      { k: 'a', v: 'c' },
      { k: 'a', v: 'r' },
    ])
    expect(fork.cur).toBe(2)
  })

  it('advancing at the frontier appends', () => {
    const grown = advance(atEnd, { k: 'a', v: 'r' })
    expect(grown.path).toHaveLength(5)
    expect(grown.cur).toBe(5)
  })
})

describe('handOver', () => {
  it('needs a board and a closed second round', () => {
    expect(handOver(CC_J_C, 4)).toBe(false) // one check: round 2 still open
    expect(handOver([...CC_J_C, { k: 'a', v: 'c' }], 5)).toBe(true) // checked down
  })

  it('a round-1 fold ends the hand without any board', () => {
    const rf: Step[] = [
      { k: 'a', v: 'r' },
      { k: 'a', v: 'f' },
    ]
    expect(handOver(rf, 2)).toBe(true)
  })
})

describe('between the rounds it is the deck to act, not a seat', () => {
  it('rewinding the opening line to closed round 1, pre-board', () => {
    // The page's own default line, cursor on the second check: "cc" is a
    // closing line, not a decision infoset — nobody is "to act".
    const s = stateAt(CC_J_C, 2)
    expect(boardPending(s)).toBe(true)
    expect(lineOver(s)).toBe(false)
  })

  it('a fold closes round 1 but ends the hand instead of awaiting a board', () => {
    const s = { l1: 'rf', board: null, l2: '' }
    expect(boardPending(s)).toBe(false)
    expect(lineOver(s)).toBe(true)
  })

  it('an open round is neither pending nor over', () => {
    expect(boardPending(stateAt(CC_J_C, 1))).toBe(false)
    expect(lineOver(stateAt(CC_J_C, 1))).toBe(false)
    expect(lineOver(stateAt(CC_J_C, 4))).toBe(false) // round 2 still open
  })
})
