import { describe, expect, it } from 'vitest'
import {
  ANTE,
  STACK,
  actorOf,
  betSymbol,
  betWord,
  costOf,
  isClosed,
  isFold,
  isThrow,
  legalBets,
  replay,
  throwIndex,
  throwWord,
} from './grammar'

/** Every complete round-1 line, grown from the grammar the way test_game.py grows them. */
function completeLines(): string[] {
  const out: string[] = []
  const grow = (line: string) => {
    const chips = replay([line])
    if (isFold(line) || isClosed(line, chips.behind)) {
      out.push(line)
      return
    }
    for (const bet of legalBets([line])) grow(line + betSymbol(bet, line))
  }
  grow('')
  return out
}

describe('the round-1 grammar', () => {
  it("grows exactly the plan's thirteen lines", () => {
    expect(completeLines().sort()).toEqual(
      ['xx', 'xpf', 'xpc', 'xppf', 'xppc', 'xpppf', 'xpppc', 'pf', 'pc', 'ppf', 'ppc', 'pppf', 'pppc'].sort(),
    )
  })

  it('offers check/bet with nothing owed, fold/call/raise facing a bet, fold/call facing the all-in', () => {
    expect(legalBets([''])).toEqual(['c', 'p'])
    expect(legalBets(['x'])).toEqual(['c', 'p'])
    expect(legalBets(['p'])).toEqual(['f', 'c', 'p'])
    expect(legalBets(['xpp'])).toEqual(['f', 'c', 'p'])
    expect(legalBets(['ppp'])).toEqual(['f', 'c'])
    expect(legalBets(['xppp'])).toEqual(['f', 'c'])
  })

  it('writes x for a check and c for a call, and names the action in context', () => {
    expect(betSymbol('c', '')).toBe('x')
    expect(betSymbol('c', 'p')).toBe('c')
    expect(betWord('c', '')).toBe('check')
    expect(betWord('c', 'xp')).toBe('call')
    expect(betWord('p', '')).toBe('bet')
    expect(betWord('p', 'p')).toBe('raise')
    expect(betWord('f', 'p')).toBe('fold')
  })

  it('gives P0 the opening of every round and alternates from there', () => {
    expect(actorOf('')).toBe(0)
    expect(actorOf('x')).toBe(1)
    expect(actorOf('xp')).toBe(0)
    expect(actorOf('xpp')).toBe(1)
  })
})

describe('the chips', () => {
  it('climb 2 → 8 → all-in from the 2-chip ante pot, and the stack truncates the third', () => {
    expect(STACK - ANTE).toBe(25)
    expect(costOf([''], 'p')).toBe(2)
    expect(costOf(['p'], 'c')).toBe(2)
    expect(costOf(['p'], 'p')).toBe(8)
    expect(costOf(['pp'], 'p')).toBe(23) // 2 already in: 25 total, the stack
    expect(costOf(['ppp'], 'c')).toBe(17)
    expect(costOf(['p'], 'f')).toBe(0)
    expect(replay(['pc'])).toEqual({ pot: 6, committed: [3, 3], behind: [23, 23], inRound: [2, 2] })
    expect(replay(['ppc'])).toEqual({ pot: 18, committed: [9, 9], behind: [17, 17], inRound: [8, 8] })
    expect(replay(['pppc'])).toEqual({ pot: 52, committed: [26, 26], behind: [0, 0], inRound: [25, 25] })
    expect(replay(['xpc'])).toEqual(replay(['pc']))
  })

  it('opens round 2 in the state round 1 left, and an all-in round 2 is closed at once', () => {
    expect(replay(['pc', ''])).toEqual({ pot: 6, committed: [3, 3], behind: [23, 23], inRound: [0, 0] })
    expect(costOf(['pc', ''], 'p')).toBe(6)
    expect(costOf(['ppc', 'p'], 'p')).toBe(17 /* all of it: 18 + 18 would exceed the stack */)
    expect(isClosed('', replay(['pppc', '']).behind)).toBe(true)
    expect(isClosed('', replay(['xx', '']).behind)).toBe(false)
    expect(isClosed('x', replay(['xx', 'x']).behind)).toBe(false)
    expect(isClosed('xx', replay(['xx', 'xx']).behind)).toBe(true)
  })

  it('stops paying for a folder where they stood', () => {
    expect(replay(['xpf']).committed).toEqual([1, 3]) // P1 bet 2 into the checked pot, P0 folded
    expect(replay(['ppf']).committed).toEqual([3, 9]) // P0 bet 2, P1 raised to 8, P0 folded
    expect(isFold('xpf')).toBe(true)
    expect(isFold('xpc')).toBe(false)
  })
})

describe('the throws', () => {
  it('name the card when given one and the position otherwise', () => {
    expect(throwWord('n')).toBe('stand pat')
    expect(throwWord('l')).toBe('throw low')
    expect(throwWord('m', 7)).toBe('throw 4d')
    expect(throwIndex('t')).toBe(2)
    expect(throwIndex('n')).toBe(-1)
  })

  it('are told apart from bets', () => {
    expect(isThrow('n')).toBe(true)
    expect(isThrow('t')).toBe(true)
    expect(isThrow('p')).toBe(false)
  })
})
