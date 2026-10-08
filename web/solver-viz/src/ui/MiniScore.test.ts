import { describe, expect, it } from 'vitest'
import { betSymbol, isClosed, isFold, legalBets, replay } from '../mini/grammar'
import { parseMini } from '../mini/cards'
import { type MiniLine, advance, decisionOf, nextOptions, toneOf } from './MiniScore'

/** Every line the score can stand on, grown from an opening board card through every decision to the hand's end. */
function everyLine(): MiniLine[] {
  const out: MiniLine[] = []
  const grow = (line: MiniLine) => {
    out.push(line)
    const d = decisionOf(line)
    if (d.kind === 'over') return
    if (d.kind === 'board2') {
      grow({ ...line, board2: parseMini('6c') })
      return
    }
    for (const o of nextOptions(line)) grow(advance(line, o.key))
  }
  grow({ board1: parseMini('4h'), r1: '', draws: [], board2: null, r2: '' })
  return out
}

describe('toneOf', () => {
  it('colours the pushing actions pink, a fold the quiet ink, and going along yellow', () => {
    expect(toneOf('p')).toBe('p')
    expect(toneOf('1')).toBe('p')
    expect(toneOf('f')).toBe('d')
    expect(toneOf('x')).toBe('c')
    expect(toneOf('c')).toBe('c')
    expect(toneOf('0')).toBe('c')
  })

  it('refuses a key it has never heard of rather than colouring it as going along', () => {
    expect(() => toneOf('2')).toThrow(/no colour/)
    expect(() => toneOf('')).toThrow(/no colour/)
  })

  it('has a colour for every option the score can offer and every chip it can have played', () => {
    const lines = everyLine()
    expect(lines.length).toBeGreaterThan(100)
    let options = 0
    for (const line of lines) {
      for (const o of nextOptions(line)) {
        expect(() => toneOf(o.key)).not.toThrow()
        options++
      }
      for (const sym of line.r1 + line.r2) expect(() => toneOf(sym)).not.toThrow()
    }
    expect(options).toBeGreaterThan(100)
  })

  it('agrees with the grammar: a played chip is the bet symbol of the option that played it', () => {
    // The chip path colours by the letter written into the line; the button path by the option key.
    // The two meet at `advance`: a pot bet is written 'p', a fold 'f', a call 'c' or a check 'x'.
    for (const line of everyLine()) {
      const d = decisionOf(line)
      if (d.kind !== 'bet') continue
      const cur = d.round === 1 ? line.r1 : line.r2
      const chips = replay(d.round === 1 ? [line.r1] : [line.r1, line.r2])
      expect(isFold(cur) || isClosed(cur, chips.behind)).toBe(false)
      for (const bet of legalBets(d.round === 1 ? [line.r1] : [line.r1, line.r2])) {
        const after = advance(line, bet)
        const written = (d.round === 1 ? after.r1 : after.r2).slice(-1)
        expect(written).toBe(betSymbol(bet, cur))
        expect(toneOf(written)).toBe(toneOf(bet))
      }
    }
  })
})
