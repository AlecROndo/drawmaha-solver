/**
 * Both layouts of the hand so far render every state the hand can stand in
 * and say the right words — the played chips, the options at the head, the
 * owed board card, the hand's end — so the backup can be swapped in for the
 * default knowing it is a complete replacement.
 */

import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { MINI_SUIT_GLYPH, miniGlyph, parseMini } from '../../mini/cards'
import { MINI_INNER_ROWS, MINI_R1_COLS } from '../../mini/regions'
import type { GameUI } from '../game'
import type { MiniLine } from '../MiniScore'
import { PathScore } from './PathScore'
import { StripScore } from './StripScore'
import { stepsOf } from './steps'

const game: GameUI = {
  rows: MINI_INNER_ROWS,
  cols: MINI_R1_COLS,
  handSize: 3,
  deck: { ranks: ['6', '5', '4', '3', '2'], suits: MINI_SUIT_GLYPH.map((glyph, s) => ({ glyph, red: s > 0 })), cardAt: (r, s) => (4 - r) * 3 + s },
  label: (c) => miniGlyph(c),
  isRed: (c) => c % 3 > 0,
}

const b1 = parseMini('4h')
const b2 = parseMini('4d')
const base: MiniLine = { board1: b1, r1: '', draws: [], board2: null, r2: '' }

/** The six states a variant has to read well in, with the words each must say. */
const STATES: { name: string; line: MiniLine; words: string[]; absent?: string[] }[] = [
  { name: 'round 1 in progress, seat 0 to act', line: base, words: ['play check', 'play bet', '4♥', 'seat 0'] },
  { name: 'the draw, seat 0 to throw', line: { ...base, r1: 'pc' }, words: ['bet', 'call', 'play stand pat', 'play throw one'] },
  { name: 'waiting on board card 2', line: { ...base, r1: 'pc', draws: [0, 1] }, words: ['stood pat', 'threw one', 'pick', 'choose board card 2'] },
  { name: 'round 2 in progress, seat 0 to act', line: { ...base, r1: 'pc', draws: [0, 1], board2: b2 }, words: ['bet', 'call', 'stood pat', 'threw one', '4♦', 'play check', 'play bet'], absent: ['pick'] },
  { name: 'ended on a fold', line: { ...base, r1: 'pf' }, words: ['bet', 'folded', 'takes'], absent: ['play '] },
  { name: 'showdown', line: { ...base, r1: 'pc', draws: [0, 1], board2: b2, r2: 'xx' }, words: ['showdown', 'check'], absent: ['play '] },
]

const VARIANTS = { PathScore, StripScore }

describe.each(Object.entries(VARIANTS))('%s', (_name, Variant) => {
  for (const s of STATES) {
    it(`says the right words at: ${s.name}`, () => {
      const html = renderToStaticMarkup(createElement(Variant, { game, line: s.line, onLine: () => {}, picking: null, onPick: () => {}, shares: { c: 0.62, p: 0.38, f: 0.1, '0': 0.7, '1': 0.3 } }))
      for (const w of s.words) expect(html, `${s.name}: wants ${JSON.stringify(w)}`).toContain(w)
      for (const w of s.absent ?? []) expect(html, `${s.name}: must not say ${JSON.stringify(w)}`).not.toContain(w)
      // The share percentages reach the option buttons.
      if (s.words.some((w) => w.startsWith('play '))) expect(html).toMatch(/\d+%/)
    })
  }

  it('fans the fifteen cards out while a picker is open', () => {
    const html = renderToStaticMarkup(createElement(Variant, { game, line: base, onLine: () => {}, picking: 1, onPick: () => {}, shares: null }))
    expect(html.match(/role="option"/g)?.length).toBe(15)
  })
})

describe('stepsOf', () => {
  it('lists the hand in play order with the pot after each bet', () => {
    const steps = stepsOf({ ...base, r1: 'pc', draws: [0, 1], board2: b2, r2: 'pc' })
    expect(steps.map((s) => (s.kind === 'act' || s.kind === 'draw' ? `${s.seat}:${s.word}:${s.pot}` : s.kind))).toEqual(['board', '0:bet:4', '1:call:6', '0:stood pat:6', '1:threw one:6', 'board', '0:bet:12', '1:call:18', 'end'])
  })

  it('names the winner of a fold', () => {
    const end = stepsOf({ ...base, r1: 'pf' }).at(-1)
    expect(end).toMatchObject({ kind: 'end', why: 'fold', winner: 0, pot: 4 })
    const end2 = stepsOf({ ...base, r1: 'xpf' }).at(-1)
    expect(end2).toMatchObject({ kind: 'end', why: 'fold', winner: 1 })
  })
})
