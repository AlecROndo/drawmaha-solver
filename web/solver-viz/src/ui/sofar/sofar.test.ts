/**
 * Both layouts of the hand so far render every state the hand can stand in
 * and say the right words — the played chips, the options at the head, the
 * owed board card, the hand's end — so the backup can be swapped in for the
 * default knowing it is a complete replacement. The clicks are driven too:
 * neither layout uses a hook, so a component is a plain function of its
 * props and its element tree can be walked and its buttons pressed without a
 * DOM.
 */

import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { MINI_SUIT_GLYPH, miniGlyph, parseMini } from '../../mini/cards'
import { MINI_INNER_ROWS, MINI_R1_COLS } from '../../mini/regions'
import type { GameUI } from '../game'
import { type MiniLine, advance, decisionOf, nextOptions } from '../MiniScore'
import { DEFAULT_LAYOUT, HandScore, scoreLayout } from './index'
import { PathScore } from './PathScore'
import { StripScore } from './StripScore'
import { type Step, rewindLabel, stepsOf } from './steps'

const game: GameUI = {
  rows: MINI_INNER_ROWS,
  cols: MINI_R1_COLS,
  handSize: 3,
  deck: { ranks: ['6', '5', '4', '3', '2'], suits: MINI_SUIT_GLYPH.map((glyph) => ({ glyph })), cardAt: (r, s) => (4 - r) * 3 + s },
  label: (c) => miniGlyph(c),
}

const b1 = parseMini('4h')
const b2 = parseMini('4d')
const base: MiniLine = { board1: b1, r1: '', draws: [], board2: null, r2: '' }

/**
 * The shares the fixture hands every state: the bet keys for a betting
 * decision, '0' / '1' for a draw. Each option button must carry its own.
 */
const SHARES = { c: 0.62, p: 0.38, f: 0.1, '0': 0.7, '1': 0.3 }
const SHARE_WORD: Record<string, string> = { 'play check': '62%', 'play call': '62%', 'play bet': '38%', 'play raise': '38%', 'play fold': '10%', 'play stand pat': '70%', 'play throw one': '30%' }

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

type Pressable = { label: string; onClick: () => void }

/**
 * Every <button aria-label> in an element tree, in document order, expanding
 * function components by calling them as plain functions. That is only valid
 * while neither layout (nor anything they render) calls a hook: the day one
 * takes a `useState`, this throws React's invalid-hook-call error from inside
 * the walk, and the fix is to render through a real renderer, not to drop the
 * click tests.
 */
function buttonsOf(node: unknown, out: Pressable[] = []): Pressable[] {
  if (node === null || node === undefined || typeof node === 'boolean' || typeof node === 'string' || typeof node === 'number') return out
  if (Array.isArray(node)) {
    for (const n of node) buttonsOf(n, out)
    return out
  }
  const el = node as { type: unknown; props: Record<string, unknown> }
  if (typeof el.type === 'function') return buttonsOf((el.type as (p: unknown) => unknown)(el.props), out)
  if (el.type === 'button' && typeof el.props['aria-label'] === 'string') out.push({ label: el.props['aria-label'], onClick: el.props.onClick as () => void })
  return buttonsOf(el.props.children, out)
}

/** The button's inner HTML in a static render, by its aria label (escaped, so a label is matched literally). */
const buttonHtml = (html: string, label: string): string | null =>
  html.match(new RegExp(`<button[^>]*aria-label="${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"[^>]*>((?:(?!</button>)[\\s\\S])*)</button>`))?.[1] ?? null

describe.each(Object.entries(VARIANTS))('%s', (_name, Variant) => {
  for (const s of STATES) {
    it(`says the right words at: ${s.name}`, () => {
      const html = renderToStaticMarkup(createElement(Variant, { game, line: s.line, onLine: () => {}, picking: null, onPick: () => {}, shares: SHARES }))
      for (const w of s.words) expect(html, `${s.name}: wants ${JSON.stringify(w)}`).toContain(w)
      for (const w of s.absent ?? []) expect(html, `${s.name}: must not say ${JSON.stringify(w)}`).not.toContain(w)
      // Each option button carries its own share, as a percentage, and the path's bar is scaled to it.
      for (const w of s.words.filter((w) => w.startsWith('play '))) {
        expect(buttonHtml(html, w), `${s.name}: the ${w} button`).toContain(`<b>${SHARE_WORD[w]}</b>`)
      }
      if (Variant === PathScore && s.words.includes('play check')) expect(html).toContain('transform:scaleX(0.62)')
      if (Variant === PathScore && s.words.includes('play stand pat')) expect(html).toContain('transform:scaleX(0.7)')
    })
  }

  it('fans the fifteen cards out while a picker is open', () => {
    const html = renderToStaticMarkup(createElement(Variant, { game, line: base, onLine: () => {}, picking: 1, onPick: () => {}, shares: null }))
    expect(html.match(/role="option"/g)?.length).toBe(15)
  })

  it('wires the clicks: a played step rewinds to just before it, an option plays it', () => {
    // Seat 1 facing a bet in round 2, after a check-bet-call round 1 and both draws.
    const line: MiniLine = { ...base, r1: 'xpc', draws: [0, 1], board2: b2, r2: 'p' }
    const got: MiniLine[] = []
    const buttons = buttonsOf(createElement(Variant, { game, line, onLine: (l: MiniLine) => got.push(l), picking: null, onPick: () => {}, shares: SHARES }))
    const played = stepsOf(line).filter((s): s is Extract<Step, { kind: 'act' | 'draw' }> => s.kind === 'act' || s.kind === 'draw')
    const rewinds = buttons.filter((b) => b.label.startsWith('rewind'))
    expect(rewinds.map((b) => b.label)).toEqual(played.map(rewindLabel))
    played.forEach((s, j) => {
      rewinds[j].onClick()
      expect(got.at(-1), `rewind at ${s.word}`).toEqual(s.back)
    })
    const options = nextOptions(line)
    expect(options.length).toBeGreaterThan(1)
    for (const o of options) {
      const b = buttons.find((x) => x.label === `play ${o.word}`)
      expect(b, `a button for ${o.word}`).toBeDefined()
      b!.onClick()
      expect(got.at(-1), `play ${o.word}`).toEqual(advance(line, o.key))
    }
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
    // Round 2: seat 0 checks, seat 1 bets, seat 0 folds — seat 1 takes it.
    const end3 = stepsOf({ ...base, r1: 'pc', draws: [0, 1], board2: b2, r2: 'xpf' }).at(-1)
    expect(end3).toMatchObject({ kind: 'end', why: 'fold', winner: 1 })
  })

  it('every played step rewinds to the hand just before it, where its own key plays it back', () => {
    const full: MiniLine = { ...base, r1: 'xpc', draws: [0, 1], board2: b2, r2: 'pc' }
    const steps = stepsOf(full)
    const isPlayed = (s: Step): s is Extract<Step, { kind: 'act' | 'draw' }> => s.kind === 'act' || s.kind === 'draw'
    const words = (ss: Step[]) => ss.filter(isPlayed).map((s) => `${s.seat}:${s.word}:${s.pot}`)
    expect(words(steps)).toHaveLength(7)
    steps.forEach((s, i) => {
      if (!isPlayed(s)) return
      // The rewound line shows exactly what was played before this step ...
      expect(words(stepsOf(s.back)), `before ${s.word}`).toEqual(words(steps.slice(0, i)))
      // ... stands on the decision this step answered ...
      expect(decisionOf(s.back), `the decision at ${s.word}`).toEqual(s.kind === 'act' ? { kind: 'bet', round: s.round, player: s.seat } : { kind: 'draw', player: s.seat })
      // ... and playing the step's key from there puts the step back.
      expect(words(stepsOf(advance(s.back, s.key))), `replaying ${s.word}`).toEqual(words(steps.slice(0, i + 1)))
      // A round-1 rewind forgets the draws and board card 2; a later one keeps them.
      expect(s.back.board2).toBe(s.kind === 'act' && s.round === 2 ? b2 : null)
      if (s.kind === 'draw') expect(s.back.draws).toEqual(full.draws.slice(0, s.seat))
    })
  })
})

describe('the layout switch', () => {
  it('reads ?score= from the query string and falls back to the default', () => {
    expect(scoreLayout('?score=strip')).toBe('strip')
    expect(scoreLayout('?score=path')).toBe('path')
    expect(scoreLayout('?score=grid')).toBe(DEFAULT_LAYOUT)
    expect(scoreLayout('?other=1')).toBe(DEFAULT_LAYOUT)
    expect(scoreLayout('')).toBe(DEFAULT_LAYOUT)
  })

  it('exports the default layout where there is no window to read a query string from', () => {
    // The precondition, stated: vitest runs these under node, so the module saw no window at import.
    expect(typeof window).toBe('undefined')
    expect(HandScore).toBe(VARIANTS[DEFAULT_LAYOUT === 'path' ? 'PathScore' : 'StripScore'])
  })
})
