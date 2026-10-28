/**
 * The hand so far, read off a `MiniLine` as one list of steps in the order
 * they were played — the shape both layouts draw from, so the path and the
 * strip differ in layout and not in what they say. Each played step carries the line to rewind to (the hand just before
 * it) and the pot after it; the head is the decision the line stands on with
 * its options already coloured.
 */

import { actorOf, replay } from '../../mini/grammar'
import type { MiniCard } from '../../mini/cards'
import { type Decision, type MiniLine, type Tone, decisionOf, linesOf, nextOptions, toneOf } from '../MiniScore'

export type Street = 'board 1' | 'round 1' | 'draw' | 'board 2' | 'round 2' | 'end'

export type Step =
  | { kind: 'board'; street: 'board 1' | 'board 2'; which: 1 | 2; card: MiniCard | null; pot: number }
  | { kind: 'act'; street: 'round 1' | 'round 2'; seat: 0 | 1; round: 1 | 2; key: string; word: string; tone: Tone; pot: number; back: MiniLine }
  | { kind: 'draw'; street: 'draw'; seat: 0 | 1; key: '0' | '1'; word: string; tone: Tone; pot: number; back: MiniLine }
  | { kind: 'end'; street: 'end'; why: 'fold' | 'showdown'; pot: number; winner: 0 | 1 | null }

export type PlayedStep = Extract<Step, { kind: 'act' | 'draw' }>
export type EndStep = Extract<Step, { kind: 'end' }>

export interface Head {
  decision: Decision
  options: { key: string; word: string; tone: Tone }[]
  /** the pot as it stands at the decision */
  pot: number
}

/** The poker word for the letter at `k` of a round, as the chip reads it. */
export function playedWord(letters: string, k: number): string {
  const sym = letters[k]
  if (sym === 'x') return 'check'
  if (sym === 'c') return 'call'
  if (sym === 'f') return 'fold'
  return letters.slice(0, k).endsWith('p') ? 'raise' : 'bet'
}

/** Past tense, for a transcript: "seat 1 called". */
export const PAST: Readonly<Record<string, string>> = { bet: 'bet', call: 'called', check: 'checked', raise: 'raised', fold: 'folded', 'stood pat': 'stood pat', 'threw one': 'threw one' }

/** Every step of the hand so far, in play order; nothing that has not happened. */
export function stepsOf(line: MiniLine): Step[] {
  const d = decisionOf(line)
  const out: Step[] = [{ kind: 'board', street: 'board 1', which: 1, card: line.board1, pot: 2 }]
  for (let k = 0; k < line.r1.length; k++) {
    out.push({
      kind: 'act',
      street: 'round 1',
      seat: actorOf(line.r1.slice(0, k)),
      round: 1,
      key: line.r1[k],
      word: playedWord(line.r1, k),
      tone: toneOf(line.r1[k]),
      pot: replay([line.r1.slice(0, k + 1)]).pot,
      back: { ...line, r1: line.r1.slice(0, k), draws: [], board2: null, r2: '' },
    })
  }
  const pot1 = replay([line.r1]).pot
  line.draws.forEach((t, seat) => {
    out.push({
      kind: 'draw',
      street: 'draw',
      seat: seat as 0 | 1,
      key: t === 1 ? '1' : '0',
      word: t === 1 ? 'threw one' : 'stood pat',
      tone: toneOf(t === 1 ? '1' : '0'),
      pot: pot1,
      back: { ...line, draws: line.draws.slice(0, seat), board2: null, r2: '' },
    })
  })
  if (line.draws.length === 2) out.push({ kind: 'board', street: 'board 2', which: 2, card: line.board2, pot: pot1 })
  if (line.board2 !== null) {
    for (let k = 0; k < line.r2.length; k++) {
      out.push({
        kind: 'act',
        street: 'round 2',
        seat: actorOf(line.r2.slice(0, k)),
        round: 2,
        key: line.r2[k],
        word: playedWord(line.r2, k),
        tone: toneOf(line.r2[k]),
        pot: replay([line.r1, line.r2.slice(0, k + 1)]).pot,
        back: { ...line, r2: line.r2.slice(0, k) },
      })
    }
  }
  if (d.kind === 'over') {
    const pot = replay(linesOf(line)).pot
    let winner: 0 | 1 | null = null
    if (d.why === 'fold') {
      const letters = line.r2.endsWith('f') ? line.r2 : line.r1
      winner = (1 - actorOf(letters.slice(0, letters.length - 1))) as 0 | 1
    }
    out.push({ kind: 'end', street: 'end', why: d.why, pot, winner })
  }
  return out
}

export function headOf(line: MiniLine): Head {
  return {
    decision: decisionOf(line),
    options: nextOptions(line).map((o) => ({ ...o, tone: toneOf(o.key) })),
    pot: replay(linesOf(line)).pot,
  }
}

/** The street the decision on screen belongs to. */
export function streetOf(d: Decision): Street {
  if (d.kind === 'bet') return d.round === 1 ? 'round 1' : 'round 2'
  if (d.kind === 'draw') return 'draw'
  if (d.kind === 'board2') return 'board 2'
  return 'end'
}

/** Whose decision, in words: "seat 0 to act", "seat 1 to draw", "board card 2 owed". */
export function headLabel(d: Decision): string {
  if (d.kind === 'bet') return `seat ${d.player} to act`
  if (d.kind === 'draw') return `seat ${d.player} to draw`
  if (d.kind === 'board2') return 'board card 2 owed'
  return d.why === 'fold' ? 'folded' : 'showdown'
}

/** The seat whose decision it is, if it is a seat's. */
export const headSeat = (d: Decision): 0 | 1 | null => (d.kind === 'bet' || d.kind === 'draw' ? d.player : null)

export const pct = (shares: Record<string, number> | null, key: string): string | null =>
  shares && shares[key] !== undefined ? `${Math.round(shares[key] * 100)}%` : null

/** How the hand ended, in one line. */
export function endWords(step: EndStep): string {
  if (step.why === 'showdown') return `showdown · pot ${step.pot}`
  return `seat ${1 - (step.winner ?? 0)} folded · seat ${step.winner} takes ${step.pot}`
}

/** Grouped by street, in play order, keeping the streets' order. */
export function byStreet(steps: Step[]): { street: Street; steps: Step[] }[] {
  const groups: { street: Street; steps: Step[] }[] = []
  for (const s of steps) {
    const last = groups[groups.length - 1]
    if (last && last.street === s.street) last.steps.push(s)
    else groups.push({ street: s.street, steps: [s] })
  }
  return groups
}

/** The aria label every variant gives a played chip, so one driver can rewind any of them. */
export const rewindLabel = (s: PlayedStep): string => (s.kind === 'act' ? `rewind to before seat ${s.seat}'s ${s.word}` : `rewind to seat ${s.seat}'s draw`)
