/**
 * Mini-drawmaha's hand so far, as the layouts under `sofar/` read it: the
 * line — board card 1, the round-1 letters, the two throw counts, board
 * card 2, the round-2 letters — and the pure rules on it. `decisionOf` says
 * where the hand stands, `nextOptions` what the actor may do there, `advance`
 * the line after doing it, `toneOf` the colour an action's chip takes, and
 * `freeCards` what a board picker may offer. The layouts draw; this decides.
 *
 * The line is the public record only — what each seat did, how many cards
 * each threw, which board cards came — never anybody's cards; that is what
 * a range is conditioned on.
 */

import { type Bet, actorOf, betWord, isClosed, isFold, legalBets, replay } from '../mini/grammar'
import { MINI_DECK, type MiniCard } from '../mini/cards'

export interface MiniLine {
  board1: MiniCard
  /** round-1 letters so far: x check, c call, p pot, f fold */
  r1: string
  /** throw counts decided so far, seat 0 first */
  draws: (0 | 1)[]
  board2: MiniCard | null
  r2: string
}

/** Where the hand stands: whose decision, or which chance card is owed, or over. */
export type Decision =
  | { kind: 'bet'; round: 1 | 2; player: 0 | 1 }
  | { kind: 'draw'; player: 0 | 1 }
  | { kind: 'board2' }
  | { kind: 'over'; why: 'fold' | 'showdown' }

/** The decision the line is standing on, read off the grammar. */
export function decisionOf(line: MiniLine): Decision {
  const chips = replay(line.r2 ? [line.r1, line.r2] : [line.r1])
  if (isFold(line.r1)) return { kind: 'over', why: 'fold' }
  if (!isClosed(line.r1, chips.behind)) return { kind: 'bet', round: 1, player: actorOf(line.r1) }
  if (line.draws.length < 2) return { kind: 'draw', player: line.draws.length as 0 | 1 }
  if (line.board2 === null) return { kind: 'board2' }
  if (isFold(line.r2)) return { kind: 'over', why: 'fold' }
  // Both stacks empty after round 1 means round 2 never opens.
  if (chips.behind[0] === 0 && chips.behind[1] === 0) return { kind: 'over', why: 'showdown' }
  if (!isClosed(line.r2, chips.behind) || line.r2 === '') return { kind: 'bet', round: 2, player: actorOf(line.r2) }
  return { kind: 'over', why: 'showdown' }
}

export const linesOf = (line: MiniLine): string[] => (line.board2 === null ? [line.r1] : [line.r1, line.r2])

/** The options at a decision: a bet's letters, or the two throw counts. */
export function nextOptions(line: MiniLine): { key: string; word: string }[] {
  const d = decisionOf(line)
  if (d.kind === 'bet') {
    const lines = linesOf(line)
    const cur = d.round === 1 ? line.r1 : line.r2
    return legalBets(lines).map((b: Bet) => ({ key: b, word: betWord(b, cur) }))
  }
  if (d.kind === 'draw') return [{ key: '0', word: 'stand pat' }, { key: '1', word: 'throw one' }]
  return []
}

/** A chip's colour: pink for the action that pushes, the quiet ink for giving up, yellow for going along. */
export type Tone = 'p' | 'd' | 'c'

/**
 * The colour of an action's chip or button, by its key — one rule for the
 * played chips and the to-act buttons, so they agree. Pink is the action that
 * pushes: a pot bet or raise ('p') puts chips in, a throw ('1') changes the
 * cards. The quiet ink is giving up ('f'). Yellow is going along with the hand
 * as it stands: a check or call ('x', 'c') or standing pat ('0'). The map is
 * closed on purpose: a key it has never heard of — a new bet size, a
 * throw-two — is refused rather than quietly coloured as "going along".
 * Own keys only, so a name the object inherits ('constructor') is refused too.
 */
const TONES: Readonly<Record<string, Tone>> = { p: 'p', '1': 'p', f: 'd', x: 'c', c: 'c', '0': 'c' }

export function toneOf(key: string): Tone {
  if (!Object.hasOwn(TONES, key)) throw new Error(`no colour for the action key ${JSON.stringify(key)}`)
  return TONES[key]
}

/** The line after taking `key` at its current decision. */
export function advance(line: MiniLine, key: string): MiniLine {
  const d = decisionOf(line)
  if (d.kind === 'bet') {
    const bet = key as Bet
    const cur = d.round === 1 ? line.r1 : line.r2
    const sym = bet === 'c' ? (cur.endsWith('p') ? 'c' : 'x') : bet
    return d.round === 1 ? { ...line, r1: line.r1 + sym } : { ...line, r2: line.r2 + sym }
  }
  if (d.kind === 'draw') return { ...line, draws: [...line.draws, key === '1' ? 1 : 0] }
  throw new Error(`nothing to advance at a ${d.kind}`)
}

/** The cards not yet on the board, for the pickers. */
export const freeCards = (line: MiniLine, except: MiniCard | null): MiniCard[] =>
  MINI_DECK.filter((c) => c !== line.board1 && c !== line.board2 || c === except)
