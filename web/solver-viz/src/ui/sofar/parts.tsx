/**
 * The parts both layouts of the hand so far share — the prop contract
 * `MiniRange` renders either layout with, the board picker's button and fan,
 * the option buttons at the head, and the hint line — so the path and the
 * strip are two layouts of one vocabulary with the same aria labels, and one
 * driver can play through either.
 */

import type { MiniCard } from '../../mini/cards'
import { CardFace } from '../bars'
import type { GameUI } from '../game'
import { type Decision, type MiniLine, advance, freeCards } from '../MiniScore'
import { type Head, pct } from './steps'

export interface ScoreProps {
  game: GameUI
  line: MiniLine
  onLine: (l: MiniLine) => void
  picking: 1 | 2 | null
  onPick: (which: 1 | 2 | null) => void
  shares: Record<string, number> | null
}

/** A board card as a button that opens its picker; "pick" while the card is owed. */
export function BoardPick({ game, which, card, picking, onPick, className = '' }: { game: GameUI; which: 1 | 2; card: MiniCard | null; picking: 1 | 2 | null; onPick: (w: 1 | 2 | null) => void; className?: string }) {
  const open = picking === which
  return (
    <button type="button" className={`bp ${open ? 'open' : ''} ${card === null ? 'owed' : ''} ${className}`} onClick={() => onPick(open ? null : which)} aria-expanded={open} aria-label={`choose board card ${which}`}>
      {card === null ? <span className="pickword">pick</span> : <CardFace game={game} card={card} />}
    </button>
  )
}

/** The fifteen cards fanned out under the score while a picker is open. */
export function Picker({ game, line, onLine, picking, onPick }: ScoreProps) {
  if (picking === null) return null
  const current = picking === 1 ? line.board1 : line.board2
  return (
    <div className="fan" role="listbox" aria-label={`board card ${picking}`}>
      <span className="fanlab">board card {picking}</span>
      {freeCards(line, current).map((c) => (
        <button
          key={c}
          type="button"
          role="option"
          aria-selected={current === c}
          className={`pk ${current === c ? 'on' : ''}`}
          onClick={() => {
            // A new board card 1 restarts the hand; a new board card 2 restarts round 2.
            onLine(picking === 1 ? { board1: c, r1: '', draws: [], board2: null, r2: '' } : { ...line, board2: c, r2: '' })
            onPick(null)
          }}
        >
          <CardFace game={game} card={c} />
        </button>
      ))}
    </div>
  )
}

/** The options at the head, each with the share of the actor's range that takes it. */
export function Opts({ head, shares, line, onLine, className = '' }: { head: Head; shares: Record<string, number> | null; line: MiniLine; onLine: (l: MiniLine) => void; className?: string }) {
  return (
    <span className={`opts ${className}`}>
      {head.options.map((o) => {
        const p = pct(shares, o.key)
        return (
          <button key={o.key} type="button" className={`opt ${o.tone}`} onClick={() => onLine(advance(line, o.key))} aria-label={`play ${o.word}`}>
            <span className="w">{o.word}</span>
            {p && <b>{p}</b>}
          </button>
        )
      })}
    </span>
  )
}

/** The one-line hint under every variant, by the hand's state. */
export function Hint({ d, extra }: { d: Decision; extra?: string }) {
  const text =
    d.kind === 'over'
      ? 'the hand is over · click a played step to rewind to it'
      : d.kind === 'board2'
        ? 'pick board card 2 to open round 2'
        : `click an action to play it (the number is how much of that seat’s range takes it) · click a played step to rewind · click a board card to change it${extra ? ` · ${extra}` : ''}`
  return <p className="legend-line hint">{text}</p>
}
