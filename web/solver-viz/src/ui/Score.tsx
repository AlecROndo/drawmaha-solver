/**
 * The full game's hand so far, as a numbered list in the tool's left
 * column: the flop, what seat 0 did, your decision, your draw. A hand of
 * Drawmaha is two players acting in turn across streets, and the list reads
 * in that order, the pot after each step at its right.
 *
 * The hand on this page is one fixed shape — flop, the other seat acts, you
 * act, the draw — because that is the part of the game the range view can
 * show today; the streets after the draw wait on the network. Seat 0's step
 * is a toggle (bet pot ↔ check): it changes the node you face. Your two
 * steps move the cursor between the two decisions the page can draw. The
 * flop is the one text field on the page; a flop that does not parse is an
 * error printed under the field, and the range on screen stays the last good
 * one.
 */

import { useState } from 'react'

import type { Card } from '../engine/cards'
import { CardFace } from './bars'
import { FULL_GAME } from './game'

export type VillainAct = 'pot' | 'check'
export type Cursor = 'you' | 'draw'

export interface HandLine {
  board: Card[]
  villainAct: VillainAct
  cursor: Cursor
}

/** The pot in antes after each street, for a 1-ante bomb pot with a pot bet called. */
export function potAfterFlop(villainAct: VillainAct): number {
  // Two antes make the pot 2; a pot bet is 2 and its call another 2.
  return villainAct === 'pot' ? 6 : 2
}

export function Score({
  line,
  boardText,
  boardError,
  onBoard,
  onToggleVillain,
  onCursor,
}: {
  line: HandLine
  boardText: string
  boardError: string | null
  onBoard: (text: string) => void
  onToggleVillain: () => void
  onCursor: (c: Cursor) => void
}) {
  const [text, setText] = useState(boardText)
  const facing = line.villainAct === 'pot'
  const pot = potAfterFlop(line.villainAct)
  return (
    <div className="score">
      <div className="score-board">
        <span className="k">Flop</span>
        <span className="hand">
          {line.board.map((c) => (
            <CardFace key={c} game={FULL_GAME} card={c} />
          ))}
        </span>
        <label className="boardfield">
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            onBlur={() => onBoard(text)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') onBoard(text)
            }}
            spellCheck={false}
            aria-label="the flop, three cards like Ks 9s 4d"
          />
        </label>
      </div>
      {boardError && <p className="err">{boardError}</p>}
      <ol className="score-steps">
        <li>
          <span className="n">1</span>
          <button
            type="button"
            className={`act ${facing ? 'p' : 'k'}`}
            onClick={onToggleVillain}
            aria-label={`seat 0 ${facing ? 'bets pot' : 'checks'}; click to make them ${facing ? 'check' : 'bet'} instead`}
          >
            <b>Seat 0</b> {facing ? 'bets the pot' : 'checks'}
            <small>switch to {facing ? 'a check' : 'a pot bet'}</small>
          </button>
          <span className="pot">pot {facing ? 4 : 2}</span>
        </li>
        <li>
          <span className="n">2</span>
          <button type="button" className={`act ${line.cursor === 'you' ? 'you' : 'c'}`} onClick={() => onCursor('you')} aria-current={line.cursor === 'you' ? 'step' : undefined}>
            <b>You, seat 1</b> {line.cursor === 'you' ? 'to act' : facing ? 'call' : 'check'}
            <small>{facing ? 'fold, call 2 or raise to 8' : 'check or bet 2'}</small>
          </button>
          <span className="pot">pot {line.cursor === 'you' ? (facing ? 4 : 2) : pot}</span>
        </li>
        <li>
          <span className="n">3</span>
          <button type="button" className={`act ${line.cursor === 'draw' ? 'you' : 'k'}`} onClick={() => onCursor('draw')} aria-current={line.cursor === 'draw' ? 'step' : undefined}>
            <b>Your draw</b> {line.cursor === 'draw' ? 'to draw' : 'after the flop'}
            <small>stand pat, or throw one to five</small>
          </button>
          <span className="pot">pot {pot}</span>
        </li>
      </ol>
      <p className="legend-line">Seat 0 throws by its own mix. The turn, the river and the showdown wait on the network.</p>
    </div>
  )
}
