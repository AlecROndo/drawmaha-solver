/**
 * The lead-up as a two-lane score: one lane per seat, one column per street,
 * the board along the top and the pot along the bottom. A hand of Drawmaha
 * is two players acting in turn across streets that have cards, and a draw
 * that has a size; a scorecard already has that shape, so you read it the
 * way you read a box score instead of decoding "P0 · check · P1 · check"
 * off a rail.
 *
 * The hand on this page is one fixed shape — flop, the other seat acts, you
 * act, the draw — because that is the part of the game the range view can
 * show today. Seat 0's flop chip is a toggle (bet pot ↔ check): it changes
 * the node you face. Your flop chip and your draw chip move the cursor
 * between the two decisions the page can draw.
 */

import { cardLabel, isRed, type Card } from '../engine/cards'
import { CardFace } from './bars'

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
  onToggleVillain,
  onCursor,
}: {
  line: HandLine
  onToggleVillain: () => void
  onCursor: (c: Cursor) => void
}) {
  const pot = potAfterFlop(line.villainAct)
  const facing = line.villainAct === 'pot'
  return (
    <div className="score" style={{ '--n': 5 } as React.CSSProperties}>
      <div className="h" />
      {['flop', 'draw', 'turn', 'river', 'showdown'].map((h) => (
        <div key={h} className="h st">
          {h}
        </div>
      ))}

      <div className="lane">board</div>
      <div className="st board">
        {line.board.map((c) => (
          <span key={c} className={`cf sm${isRed(c) ? ' red' : ''}`}>
            {cardLabel(c)}
          </span>
        ))}
      </div>
      <div className="st">
        <span className="none">—</span>
      </div>
      <div className="st board">
        <CardFace ghost />
      </div>
      <div className="st board">
        <CardFace ghost />
      </div>
      <div className="st">
        <span className="none">—</span>
      </div>

      <div className="lane">seat 0</div>
      <div className="st">
        <button
          type="button"
          className={`act ${facing ? 'p' : ''}`}
          onClick={onToggleVillain}
          aria-label={`seat 0 ${facing ? 'bets pot' : 'checks'}; click to make them ${facing ? 'check' : 'bet'} instead`}
        >
          {facing ? 'bet' : 'check'}
          {facing && <small>pot · 2</small>}
        </button>
      </div>
      <div className="st">
        <span className="act d">
          throws
          <small>decided by their mix</small>
        </span>
      </div>
      <div className="st">
        <span className="none">·</span>
      </div>
      <div className="st">
        <span className="none">·</span>
      </div>
      <div className="st">
        <span className="none">·</span>
      </div>

      <div className="lane">
        <b>you</b> · seat 1
      </div>
      <div className="st">
        <button
          type="button"
          className={`act ${line.cursor === 'you' ? 'you' : 'c'}`}
          onClick={() => onCursor('you')}
          aria-current={line.cursor === 'you' ? 'step' : undefined}
        >
          {line.cursor === 'you' ? 'to act' : facing ? 'call' : 'check'}
          <small>{line.cursor === 'you' ? (facing ? 'fold / call / pot 8' : 'check / pot 2') : facing ? '2' : ''}</small>
        </button>
      </div>
      <div className="st">
        <button
          type="button"
          className={`act ${line.cursor === 'draw' ? 'you' : 'd'}`}
          onClick={() => onCursor('draw')}
          aria-current={line.cursor === 'draw' ? 'step' : undefined}
        >
          {line.cursor === 'draw' ? 'to draw' : 'throw'}
          <small>{line.cursor === 'draw' ? 'stand pat … throw 5' : 'after the flop is settled'}</small>
        </button>
      </div>
      <div className="st">
        <span className="none">·</span>
      </div>
      <div className="st">
        <span className="none">·</span>
      </div>
      <div className="st">
        <span className="none">·</span>
      </div>

      <div className="lane">pot</div>
      <div className="st pot">
        <b>{facing ? `2 → ${pot}` : '2'}</b>
      </div>
      <div className="st pot">
        <b>{pot}</b>
      </div>
      <div className="st pot">
        <span className="none">·</span>
      </div>
      <div className="st pot">
        <span className="none">·</span>
      </div>
      <div className="st pot">
        <span className="none">·</span>
      </div>
    </div>
  )
}

/** The score collapsed to one line of notation, for the range window's title bar. */
export function OneLine({ line }: { line: HandLine }) {
  const facing = line.villainAct === 'pot'
  return (
    <span className="oneline">
      <span className="hand">
        {line.board.map((c) => (
          <CardFace key={c} card={c} />
        ))}
      </span>
      <span className={`pill ${facing ? 'p' : ''}`}>{facing ? 'bet 2' : 'check'}</span>
      {line.cursor === 'draw' && <span className={`pill c`}>{facing ? 'call 2' : 'check'}</span>}
      <span className="pill here">{line.cursor === 'draw' ? 'your draw' : 'you'}</span>
    </span>
  )
}
