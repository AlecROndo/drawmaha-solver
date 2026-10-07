/**
 * The deck as the filter: the 49 live cards laid out by rank and suit, the
 * board cards face up and inert. Tap a card to hold it; the range narrows
 * to the holdings that contain every held card, five cards is a hand, none
 * is the range. Each card wears a small bar for how over-represented it is
 * in the raising range right now, so the deck is also a map of what the
 * policy likes. Opened from the header's cards button.
 */

import { cardLabel, isRed, type Card } from '../engine/cards'

const RANKS_HIGH_FIRST = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
const SUITS = ['♠', '♥', '♦', '♣']

export function Deck({
  board,
  held,
  potLift,
  onToggle,
}: {
  board: readonly Card[]
  held: readonly Card[]
  /** 52 entries, 1 = neutral; from deckStats */
  potLift: Float32Array
  onToggle: (c: Card) => void
}) {
  const onBoard = new Set(board)
  const isHeld = new Set(held)
  const full = held.length >= 5
  // The bar is relative to the deck's own spread: the most over-represented live card fills it.
  const lifts = Array.from(potLift).filter((_, c) => !onBoard.has(c) && Number.isFinite(potLift[c]) && potLift[c] > 0)
  const lo = Math.min(...lifts)
  const hi = Math.max(...lifts)
  return (
    <div className="deckwrap">
      <div className="deck">
        <div />
        {RANKS_HIGH_FIRST.map((r) => (
          <div key={r} className="rh">
            {'23456789TJQKA'[r]}
          </div>
        ))}
        {SUITS.map((s, si) => (
          <DeckRow key={s} si={si} />
        ))}
      </div>
      <p className="legend-line">
        bar under a card: how over-represented it is among the holdings that raise, relative to the other live cards (full bar = the most) · tap to hold · {held.length} of 5 placed
      </p>
    </div>
  )

  function DeckRow({ si }: { si: number }) {
    return (
      <>
        <div className={`sh ${si === 1 || si === 2 ? 'red' : ''}`}>{SUITS[si]}</div>
        {RANKS_HIGH_FIRST.map((r) => {
          const c = r * 4 + si
          if (onBoard.has(c)) {
            return (
              <div key={c} className={`dcard board ${isRed(c) ? 'red' : ''}`} aria-label={`${cardLabel(c)}, on the board`}>
                {cardLabel(c)}
              </div>
            )
          }
          const lift = potLift[c]
          const w = hi > lo ? Math.max(0, Math.min(1, (lift - lo) / (hi - lo))) : 0
          const held = isHeld.has(c)
          return (
            <button
              key={c}
              type="button"
              className={`dcard ${held ? 'held' : ''} ${isRed(c) ? 'red' : ''}`}
              disabled={full && !held}
              onClick={() => onToggle(c)}
              aria-pressed={held}
              aria-label={`${cardLabel(c)}${held ? ', held' : ''}`}
            >
              {'23456789TJQKA'[r]}
              <span className="bar" aria-hidden>
                <i style={{ width: `${(w * 100).toFixed(0)}%` }} />
              </span>
            </button>
          )
        })}
      </>
    )
  }
}
