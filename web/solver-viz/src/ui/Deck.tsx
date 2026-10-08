/**
 * The deck as the filter: the 49 live cards laid out by rank and suit, the
 * board cards face up and inert. Tap a card to hold it; the range narrows
 * to the holdings that contain every held card, five cards is a hand, none
 * is the range. Each card wears a small bar for how over-represented it is
 * in the raising range right now, so the deck is also a map of what the
 * policy likes. Opened from the header's cards button.
 */

import type { GameUI } from './game'

export function Deck({
  game,
  board,
  held,
  potLift,
  onToggle,
}: {
  game: GameUI
  board: readonly number[]
  held: readonly number[]
  /** one entry per card of the deck, 1 = neutral; from deckStats */
  potLift: Float32Array
  onToggle: (c: number) => void
}) {
  const { ranks, suits, cardAt } = game.deck
  const live: number[] = []
  for (let r = 0; r < ranks.length; r++) for (let s = 0; s < suits.length; s++) live.push(cardAt(r, s))
  const onBoard = new Set(board)
  const isHeld = new Set(held)
  const full = held.length >= game.handSize
  // The bar is relative to the deck's own spread: the most over-represented live card fills it.
  const lifts = live.filter((c) => !onBoard.has(c) && Number.isFinite(potLift[c]) && potLift[c] > 0).map((c) => potLift[c])
  const lo = Math.min(...lifts)
  const hi = Math.max(...lifts)
  return (
    <div className="deckwrap">
      <div className={`deck ${ranks.length <= 5 ? 'compact' : ''}`} style={{ gridTemplateColumns: `28px repeat(${ranks.length}, 1fr)` }}>
        <div />
        {ranks.map((r) => (
          <div key={r} className="rh">
            {r}
          </div>
        ))}
        {suits.map((s, si) => (
          <DeckRow key={s.glyph} si={si} />
        ))}
      </div>
      <p className="legend-line">
        bar under a card: how over-represented it is among the holdings that raise, relative to the other live cards (full bar = the most) · tap to hold · {held.length} of {game.handSize} placed
      </p>
    </div>
  )

  function DeckRow({ si }: { si: number }) {
    return (
      <>
        <div className={`sh ${suits[si].red ? 'red' : ''}`}>{suits[si].glyph}</div>
        {ranks.map((r, ri) => {
          const c = cardAt(ri, si)
          if (onBoard.has(c)) {
            return (
              <div key={c} className={`dcard board ${game.isRed(c) ? 'red' : ''}`} aria-label={`${game.label(c)}, on the board`}>
                {game.label(c)}
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
              className={`dcard ${held ? 'held' : ''} ${game.isRed(c) ? 'red' : ''}`}
              disabled={full && !held}
              onClick={() => onToggle(c)}
              aria-pressed={held}
              aria-label={`${game.label(c)}${held ? ', held' : ''}`}
            >
              {r}
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
