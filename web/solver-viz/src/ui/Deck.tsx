/**
 * The deck as the filter, in the tool's left column: the live cards laid
 * out by rank and suit, the board cards shown and inert. Press a card to
 * hold it; the range narrows to the holdings that contain every held card,
 * a full hand is one hand, none is the range. Each card wears a small bar
 * for how over-represented it is among the holdings that bet the pot right
 * now, so the deck is also a map of what the strategy likes.
 */

import { suitKey } from './bars'
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
      <div className={`deck ${ranks.length <= 5 ? 'compact' : ''}`} style={{ gridTemplateColumns: `16px repeat(${ranks.length}, minmax(0, 1fr))` }}>
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
        The bar under a card is how over-represented it is among the holdings that bet the pot; a full bar is the most. {held.length} of {game.handSize} placed.
      </p>
    </div>
  )

  function DeckRow({ si }: { si: number }) {
    return (
      <>
        <div className={`sh su-${suitKey(suits[si].glyph)}`}>{suits[si].glyph}</div>
        {ranks.map((r, ri) => {
          const c = cardAt(ri, si)
          if (onBoard.has(c)) {
            return (
              <div key={c} className={`dcard cardface board su-${suitKey(suits[si].glyph)}`} role="img" aria-label={`${game.label(c)}, on the board`}>
                <b>{r}</b>
                <i aria-hidden="true">{suits[si].glyph}</i>
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
              className={`dcard cardface ${held ? 'held' : ''} su-${suitKey(suits[si].glyph)}`}
              disabled={full && !held}
              onClick={() => onToggle(c)}
              aria-pressed={held}
              aria-label={`${game.label(c)}${held ? ', held' : ''}`}
            >
              <b>{r}</b>
              <i aria-hidden="true">{suits[si].glyph}</i>
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
