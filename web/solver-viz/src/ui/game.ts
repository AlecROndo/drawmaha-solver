/**
 * What the shared views need to know about the game they are drawing: the
 * two axes of the grid, how many cards a hand holds, and how a card index
 * is spelled and laid out in a deck.
 *
 * Terms: the *rows* are the inner-hand regions and the *cols* the outer-hand
 * regions (`engine/regions.ts` for the full game, `mini/regions.ts` for
 * mini-drawmaha); a *deck layout* is the rank order (high first) and the
 * suit order the deck sheet draws, with `cardAt(rank, suit)` the index the
 * engine uses for that card. The components take one of these and nothing
 * else about the game, so the same grid, ledger, deck and draw board draw
 * a 52-card illustrative range and a 15-card solved one.
 */

import { cardLabel, isRed, type Card } from '../engine/cards'
import { INNER_ROWS, OUTER_COLS, type Region } from '../engine/regions'

export interface DeckLayout {
  /** rank labels, high first, as the deck sheet's columns */
  ranks: readonly string[]
  /** suit glyphs as the deck sheet's rows, with whether each draws red */
  suits: readonly { glyph: string; red: boolean }[]
  /** the engine's card index for the rank at `rankIndex` (into `ranks`) and the suit at `suitIndex` */
  cardAt(rankIndex: number, suitIndex: number): number
}

export interface GameUI {
  rows: readonly Region[]
  cols: readonly Region[]
  handSize: number
  deck: DeckLayout
  label(card: number): string
  isRed(card: number): boolean
}

const FULL_RANKS = 'A K Q J T 9 8 7 6 5 4 3 2'.split(' ')

/** The full game: 52 cards, nine rows by eight columns, five to a hand. */
export const FULL_GAME: GameUI = {
  rows: INNER_ROWS,
  cols: OUTER_COLS,
  handSize: 5,
  deck: {
    ranks: FULL_RANKS,
    suits: [
      { glyph: '♠', red: false },
      { glyph: '♥', red: true },
      { glyph: '♦', red: true },
      { glyph: '♣', red: false },
    ],
    // rank index 0 is the ace = engine rank 12; suits are ♠ ♥ ♦ ♣ = 0..3
    cardAt: (rankIndex, suitIndex) => (12 - rankIndex) * 4 + suitIndex,
  },
  label: (c) => cardLabel(c as Card),
  isRed: (c) => isRed(c as Card),
}
