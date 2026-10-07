/**
 * Cards as small integers: their spelling, their reading, and the deck with
 * a board taken out.
 *
 * Terms the engine uses throughout. A *card* is 0..51 = rank × 4 + suit, with
 * ranks 0..12 standing for 2..A and suits 0..3 for ♠ ♥ ♦ ♣. A *holding* is a
 * player's five hole cards. The *board* is the shared cards (three on the
 * flop, where this app lives). The *inner* hand is the holding read on its own
 * as a five-card poker hand; the *outer* hand is the best Omaha hand made of
 * exactly two hole cards and three board cards. Drawmaha splits the pot
 * between the two. A *region* is a cell of the composite grid, an inner class
 * crossed with an outer class (`regions.ts`). The *illustrative policy* is a
 * smooth stand-in for the untrained rung-4 solver (`policy.ts`), labelled as
 * such wherever it is drawn.
 */

export type Card = number

const RANK_CHARS = '23456789TJQKA'
const SUIT_CHARS = 'shdc'
const SUIT_GLYPHS: readonly string[] = ['♠', '♥', '♦', '♣']

export const rankOf = (c: Card): number => Math.floor(c / 4)
export const suitOf = (c: Card): number => c % 4

/**
 * Read one card: a rank (2–9, T or 10, J, Q, K, A; any case) followed by a
 * suit letter (s h d c; any case) or glyph (♠ ♥ ♦ ♣). `'Ks'`, `'K♠'`, `'ks'`,
 * `'10s'` all read as 47. Throws an Error naming the token otherwise.
 */
export function parseCard(s: string): Card {
  const t = s.trim()
  // '10' is the one two-character rank; everything else is one character.
  const rankChars = t.startsWith('10') ? 2 : 1
  const rankToken = t.slice(0, rankChars)
  const suitToken = t.slice(rankChars)
  const rank = rankToken === '10' ? 8 : rankToken.length === 1 ? RANK_CHARS.indexOf(rankToken.toUpperCase()) : -1
  const suit =
    suitToken.length === 1 ? Math.max(SUIT_CHARS.indexOf(suitToken.toLowerCase()), SUIT_GLYPHS.indexOf(suitToken)) : -1
  if (rank < 0 || suit < 0) {
    throw new Error(`"${s}" is not a card: expected a rank (2–9, T, J, Q, K, A) and a suit (s h d c or ♠ ♥ ♦ ♣)`)
  }
  return rank * 4 + suit
}

/**
 * Read a flop: exactly three distinct cards separated by spaces or commas.
 * Throws an Error naming the unreadable token, the card that appears twice,
 * or the count when it is not three.
 */
export function parseBoard(s: string): Card[] {
  const tokens = s.trim().split(/[\s,]+/).filter((t) => t.length > 0)
  if (tokens.length !== 3) throw new Error(`a board is 3 cards; "${s}" has ${tokens.length}`)
  const cards = tokens.map(parseCard)
  assertDistinct(cards, 'board')
  return cards
}

/** `'K♠'`: the rank letter and the suit glyph. */
export function cardLabel(c: Card): string {
  assertCard(c)
  return RANK_CHARS[rankOf(c)] + SUIT_GLYPHS[suitOf(c)]
}

/** Hearts and diamonds. */
export function isRed(c: Card): boolean {
  const suit = suitOf(c)
  return suit === 1 || suit === 2
}

/** The 52 − |board| cards a holding can be dealt from, in deck order. */
export function unseen(board: readonly Card[]): Card[] {
  assertDistinct(board, 'board')
  const onBoard = new Set(board)
  const rest: Card[] = []
  for (let c = 0; c < 52; c++) if (!onBoard.has(c)) rest.push(c)
  return rest
}

/** Throws unless `c` is an integer in 0..51. */
export function assertCard(c: Card): void {
  if (!Number.isInteger(c) || c < 0 || c > 51) throw new Error(`${c} is not a card (0..51)`)
}

/** Throws naming the first card that appears twice in `cards`, or any non-card. */
export function assertDistinct(cards: readonly Card[], what: string): void {
  for (let i = 0; i < cards.length; i++) {
    assertCard(cards[i])
    for (let j = 0; j < i; j++) {
      if (cards[i] === cards[j]) throw new Error(`${cardLabel(cards[i])} appears twice in the ${what}`)
    }
  }
}
