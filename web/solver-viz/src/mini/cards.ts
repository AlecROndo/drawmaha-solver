/**
 * Mini-drawmaha's 15-card deck as small integers: spelling, reading, and the
 * two coordinates every other module in `src/mini` takes a card apart into.
 *
 * Terms. A *mini card* is 0..14 = rank × 3 + suit, with ranks 0..4 standing
 * for 2 3 4 5 6 and suits 0..2 for clubs, diamonds, hearts — the rung-3 deal
 * pack's convention, and the one the Python exporter writes into every key
 * table and fixture. Ranks have no faces: rank 4 is the top of the deck and
 * rank 0 the floor, and they are not neighbours (no wheel, see `ranking.ts`).
 * Three suits is what makes trips possible and quads impossible. A *label* is
 * the two-letter spelling the Python suite and CLI use (`4h`); a *glyph* is
 * the same card drawn for a face (`4♥`).
 *
 * This module is a transcript of `src/drawmaha_solver/minidrawmaha/cards.py`'s
 * deck half; the suit relabelling lives in `canonical.ts`.
 */

/** A card as its index in the deck: rank × 3 + suit. */
export type MiniCard = number

export const MINI_RANKS = '23456'
export const MINI_SUITS = 'cdh'
export const MINI_SUIT_GLYPH: readonly string[] = ['♣', '♦', '♥']

export const MINI_DECK_SIZE = MINI_RANKS.length * MINI_SUITS.length

/** All fifteen cards, ordered by (rank, suit) so `MINI_DECK.slice(0, 3)` is the whole of the lowest rank. */
export const MINI_DECK: readonly MiniCard[] = Array.from({ length: MINI_DECK_SIZE }, (_, i) => i)

export const rankOf = (c: MiniCard): number => Math.floor(c / MINI_SUITS.length)
export const suitOf = (c: MiniCard): number => c % MINI_SUITS.length

/**
 * Throws unless `c` is an integer in 0..14. Every entry point in `src/mini`
 * runs its cards through this, because a card outside the deck would not
 * crash the arithmetic downstream — it would quietly become a 16th card with
 * a rank of 5 and land in some other hand's key.
 */
export function assertMiniCard(c: unknown): asserts c is MiniCard {
  if (typeof c !== 'number' || !Number.isInteger(c) || c < 0 || c >= MINI_DECK_SIZE) {
    throw new Error(`${String(c)} is not a card in the 15-card deck (0..14)`)
  }
}

/** `4h`: the spelling the Python suite, the CLI and the fixtures use. */
export function miniLabel(c: MiniCard): string {
  assertMiniCard(c)
  return MINI_RANKS[rankOf(c)] + MINI_SUITS[suitOf(c)]
}

/** `4♥`: the same card drawn for a face. */
export function miniGlyph(c: MiniCard): string {
  assertMiniCard(c)
  return MINI_RANKS[rankOf(c)] + MINI_SUIT_GLYPH[suitOf(c)]
}

/**
 * Read one card from its label (`4h`, any case) or its glyph form (`4♥`).
 * Throws an Error naming the token for anything the deck does not hold,
 * rather than inventing a card — a typo that silently parsed would pin the
 * wrong answer.
 */
export function parseMini(s: string): MiniCard {
  const t = s.trim()
  if (t.length !== 2) throw new Error(`${JSON.stringify(s)} is not a mini card (want a rank 2-6 and a suit c/d/h)`)
  const rank = MINI_RANKS.indexOf(t[0])
  const suit = Math.max(MINI_SUITS.indexOf(t[1].toLowerCase()), MINI_SUIT_GLYPH.indexOf(t[1]))
  if (rank < 0 || suit < 0) throw new Error(`${JSON.stringify(s)} is not a mini card (want a rank 2-6 and a suit c/d/h)`)
  return rank * MINI_SUITS.length + suit
}
