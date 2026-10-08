/**
 * The joint suit relabelling that turns a player's physical cards into the
 * key the frozen strategy is indexed by, and the draw order that depends on
 * it. A transcript of `cards.canonical` / `cards.canonical_relabelling` and
 * `game.canonical_picture` / `game.draw_order` in
 * `src/drawmaha_solver/minidrawmaha`, pinned against Python-generated
 * fixtures in `canonical.test.ts`.
 *
 * Terms. A *picture* is everything the acting player can see of their own
 * position: the *hole* (three private cards), the cards they have *thrown* at
 * the draw (none or one) and the *board* (one or two shared cards, in the
 * order dealt). A *relabelling* is a permutation of the three suits, read
 * `newSuit = relabelling[oldSuit]`. The *key* is the picture under the
 * relabelling whose image is lexicographically smallest, taken over the whole
 * picture at once — "jointly" is load-bearing: canonicalising the hole and the
 * board separately would forget whether they share a suit, which is exactly
 * what a flush is, and two different positions would collapse into one key.
 *
 * The board stays in dealt order inside the key (each board card is its own
 * one-card group in the search), because round 1 was played against the first
 * card alone and the player remembers which card that was. The hole and the
 * thrown cards are sorted, because the order they arrived in is not a fact
 * about the position.
 *
 * The *draw order* is the physical hole sorted by (rank, canonical suit); the
 * strategy's throw columns (low, mid, top) index it. Under physical suit
 * numbers two positions that differ by a suit swap would sort their
 * equal-ranked cards in opposite orders, and one ledger's "throw the lowest"
 * would name the board-suited card in one position and the offsuit card in the
 * other.
 */

import { MINI_SUITS, assertMiniCard, rankOf, suitOf, type MiniCard } from './cards'

export interface Picture {
  hole: MiniCard[]
  thrown: MiniCard[]
  board: MiniCard[]
}

export type Relabelling = [number, number, number]

/**
 * All six permutations of the suits in `itertools.permutations((0, 1, 2))`
 * order. The order is part of the contract, not a convenience: when two
 * relabellings produce the same image the FIRST wins (strict `<` below), and
 * the Python side breaks the same tie the same way, so the relabelling — and
 * with it the draw order — agrees across the two implementations.
 */
export const SUIT_RELABELLINGS: readonly Relabelling[] = [
  [0, 1, 2],
  [0, 2, 1],
  [1, 0, 2],
  [1, 2, 0],
  [2, 0, 1],
  [2, 1, 0],
]

const relabel = (c: MiniCard, relabelling: Relabelling): MiniCard =>
  rankOf(c) * MINI_SUITS.length + relabelling[suitOf(c)]

const ascending = (a: number, b: number): number => a - b

/** The groups `canonical` relabels: the hole, the thrown cards, then ONE group per board card. */
function groupsOf(p: Picture): MiniCard[][] {
  return [p.hole, p.thrown, ...p.board.map((c) => [c])]
}

function assertPicture(p: Picture): void {
  const seen = new Set<MiniCard>()
  for (const group of [p.hole, p.thrown, p.board]) {
    for (const c of group) {
      assertMiniCard(c)
      if (seen.has(c)) throw new Error(`card ${c} appears twice in the picture ${JSON.stringify(p)}`)
      seen.add(c)
    }
  }
}

/**
 * Lexicographic order on two images of the same picture. Both are the same
 * groups under different relabellings, so group lengths line up and a flat
 * element-wise walk is the same comparison Python makes between its tuples
 * of tuples — a card's integer order is its (rank, suit) order because
 * suit < 3.
 */
function less(a: readonly MiniCard[], b: readonly MiniCard[]): boolean {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return a[i] < b[i]
  }
  return false
}

/**
 * The picture under its canonical relabelling, and the relabelling itself.
 * `key.hole` and `key.thrown` are sorted ascending; `key.board` keeps dealt
 * order. Throws on a card outside 0..14 or a card that appears twice across
 * the three groups.
 */
export function canonicalPicture(p: Picture): { key: Picture; relabelling: Relabelling } {
  assertPicture(p)
  const groups = groupsOf(p)
  let bestFlat: MiniCard[] | null = null
  let bestGroups: MiniCard[][] = []
  let bestRelabelling: Relabelling = SUIT_RELABELLINGS[0]
  for (const relabelling of SUIT_RELABELLINGS) {
    const image = groups.map((g) => g.map((c) => relabel(c, relabelling)).sort(ascending))
    const flat = image.flat()
    if (bestFlat === null || less(flat, bestFlat)) {
      bestFlat = flat
      bestGroups = image
      bestRelabelling = relabelling
    }
  }
  const [hole, thrown, ...streets] = bestGroups
  return { key: { hole, thrown, board: streets.flat() }, relabelling: bestRelabelling }
}

/**
 * The actor's PHYSICAL hole cards in the order the throw actions index them:
 * by rank, ties broken by the canonical suit label of the whole picture.
 * Position 0 is `THROW_LOW`, 1 `THROW_MID`, 2 `THROW_TOP`.
 */
export function drawOrder(p: Picture): MiniCard[] {
  const { relabelling } = canonicalPicture(p)
  const under = (c: MiniCard): number => relabel(c, relabelling)
  return [...p.hole].sort((a, b) => under(a) - under(b))
}

/**
 * A canonical key as a compact, unambiguous string for use as a Map key:
 * hole cards joined by `,`, then `/`, the thrown cards, `|`, the board. Two
 * keys collide iff every group matches card for card.
 */
export function keyString(key: Picture): string {
  return `${key.hole.join(',')}/${key.thrown.join(',')}|${key.board.join(',')}`
}
