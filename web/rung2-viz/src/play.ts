/**
 * Enough of Leduc to play one hand in the browser: the deck, the showdown
 * ladder, the settlement, and sampling the committed solve's mixed strategy.
 * A transcript of the payout half of `src/drawmaha_solver/leduc/game.py`,
 * pinned by `play.test.ts` against the same numbers the referee fixture pins
 * — the ±3 fold-to-a-re-raise included. The solver itself stays in Python.
 */

import { ANTE, BETS, type Act, type Rank } from './leduc'

export interface Card {
  rank: Rank
  /** two indistinguishable copies per rank; the suit only names the card */
  suit: 0 | 1
}

export const DECK: Card[] = (['J', 'Q', 'K'] as Rank[]).flatMap((rank) =>
  ([0, 1] as const).map((suit) => ({ rank, suit })),
)

const RANK_VALUE: Record<Rank, number> = { J: 0, Q: 1, K: 2 }
const PAIR_STRENGTH = 3

/** Pairing the board beats every unpaired card; otherwise rank order. */
export const strength = (privateRank: Rank, board: Rank): number =>
  privateRank === board ? PAIR_STRENGTH : RANK_VALUE[privateRank]

/** Chips (P0, P1) put in during one round; a folder stops where they stood. */
function contributions(line: string, size: number): [number, number] {
  const paid: [number, number] = [0, 0]
  let raises = 0
  for (let i = 0; i < line.length; i++) {
    if (line[i] === 'f') break
    if (line[i] === 'r') raises += 1
    paid[i % 2] = raises * size
  }
  return paid
}

/**
 * Chips to P0 at a finished hand (P1's are the negation).
 *
 * A fold pays the winner everything the folder had put in, ante included; a
 * showdown pays the loser's (equal) stake to the stronger hand, and equal
 * unpaired ranks push.
 */
export function settle(l1: string, board: Rank | null, l2: string, r0: Rank, r1: Rank): number {
  const one = contributions(l1, BETS[0])
  const two = board === null ? [0, 0] : contributions(l2, BETS[1])
  const paid: [number, number] = [ANTE + one[0] + two[0], ANTE + one[1] + two[1]]

  if (l1.includes('f') || l2.includes('f')) {
    // Seats alternate within each round, so the folder's seat is the fold's
    // index parity in its own round's line.
    const folder = (l1.includes('f') ? l1.indexOf('f') : l2.indexOf('f')) % 2
    return folder === 0 ? -paid[0] : paid[1]
  }

  if (board === null) throw new Error('a hand without a board can only end in a fold')
  const s0 = strength(r0, board)
  const s1 = strength(r1, board)
  if (s0 === s1) return 0
  return s0 > s1 ? paid[1] : -paid[0]
}

/**
 * One action drawn from the solve's mixed strategy at `key`, using the
 * supplied uniform draw — injected so tests replay exact hands. A spot the
 * export somehow lacks falls back to uniform over the legal actions.
 */
export function sampleAction(
  strategy: Record<string, Partial<Record<Act, number>>>,
  key: string,
  legalActs: Act[],
  uniform: () => number = Math.random,
): Act {
  const row = strategy[key]
  const u = uniform()
  if (!row) return legalActs[Math.min(Math.floor(u * legalActs.length), legalActs.length - 1)]
  let cumulative = 0
  for (const act of legalActs) {
    cumulative += row[act] ?? 0
    if (u < cumulative) return act
  }
  return legalActs[legalActs.length - 1]
}

/** A uniformly shuffled copy of the deck. */
export function shuffled(uniform: () => number = Math.random): Card[] {
  const cards = [...DECK]
  for (let i = cards.length - 1; i > 0; i--) {
    const j = Math.floor(uniform() * (i + 1))
    ;[cards[i], cards[j]] = [cards[j], cards[i]]
  }
  return cards
}
