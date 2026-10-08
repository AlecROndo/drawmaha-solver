/**
 * A filter on the range — chosen rows, chosen columns, cards that must be
 * held — compiled to a 0/1 mask over the sample.
 *
 * Terms: a *holding* is a sampled hand of five hole cards; a *region* is an
 * inner row or outer column of the grid (`regions.ts`); the *inner* hand is
 * the holding as a poker hand, the *outer* its best Omaha hand on the board.
 * A hand passes the filter when its row is among the chosen rows (or none
 * are chosen), its column among the chosen columns (or none), and it holds
 * every held card. Every aggregate (`aggregate.ts`) takes the mask, and the
 * *illustrative policy* is averaged over the hands that pass.
 */

import { type Card, assertDistinct } from './cards'
import type { RangeSample } from './sample'

export interface Filter {
  rows: ReadonlySet<number> | null
  cols: ReadonlySet<number> | null
  held: readonly Card[]
}

export const EMPTY_FILTER: Filter = Object.freeze({ rows: null, cols: null, held: [] })

/**
 * The mask of hands passing `f`, or null when `f` restricts nothing (rows
 * and cols null, no held card) so callers can skip the per-hand test.
 */
export function maskFor(s: RangeSample, f: Filter): Uint8Array | null {
  if (f.rows === null && f.cols === null && f.held.length === 0) return null
  assertDistinct(f.held, 'held cards')
  const mask = new Uint8Array(s.n)
  for (let i = 0; i < s.n; i++) {
    if (f.rows !== null && !f.rows.has(s.row[i])) continue
    if (f.cols !== null && !f.cols.has(s.col[i])) continue
    let holdsAll = true
    for (const c of f.held) {
      const base = i * 5
      if (s.cards[base] !== c && s.cards[base + 1] !== c && s.cards[base + 2] !== c && s.cards[base + 3] !== c && s.cards[base + 4] !== c) {
        holdsAll = false
        break
      }
    }
    if (holdsAll) mask[i] = 1
  }
  return mask
}
