/**
 * From a spot the user has walked to, to the public point the export indexes
 * it by.
 *
 * Terms. A *spot* is the position as the interface holds it: the board cards
 * dealt so far, the betting lines (one per round begun, in `x`/`c`/`p`/`f`
 * symbols), the public draw counts so far and who is to act. A *public point*
 * is the same thing with the cards reduced to a count (`data.ts`). The two
 * agree on every field a point has, so matching is equality on those fields
 * — and a spot that is not a decision (a fold has ended the hand, round 2 is
 * closed, the deck is to act, or it is the other seat's draw) matches nothing
 * and throws, which is how the interface learns to stop at the last decision.
 */

import type { MiniIndex, PointMeta } from './data'
import { miniLabel, type MiniCard } from './cards'

export interface MiniSpot {
  board: MiniCard[]
  lines: string[]
  /** the `DrawSignal.count`s present, P0's first */
  draws: number[]
  player: 0 | 1
}

/** A spot, spelled for an error message: `P1 on 4h 2c after xp|'' with draws 1,0`. */
export function describeSpot(spot: MiniSpot): string {
  const board = spot.board.map(miniLabel).join(' ')
  const lines = spot.lines.map((l) => (l === '' ? "''" : l)).join('|')
  const draws = spot.draws.length ? ` with draws ${spot.draws.join(',')}` : ''
  return `P${spot.player} on ${board || '(no board)'} after ${lines || '(no betting)'}${draws}`
}

const sameNumbers = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((x, i) => x === b[i])

const sameStrings = (a: readonly string[], b: readonly string[]): boolean =>
  a.length === b.length && a.every((x, i) => x === b[i])

/**
 * The one public point whose player, board-card count, draw counts and
 * betting lines are the spot's. Throws, naming the spot, when none matches
 * (the spot is not a decision) or more than one does (the export is broken).
 */
export function pointFor(index: MiniIndex, spot: MiniSpot): PointMeta {
  const hits = index.points.filter(
    (p) =>
      p.player === spot.player &&
      p.boardCards === spot.board.length &&
      sameNumbers(p.draws, spot.draws) &&
      sameStrings(p.lines, spot.lines),
  )
  if (hits.length === 1) return hits[0]
  if (hits.length === 0) throw new Error(`no public decision point for ${describeSpot(spot)}`)
  throw new Error(`${hits.length} public points match ${describeSpot(spot)}: ids ${hits.map((p) => p.id).join(', ')}`)
}
