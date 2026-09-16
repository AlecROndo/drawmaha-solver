/**
 * The timeline's state: the whole explored line (PATH) and where along it the
 * reader is looking (CUR). Rewinding moves CUR only — walking back and then
 * forward again returns to the same stations instead of deleting them. Taking
 * a DIFFERENT action at the cursor is the one genuine fork, and only then is
 * the tail dropped, because those stations no longer exist on the line being
 * looked at.
 */

import { closed, type Act, type Rank } from './leduc'

export type Step = { k: 'a'; v: Act } | { k: 'b'; v: Rank }

export interface Line {
  l1: string
  board: Rank | null
  l2: string
}

export interface Walk {
  path: Step[]
  cur: number
}

/** The (round-1 line, board, round-2 line) reached after `n` steps. */
export function stateAt(path: Step[], n: number): Line {
  let l1 = ''
  let board: Rank | null = null
  let l2 = ''
  for (const step of path.slice(0, n)) {
    if (step.k === 'b') board = step.v
    else if (board === null) l1 += step.v
    else l2 += step.v
  }
  return { l1, board, l2 }
}

export const jump = (walk: Walk, n: number): Walk => ({ path: walk.path, cur: n })

/** Take `step` from where we are looking: continue the line, or fork it. */
export function advance(walk: Walk, step: Step): Walk {
  const onPath = walk.path[walk.cur]
  const path =
    onPath && onPath.k === step.k && onPath.v === step.v
      ? walk.path
      : [...walk.path.slice(0, walk.cur), step]
  return { path, cur: walk.cur + 1 }
}

/**
 * Whether round 1 has closed with the board still to come. A fold also
 * closes the line, but it ends the hand — there is no board after it.
 */
export const boardPending = (s: Line): boolean =>
  s.board === null && closed(s.l1) && !s.l1.endsWith('f')

/** Whether the hand has ended on this line: a round-1 fold, or a closed round 2. */
export const lineOver = (s: Line): boolean =>
  s.board === null ? s.l1.endsWith('f') : closed(s.l2)

/** Whether the hand has ended by step `n`. */
export const handOver = (path: Step[], n: number): boolean => lineOver(stateAt(path, n))
