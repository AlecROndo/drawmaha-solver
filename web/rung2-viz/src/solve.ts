import data from './data/solve.json'

import type { Strategy } from './leduc'

/**
 * The committed solve this page renders — `leduc-analysis --json` output.
 *
 * Only the final average strategy ships: the page walks and rewinds one
 * betting line at a time, so it needs the answer, not the trajectory. The
 * per-checkpoint frames arrive with the playback PR, and the numbers here are
 * the same ones the Python suite pins against the referee fixture.
 */
export interface Solve {
  iterations: number
  exploitabilityAverage: number
  exploitabilityCurrent: number
  gameValue: number
  gameValueExact: number
  /** infoset label ("K:", "J:cc|Q:r", …) → mixed strategy over legal actions */
  strategy: Strategy
}

export const SOLVE = data as unknown as Solve
