/**
 * Which layout draws the hand so far. The path (`PathScore`) is the default;
 * the strip (`StripScore`) is kept as the backup, one constant away. Either
 * can be previewed without a rebuild by `?score=path` or `?score=strip` in
 * the query string; an unknown value falls back to the default.
 */

import { PathScore } from './PathScore'
import { StripScore } from './StripScore'

export type ScoreLayout = 'path' | 'strip'

export const DEFAULT_LAYOUT: ScoreLayout = 'path'

const LAYOUTS = { path: PathScore, strip: StripScore } as const

export const isScoreLayout = (s: string | null): s is ScoreLayout => s === 'path' || s === 'strip'

/** The layout named by the page's query string, else the default. */
export function scoreLayout(search: string): ScoreLayout {
  const q = new URLSearchParams(search).get('score')
  return isScoreLayout(q) ? q : DEFAULT_LAYOUT
}

export const HandScore = LAYOUTS[scoreLayout(typeof window === 'undefined' ? '' : window.location.search)]
