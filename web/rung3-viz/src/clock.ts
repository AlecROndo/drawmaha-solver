/**
 * The clock chapter's numbers: what the race's 2M checkpoints say about how
 * idle a sampled row is (`figures/rung3/visit_forensics.json`, written by
 * `scripts/minidrawmaha/checkpoint_forensics.py`) and the visit-clock DCFR
 * arm's grades (`figures/rung3/grades_visit_clock.json`, written by
 * `scripts/minidrawmaha/grade_visit_clock.py`), read from the copies under
 * `src/data/`. Chips per hand, as the files are.
 */

import { series, type Grade } from './research'

import forensics from './data/visit_forensics.json'
import visitGrades from './data/grades_visit_clock.json'

export interface Forensics {
  rows: number
  never_visited_by_2M: number
  never_visited_by_100k: number
  visited_by_10k: number
  visited_by_100k: number
  visited_by_2M: number
  first_visit_after_100k_given_visited: number
  age_quantiles_iters: Record<'p10' | 'p25' | 'p50' | 'p75' | 'p90' | 'p99', number>
  age_ge_1: number
  age_ge_10: number
  age_ge_50: number
  age_ge_1075: number
  visited_by_10k_reach_weighted: number
  age_quantiles_reach_weighted: Record<'p10' | 'p50' | 'p90', number>
}

export const FORENSICS = forensics as unknown as Forensics

/** Halvings that take a float64 regret to exactly zero: 2⁻¹⁰⁷⁵ underflows, so the 1,075th halving of 1 is 0. */
export const HALVINGS_TO_ZERO = 1075

/** DCFR's positive-side factor k^1.5 ÷ (k^1.5 + 1) at global tick k. */
export const positiveFactor = (k: number): number => k ** 1.5 / (k ** 1.5 + 1)

export type VisitColumn = 'linear' | 'quadratic'

export interface VisitPoint extends Grade {
  iteration: number
  column: VisitColumn
  hands: number
}

const VISIT_KEY = /^dcfr-visit\/(\d+)\/(linear|quadratic)$/

/** The visit-clock arm's grades at its four checkpoints, W = 20, both averages, in order. */
export const VISIT_CLOCK: VisitPoint[] = Object.entries(visitGrades as Record<string, Grade & { workers: number }>)
  .flatMap(([key, grade]) => {
    const m = VISIT_KEY.exec(key)
    if (!m) return []
    const iteration = Number(m[1])
    return [{ ...grade, iteration, column: m[2] as VisitColumn, hands: iteration * grade.workers }]
  })
  .sort((a, b) => a.iteration - b.iteration || a.column.localeCompare(b.column))

export const visitClockAt = (hands: number, column: VisitColumn): number => {
  const p = VISIT_CLOCK.find((v) => v.hands === hands && v.column === column)
  if (!p) throw new Error(`the visit-clock arm has no ${column} grade at ${hands} hands`)
  return p.exploitability
}

/** The race's own DCFR, on the global clock, at the same hands per seat. */
export const globalDcfrAt = (hands: number): number => {
  const p = series('dcfr').find((q) => q.hands === hands)
  if (!p) throw new Error(`the race has no DCFR grade at ${hands} hands`)
  return p.exploitability
}
