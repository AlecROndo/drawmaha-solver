/**
 * The one shape every view of a range reads: a table of hands with their
 * weights, their region, their mix and their equities, whichever game and
 * whichever source produced them.
 *
 * Terms: a *hand* is one row — its cards, how much of the range it is
 * (`weight`, or an equal share when `weight` is null), the *region* it falls
 * in (an inner `row` crossed with an outer `col`, the axes' sizes in `rows`
 * and `cols`), its *mix* over the three betting actions (fold, check/call,
 * pot — a node without a fold has f = 0 everywhere), and its two *equities*
 * (inner, outer; outer null when the street cannot score it) plus the
 * single number the strength ribbon sorts by (`eq`). The full game builds
 * one of these from a sampled range and its illustrative policy
 * (`tableOf`); mini-drawmaha builds one from the exact reach-weighted
 * enumeration of the solver's own strategy. `aggregate.ts` reads nothing
 * else.
 */

import type { Policy } from './policy'
import { INNER_ROWS, OUTER_COLS } from './regions'
import type { RangeSample } from './sample'

export interface RangeTable {
  n: number
  rows: number
  cols: number
  /** cards per hand; `cards` holds `n * handSize` indices into a deck of `deckSize` */
  handSize: number
  deckSize: number
  cards: Uint8Array
  /** how much of the range each hand is; null means every hand counts equally */
  weight: Float32Array | null
  row: Uint8Array
  col: Uint8Array
  f: Float32Array
  c: Float32Array
  p: Float32Array
  /** the ribbon's default key: one number per hand in [0, 1] */
  eq: Float32Array
  eqI: Float32Array
  eqO: Float32Array | null
}

/** The full game's sample and its policy, seen as a table; the arrays are shared, not copied. */
export function tableOf(s: RangeSample, pol: Policy): RangeTable {
  if (pol.p.length !== s.n) throw new Error(`policy has ${pol.p.length} entries for a sample of ${s.n}`)
  return {
    n: s.n,
    rows: INNER_ROWS.length,
    cols: OUTER_COLS.length,
    handSize: 5,
    deckSize: 52,
    cards: s.cards,
    weight: null,
    row: s.row,
    col: s.col,
    f: pol.f,
    c: pol.c,
    p: pol.p,
    eq: pol.eq,
    eqI: s.innerEquity,
    eqO: s.outerEquity,
  }
}

/** The weight of hand `i`: its entry, or 1/n when the table is unweighted. */
export const weightOf = (t: RangeTable, i: number): number => (t.weight === null ? 1 / t.n : t.weight[i])

export function assertTable(t: RangeTable): void {
  const want = (name: string, len: number, expected: number) => {
    if (len !== expected) throw new Error(`${name} has ${len} entries for a table of ${t.n} hands (wanted ${expected})`)
  }
  want('cards', t.cards.length, t.n * t.handSize)
  want('row', t.row.length, t.n)
  want('col', t.col.length, t.n)
  want('f', t.f.length, t.n)
  want('c', t.c.length, t.n)
  want('p', t.p.length, t.n)
  want('eq', t.eq.length, t.n)
  want('eqI', t.eqI.length, t.n)
  if (t.eqO !== null) want('eqO', t.eqO.length, t.n)
  if (t.weight !== null) want('weight', t.weight.length, t.n)
}
