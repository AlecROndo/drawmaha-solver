/**
 * The two-half grid: rows are what the cards in hand make on their own (the
 * inner half), columns what two of them make with the board (the outer
 * half). A cell is one region of the range, drawn the way GTO Wizard draws
 * a hand class: tinted by the action most of the region takes, the tint
 * deeper the more of the range lives there, its share printed in the corner,
 * the action's word under it, and the region's whole mix as a thin bar along
 * its foot. The headers carry each row's and column's own share and mix.
 * Drawmaha's answer to the hold'em 13×13.
 *
 * The depth is sized against the view, not against the whole range. The
 * largest region in view takes the deepest tint and every other region the
 * depth a square of its area would have beside that one's: depth ∝
 * √(share / maxShare), so a region a quarter the size is half as deep, and a
 * region too small to draw honestly keeps a floor tint (`MARK.min` of
 * `MARK.max`). Sized against the whole range instead, a node whose weight
 * is spread over forty regions would tint forty cells alike and the eye
 * could not compare them. `maxShare` is taken over `view.cells`, which
 * `gridView` computes under the row/column mask, so a filtered-off region
 * has share 0 and never sets the maximum: narrowing the filter is a zoom,
 * the survivors deepen. Every node and every filter normalises on its own,
 * so a tint says nothing across spots, only across the cells beside it —
 * the percentage in the corner is the number that carries over.
 *
 * The filters that keep whole rows or columns are `Filters.tsx`, in the
 * tool's left column; here a filtered-off row or column is dimmed. The
 * pinned cell is the one the ledger is open at. Columns are made-hand ×
 * draw composites on purpose: on most flops of the full game the made-hand
 * axis alone is four columns wide and the draws are where the decisions are.
 */

import type { CSSProperties } from 'react'

import type { Cell, GridView } from '../engine/aggregate'
import type { Node } from '../engine/policy'
import { MixBar, actionWord, pct, pct0 } from './bars'
import type { GameUI } from './game'

export interface Pin {
  row: number
  col: number
}

/** The action most of a cell takes, for the small label in its corner. */
export const dominant = (c: Cell): 'f' | 'c' | 'p' => (c.p > c.c && c.p > c.f ? 'p' : c.f > c.c ? 'f' : 'c')

/**
 * The scale the depth law is written in: the side of a square, in px, with
 * the largest region's square `max` on a side and the floor `min`. A cell's
 * tint depth is `sideOf(share, maxShare) / MARK.max`, between `min / max`
 * (9%, so a <0.1% region is still a visible tint and not an empty cell) and
 * 1. The numbers are the square this grid drew before its cells were
 * tinted; the law, not the pixels, is what they carry now.
 */
export const MARK = { max: 56, min: 5 } as const

/**
 * The side of a region's square, in px, from its share and the largest share
 * in view — the grid's tint depth times `MARK.max`: `max` at the largest region, `min` at nothing, and in between
 * side ∝ √(share / maxShare), so the *area* is the honest comparison — a
 * region with a quarter of the largest share gets half its side. A share at
 * or above `maxShare`, or a `maxShare` of zero, is clamped to `max` rather
 * than left to divide by nothing.
 *
 * The grid reads it as a fraction of `max`: the depth of a cell's tint.
 */
export function sideOf(share: number, maxShare: number, max: number = MARK.max, min: number = MARK.min): number {
  if (!(share > 0)) return min
  if (!(maxShare > 0) || share >= maxShare) return max
  return Math.min(max, Math.max(min, Math.sqrt(share / maxShare) * max))
}

/**
 * The largest share in view, the `maxShare` every cell's `sideOf` is sized
 * against. It is taken over the cells `gridView` returns under the mask, so a
 * masked-off region (share 0) never sets it and narrowing the filter is a
 * zoom; the test pins that through this same function rather than a copy.
 */
export const maxShareOf = (view: GridView): number => Math.max(0, ...view.cells.flat().map((c) => c.share))

/** The depth of a region's tint, 0.09..1: `sideOf` as a fraction of the largest region's side. */
export const depthOf = (share: number, maxShare: number): number => sideOf(share, maxShare) / MARK.max

/** The tint a cell takes: its dominant action's colour mixed into the paper, 6% at the floor to 30% at the deepest. */
export const tintOf = (dom: 'f' | 'c' | 'p', depth: number): string => `color-mix(in srgb, var(--rx-${dom}) ${(6 + 24 * depth).toFixed(1)}%, var(--paper))`

export function Grid({
  game,
  view,
  node,
  rows,
  cols,
  pin,
  onPin,
}: {
  game: GameUI
  view: GridView
  node: Node
  /** the inner rows the filter keeps, or null for all; the rest are dimmed */
  rows: ReadonlySet<number> | null
  /** the outer columns the filter keeps, or null for all */
  cols: ReadonlySet<number> | null
  pin: Pin | null
  onPin: (p: Pin | null) => void
}) {
  const INNER_ROWS = game.rows
  const OUTER_COLS = game.cols
  const maxShare = maxShareOf(view)
  const keys = node === 'facing' ? (['f', 'c', 'p'] as const) : (['c', 'p'] as const)
  const nums = (m: Cell) => keys.map((k) => `${actionWord(k, node)} ${Math.round(m[k] * 100)}`).join(' · ')
  const inHand = game.handSize === 3 ? 'three' : 'five'

  return (
    <div className="gridblock">
      <div className="grid2" style={{ gridTemplateColumns: `144px repeat(${OUTER_COLS.length}, minmax(0, 1fr))` }}>
        <div className="corner">inner ↓ · outer →</div>
        {OUTER_COLS.map((c, j) => (
          <div key={c.label} className={`ch ${cols && !cols.has(j) ? 'off' : ''}`} title={c.long}>
            <b>{c.label}</b>
            <span className="sh">{pct(view.cols[j].share)}</span>
            <MixBar mix={view.cols[j]} h={3} />
          </div>
        ))}
        {INNER_ROWS.map((r, i) => (
          <RowOf key={r.label} i={i} />
        ))}
      </div>
      <p className="legend-line">
        Rows: what the {inHand} in hand make. Columns: what two of them make with the board. A cell’s tint is the action most of its region takes,
        deeper for a larger region; the bar along its foot is the region’s whole mix.
      </p>
    </div>
  )

  function RowOf({ i }: { i: number }) {
    const r = INNER_ROWS[i]
    const off = rows && !rows.has(i)
    return (
      <>
        <div className={`rh ${off ? 'off' : ''}`} title={r.long}>
          <span className="rh-top">
            <b>{r.label}</b>
            <span className="sh">{pct(view.rows[i].share)}</span>
          </span>
          <MixBar mix={view.rows[i]} h={4} />
          <span className="rh-nums">{view.rows[i].n > 0 ? nums(view.rows[i]) : 'none in view'}</span>
        </div>
        {OUTER_COLS.map((c, j) => {
          const cell = view.cells[i][j]
          const pinned = pin?.row === i && pin?.col === j
          if (cell.n === 0) return <div key={c.label} className="cell empty" aria-hidden />
          const dom = dominant(cell)
          const style = { background: tintOf(dom, depthOf(cell.share, maxShare)) } as CSSProperties
          return (
            <button
              key={c.label}
              type="button"
              className={`cell ${pinned ? 'pin' : ''} ${off || (cols && !cols.has(j)) ? 'off' : ''}`}
              style={style}
              onClick={() => onPin(pinned ? null : { row: i, col: j })}
              aria-pressed={pinned}
              aria-label={`${r.long} × ${c.long}: ${pct(cell.share)} of the range; ${keys.map((k) => `${actionWord(k, node)} ${pct0(cell[k])}`).join(', ')}`}
            >
              <span className="pct">{cell.share >= 0.001 ? pct(cell.share) : '<0.1%'}</span>
              <span className={`dom ${dom}`}>{actionWord(dom, node)}</span>
              <MixBar mix={cell} h={3} className="cellmix" />
            </button>
          )
        })}
      </>
    )
  }
}
