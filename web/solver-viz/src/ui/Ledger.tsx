/**
 * The ledger: hairline rows that open. Each row is a region of the range
 * with its share and its mix; a row opens into the rows beneath it (an
 * inner class → the outer columns within it → the hands in that cell), so
 * the view and the path down to a single hand are the same object. The
 * breadcrumb is the filter you have built by opening rows. Nothing hides
 * behind a hover: every number is printed.
 *
 * Opened from the header's action buttons (sorted by that action's
 * frequency) or by pinning a grid cell (opened at that row and column).
 */

import type { Cell, GridView } from '../engine/aggregate'
import type { Card } from '../engine/cards'
import type { Node } from '../engine/policy'
import { INNER_ROWS, OUTER_COLS } from '../engine/regions'
import { Hand, MixBar, actionWord, pct, pct0 } from './bars'
import type { Pin } from './Grid'
import type { SortKey } from './SpotHead'

export interface Example {
  cards: Card[]
  mix: { f: number; c: number; p: number }
  eqI: number
  eqO: number
}

export function Ledger({
  view,
  node,
  sort,
  pin,
  onPin,
  examplesFor,
}: {
  view: GridView
  node: Node
  sort: SortKey
  pin: Pin | null
  onPin: (p: Pin | null) => void
  examplesFor: (row: number, col: number) => Example[]
}) {
  const order = INNER_ROWS.map((_, i) => i).filter((i) => view.rows[i].n > 0)
  if (sort) order.sort((a, b) => view.rows[b][sort] - view.rows[a][sort])
  const openRow = pin?.row ?? null

  const nums = (c: Cell) => (
    <span className="nums">
      {node === 'facing' && (
        <>
          f <b>{pct0(c.f)}</b>
        </>
      )}{' '}
      {node === 'facing' ? 'c' : 'x'} <b>{pct0(c.c)}</b> p <b>{pct0(c.p)}</b>
    </span>
  )

  return (
    <div className="ledger">
      <div className="crumbs">
        <span>all</span>
        {openRow !== null && (
          <>
            <i>›</i>
            <span>{INNER_ROWS[openRow].long}</span>
          </>
        )}
        {pin && (
          <>
            <i>›</i>
            <span className="on">{OUTER_COLS[pin.col].long}</span>
          </>
        )}
        {sort && <em>sorted by {actionWord(sort, node)}</em>}
      </div>
      <div className="row head">
        <span />
        <span>region of the range</span>
        <span className="share">share</span>
        <span>mix</span>
        <span />
      </div>
      {order.map((i) => {
        const r = view.rows[i]
        const open = openRow === i
        return (
          <div key={i} className="group">
            <button type="button" className={`lrow l1 ${open ? 'open' : ''}`} onClick={() => onPin(open ? null : { row: i, col: firstCol(view, i) })} aria-expanded={open}>
              <span className="tog">{open ? '▾' : '▸'}</span>
              <span className="name">{INNER_ROWS[i].long}</span>
              <span className="share">
                <b>{pct(r.share)}</b>
              </span>
              <MixBar mix={r} />
              {nums(r)}
            </button>
            {open &&
              OUTER_COLS.map((c, j) => {
                const cell = view.cells[i][j]
                if (cell.n === 0) return null
                const open2 = pin?.col === j
                return (
                  <div key={j}>
                    <button type="button" className={`lrow l2 ${open2 ? 'open' : ''}`} onClick={() => onPin(open2 ? { row: i, col: -1 } : { row: i, col: j })} aria-expanded={open2}>
                      <span className="tog">{open2 ? '▾' : '▸'}</span>
                      <span className="name">outer: {c.long}</span>
                      <span className="share">
                        <b>{pct(cell.share)}</b>
                      </span>
                      <MixBar mix={cell} />
                      {nums(cell)}
                    </button>
                    {open2 &&
                      examplesFor(i, j).map((ex, k) => (
                        <div key={k} className="lrow l3">
                          <span className="tog" />
                          <span className="name">
                            <Hand cards={ex.cards} />
                            <small>
                              inner {pct0(ex.eqI)} · outer {pct0(ex.eqO)}
                            </small>
                          </span>
                          <span className="share" />
                          <MixBar mix={ex.mix} h={6} />
                          {nums({ ...ex.mix, share: 0, n: 1 })}
                        </div>
                      ))}
                  </div>
                )
              })}
          </div>
        )
      })}
    </div>
  )
}

/** The first non-empty column of a row: where the row opens when no column was chosen. */
function firstCol(view: GridView, i: number): number {
  const j = view.cells[i].findIndex((c) => c.n > 0)
  return j < 0 ? -1 : j
}
