/**
 * The ledger: the range as hairline rows that open. Each row is a region of
 * the range with its share and its mix, printed as numbers; a row opens
 * into the rows beneath it (an inner class → the outer columns within it →
 * the hands in that cell that pot most), so the table and the path down to
 * a single hand are the same object. Nothing hides behind a hover.
 *
 * Pinning a grid cell opens the ledger at that row and column; an action in
 * the spot's key sorts the classes by that action's frequency, most first.
 */

import { Fragment } from 'react'

import type { Cell, Example, GridView } from '../engine/aggregate'
import type { Node } from '../engine/policy'
import { Hand, MixBar, actionWord, pct, pct0 } from './bars'
import type { GameUI } from './game'
import type { Pin } from './Grid'
import type { SortKey } from './SpotHead'

export function Ledger({
  game,
  view,
  node,
  sort,
  pin,
  onPin,
  examplesFor,
}: {
  game: GameUI
  view: GridView
  node: Node
  sort: SortKey
  pin: Pin | null
  onPin: (p: Pin | null) => void
  examplesFor: (row: number, col: number) => Example[]
}) {
  const INNER_ROWS = game.rows
  const OUTER_COLS = game.cols
  const keys = node === 'facing' ? (['f', 'c', 'p'] as const) : (['c', 'p'] as const)
  const order = INNER_ROWS.map((_, i) => i).filter((i) => view.rows[i].n > 0)
  if (sort) order.sort((a, b) => view.rows[b][sort] - view.rows[a][sort])
  const openRow = pin?.row ?? null

  const nums = (c: { f: number; c: number; p: number }) =>
    keys.map((k) => (
      <td key={k} className="r n">
        {pct0(c[k])}
      </td>
    ))

  return (
    <table className="hair rx-ledger">
      <thead>
        <tr>
          <th>Hand</th>
          <th className="r">Share</th>
          {keys.map((k) => (
            <th key={k} className={`r ${sort === k ? 'sorted' : ''}`}>
              {actionWord(k, node)}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {order.map((i) => {
          const r = view.rows[i]
          const open = openRow === i
          return (
            <Fragment key={i}>
              <tr className={`l1 ${open ? 'open' : ''}`}>
                <td className="k">
                  <button type="button" className="lrow" onClick={() => onPin(open ? null : { row: i, col: firstCol(view, i) })} aria-expanded={open} title={INNER_ROWS[i].long}>
                    <span className="tog" aria-hidden>
                      {open ? '▾' : '▸'}
                    </span>
                    {INNER_ROWS[i].label}
                  </button>
                  <MixBar mix={r} h={3} />
                </td>
                <td className="r n">{pct(r.share)}</td>
                {nums(r)}
              </tr>
              {open &&
                OUTER_COLS.map((c, j) => {
                  const cell: Cell = view.cells[i][j]
                  if (cell.n === 0) return null
                  const open2 = pin?.col === j
                  return (
                    <Fragment key={j}>
                      <tr className={`l2 ${open2 ? 'open' : ''}`}>
                        <td className="k">
                          <button type="button" className="lrow" onClick={() => onPin(open2 ? { row: i, col: -1 } : { row: i, col: j })} aria-expanded={open2} title={`with the board: ${c.long}`}>
                            <span className="tog" aria-hidden>
                              {open2 ? '▾' : '▸'}
                            </span>
                            {c.label}
                          </button>
                        </td>
                        <td className="r n">{pct(cell.share)}</td>
                        {nums(cell)}
                      </tr>
                      {open2 &&
                        examplesFor(i, j).map((ex, k) => (
                          <tr key={k} className="l3">
                            <td className="k">
                              <Hand game={game} cards={ex.cards} />
                              <small>
                                inner {pct0(ex.eqI)}
                                {Number.isNaN(ex.eqO) ? '' : ` · outer ${pct0(ex.eqO)}`}
                              </small>
                            </td>
                            <td className="r" />
                            {nums(ex.mix)}
                          </tr>
                        ))}
                    </Fragment>
                  )
                })}
            </Fragment>
          )
        })}
      </tbody>
    </table>
  )
}

/** The first non-empty column of a row: where the row opens when no column was chosen. */
function firstCol(view: GridView, i: number): number {
  const j = view.cells[i].findIndex((c) => c.n > 0)
  return j < 0 ? -1 : j
}
