/**
 * The two-half grid: rows are what the five cards in hand make on their own
 * (the inner half), columns what two of them make with the board (the outer
 * half). A cell is one region of the range: its square's area is how much
 * of the range lives there, the square's bands are what that region does,
 * and the margins carry each row's and column's own mix. Drawmaha's answer
 * to the hold'em 13×13.
 *
 * The filter chips above the grid keep whole rows or columns; the pinned
 * cell is the one the ledger is open at. Columns are made-hand × draw
 * composites on purpose: on most flops the made-hand axis alone is four
 * columns wide and the draws are where the decisions are.
 */

import type { Cell, GridView } from '../engine/aggregate'
import type { Node } from '../engine/policy'
import { MixBar, pct } from './bars'
import type { GameUI } from './game'

export interface Pin {
  row: number
  col: number
}

/** The action most of a cell takes, for the small label in its corner. */
export const dominant = (c: Cell): 'f' | 'c' | 'p' => (c.p > c.c && c.p > c.f ? 'p' : c.f > c.c ? 'f' : 'c')

export function Grid({
  game,
  view,
  node,
  rows,
  cols,
  pin,
  onRows,
  onCols,
  onPin,
}: {
  game: GameUI
  view: GridView
  node: Node
  rows: ReadonlySet<number> | null
  cols: ReadonlySet<number> | null
  pin: Pin | null
  onRows: (r: ReadonlySet<number> | null) => void
  onCols: (c: ReadonlySet<number> | null) => void
  onPin: (p: Pin | null) => void
}) {
  const INNER_ROWS = game.rows
  const OUTER_COLS = game.cols
  const maxShare = Math.max(1e-9, ...view.cells.flat().map((c) => c.share))
  const toggle = (set: ReadonlySet<number> | null, i: number, size: number): ReadonlySet<number> | null => {
    const next = new Set(set ?? [])
    if (next.has(i)) next.delete(i)
    else next.add(i)
    return next.size === 0 || next.size === size ? null : next
  }
  const word = (k: 'f' | 'c' | 'p') => (k === 'f' ? 'fold' : k === 'c' ? (node === 'facing' ? 'call' : 'check') : 'pot')
  const inHand = game.handSize === 3 ? 'three' : 'five'

  return (
    <div className="gridblock">
      <div className="filters">
        <div>
          <div className="lbl">inner · the {inHand} in hand</div>
          <div className="seg">
            <button type="button" className={rows === null ? 'on' : ''} onClick={() => onRows(null)}>
              all
            </button>
            {INNER_ROWS.map((r, i) => (
              <button key={r.label} type="button" className={rows?.has(i) ? 'on' : ''} onClick={() => onRows(toggle(rows, i, INNER_ROWS.length))}>
                {r.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="lbl">outer · with the board</div>
          <div className="seg">
            <button type="button" className={cols === null ? 'on' : ''} onClick={() => onCols(null)}>
              all
            </button>
            {OUTER_COLS.map((c, j) => (
              <button key={c.label} type="button" className={cols?.has(j) ? 'on' : ''} onClick={() => onCols(toggle(cols, j, OUTER_COLS.length))}>
                {c.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid2" style={{ gridTemplateColumns: `150px repeat(${OUTER_COLS.length}, minmax(0, 1fr))` }}>
        <div className="corner">inner ↓ · outer →</div>
        {OUTER_COLS.map((c, j) => (
          <div key={c.label} className={`ch ${cols && !cols.has(j) ? 'off' : ''}`} title={c.long}>
            <b>{c.label}</b>
            <MixBar mix={view.cols[j]} h={4} />
            <span className="sh">{pct(view.cols[j].share)}</span>
          </div>
        ))}
        {INNER_ROWS.map((r, i) => (
          <RowOf key={r.label} i={i} />
        ))}
      </div>
      <p className="legend-line">
        rows: the {inHand} in hand · columns: two of them with the board · a square's area is the region's share of the whole range · its bands are the mix ·
        corner word: what most of the region does
      </p>
    </div>
  )

  function RowOf({ i }: { i: number }) {
    const r = INNER_ROWS[i]
    const off = rows && !rows.has(i)
    return (
      <>
        <div className={`rh ${off ? 'off' : ''}`} title={r.long}>
          <b>{r.label}</b>
          <MixBar mix={view.rows[i]} h={4} />
          <span className="sh">{pct(view.rows[i].share)}</span>
        </div>
        {OUTER_COLS.map((c, j) => {
          const cell = view.cells[i][j]
          const pinned = pin?.row === i && pin?.col === j
          if (cell.n === 0) return <div key={c.label} className="cell empty" aria-hidden />
          const side = Math.max(9, Math.sqrt(cell.share / maxShare) * 50)
          const dom = dominant(cell)
          return (
            <button
              key={c.label}
              type="button"
              className={`cell ${pinned ? 'pin' : ''} ${(off || (cols && !cols.has(j))) ? 'off' : ''}`}
              onClick={() => onPin(pinned ? null : { row: i, col: j })}
              aria-pressed={pinned}
              aria-label={`${r.long} × ${c.long}: ${pct(cell.share)} of the range; fold ${pct(cell.f)}, ${word('c')} ${pct(cell.c)}, pot ${pct(cell.p)}`}
            >
              <span className={`dom ${dom}`}>{word(dom)}</span>
              <span className="sq" style={{ width: side, height: side }}>
                <i className="p" style={{ flex: cell.p }} />
                <i className="c" style={{ flex: cell.c }} />
                <i className="f" style={{ flex: cell.f }} />
              </span>
              <span className="pct">{cell.share >= 0.001 ? pct(cell.share) : '<0.1%'}</span>
            </button>
          )
        })}
      </>
    )
  }
}
