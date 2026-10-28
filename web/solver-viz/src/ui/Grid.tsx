/**
 * The two-half grid: rows are what the five cards in hand make on their own
 * (the inner half), columns what two of them make with the board (the outer
 * half). A cell is one region of the range: its square's area is how much
 * of the range lives there, the square's bands are what that region does,
 * and the margins carry each row's and column's own mix. Drawmaha's answer
 * to the hold'em 13×13.
 *
 * The squares are sized against the view, not against the whole range. The
 * largest region in view fills its cell (`MARK.max` on a side) and every
 * other region's square has the area it deserves relative to that one:
 * side ∝ √(share / maxShare), so a region a quarter the size is half as
 * wide, and a region too small to draw honestly is still a `MARK.min` chip.
 * Sized against the whole range instead, a node whose weight is spread over
 * forty regions drew forty tiny squares in forty wide cells and the eye
 * could not compare them; sized against the largest, the comparison is the
 * whole figure. `maxShare` is taken over `view.cells`, which `gridView`
 * computes under the row/column mask, so a filtered-off region has share 0
 * and never sets the maximum: narrowing the filter is a zoom, the survivors
 * grow to fill the frame. Every node and every filter normalises on its own,
 * so a square's size says nothing across spots, only across the cells beside
 * it — the percentage in the corner is the number that carries over.
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

/**
 * The mark's sizes in px. `max` is the side of the largest region's square.
 * The cell is 84 px tall, border-box: less its 1 px borders, the 4 px of
 * padding above and below and the 13 px label strip (the corner word and
 * the percentage share one line across the top), 61 px are left for the
 * square, and it takes 56 of them so a hair of well shows above and below
 * it and it never runs under a label. Across, 56 px plus the 12 px of
 * padding and 2 px of border is 70 px, so the square fills its slot at any
 * cell width of 70 px and up (the columns are `minmax(0, 1fr)`, wider than
 * that at every width the page is laid out for). `min` is the chip a region
 * keeps however small its share, so a <0.1% region is a visible dot and not
 * a missing cell; it is 9% of `max`, where the rule this replaced floored at
 * 9 px of a 50 px maximum, 18%.
 */
export const MARK = { max: 56, min: 5 } as const

/**
 * The side of a region's square, in px, from its share and the largest share
 * in view: `max` at the largest region, `min` at nothing, and in between
 * side ∝ √(share / maxShare), so the *area* is the honest comparison — a
 * region with a quarter of the largest share gets half its side. A share at
 * or above `maxShare`, or a `maxShare` of zero, is clamped to `max` rather
 * than left to divide by nothing.
 *
 * The rectangle variant — fill the cell's inner width as well as its height,
 * with the same √ law on both — is `{ width: `${100 * side / max}%`, height:
 * side }` on the same `side`; it was not taken because similar rectangles
 * change shape with the viewport and the bands of a wide, short mark read as
 * a second mix bar rather than as a square's fill.
 */
export function sideOf(share: number, maxShare: number, max: number = MARK.max, min: number = MARK.min): number {
  if (!(share > 0)) return min
  if (!(maxShare > 0) || share >= maxShare) return max
  return Math.min(max, Math.max(min, Math.sqrt(share / maxShare) * max))
}

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
  // Over the cells in view: under a filter the masked-off regions have share 0 (see the docstring).
  const maxShare = Math.max(0, ...view.cells.flat().map((c) => c.share))
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
        rows: the {inHand} in hand · columns: two of them with the board · a square's area is the region's share, relative to the largest region in view · its
        bands are the mix · corner word: what most of the region does
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
          const side = sideOf(cell.share, maxShare)
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
              <span className="lab">
                <span className={`dom ${dom}`}>{word(dom)}</span>
                <span className="pct">{cell.share >= 0.001 ? pct(cell.share) : '<0.1%'}</span>
              </span>
              <span className="sq" style={{ width: side, height: side }}>
                <i className="p" style={{ flex: cell.p }} />
                <i className="c" style={{ flex: cell.c }} />
                <i className="f" style={{ flex: cell.f }} />
              </span>
            </button>
          )
        })}
      </>
    )
  }
}
