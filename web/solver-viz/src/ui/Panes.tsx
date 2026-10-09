/**
 * One pane per action: the two-half map drawn once per action, each shaded
 * by that action's frequency alone, with the regions that take it most
 * listed beneath. You compare panes instead of decoding bands; "which hands
 * raise here?" is one pane, read directly. Opened from the header's mix bar.
 */

import type { GridView } from '../engine/aggregate'
import type { Node } from '../engine/policy'
import { actionWord, pct0 } from './bars'
import type { GameUI } from './game'

const INK: Record<'f' | 'c' | 'p', string> = { f: '245,240,230', c: '233,184,98', p: '216,86,111' }

export function Panes({ game, view, node }: { game: GameUI; view: GridView; node: Node }) {
  const INNER_ROWS = game.rows
  const OUTER_COLS = game.cols
  const keys = node === 'facing' ? (['f', 'c', 'p'] as const) : (['c', 'p'] as const)
  return (
    <div className="panes" style={{ gridTemplateColumns: `repeat(${keys.length}, 1fr)` }}>
      {keys.map((k) => {
        const tops = INNER_ROWS.flatMap((r, i) =>
          OUTER_COLS.map((c, j) => ({ name: `${r.label} × ${c.label}`, v: view.cells[i][j][k], share: view.cells[i][j].share })),
        )
          .filter((t) => t.share > 0.002)
          .sort((a, b) => b.v - a.v)
          .slice(0, 5)
        return (
          <div key={k} className="one">
            <div className={`lbl ${k}`}>
              <span>{actionWord(k, node)}</span>
              <b>{pct0(view.all[k])} of the range</b>
            </div>
            <div className="mini" style={{ gridTemplateColumns: `repeat(${OUTER_COLS.length}, 1fr)` }} aria-hidden>
              {INNER_ROWS.map((_, ri) => {
                const i = INNER_ROWS.length - 1 - ri
                return OUTER_COLS.map((_, j) => {
                  const cell = view.cells[i][j]
                  const v = cell.n === 0 ? 0 : cell[k]
                  const alpha = cell.n === 0 ? 0 : 0.06 + v * (k === 'f' ? 0.5 : 0.94)
                  return <i key={`${i}-${j}`} style={{ background: `rgba(${INK[k]},${alpha.toFixed(2)})` }} />
                })
              })}
            </div>
            <div className="toplist">
              {tops.map((t) => (
                <div key={t.name}>
                  <span>{t.name}</span>
                  <span className="v">{pct0(t.v)}</span>
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}
