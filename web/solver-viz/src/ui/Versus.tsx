/**
 * Both seats: the other seat's range beside yours, theirs already filtered
 * by what they did on the flop (their betting range when they bet, their
 * checking range when they checked), and in the middle your hovered
 * region's equity against their whole range, split into the two halves and
 * the scoop chance. Under it the filmstrip: the lead-up as a strip of range
 * thumbnails, one per event, so the hand so far reads as two ranges
 * shrinking.
 *
 * Card removal between the seats is ignored: each seat's range is sampled
 * from the same 49 cards. The page says so.
 */

import { useState } from 'react'

import type { Cell, GridView } from '../engine/aggregate'
import type { Node } from '../engine/policy'
import { INNER_ROWS, OUTER_COLS } from '../engine/regions'
import { dominant } from './Grid'
import { pct, pct0 } from './bars'
import type { VillainAct } from './Score'

export interface Equity {
  inner: number
  outer: number
  total: number
  scoop: number
  scooped: number
}

const INK: Record<'f' | 'c' | 'p', string> = { f: '245,240,230', c: '233,184,98', p: '216,86,111' }

/** A thumbnail of the two-half grid: ink = `weight(cell)`, colour = the cell's dominant action (or ivory when `plain`). */
export function Thumb({ view, weight, plain, hover, onHover }: { view: GridView; weight: (c: Cell) => number; plain?: boolean; hover?: { row: number; col: number } | null; onHover?: (p: { row: number; col: number } | null) => void }) {
  const max = Math.max(1e-9, ...view.cells.flat().map(weight))
  return (
    <div className="heat" style={{ gridTemplateColumns: `repeat(${OUTER_COLS.length}, 1fr)` }} onMouseLeave={() => onHover?.(null)}>
      {INNER_ROWS.map((_, ri) => {
        const i = INNER_ROWS.length - 1 - ri
        return OUTER_COLS.map((_, j) => {
          const c = view.cells[i][j]
          if (c.n === 0) return <i key={`${i}-${j}`} className="void" />
          const a = Math.pow(weight(c) / max, 0.5)
          const ink = plain ? INK.f : INK[dominant(c)]
          const hov = hover && hover.row === i && hover.col === j
          return (
            <i
              key={`${i}-${j}`}
              className={hov ? 'hov' : ''}
              style={{ background: `rgba(${ink},${(0.05 + 0.9 * Math.min(1, a)).toFixed(2)})` }}
              onMouseEnter={() => onHover?.({ row: i, col: j })}
              title={`${INNER_ROWS[i].label} × ${OUTER_COLS[j].label} · ${pct(c.share)}`}
            />
          )
        })
      })}
    </div>
  )
}

export function Versus({
  mine,
  theirs,
  node,
  villainAct,
  equityOf,
  throwsMine,
  throwsTheirs,
}: {
  mine: GridView
  theirs: GridView
  node: Node
  villainAct: VillainAct
  /** equity of one of your cells against their filtered range */
  equityOf: (row: number, col: number) => Equity
  throwsMine: number[]
  throwsTheirs: number[]
}) {
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null)
  const theirWeight = (c: Cell) => c.share * (villainAct === 'pot' ? c.p : c.c)
  const myWeight = (c: Cell) => c.share * c.c
  const theirKept = theirs.all[villainAct === 'pot' ? 'p' : 'c']
  const eq = hover ? equityOf(hover.row, hover.col) : null
  const yourAct = node === 'facing' ? 'call' : 'check'

  return (
    <div className="versus">
      <div className="vs">
        <div className="side">
          <div className="lbl">
            <span>seat 0 · their {villainAct === 'pot' ? 'betting' : 'checking'} range</span>
            <span className="dimmer">{pct0(theirKept)} kept</span>
          </div>
          <Thumb view={theirs} weight={theirWeight} />
          <p className="legend-line">ink = how much of their range is here after they {villainAct === 'pot' ? 'bet' : 'checked'} · colour = what the region mostly did</p>
        </div>
        <div className="mid">
          <div className="vsn">vs</div>
          <div className="eq">
            <span className="dimmer">{hover ? `${INNER_ROWS[hover.row].label} × ${OUTER_COLS[hover.col].label}` : 'hover a cell on your side'}</span>
            <b>{eq ? pct0(eq.total) : '—'}</b>
            <span className="dimmer">total equity vs their range</span>
            <div className="two">
              <span>
                inner <b>{eq ? pct0(eq.inner) : '—'}</b>
              </span>
              <span>
                outer <b>{eq ? pct0(eq.outer) : '—'}</b>
              </span>
            </div>
            <div className="two">
              <span>
                scoop <b>{eq ? pct0(eq.scoop) : '—'}</b>
              </span>
              <span>
                scooped <b>{eq ? pct0(eq.scooped) : '—'}</b>
              </span>
            </div>
          </div>
          <span className="dimmer small">card removal between the seats is ignored</span>
        </div>
        <div className="side">
          <div className="lbl">
            <span>you · seat 1 · whole range</span>
            <span className="dimmer">colour = your mix</span>
          </div>
          <Thumb view={mine} weight={(c) => c.share} hover={hover} onHover={setHover} />
          <p className="legend-line">ink = share of your range · colour = what the region mostly does now</p>
        </div>
      </div>

      <div className="lbl" style={{ marginTop: 22 }}>
        the hand so far, as ranges
      </div>
      <div className="film">
        <Frame who="deal · flop" what="both ranges" kept="100%">
          <Thumb view={mine} weight={(c) => c.share} plain />
        </Frame>
        <Frame who="seat 0" what={villainAct === 'pot' ? 'bets pot' : 'checks'} cls={villainAct === 'pot' ? 'p' : 'c'} kept={pct0(theirKept)} keptLabel="their range kept">
          <Thumb view={theirs} weight={theirWeight} plain />
        </Frame>
        <Frame who="you" what={yourAct} cls="c" kept={pct0(mine.all.c)} keptLabel="your range kept" now>
          <Thumb view={mine} weight={myWeight} plain />
        </Frame>
        <Frame who="seat 0 · draw" what="throws" kept="" keptLabel="throw-count mix">
          <ThrowBars t={throwsTheirs} />
        </Frame>
        <Frame who="you · draw" what="throw" kept="" keptLabel="throw-count mix">
          <ThrowBars t={throwsMine} />
        </Frame>
        <Frame who="turn" what="needs rung 4" kept="—">
          <div className="heat" style={{ gridTemplateColumns: `repeat(${OUTER_COLS.length}, 1fr)` }}>
            {Array.from({ length: INNER_ROWS.length * OUTER_COLS.length }, (_, k) => (
              <i key={k} className="void" />
            ))}
          </div>
        </Frame>
      </div>
    </div>
  )
}

function Frame({ who, what, cls, kept, keptLabel, now, children }: { who: string; what: string; cls?: string; kept: string; keptLabel?: string; now?: boolean; children: React.ReactNode }) {
  return (
    <div className={`frame ${now ? 'now' : ''}`}>
      <div className="who">{who}</div>
      <div className={`what ${cls ?? ''}`}>
        <b>{what}</b>
      </div>
      {children}
      <div className="kept">
        <span>{keptLabel ?? ''}</span>
        <b>{kept}</b>
      </div>
    </div>
  )
}

function ThrowBars({ t }: { t: number[] }) {
  return (
    <div className="throwbars" aria-label="how many cards the range throws, stand pat through five">
      {t.map((v, k) => (
        <i key={k} style={{ height: `${Math.max(4, v * 100)}%`, background: `rgba(233,184,98,${(0.3 + k * 0.13).toFixed(2)})` }} title={`${k === 0 ? 'stand pat' : `throw ${k}`} ${pct0(v)}`} />
      ))}
    </div>
  )
}
