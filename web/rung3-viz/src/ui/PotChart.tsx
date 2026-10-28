import { useState } from 'react'

import { POT_SERIES } from '../deep'
import { perHundred } from '../research'
import { fmtHands, logScale } from './scales'

/**
 * The regret targets divided by pot^k before the refit, as curves: each
 * arm's exact average against hands per seat on the page's log axes, the
 * same 4 × 256 net and 16,000-step refit throughout. The raw arm at 5e-4 is
 * drawn out to its five-million-hand floor so the pot arms, stopped at two
 * million, can be read against it. Raw arms wear ivory, pot arms the ink
 * the page points with; within each pair the line style tells the arms
 * apart, every line is named at its end, and hovering or arrowing a mark
 * reads every arm's grade at that column.
 */

const W = 760
const H = 400
const M = { l: 58, r: 132, t: 24, b: 44 }
const PLOT_BOTTOM = H - M.b
/** the gap two end labels keep when their lines end at the same mark */
const LABEL_H = 15

const x = logScale(1e5, 5e6, M.l, W - M.r)
const y = logScale(7, 100, PLOT_BOTTOM, M.t)

const kWord = (k: number) => (k === 0 ? 'raw chips' : k === 0.5 ? '÷ √pot' : k === 1 ? '÷ pot' : `÷ pot^${k}`)
const lrWord = (lr: number) => (lr === 0.001 ? '1e-3' : lr === 0.0005 ? '5e-4' : String(lr))
const ink = (k: number, lr: number) => (k > 0 ? 'var(--yellow)' : lr === 0.0005 ? 'var(--iv-80)' : 'var(--iv-45)')
const dash = (k: number, lr: number) => (k === 1 || (k === 0 && lr !== 0.0005) ? '6 4' : undefined)
const name = (k: number, lr: number) => `${kWord(k)} · lr ${lrWord(lr)}`

/** every mark any arm was graded at, in order: the hover columns */
const MARKS = [...new Set(POT_SERIES.flatMap((s) => s.points.map((p) => p.t)))].sort((a, b) => a - b)

export function PotChart() {
  const [at, setAt] = useState(MARKS.indexOf(200))
  const t = MARKS[at]
  const xAt = x(t * 10_000)
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setAt((i) => Math.max(0, i - 1))
    if (e.key === 'ArrowRight') setAt((i) => Math.min(MARKS.length - 1, i + 1))
  }
  const gradeAt = (s: (typeof POT_SERIES)[number], mark: number) => s.points.find((p) => p.t === mark)?.exact
  // end labels: lines that end at the same mark have their labels stacked top to bottom, LABEL_H apart
  const ends = POT_SERIES.map((s) => ({ s, last: s.points.at(-1)!, ly: y(perHundred(s.points.at(-1)!.exact)) + 4 }))
  for (const hands of new Set(ends.map((e) => e.last.hands))) {
    const group = ends.filter((e) => e.last.hands === hands).sort((a, b) => a.ly - b.ly)
    for (let i = 1; i < group.length; i++) group[i].ly = Math.max(group[i].ly, group[i - 1].ly + LABEL_H)
  }
  return (
    <>
      <div className="legend">
        {POT_SERIES.map((s) => (
          <span key={s.arm.id} className="item">
            <span className="swatch" style={{ background: dash(s.potPower, s.lr) ? `repeating-linear-gradient(90deg, ${ink(s.potPower, s.lr)} 0 4px, transparent 4px 7px)` : ink(s.potPower, s.lr) }} />
            {name(s.potPower, s.lr)}
          </span>
        ))}
      </div>
      <svg className="chart pot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Exact average of the regret targets over pot^k against hands per seat" tabIndex={0} onKeyDown={onKey}>
        {[10, 100].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke="var(--panel-hair)" />
            <text x={M.l - 8} y={y(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {v}
            </text>
          </g>
        ))}
        {[1e5, 1e6, 2e6, 5e6].map((h) => (
          <g key={h}>
            <line x1={x(h)} x2={x(h)} y1={M.t} y2={PLOT_BOTTOM} stroke="var(--panel-hair)" />
            <text x={x(h)} y={PLOT_BOTTOM + 16} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
              {fmtHands(h)}
            </text>
          </g>
        ))}
        <text x={4} y={M.t - 10} fontSize="10" fill="var(--panel-dim)">
          chips / 100 hands, the exact average, log
        </text>
        <text x={W - M.r} y={PLOT_BOTTOM + 32} textAnchor="end" fontSize="10" fill="var(--panel-dim)">
          hands / seat, log
        </text>
        {POT_SERIES.map((s) => (
          <g key={s.arm.id}>
            <path
              d={s.points.map((p, i) => `${i ? 'L' : 'M'} ${x(p.hands).toFixed(1)} ${y(perHundred(p.exact)).toFixed(1)}`).join(' ')}
              fill="none"
              stroke={ink(s.potPower, s.lr)}
              strokeWidth={s.potPower === 0.5 ? 2.2 : 1.6}
              strokeDasharray={dash(s.potPower, s.lr)}
            />
            {s.points.map((p) => (
              <circle key={p.t} cx={x(p.hands)} cy={y(perHundred(p.exact))} r={3} fill={ink(s.potPower, s.lr)}>
                <title>{`${name(s.potPower, s.lr)}, ${fmtHands(p.hands)} hands per seat: ${perHundred(p.exact).toFixed(2)} chips per hundred hands`}</title>
              </circle>
            ))}
          </g>
        ))}
        {ends.map(({ s, last, ly }) => (
          <text key={s.arm.id} x={x(last.hands) + 8} y={ly} fontSize="11" fill={ink(s.potPower, s.lr)}>
            {kWord(s.potPower)} {perHundred(last.exact).toFixed(1)}
          </text>
        ))}
        {/* the selected mark */}
        <line x1={xAt} x2={xAt} y1={M.t} y2={PLOT_BOTTOM} stroke="var(--panel-mark)" strokeWidth={1} strokeOpacity={0.6} />
        {POT_SERIES.map((s) => {
          const v = gradeAt(s, t)
          return v === undefined ? null : <circle key={s.arm.id} cx={xAt} cy={y(perHundred(v))} r={s.potPower === 0.5 ? 4.5 : 4} fill={ink(s.potPower, s.lr)} />
        })}
        {MARKS.map((mark, i) => {
          const cx = x(mark * 10_000)
          const left = i === 0 ? M.l : (x(MARKS[i - 1] * 10_000) + cx) / 2
          const right = i === MARKS.length - 1 ? W - M.r : (x(MARKS[i + 1] * 10_000) + cx) / 2
          return <rect key={mark} x={left} y={M.t} width={right - left} height={PLOT_BOTTOM - M.t} fill="transparent" onMouseEnter={() => setAt(i)} onFocus={() => setAt(i)} />
        })}
      </svg>
      <dl className="stat">
        <div>
          <dt>hands per seat</dt>
          <dd>{(t * 10_000).toLocaleString()}</dd>
        </div>
        {POT_SERIES.map((s) => {
          const v = gradeAt(s, t)
          return (
            <div key={s.arm.id}>
              <dt style={{ color: ink(s.potPower, s.lr) }}>{name(s.potPower, s.lr)}</dt>
              <dd>{v === undefined ? <small>not graded</small> : perHundred(v).toFixed(1)}</dd>
            </div>
          )
        })}
      </dl>
    </>
  )
}
