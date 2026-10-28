import { useState } from 'react'

import { BEST, OBJECTIVES, OBJECTIVE_NAME, type Objective } from '../deep'
import { perHundred } from '../research'
import { fmtHands, logScale } from './scales'

/**
 * The three averages of one arm, mark by mark: the exact own-reach average,
 * Π fit fresh on the policy memory, and the current iterate. Hover or arrow
 * a mark to read each object's grade and both seats' best responses.
 */

const W = 760
const H = 300
const M = { l: 58, r: 110, t: 22, b: 34 }
const x = logScale(10, 500, M.l, W - M.r)
const y = logScale(8, 80, H - M.b, M.t)
const INK: Record<Objective, string> = { exact: 'var(--obj-exact)', pi: 'var(--obj-pi)', current: 'var(--obj-current)' }

const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`

export function AveragesChart() {
  const [at, setAt] = useState(BEST.marks.length - 1)
  const mark = BEST.marks[at]
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setAt((i) => Math.max(0, i - 1))
    if (e.key === 'ArrowRight') setAt((i) => Math.min(BEST.marks.length - 1, i + 1))
  }
  const path = (o: Objective) => BEST.marks.map((m, i) => `${i ? 'L' : 'M'} ${x(m.t).toFixed(1)} ${y(perHundred(m[o].exploitability)).toFixed(1)}`).join(' ')
  return (
    <>
      <svg className="chart averages" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Three graded objects of one Deep CFR arm against iteration" tabIndex={0} onKeyDown={onKey}>
        {[10, 20, 50].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke="var(--panel-hair)" />
            <text x={M.l - 8} y={y(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {v}
            </text>
          </g>
        ))}
        {BEST.marks.map((m) => (
          <g key={m.t}>
            <line x1={x(m.t)} x2={x(m.t)} y1={M.t} y2={H - M.b} stroke="var(--panel-hair)" />
            <text x={x(m.t)} y={H - 12} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
              {fmtHands(m.hands)}
            </text>
          </g>
        ))}
        <text x={4} y={M.t - 8} fontSize="10" fill="var(--panel-dim)">
          chips / 100 hands
        </text>
        <text x={W - M.r + 8} y={H - 12} fontSize="10" fill="var(--panel-dim)">
          hands / seat
        </text>
        {OBJECTIVES.map((o) => (
          <path key={o} d={path(o)} fill="none" stroke={INK[o]} strokeWidth={o === 'exact' ? 2.2 : 1.5} strokeDasharray={o === 'current' ? '6 4' : undefined} />
        ))}
        {OBJECTIVES.map((o) => (
          <text key={o} x={x(500) + 8} y={y(perHundred(BEST.marks.at(-1)![o].exploitability)) + (o === 'exact' ? 10 : o === 'pi' ? -3 : 4)} fontSize="11" fill={INK[o]}>
            {o === 'exact' ? 'exact' : o === 'pi' ? 'Π' : 'current'} {perHundred(BEST.marks.at(-1)![o].exploitability).toFixed(1)}
          </text>
        ))}
        <line x1={x(mark.t)} x2={x(mark.t)} y1={M.t} y2={H - M.b} stroke="var(--panel-mark)" strokeOpacity={0.6} />
        {OBJECTIVES.map((o) => (
          <circle key={o} cx={x(mark.t)} cy={y(perHundred(mark[o].exploitability))} r={o === 'exact' ? 4 : 3} fill={INK[o]} />
        ))}
        {BEST.marks.map((m, i) => {
          const cx = x(m.t)
          const left = i === 0 ? M.l : (x(BEST.marks[i - 1].t) + cx) / 2
          const right = i === BEST.marks.length - 1 ? W - M.r : (x(BEST.marks[i + 1].t) + cx) / 2
          return <rect key={m.t} x={left} y={M.t} width={right - left} height={H - M.t - M.b} fill="transparent" onMouseEnter={() => setAt(i)} onFocus={() => setAt(i)} />
        })}
      </svg>
      <dl className="stat">
        <div>
          <dt>iteration · hands per seat</dt>
          <dd>
            {mark.t} · {mark.hands.toLocaleString()}
          </dd>
        </div>
        {OBJECTIVES.map((o) => (
          <div key={o}>
            <dt style={{ color: INK[o] }}>{OBJECTIVE_NAME[o]}</dt>
            <dd>
              {perHundred(mark[o].exploitability).toFixed(1)}{' '}
              <small>
                BR₀ {signed(mark[o].br0)} · BR₁ {signed(mark[o].br1)}
              </small>
            </dd>
          </div>
        ))}
      </dl>
    </>
  )
}
