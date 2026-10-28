import { useState } from 'react'

import { ARMS, BEST, OBJECTIVES, OBJECTIVE_NAME, lcfrAt, type Objective } from '../deep'
import { UNIFORM_RANDOM, perHundred, series } from '../research'
import { fmtHands, logScale } from './scales'

/**
 * The race at equal hands: rung 3's LCFR line with the best Deep CFR arm's
 * three graded objects on the same axes, and under them the price — the
 * net's exact average divided by the table's at the same hands. Hover or
 * arrow a mark to read every number at that column. The spec's one
 * question is the lower strip.
 */

const W = 760
const H = 440
const M = { l: 58, r: 120, t: 24, b: 30 }
const PLOT_BOTTOM = 300
const STRIP_TOP = 338
const STRIP_BOTTOM = H - M.b

const x = logScale(1e5, 2e7, M.l, W - M.r)
const y = logScale(5, 600, PLOT_BOTTOM, M.t)
const yPrice = logScale(0.05, 1, STRIP_BOTTOM, STRIP_TOP)

const OBJ_INK: Record<Objective, string> = { exact: 'var(--obj-exact)', pi: 'var(--obj-pi)', current: 'var(--obj-current)' }
const SMALL = ARMS['mac-3x64']

const path = (points: { hands: number; v: number }[]) =>
  points.map((p, i) => `${i ? 'L' : 'M'} ${x(p.hands).toFixed(1)} ${y(perHundred(p.v)).toFixed(1)}`).join(' ')

export function DeepRaceChart() {
  const lcfr = series('lcfr').map((p) => ({ hands: p.hands, v: p.exploitability }))
  const objects = OBJECTIVES.map((o) => ({ o, points: BEST.marks.map((m) => ({ hands: m.hands, v: m[o].exploitability })) }))
  const small = SMALL.marks.map((m) => ({ hands: m.hands, v: m.exact.exploitability }))
  const price = BEST.marks.map((m) => ({ hands: m.hands, v: m.exact.exploitability / lcfrAt(m.hands) }))
  const [at, setAt] = useState(BEST.marks.length - 1)
  const mark = BEST.marks[at]
  const xAt = x(mark.hands)

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setAt((i) => Math.max(0, i - 1))
    if (e.key === 'ArrowRight') setAt((i) => Math.min(BEST.marks.length - 1, i + 1))
  }

  return (
    <>
      <div className="legend">
        <span className="item">
          <span className="swatch" style={{ background: 'var(--table-ink)' }} />
          LCFR, the table (rung 3)
        </span>
        {OBJECTIVES.map((o) => (
          <span key={o} className="item">
            <span className="swatch" style={{ background: OBJ_INK[o] }} />
            Deep CFR {BEST.net}: {OBJECTIVE_NAME[o]}
          </span>
        ))}
        <span className="item">
          <span className="swatch" style={{ background: 'var(--iv-30)' }} />
          Deep CFR {SMALL.net}, the spec's default: exact average
        </span>
      </div>
      <svg
        className="chart deeprace"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Exploitability of LCFR and Deep CFR against hands per seat, and their ratio"
        tabIndex={0}
        onKeyDown={onKey}
      >
        {[10, 100].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke="var(--panel-hair)" />
            <text x={M.l - 8} y={y(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {v}
            </text>
          </g>
        ))}
        {[1e5, 1e6, 1e7].map((h) => (
          <g key={h}>
            <line x1={x(h)} x2={x(h)} y1={M.t} y2={PLOT_BOTTOM} stroke="var(--panel-hair)" />
            <line x1={x(h)} x2={x(h)} y1={STRIP_TOP} y2={STRIP_BOTTOM} stroke="var(--panel-hair)" />
            <text x={x(h)} y={STRIP_BOTTOM + 16} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
              {fmtHands(h)}
            </text>
          </g>
        ))}
        <text x={4} y={M.t - 10} fontSize="10" fill="var(--panel-dim)">
          chips / 100 hands
        </text>
        <text x={W - M.r + 8} y={STRIP_BOTTOM + 16} fontSize="10" fill="var(--panel-dim)">
          hands / seat
        </text>

        {/* uniform random: the scale */}
        <line
          x1={M.l}
          x2={W - M.r}
          y1={y(perHundred(UNIFORM_RANDOM.exploitability))}
          y2={y(perHundred(UNIFORM_RANDOM.exploitability))}
          stroke="var(--panel-dim)"
          strokeDasharray="4 4"
        />
        <text x={W - M.r + 8} y={y(perHundred(UNIFORM_RANDOM.exploitability)) + 4} fontSize="10.5" fill="var(--panel-dim)">
          random {perHundred(UNIFORM_RANDOM.exploitability).toFixed(0)}
        </text>

        <path d={path(lcfr)} fill="none" stroke="var(--table-ink)" strokeWidth={1.6} />
        <path d={path(small)} fill="none" stroke="var(--iv-30)" strokeWidth={1.4} strokeDasharray="2 4" />
        {objects.map(({ o, points }) => (
          <path
            key={o}
            d={path(points)}
            fill="none"
            stroke={OBJ_INK[o]}
            strokeWidth={o === 'exact' ? 2.2 : 1.5}
            strokeDasharray={o === 'current' ? '6 4' : undefined}
          />
        ))}
        <text x={x(2e7) + 8} y={y(perHundred(lcfr.at(-1)!.v)) + 4} fontSize="11" fill="var(--table-ink)">
          LCFR {perHundred(lcfr.at(-1)!.v).toFixed(1)}
        </text>
        {objects.map(({ o, points }) => (
          <text
            key={o}
            x={x(points.at(-1)!.hands) + 8}
            y={y(perHundred(points.at(-1)!.v)) + (o === 'exact' ? 12 : o === 'pi' ? -4 : 4)}
            fontSize="11"
            fill={OBJ_INK[o]}
          >
            {o === 'exact' ? 'exact' : o === 'pi' ? 'Π' : 'current'} {perHundred(points.at(-1)!.v).toFixed(1)}
          </text>
        ))}

        {/* the price strip */}
        {[0.1, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={yPrice(v)} y2={yPrice(v)} stroke="var(--panel-hair)" strokeDasharray={v === 1 ? '4 4' : undefined} />
            <text x={M.l - 8} y={yPrice(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {v}
            </text>
          </g>
        ))}
        <text x={4} y={STRIP_TOP - 10} fontSize="10" fill="var(--panel-dim)">
          price: net ÷ table, exact average
        </text>
        <path
          d={price.map((p, i) => `${i ? 'L' : 'M'} ${x(p.hands).toFixed(1)} ${yPrice(p.v).toFixed(1)}`).join(' ')}
          fill="none"
          stroke="var(--obj-exact)"
          strokeWidth={2}
        />
        {price.map((p) => (
          <circle key={p.hands} cx={x(p.hands)} cy={yPrice(p.v)} r={3} fill="var(--obj-exact)" />
        ))}

        {/* the selected mark */}
        <line x1={xAt} x2={xAt} y1={M.t} y2={STRIP_BOTTOM} stroke="var(--panel-mark)" strokeWidth={1} strokeOpacity={0.6} />
        {OBJECTIVES.map((o) => (
          <circle key={o} cx={xAt} cy={y(perHundred(mark[o].exploitability))} r={o === 'exact' ? 4 : 3} fill={OBJ_INK[o]} />
        ))}
        <circle cx={xAt} cy={y(perHundred(lcfrAt(mark.hands)))} r={3.5} fill="var(--table-ink)" />
        {BEST.marks.map((m, i) => {
          const cx = x(m.hands)
          const left = i === 0 ? M.l : (x(BEST.marks[i - 1].hands) + cx) / 2
          const right = i === BEST.marks.length - 1 ? x(1e7) : (x(BEST.marks[i + 1].hands) + cx) / 2
          return (
            <rect key={m.t} x={left} y={M.t} width={right - left} height={STRIP_BOTTOM - M.t} fill="transparent" onMouseEnter={() => setAt(i)} onFocus={() => setAt(i)} />
          )
        })}
      </svg>
      <dl className="stat">
        <div>
          <dt>hands per seat</dt>
          <dd>{mark.hands.toLocaleString()}</dd>
        </div>
        <div>
          <dt style={{ color: 'var(--table-ink)' }}>LCFR</dt>
          <dd>{perHundred(lcfrAt(mark.hands)).toFixed(1)}</dd>
        </div>
        {OBJECTIVES.map((o) => (
          <div key={o}>
            <dt style={{ color: OBJ_INK[o] }}>{OBJECTIVE_NAME[o]}</dt>
            <dd>{perHundred(mark[o].exploitability).toFixed(1)}</dd>
          </div>
        ))}
        <div>
          <dt>price, exact</dt>
          <dd>{(mark.exact.exploitability / lcfrAt(mark.hands)).toFixed(2)}</dd>
        </div>
      </dl>
    </>
  )
}
