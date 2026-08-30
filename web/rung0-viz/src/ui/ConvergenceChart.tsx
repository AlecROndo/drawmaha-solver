import { useRef, useState } from 'react'
import type { Engine, HistoryPoint } from '../sim/engine'
import { ACTIONS } from '../sim/game'
import { ACTION_COLORS, fmt, fmtIter } from './format'
import { decadeLabel, makeLogX, nearestPoint, spreadLabels } from './chartScale'

const W = 560
const H = 250
const M = { l: 36, r: 76, t: 12, b: 26 }

/**
 * Average-strategy components vs iteration, log x, dashed reference at ⅓.
 */
export function ConvergenceChart({ engine }: { engine: Engine }) {
  const history = engine.history
  const wrapRef = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<HistoryPoint | null>(null)

  const scale = makeLogX(engine.iteration, M.l, W - M.r)
  const y = (v: number) => M.t + (1 - v) * (H - M.t - M.b)

  const path = (idx: number) =>
    history.map((h, i) => `${i === 0 ? 'M' : 'L'}${scale.x(h.t).toFixed(1)},${y(h.avg[idx]).toFixed(1)}`).join('')

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const frac = Math.min(Math.max((px - M.l) / (W - M.l - M.r), 0), 1)
    setHover(nearestPoint(history, 10 ** (frac * Math.log10(scale.maxT))))
  }

  const last = history[history.length - 1]

  return (
    <section className="panel" aria-label="Convergence of the average strategy">
      <h2>Average strategy → (⅓, ⅓, ⅓)</h2>
      <p className="sub">each component of S/n vs iteration (log x)</p>
      <div className="legend">
        {ACTIONS.map((name, i) => (
          <span className="item" key={name}>
            <span className="swatch" style={{ background: ACTION_COLORS[i] }} />
            {name}
          </span>
        ))}
      </div>
      <div className="chart-wrap" ref={wrapRef}>
        <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
          {/* gridlines at decades */}
          {scale.decades.map((d) => (
            <g key={d}>
              <line x1={scale.x(d)} x2={scale.x(d)} y1={M.t} y2={H - M.b} stroke="var(--grid)" />
              <text x={scale.x(d)} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--muted)">
                {decadeLabel(d)}
              </text>
            </g>
          ))}
          {/* y axis: 0, ⅓ (dashed), 1 */}
          {[0, 1].map((v) => (
            <g key={v}>
              <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
              <text x={M.l - 6} y={y(v) + 3} textAnchor="end" fontSize="10" fill="var(--muted)">
                {v}
              </text>
            </g>
          ))}
          <line x1={M.l} x2={W - M.r} y1={y(1 / 3)} y2={y(1 / 3)} stroke="var(--muted)" strokeDasharray="4 3" />
          <text x={M.l - 6} y={y(1 / 3) + 3} textAnchor="end" fontSize="10" fill="var(--muted)">
            ⅓
          </text>
          {history.length > 1 &&
            [0, 1, 2].map((i) => (
              <path key={i} d={path(i)} fill="none" stroke={ACTION_COLORS[i]} strokeWidth="2" strokeLinejoin="round" />
            ))}
          {/* direct labels at the right edge, text ink + colored tick, spread to avoid collisions */}
          {last &&
            (() => {
              const ys = spreadLabels(
                [0, 1, 2].map((i) => y(last.avg[i])),
                12,
                M.t + 6,
                H - M.b - 4,
              )
              return ACTIONS.map((name, i) => (
                <g key={name}>
                  <rect x={W - M.r + 4} y={ys[i] - 1.5} width={10} height={3} rx={1.5} fill={ACTION_COLORS[i]} />
                  <text x={W - M.r + 18} y={ys[i] + 3} fontSize="10" fill="var(--ink-2)">
                    {name}
                  </text>
                </g>
              ))
            })()}
          {hover && (
            <g>
              <line x1={scale.x(hover.t)} x2={scale.x(hover.t)} y1={M.t} y2={H - M.b} stroke="var(--axis)" />
              {[0, 1, 2].map((i) => (
                <circle key={i} cx={scale.x(hover.t)} cy={y(hover.avg[i])} r={3} fill={ACTION_COLORS[i]} stroke="var(--surface)" strokeWidth="1.5" />
              ))}
            </g>
          )}
        </svg>
        {hover && wrapRef.current && (
          <div
            className="chart-tooltip"
            style={{
              left: Math.min((scale.x(hover.t) / W) * wrapRef.current.clientWidth + 10, wrapRef.current.clientWidth - 130),
              top: 8,
            }}
          >
            <div className="row">
              <span>iteration</span>
              <span className="val">{fmtIter(hover.t)}</span>
            </div>
            {ACTIONS.map((name, i) => (
              <div className="row" key={name}>
                <span className="chip">
                  <span className="dot" style={{ background: ACTION_COLORS[i], width: 7, height: 7 }} />
                  {name}
                </span>
                <span className="val">{fmt(hover.avg[i])}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
