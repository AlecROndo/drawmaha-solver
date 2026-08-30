import type { Engine } from '../sim/engine'
import { ACTIONS } from '../sim/game'
import { ACTION_COLORS, fmt } from './format'
import { makeLogX, spreadLabels } from './chartScale'
import {
  CHART_H as H,
  CHART_W as W,
  ChartTooltip,
  DecadeGridlines,
  EdgeLabel,
  HoverCrosshair,
  YGridline,
  useChartHover,
  type ChartMargins,
} from './chartChrome'

const M: ChartMargins = { l: 36, r: 76, t: 12, b: 26 }

/**
 * Average-strategy components vs iteration, log x, dashed reference at ⅓.
 */
export function ConvergenceChart({ engine }: { engine: Engine }) {
  const history = engine.history
  const scale = makeLogX(engine.iteration, M.l, W - M.r)
  const { wrapRef, hover, onMove, onLeave } = useChartHover(history, scale, M)
  const y = (v: number) => M.t + (1 - v) * (H - M.t - M.b)

  const path = (idx: number) =>
    history.map((h, i) => `${i === 0 ? 'M' : 'L'}${scale.x(h.t).toFixed(1)},${y(h.avg[idx]).toFixed(1)}`).join('')

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
        <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={onLeave}>
          <DecadeGridlines scale={scale} m={M} />
          {/* y axis: 0, ⅓ (dashed), 1 */}
          {[0, 1].map((v) => (
            <YGridline key={v} y={y(v)} label={v} m={M} />
          ))}
          <line x1={M.l} x2={W - M.r} y1={y(1 / 3)} y2={y(1 / 3)} stroke="var(--muted)" strokeDasharray="4 3" />
          <text x={M.l - 6} y={y(1 / 3) + 3} textAnchor="end" fontSize="10" fill="var(--muted)">
            ⅓
          </text>
          {history.length > 1 &&
            [0, 1, 2].map((i) => (
              <path key={i} d={path(i)} fill="none" stroke={ACTION_COLORS[i]} strokeWidth="2" strokeLinejoin="round" />
            ))}
          {/* direct labels at the right edge, spread to avoid collisions */}
          {last &&
            (() => {
              const ys = spreadLabels(
                [0, 1, 2].map((i) => y(last.avg[i])),
                12,
                M.t + 6,
                H - M.b - 4,
              )
              return ACTIONS.map((name, i) => (
                <EdgeLabel key={name} y={ys[i]} color={ACTION_COLORS[i]} text={name} m={M} />
              ))
            })()}
          {hover && (
            <g>
              <HoverCrosshair x={scale.x(hover.t)} m={M} />
              {[0, 1, 2].map((i) => (
                <circle key={i} cx={scale.x(hover.t)} cy={y(hover.avg[i])} r={3} fill={ACTION_COLORS[i]} stroke="var(--surface)" strokeWidth="1.5" />
              ))}
            </g>
          )}
        </svg>
        {hover && wrapRef.current && (
          <ChartTooltip x={scale.x(hover.t)} wrap={wrapRef.current} clamp={130} t={hover.t}>
            {ACTIONS.map((name, i) => (
              <div className="row" key={name}>
                <span className="chip">
                  <span className="dot" style={{ background: ACTION_COLORS[i], width: 7, height: 7 }} />
                  {name}
                </span>
                <span className="val">{fmt(hover.avg[i])}</span>
              </div>
            ))}
          </ChartTooltip>
        )}
      </div>
    </section>
  )
}
