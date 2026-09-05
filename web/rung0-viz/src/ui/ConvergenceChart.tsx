import type { Engine } from '../sim/engine'
import { ACTIONS } from '../sim/game'
import { ACTION_COLORS, fmt } from './format'
import { makeLogX, spreadLabels } from './chartScale'
import {
  ChartTooltip,
  DecadeGridlines,
  EdgeLabel,
  FULL,
  HoverCrosshair,
  YGridline,
  useChartHover,
  type ChartBox,
} from './chartChrome'
import { FigureHead } from './site'

const BOX: ChartBox = { ...FULL, l: 50, r: 90, t: 14, b: 30 }

/**
 * Average-strategy components vs iteration, log x, dashed reference at ⅓.
 * Runs the full width of the container: it is one long time axis, and cutting
 * it to half a column would throw away the decades that make the walk read.
 */
export function ConvergenceChart({ engine }: { engine: Engine }) {
  const history = engine.history
  const scale = makeLogX(engine.iteration, BOX.l, BOX.w - BOX.r)
  const { wrapRef, hover, onMove, onLeave } = useChartHover(history, scale, BOX)
  const y = (v: number) => BOX.t + (1 - v) * (BOX.h - BOX.t - BOX.b)

  const path = (idx: number) =>
    history
      .map((h, i) => `${i === 0 ? 'M' : 'L'}${scale.x(h.t).toFixed(1)},${y(h.avg[idx]).toFixed(1)}`)
      .join('')

  const last = history[history.length - 1]

  return (
    <section className="sect" aria-label="Convergence of the average strategy">
      <FigureHead n="Fig. 5" title="Each component of the average walks to one third">
        The running average of σ, per action, on a log time axis.
      </FigureHead>
      <div className="legend">
        {ACTIONS.map((name, i) => (
          <span className="item" key={name}>
            <span className="swatch" style={{ background: ACTION_COLORS[i] }} />
            {name}
          </span>
        ))}
        <span className="item">
          <span className="swatch" style={{ background: 'var(--muted)' }} />⅓, the uniform Nash
        </span>
      </div>
      <div className="chart-wrap" ref={wrapRef}>
        <svg viewBox={`0 0 ${BOX.w} ${BOX.h}`} onMouseMove={onMove} onMouseLeave={onLeave}>
          <DecadeGridlines scale={scale} box={BOX} />
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <YGridline key={v} y={y(v)} label={v} box={BOX} />
          ))}
          <line
            x1={BOX.l}
            x2={BOX.w - BOX.r}
            y1={y(1 / 3)}
            y2={y(1 / 3)}
            stroke="var(--muted)"
            strokeDasharray="4 3"
          />
          {history.length > 1 &&
            [0, 1, 2].map((i) => (
              <path
                key={i}
                d={path(i)}
                fill="none"
                stroke={ACTION_COLORS[i]}
                strokeWidth="2"
                strokeLinejoin="round"
              />
            ))}
          {/* direct labels at the right edge, spread to avoid collisions */}
          {last &&
            (() => {
              const ys = spreadLabels(
                [0, 1, 2].map((i) => y(last.avg[i])),
                13,
                BOX.t + 6,
                BOX.h - BOX.b - 4,
              )
              return ACTIONS.map((name, i) => (
                <EdgeLabel key={name} y={ys[i]} color={ACTION_COLORS[i]} text={name} box={BOX} />
              ))
            })()}
          {hover && (
            <g>
              <HoverCrosshair x={scale.x(hover.t)} box={BOX} />
              {[0, 1, 2].map((i) => (
                <circle
                  key={i}
                  cx={scale.x(hover.t)}
                  cy={y(hover.avg[i])}
                  r={3}
                  fill={ACTION_COLORS[i]}
                  stroke="var(--surface)"
                  strokeWidth="1.5"
                />
              ))}
            </g>
          )}
        </svg>
        {hover && wrapRef.current && (
          <ChartTooltip x={scale.x(hover.t)} wrap={wrapRef.current} clamp={130} t={hover.t} box={BOX}>
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
