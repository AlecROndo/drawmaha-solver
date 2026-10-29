import type { Engine } from '../sim/engine'
import { ACTIONS } from '../sim/game'
import { ACTION_DASH, AVG, fmt } from './format'
import { makeLogX, spreadLabels } from './chartScale'
import {
  ChartTooltip,
  DecadeGridlines,
  EdgeLabel,
  FULL,
  HoverCrosshair,
  LegendKey,
  YGridline,
  useChartHover,
  type ChartBox,
} from './chartChrome'
import { Figure } from './site'

const BOX: ChartBox = { ...FULL, l: 50, r: 90, t: 14, b: 30 }

/**
 * Average-strategy components vs iteration, log x, dashed reference at ⅓.
 * Runs the full width of the container: it is one long time axis, and cutting
 * it to half a column would throw away the decades that make the walk read.
 * All three lines are the average, so all three are the accent; the actions
 * are told apart by dash and by the label at the right edge.
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
  const mode = engine.config.mode
  const title =
    mode === 'self-play'
      ? "Each action's share of the average walks to one third"
      : mode === 'vs-fixed'
        ? 'Against a fixed opponent the average walks to the action that beats it'
        : 'Against you the average moves toward what beats your throws'

  return (
    <Figure
      n="Figure 5"
      wide
      title={title}
      sub="The average S/n per action, against rounds on a log axis. The grey dash is ⅓, the uniform Nash."
      label="Convergence of the average strategy"
    >
      <div className="legend">
        {ACTIONS.map((name, i) => (
          <LegendKey key={name} color={AVG} dash={ACTION_DASH[i]}>
            {name}
          </LegendKey>
        ))}
        <LegendKey color="var(--ink-3)" dash="4 3">
          ⅓, the uniform Nash
        </LegendKey>
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
            stroke="var(--ink-3)"
            strokeDasharray="4 3"
          />
          {history.length > 1 &&
            [0, 1, 2].map((i) => (
              <path
                key={i}
                d={path(i)}
                fill="none"
                stroke={AVG}
                strokeWidth="2"
                strokeDasharray={ACTION_DASH[i]}
                strokeLinejoin="round"
              />
            ))}
          {/* direct labels at the right edge, spread to avoid collisions */}
          {last &&
            (() => {
              const ys = spreadLabels(
                [0, 1, 2].map((i) => y(last.avg[i])),
                14,
                BOX.t + 6,
                BOX.h - BOX.b - 4,
              )
              return ACTIONS.map((name, i) => (
                <EdgeLabel key={name} y={ys[i]} color={AVG} dash={ACTION_DASH[i]} text={name} box={BOX} />
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
                  r={4}
                  fill={AVG}
                  stroke="var(--paper)"
                  strokeWidth="2"
                />
              ))}
            </g>
          )}
        </svg>
        {hover && wrapRef.current && (
          <ChartTooltip x={scale.x(hover.t)} wrap={wrapRef.current} clamp={150} t={hover.t} box={BOX}>
            {ACTIONS.map((name, i) => (
              <div className="row" key={name}>
                <span>{name}</span>
                <span className="val">{fmt(hover.avg[i])}</span>
              </div>
            ))}
          </ChartTooltip>
        )}
      </div>
    </Figure>
  )
}
