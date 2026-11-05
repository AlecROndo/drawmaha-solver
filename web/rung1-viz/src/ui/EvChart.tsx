import { Figure } from './site'
import type { ExploitRun } from './useExploit'

/**
 * What the counter-strategy earns, rising toward the exact ceiling.
 *
 * Two horizontal rules carry the whole argument. The upper one is the exact
 * best-response value, enumerated by `exploitability.py` — CFR's curve must
 * arrive there and cannot pass it. The lower one is what an EQUILIBRIUM player
 * earns against the same locked strategy: unbeatable, but not punishing. The
 * band between them is the money exploitative play collects and Nash leaves.
 *
 * CFR's curve is the thing pointed at, so it wears the accent; the ceiling is
 * ink, the known answer; equilibrium play is the muted ink, dashed.
 */

const W = 560
const H = 280
const M = { l: 56, r: 16, t: 28, b: 40 }

const NICE_PADDING = 0.08

/** A chip count with a true minus sign. */
const num = (v: number, digits: number) => (v < 0 ? `−${Math.abs(v).toFixed(digits)}` : v.toFixed(digits))

export function EvChart({ run, index }: { run: ExploitRun; index: number }) {
  const { iterations, ev, ceiling, nashValue, evSeat } = run
  const span = Math.log10(Math.max(iterations[iterations.length - 1], 10))
  const x = (t: number) => M.l + (Math.log10(Math.max(t, 1)) / span) * (W - M.l - M.r)

  const marks = [
    ...ev,
    ...(ceiling === null ? [] : [ceiling]),
    ...(nashValue === null ? [] : [nashValue]),
  ]
  const lo = Math.min(...marks)
  const hi = Math.max(...marks)
  // A flat run would collapse the axis to a single line; pad by a fixed floor
  // so the curve still reads as a curve.
  const pad = Math.max((hi - lo) * NICE_PADDING, 0.02)
  const y = (v: number) => M.t + ((hi + pad - v) / (hi - lo + 2 * pad)) * (H - M.t - M.b)

  const path = ev.map((v, i) => `${i ? 'L' : 'M'} ${x(iterations[i])} ${y(v)}`).join(' ')
  const decades = [1, 10, 100, 1_000, 10_000, 100_000].filter(
    (d) => d <= iterations[iterations.length - 1],
  )

  return (
    <Figure
      n="Figure 2"
      title="The counter-strategy rises to the exact ceiling"
      sub={`Chips per hand to P${evSeat}, iterations on a log axis.`}
      caption="The solid rule is the exact best response; the dashed one is what equilibrium play earns against the same locks. The band between them is what Nash leaves on the table."
      label="Counter-strategy EV over the run"
    >
      <div className="legend">
        <span className="item">
          <span className="swatch" style={{ background: 'var(--series-2)' }} />
          CFR's counter-strategy
        </span>
        <span className="item">
          <span className="swatch" style={{ background: 'var(--series-1)' }} />
          Exact best response
        </span>
        <span className="item">
          <span className="swatch" style={{ background: 'var(--series-3)' }} />
          Equilibrium play
        </span>
      </div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Counter-strategy EV against iterations">
        <text className="axis" x={M.l} y={12}>
          chips/hand to P{evSeat}
        </text>
        {/* Chips are linear and the range is data-driven, so the axis is
            labelled at its own ends rather than at round numbers that might
            fall outside the run entirely. */}
        {[hi + pad, (hi + lo) / 2, lo - pad].map((v) => (
          <g key={v}>
            <line className="grid" x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} />
            <text className="axis" x={M.l - 8} y={y(v) + 4} textAnchor="end">
              {num(v, 3)}
            </text>
          </g>
        ))}
        <line className="axis-line" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        {decades.map((d) => (
          <g key={d}>
            <line className="axis-line" x1={x(d)} x2={x(d)} y1={H - M.b} y2={H - M.b + 4} />
            <text className="axis" x={x(d)} y={H - M.b + 18} textAnchor="middle">
              {d >= 1000 ? `${d / 1000}k` : d}
            </text>
          </g>
        ))}
        <text className="axis" x={W - M.r} y={H - 4} textAnchor="end">
          iterations
        </text>

        {nashValue !== null && (
          <>
            <line
              x1={M.l}
              x2={W - M.r}
              y1={y(nashValue)}
              y2={y(nashValue)}
              stroke="var(--series-3)"
              strokeWidth={1.5}
              strokeDasharray="4 4"
            />
            {/* The best response never earns less than equilibrium play, so
                the two rules can sit a pixel apart: the ceiling is labelled
                above its rule and this one below, and they never collide. */}
            <text className="lbl" x={W - M.r - 8} y={y(nashValue) + 16} textAnchor="end" fill="var(--series-3)">
              Equilibrium play<tspan> {num(nashValue, 4)}</tspan>
            </text>
          </>
        )}
        {ceiling !== null && (
          <>
            <line x1={M.l} x2={W - M.r} y1={y(ceiling)} y2={y(ceiling)} stroke="var(--series-1)" strokeWidth={1.5} />
            <text className="lbl" x={W - M.r - 8} y={y(ceiling) - 7} textAnchor="end" fill="var(--series-1)">
              Exact best response<tspan> {num(ceiling, 4)}</tspan>
            </text>
          </>
        )}

        <path d={path} fill="none" stroke="var(--series-2)" strokeWidth={2.5} strokeLinejoin="round" />
        <line className="playhead" x1={x(iterations[index])} x2={x(iterations[index])} y1={M.t} y2={H - M.b} />
        <circle
          cx={x(iterations[index])}
          cy={y(ev[index])}
          r={4}
          fill="var(--series-2)"
          stroke="var(--paper)"
          strokeWidth={2}
        />
      </svg>
      <dl className="stats">
        <div className="stat">
          <dt>Earning now</dt>
          <dd>{num(ev[index], 5)}</dd>
        </div>
        <div className="stat">
          <dt>Exact ceiling</dt>
          <dd>{ceiling === null ? '—' : num(ceiling, 5)}</dd>
        </div>
        <div className="stat">
          <dt>Gap to it</dt>
          <dd>{ceiling === null ? '—' : num(ceiling - ev[index], 5)}</dd>
        </div>
      </dl>
    </Figure>
  )
}
