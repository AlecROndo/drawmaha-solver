import type { ExploitRun } from './useExploit'

/**
 * What the counter-strategy earns, climbing toward the exact ceiling.
 *
 * Two horizontal rules carry the whole argument. The upper one is the exact
 * best-response value, enumerated by `exploitability.py` — CFR's curve must
 * arrive there and cannot pass it. The lower one is what an EQUILIBRIUM player
 * earns against the same locked strategy: unbeatable, but not punishing. The
 * band between them is the money exploitative play collects and Nash leaves.
 */

const W = 560
const H = 260
const M = { l: 52, r: 16, t: 14, b: 30 }

const NICE_PADDING = 0.08

export function EvChart({ run, index }: { run: ExploitRun; index: number }) {
  const { iterations, ev, ceiling, nashValue, evSeat } = run
  const span = Math.log10(Math.max(iterations[iterations.length - 1], 10))
  const x = (t: number) => M.l + (Math.log10(Math.max(t, 1)) / span) * (W - M.l - M.r)

  const marks = [...ev, ...(ceiling === null ? [] : [ceiling]), ...(nashValue === null ? [] : [nashValue])]
  const lo = Math.min(...marks)
  const hi = Math.max(...marks)
  // A flat run would collapse the axis to a single line; pad by a fixed floor
  // so the curve still reads as a curve.
  const pad = Math.max((hi - lo) * NICE_PADDING, 0.02)
  const y = (v: number) =>
    M.t + ((hi + pad - v) / (hi - lo + 2 * pad)) * (H - M.t - M.b)

  const path = ev.map((v, i) => `${i ? 'L' : 'M'} ${x(iterations[i])} ${y(v)}`).join(' ')
  const decades = [1, 10, 100, 1_000, 10_000, 100_000].filter(
    (d) => d <= iterations[iterations.length - 1],
  )

  return (
    <section className="panel" aria-label="Counter-strategy EV over the run">
      <h2>Fig. 2 · The counter-strategy climbs to the exact ceiling</h2>
      <p className="sub">
        chips/hand to P{evSeat} · x log · upper rule is the exact best response, lower rule is
        what equilibrium play earns here
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Counter-strategy EV against iterations">
        {/* Chips are linear and the range is data-driven, so the axis is
            labelled at its own ends rather than at round numbers that might
            fall outside the run entirely. */}
        {[hi + pad, (hi + lo) / 2, lo - pad].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
            <text x={M.l - 6} y={y(v) + 3} textAnchor="end" fontSize="9.5" fill="var(--muted)">
              {v.toFixed(3)}
            </text>
          </g>
        ))}
        {decades.map((d) => (
          <g key={d}>
            <line x1={x(d)} x2={x(d)} y1={M.t} y2={H - M.b} stroke="var(--grid)" />
            <text x={x(d)} y={H - 10} textAnchor="middle" fontSize="9.5" fill="var(--muted)">
              {d >= 1000 ? `${d / 1000}k` : d}
            </text>
          </g>
        ))}

        {nashValue !== null && (
          <>
            <line
              x1={M.l}
              x2={W - M.r}
              y1={y(nashValue)}
              y2={y(nashValue)}
              stroke="var(--cur-obj)"
              strokeWidth={1.2}
              strokeDasharray="3 3"
            />
            <text x={W - M.r} y={y(nashValue) - 5} textAnchor="end" fontSize="9.5" fill="var(--cur-obj)">
              equilibrium play {nashValue.toFixed(4)}
            </text>
          </>
        )}
        {ceiling !== null && (
          <>
            <line
              x1={M.l}
              x2={W - M.r}
              y1={y(ceiling)}
              y2={y(ceiling)}
              stroke="var(--ink-2)"
              strokeWidth={1.2}
            />
            <text x={W - M.r} y={y(ceiling) - 5} textAnchor="end" fontSize="9.5" fill="var(--ink-2)">
              exact best response {ceiling.toFixed(4)}
            </text>
          </>
        )}

        <path d={path} fill="none" stroke="var(--avg-obj)" strokeWidth={2} />
        <line x1={x(iterations[index])} x2={x(iterations[index])} y1={M.t} y2={H - M.b} stroke="var(--ink)" strokeWidth={1} />
        <circle cx={x(iterations[index])} cy={y(ev[index])} r={3.5} fill="var(--avg-obj)" />
      </svg>
      <dl className="stat-row">
        <div>
          <dt>
            <span className="dot" style={{ background: 'var(--avg-obj)' }} /> earning now
          </dt>
          <dd>{ev[index].toFixed(5)}</dd>
        </div>
        <div>
          <dt>exact ceiling</dt>
          <dd>{ceiling === null ? '—' : ceiling.toFixed(5)}</dd>
        </div>
        <div>
          <dt>gap to it</dt>
          <dd>{ceiling === null ? '—' : (ceiling - ev[index]).toFixed(5)}</dd>
        </div>
      </dl>
    </section>
  )
}
