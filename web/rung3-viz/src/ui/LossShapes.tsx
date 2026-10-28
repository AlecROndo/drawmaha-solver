import { linScale } from './scales'

/**
 * Why cross-entropy: how squared error and cross-entropy price the same
 * 0.03 miss at a rare action and at a coin flip. Two curves per panel,
 * computed here from the two losses' definitions (nothing read from a
 * file): squared error (q − p)² + ((1 − q) − (1 − p))², and cross-entropy
 * minus its floor, p·log(p/q) + (1 − p)·log((1 − p)/(1 − q)), which is the
 * KL divergence.
 */

const W = 760
const H = 250
const PANEL = { l: 54, r: 24, t: 26, b: 36 }
const HALF = W / 2

const mse = (p: number, q: number) => (q - p) ** 2 + (1 - q - (1 - p)) ** 2
const kl = (p: number, q: number) => p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q))

const PANELS = [
  { p: 0.04, miss: 0.01, lo: 0.002, hi: 0.12, title: 'the action happens 4% of the time; Π says 1%' },
  { p: 0.5, miss: 0.53, lo: 0.38, hi: 0.62, title: 'the action happens 50% of the time; Π says 53%' },
]
const Y_MAX = 0.06

export function LossShapes() {
  const ratio = kl(0.04, 0.01) / mse(0.04, 0.01)
  return (
    <>
      <div className="legend">
        <span className="item">
          <span className="swatch" style={{ background: 'var(--loss-mse)' }} />
          squared error
        </span>
        <span className="item">
          <span className="swatch" style={{ background: 'var(--loss-ce)' }} />
          cross-entropy, minus its floor
        </span>
        <span className="item">
          <small>the same 0.03 miss costs {ratio.toFixed(0)}× more under cross-entropy at the rare action, and the same at the coin flip</small>
        </span>
      </div>
      <svg className="chart lossshapes" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Squared error and cross-entropy for a 0.03 miss at a 4 percent action and at a 50 percent action">
        {PANELS.map((panel, k) => {
          const x0 = k === 0 ? PANEL.l : HALF + PANEL.l
          const x1 = k === 0 ? HALF - PANEL.r : W - PANEL.r
          const x = linScale(panel.lo, panel.hi, x0, x1)
          const y = linScale(0, Y_MAX, H - PANEL.b, PANEL.t)
          const qs = Array.from({ length: 120 }, (_, i) => panel.lo + ((panel.hi - panel.lo) * i) / 119)
          const curve = (f: (p: number, q: number) => number) => qs.map((q, i) => `${i ? 'L' : 'M'} ${x(q).toFixed(1)} ${y(Math.min(f(panel.p, q), Y_MAX)).toFixed(1)}`).join(' ')
          return (
            <g key={panel.p}>
              <text x={x0} y={PANEL.t - 10} fontSize="10.5" fill="var(--panel-dim)">
                {panel.title}
              </text>
              {[0, 0.02, 0.04, 0.06].map((v) => (
                <g key={v}>
                  <line x1={x0} x2={x1} y1={y(v)} y2={y(v)} stroke="var(--panel-hair)" />
                  <text x={x0 - 8} y={y(v) + 3.5} textAnchor="end" fontSize="10.5" fill="var(--panel-dim)">
                    {v.toFixed(2)}
                  </text>
                </g>
              ))}
              <line x1={x(panel.p)} x2={x(panel.p)} y1={PANEL.t} y2={H - PANEL.b} stroke="var(--panel-dim)" strokeDasharray="2 3" />
              <text x={x(panel.p)} y={H - 14} textAnchor="middle" fontSize="10.5" fill="var(--panel-dim)">
                truth {panel.p}
              </text>
              <text x={x1} y={H - 14} textAnchor="end" fontSize="10" fill="var(--panel-dim)">
                Π's probability
              </text>
              <path d={curve(mse)} fill="none" stroke="var(--loss-mse)" strokeWidth={1.8} />
              <path d={curve(kl)} fill="none" stroke="var(--loss-ce)" strokeWidth={2.2} />
              {[
                { f: mse, ink: 'var(--loss-mse)', dy: 14 },
                { f: kl, ink: 'var(--loss-ce)', dy: -8 },
              ].map(({ f, ink, dy }) => (
                <g key={ink}>
                  <circle cx={x(panel.miss)} cy={y(Math.min(f(panel.p, panel.miss), Y_MAX))} r={4} fill={ink} />
                  <text x={x(panel.miss) + 8} y={y(Math.min(f(panel.p, panel.miss), Y_MAX)) + dy} fontSize="10.5" fill={ink}>
                    {f(panel.p, panel.miss).toFixed(4)}
                  </text>
                </g>
              ))}
            </g>
          )
        })}
      </svg>
    </>
  )
}
