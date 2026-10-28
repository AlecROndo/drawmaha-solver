import { RULE_SCREEN, SIGN_FLIPS } from '../deep'
import { UNIFORM_RANDOM, perHundred } from '../research'
import { linScale, logScale } from './scales'

/**
 * The per-sample regret rules under a net, two panels. Left: the exact
 * average of the 4 × 256 net under linear, DCFR and CFR+ weighting against
 * iteration, log axes. Right: the mechanism — the share of seat 0's bad
 * cells (negative under linear weights) that each rule turns positive, by
 * mark. Regret matching plays every positive cell.
 */

const W = 760
const H = 300
const PANEL = { l: 54, r: 24, t: 26, b: 40 }
const HALF = W / 2
const RULE_INK = { linear: 'var(--rule-lcfr)', dcfr: 'var(--rule-dcfr)', cfrplus: 'var(--rule-cfrplus)' } as const
const RULE_WORD = { linear: 'linear', dcfr: 'DCFR, per sample', cfrplus: 'CFR+, per sample' } as const

export function RuleScreenChart() {
  const x = logScale(10, 100, PANEL.l, HALF - PANEL.r)
  const y = logScale(10, 600, H - PANEL.b, PANEL.t)
  const rules = ['linear', 'dcfr', 'cfrplus'] as const
  const ts = [10, 20, 50]
  const xb = linScale(0, ts.length, HALF + PANEL.l, W - PANEL.r)
  const yb = linScale(0, 0.45, H - PANEL.b, PANEL.t)
  const band = (xb(1) - xb(0)) / 1
  const barW = band / 3.2
  const flips = (rule: 'dcfr' | 'cfrplus', t: number) => SIGN_FLIPS[rule].find((f) => f.t === t && f.seat === 0)!
  return (
    <>
      <div className="legend">
        {rules.map((r) => (
          <span key={r} className="item">
            <span className="swatch" style={{ background: RULE_INK[r] }} />
            {RULE_WORD[r]}
          </span>
        ))}
      </div>
      <svg className="chart rulescreen" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-sample regret rules under a net: exploitability by iteration, and the share of bad cells turned positive">
        <text x={PANEL.l} y={PANEL.t - 10} fontSize="10.5" fill="var(--panel-dim)">
          exact average, 4 × 256 · chips / 100 hands, log
        </text>
        {[10, 100].map((v) => (
          <g key={v}>
            <line x1={PANEL.l} x2={HALF - PANEL.r} y1={y(v)} y2={y(v)} stroke="var(--panel-hair)" />
            <text x={PANEL.l - 8} y={y(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {v}
            </text>
          </g>
        ))}
        {[10, 20, 50, 100].map((t) => (
          <text key={t} x={x(t)} y={H - 16} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
            t = {t}
          </text>
        ))}
        <line x1={PANEL.l} x2={HALF - PANEL.r} y1={y(perHundred(UNIFORM_RANDOM.exploitability))} y2={y(perHundred(UNIFORM_RANDOM.exploitability))} stroke="var(--panel-dim)" strokeDasharray="4 4" />
        <text x={HALF - PANEL.r} y={y(perHundred(UNIFORM_RANDOM.exploitability)) - 5} textAnchor="end" fontSize="10.5" fill="var(--panel-dim)">
          random {perHundred(UNIFORM_RANDOM.exploitability).toFixed(0)}
        </text>
        {rules.map((r) => {
          const pts = RULE_SCREEN[r]
          return (
            <g key={r}>
              <path d={pts.map((p, i) => `${i ? 'L' : 'M'} ${x(p.t).toFixed(1)} ${y(perHundred(p.exact)).toFixed(1)}`).join(' ')} fill="none" stroke={RULE_INK[r]} strokeWidth={r === 'linear' ? 2.2 : 1.6} strokeDasharray={r === 'dcfr' ? '6 4' : undefined} />
              {pts.map((p) => (
                <circle key={p.t} cx={x(p.t)} cy={y(perHundred(p.exact))} r={3} fill={RULE_INK[r]} />
              ))}
              <text x={x(pts.at(-1)!.t) + (r === 'linear' ? 8 : -8)} y={y(perHundred(pts.at(-1)!.exact)) + (r === 'cfrplus' ? -6 : r === 'dcfr' ? 14 : 4)} textAnchor={r === 'linear' ? 'start' : 'end'} fontSize="11" fill={RULE_INK[r]}>
                {perHundred(pts.at(-1)!.exact).toFixed(1)}
              </text>
            </g>
          )
        })}
        {/* right: flipped share */}
        <text x={HALF + PANEL.l} y={PANEL.t - 10} fontSize="10.5" fill="var(--panel-dim)">
          seat 0: bad cells turned positive, by mark
        </text>
        {[0.1, 0.2, 0.3, 0.4].map((v) => (
          <g key={v}>
            <line x1={HALF + PANEL.l} x2={W - PANEL.r} y1={yb(v)} y2={yb(v)} stroke="var(--panel-hair)" />
            <text x={HALF + PANEL.l - 8} y={yb(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {ts.map((t, i) => (
          <g key={t}>
            <text x={xb(i) + band / 2} y={H - 16} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
              t = {t}
            </text>
            {(['dcfr', 'cfrplus'] as const).map((r, j) => {
              const f = flips(r, t)
              const bx = xb(i) + band / 2 + (j - 1) * barW + barW * 0.1
              return (
                <g key={r}>
                  <rect x={bx} y={yb(f.flipped)} width={barW * 0.8} height={yb(0) - yb(f.flipped)} fill={RULE_INK[r]} fillOpacity={r === 'dcfr' ? 0.7 : 1}>
                    <title>{`${RULE_WORD[r]}, t = ${t}: ${(f.flipped * 100).toFixed(1)}% of bad cells flipped; ${(f.negativeKept * 100).toFixed(1)}% of negative regret mass kept`}</title>
                  </rect>
                  <text x={bx + barW * 0.4} y={yb(f.flipped) - 5} textAnchor="middle" fontSize="10.5" fill="var(--panel-mark)">
                    {Math.round(f.flipped * 100)}%
                  </text>
                </g>
              )
            })}
          </g>
        ))}
      </svg>
    </>
  )
}
