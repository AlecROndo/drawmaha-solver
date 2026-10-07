import { FINAL_COLUMNS, FINAL_ITERATION, RULES, RULE_NAME, WORKERS, perHundred, type Column } from '../research'
import { RULE_INK } from '../inks'
import { Panel } from './site'

/**
 * Which average to read out of a run. Every rule banks its primary, linear
 * (weight t) average and may bank further columns of the very same play at
 * other weights — uniform (weight 1), quadratic (weight t²) — for free,
 * since nothing reads an average while training. Here they are at the end
 * of the race: linear beats uniform under vanilla and CFR+, and DCFR's own
 * quadratic average edges its linear one.
 */

const W = 560
const H = 230
const M = { l: 54, r: 20, t: 18, b: 44 }

const Y_MAX = 40
const y = (perHundredHands: number) => M.t + (1 - Math.min(perHundredHands, Y_MAX) / Y_MAX) * (H - M.t - M.b)

const COLUMN_ORDER: Column[] = ['linear', 'uniform', 'quadratic']
const COLUMN_WORD: Record<Column, string> = { linear: 'linear · t', uniform: 'uniform · 1', quadratic: 'quadratic · t²' }

export function ColumnsChart() {
  const groups = RULES.map((rule) => ({
    rule,
    bars: COLUMN_ORDER.map((column) => FINAL_COLUMNS.find((p) => p.rule === rule && p.column === column)).filter(
      (p): p is NonNullable<typeof p> => p !== undefined,
    ),
  }))
  const groupW = (W - M.l - M.r) / groups.length
  const barW = 34
  const gap = 8

  return (
    <Panel
      n="02"
      k="Fig. 2 · the averaging columns"
      title="Linear averaging beats uniform; the rule is still what decides."
      say={
        <>
          Exploitability at {((FINAL_ITERATION * WORKERS) / 1e6).toFixed(0)}M hands per seat of each average a run banked,
          chips per hundred hands. The columns cost nothing and cannot change the play.
        </>
      }
      label="Exploitability of each averaging column at the end of the race"
    >
      <svg className="chart columns" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Averaging columns at the end of the race">
        {[0, 10, 20, 30, 40].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke="var(--panel-hair)" />
            <text x={M.l - 8} y={y(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {v}
            </text>
          </g>
        ))}
        {groups.map(({ rule, bars }, g) => {
          const centre = M.l + groupW * g + groupW / 2
          const total = bars.length * barW + (bars.length - 1) * gap
          return (
            <g key={rule}>
              {bars.map((p, i) => {
                const x = centre - total / 2 + i * (barW + gap)
                const v = perHundred(p.exploitability)
                return (
                  <g key={p.column}>
                    <rect
                      x={x}
                      y={y(v)}
                      width={barW}
                      height={y(0) - y(v)}
                      fill={RULE_INK[rule]}
                      fillOpacity={p.column === 'linear' ? 1 : 0.45}
                      stroke={p.column === 'linear' ? 'none' : RULE_INK[rule]}
                      strokeDasharray={p.column === 'quadratic' ? '3 3' : undefined}
                    />
                    <text x={x + barW / 2} y={y(v) - 6} textAnchor="middle" fontSize="11" fill="var(--panel-mark)">
                      {v.toFixed(1)}
                    </text>
                    <text x={x + barW / 2} y={y(0) + 13} textAnchor="middle" fontSize="9.5" fill="var(--panel-dim)">
                      {p.column === 'linear' ? 't' : p.column === 'uniform' ? '1' : 't²'}
                    </text>
                  </g>
                )
              })}
              <text x={centre} y={H - 10} textAnchor="middle" fontSize="11.5" fill={RULE_INK[rule]}>
                {RULE_NAME[rule]}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="legend">
        {COLUMN_ORDER.map((c) => (
          <span key={c} className="item">
            <span className={`swatch col-${c}`} />
            {COLUMN_WORD[c]}
          </span>
        ))}
      </div>
    </Panel>
  )
}
