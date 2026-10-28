import { REFIT_ARMS } from '../deep'
import { perHundred } from '../research'
import { linScale } from './scales'

/**
 * Refit steps and learning rate at one million hands: the same net on the
 * same hands, only the fit changed. Two groups, one per net; within a group
 * the shade deepens with the step count.
 */

const W = 760
const ROW = 30
const GAP = 22
const M = { l: 262, r: 150, t: 12 }
const X_MAX = 30
const x = linScale(0, X_MAX, M.l, W - M.r)

const cut = (from: number, to: number) => `−${Math.round((1 - to / from) * 100)}%`
const shade = (steps: number) => (steps >= 16_000 ? 0.95 : steps >= 8_000 ? 0.7 : 0.45)
const lrWord = (lr: number) => (lr === 0.001 ? '1e-3' : lr === 0.0005 ? '5e-4' : String(lr))

export function RefitChart() {
  const nets = [...new Set(REFIT_ARMS.map((a) => a.net))]
  const rows = REFIT_ARMS.map((a, i) => ({ ...a, y: M.t + i * ROW + nets.indexOf(a.net) * GAP }))
  const H = rows.at(-1)!.y + ROW + 46
  const note = (a: (typeof rows)[number]) => {
    const group = rows.filter((r) => r.net === a.net)
    const i = group.indexOf(a)
    if (i === 0) return ''
    const prev = group[i - 1]
    return `${cut(prev.exact, a.exact)} against the ${prev.steps.toLocaleString()}-step fit`
  }
  return (
    <svg className="chart refit" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Exact average of five refit settings at one million hands per seat">
      {[0, 10, 20, 30].map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={M.t} y2={H - 46} stroke="var(--panel-hair)" />
          <text x={x(v)} y={H - 30} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
            {v}
          </text>
        </g>
      ))}
      <text x={M.l} y={H - 10} fontSize="10" fill="var(--panel-dim)">
        chips / 100 hands
      </text>
      {rows.map((a) => {
        const v = perHundred(a.exact)
        const first = rows.find((r) => r.net === a.net) === a
        return (
          <g key={a.arm}>
            {first && (
              <text x={16} y={a.y + 16} fontSize="12" fill="var(--ivory)">
                {a.net}
              </text>
            )}
            <text x={M.l - 10} y={a.y + 16} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {a.steps.toLocaleString()} steps · lr {lrWord(a.lr)}
            </text>
            <rect x={M.l} y={a.y + 4} width={x(v) - M.l} height={ROW - 10} fill="var(--ivory)" fillOpacity={shade(a.steps)} />
            <text x={x(v) + 7} y={a.y + 16} fontSize="12" fill="var(--panel-mark)">
              {v.toFixed(1)}
            </text>
            {note(a) && (
              <text x={x(v) + 48} y={a.y + 16} fontSize="10.5" fill="var(--yellow)">
                {note(a)}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}
