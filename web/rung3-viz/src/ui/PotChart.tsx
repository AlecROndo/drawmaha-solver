import { POT_ARMS } from '../deep'
import { perHundred } from '../research'
import { linScale } from './scales'

/**
 * The regret targets divided by pot^k before the refit, at one and two
 * million hands: the same 4 × 256 net, the same 16,000-step refit, the same
 * hands; only the targets' scale changes. Two groups, one per mark; within a
 * group the raw arms are ivory, the pot arms the site's winning ink, and
 * every bar is named at its left and valued at its right, so the shades
 * carry no information a reader cannot also read.
 */

const W = 760
const ROW = 30
const GAP = 22
const M = { l: 214, r: 170, t: 12 }
const X_MAX = 15
const x = linScale(0, X_MAX, M.l, W - M.r)
const MARKS = [100, 200] as const

const lrWord = (lr: number) => (lr === 0.001 ? '1e-3' : lr === 0.0005 ? '5e-4' : String(lr))
const kWord = (k: number) => (k === 0 ? 'raw chips' : k === 0.5 ? '÷ √pot' : k === 1 ? '÷ pot' : `÷ pot^${k}`)
const cut = (from: number, to: number) => `−${Math.round((1 - to / from) * 100)}%`

export function PotChart() {
  const rows = MARKS.flatMap((t, g) =>
    POT_ARMS.map((a, i) => ({ ...a, t, value: a.exact[t], y: M.t + (g * POT_ARMS.length + i) * ROW + g * GAP })),
  )
  const H = rows.at(-1)!.y + ROW + 46
  const rawAt = (t: 100 | 200) => POT_ARMS.find((a) => a.potPower === 0 && a.lr === 0.0005)!.exact[t]
  return (
    <svg className="chart pot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Exact average of the regret targets over pot^k at one and two million hands per seat">
      {[0, 5, 10, 15].map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={M.t} y2={H - 46} stroke="var(--panel-hair)" />
          <text x={x(v)} y={H - 30} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
            {v}
          </text>
        </g>
      ))}
      <text x={M.l} y={H - 10} fontSize="10" fill="var(--panel-dim)">
        chips / 100 hands, the exact average
      </text>
      {rows.map((a) => {
        const v = perHundred(a.value)
        const first = rows.find((r) => r.t === a.t) === a
        const pot = a.potPower > 0
        const note = pot ? `${cut(rawAt(a.t), a.value)} against raw` : ''
        return (
          <g key={`${a.arm}@${a.t}`}>
            {first && (
              <text x={16} y={a.y + 16} fontSize="12" fill="var(--ivory)">
                {a.t / 100}M hands
              </text>
            )}
            <text x={M.l - 10} y={a.y + 16} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {kWord(a.potPower)} · lr {lrWord(a.lr)}
            </text>
            <rect x={M.l} y={a.y + 4} width={x(v) - M.l} height={ROW - 10} fill={pot ? 'var(--yellow)' : 'var(--ivory)'} fillOpacity={pot ? 0.9 : a.lr === 0.0005 ? 0.6 : 0.35}>
              <title>{`${kWord(a.potPower)}, lr ${lrWord(a.lr)}, ${a.t / 100}M hands per seat: ${v.toFixed(2)} chips per hundred hands`}</title>
            </rect>
            <text x={x(v) + 7} y={a.y + 16} fontSize="12" fill="var(--panel-mark)">
              {v.toFixed(1)}
            </text>
            {note && (
              <text x={x(v) + 44} y={a.y + 16} fontSize="10.5" fill="var(--yellow)">
                {note}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}
