import { SIZE_SWEEP, SWEEP_RULE, lcfrAt } from '../deep'
import { perHundred } from '../research'
import { linScale } from './scales'

/**
 * Width against depth at one million hands: five nets, one bar each, the
 * exact average in chips per hundred hands. Width is told by the row and
 * depth by shade (four layers darker, three lighter); the lowest bar, the
 * finding, is the accent. The percentages the finding rests on are computed
 * from the bars they sit beside.
 */

const W = 860
const ROW = 34
const M = { l: 92, r: 150, t: 18 }
const H = M.t + SIZE_SWEEP.length * ROW + 58
const X_MAX = 45
const x = linScale(0, X_MAX, M.l, W - M.r)

const cut = (from: number, to: number) => `−${Math.round((1 - to / from) * 100)}%`

export function SizeChart() {
  const at = (net: string) => SIZE_SWEEP.find((s) => s.net === net)!
  // the lowest grade is the finding, in the accent; depth is told by shade, four layers darker
  const best = SIZE_SWEEP.reduce((a, b) => (b.exact < a.exact ? b : a))
  const notes: Record<string, string> = {
    '4 × 64': `${cut(at('3 × 64').exact, at('4 × 64').exact)} from a fourth layer`,
    '3 × 128': `${cut(at('3 × 64').exact, at('3 × 128').exact)} from doubling the width`,
    '4 × 128': `${cut(at('3 × 128').exact, at('4 × 128').exact)} from a fourth layer`,
    '4 × 256': `${cut(at('4 × 128').exact, at('4 × 256').exact)} from doubling again`,
  }
  return (
    <svg className="chart size" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Exact average of five net sizes at one million hands per seat">
      {[0, 10, 20, 30, 40].map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={M.t} y2={M.t + SIZE_SWEEP.length * ROW} stroke="var(--grid)" />
          <text x={x(v)} y={M.t + SIZE_SWEEP.length * ROW + 16} textAnchor="middle" fontSize="11" fill="var(--ink-3)">
            {v}
          </text>
        </g>
      ))}
      <text x={M.l} y={M.t + SIZE_SWEEP.length * ROW + 36} fontSize="10" fill="var(--ink-3)">
        chips / 100 hands · LCFR at the same hands: {perHundred(lcfrAt(1_000_000)).toFixed(1)}, off the scale
      </text>
      {SIZE_SWEEP.map((s, i) => {
        const yTop = M.t + i * ROW + 5
        const v = perHundred(s.exact)
        return (
          <g key={s.net}>
            <text x={M.l - 10} y={yTop + 17} textAnchor="end" fontSize="12" fill={s.net === SWEEP_RULE.winner.split('x').reverse().join(' × ') ? 'var(--ink)' : 'var(--ink-3)'}>
              {s.net}
            </text>
            <rect x={M.l} y={yTop} width={x(v) - M.l} height={ROW - 10} fill={s === best ? 'var(--series-2)' : s.depth === 4 ? 'var(--ink-2)' : 'var(--rule-2)'} />
            <text x={x(v) + 7} y={yTop + 17} fontSize="12" fill="var(--ink)">
              {v.toFixed(1)}
            </text>
            {notes[s.net] && (
              <text x={x(v) + 48} y={yTop + 17} fontSize="10.5" fill="var(--series-2)">
                {notes[s.net]}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}
