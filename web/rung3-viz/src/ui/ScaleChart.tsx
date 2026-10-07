import { DEALS, DECISION_NODES, FINAL_ITERATION, SIZES, WORKERS } from '../research'
import { Panel } from './site'

/**
 * How big the game is, on one log axis: the information sets of each rung
 * the ladder has climbed, mini-drawmaha's full decision tree, and how many
 * hands the race actually sampled against it. The sampler saw a few hundredths
 * of a percent of the tree per seat and still drove exploitability to 6 chips
 * per hundred hands — which is the whole case for sampling.
 */

const W = 560
const H = 240
const M = { l: 210, r: 70, t: 20, b: 36 }
const X_MAX = 1e12
const x = (v: number) => M.l + (Math.log10(Math.max(v, 1)) / Math.log10(X_MAX)) * (W - M.l - M.r)

const say = (n: number): string => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`
  return String(n)
}

export function ScaleChart() {
  const sampled = FINAL_ITERATION * WORKERS * 2
  const rows: { label: string; value: number; ink: string }[] = [
    ...SIZES.map((s) => ({ label: `${s.game} · infosets`, value: s.infosets, ink: s.rung === 3 ? 'var(--rule-lcfr)' : 'var(--panel-dim)' })),
    { label: 'mini-drawmaha · deals', value: DEALS, ink: 'var(--panel-dim)' },
    { label: 'mini-drawmaha · tree nodes', value: DECISION_NODES, ink: 'var(--rule-cfrplus)' },
    { label: 'hands sampled per rule', value: sampled, ink: 'var(--rule-lcfr)' },
  ]
  const rh = (H - M.t - M.b) / rows.length

  return (
    <Panel
      n="06"
      k="Fig. 6 · how big it is"
      title="A table 21,000 times Leduc’s, and a tree no walk could cross."
      say={
        <>
          Log axis. One pass over the whole tree is out of the question, so the learner samples: {WORKERS} hands
          per seat per iteration, every touched row banked once.
        </>
      }
      label="The size of each rung's game on a log axis"
    >
      <svg className="chart scale" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Game sizes on a log axis">
        {Array.from({ length: 7 }, (_, i) => i * 2).map((p) => (
          <g key={p}>
            <line x1={x(10 ** p)} x2={x(10 ** p)} y1={M.t} y2={H - M.b} stroke="var(--panel-hair)" />
            <text x={x(10 ** p)} y={H - M.b + 16} textAnchor="middle" fontSize="10.5" fill="var(--panel-dim)">
              10{String(p).replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(d)])}
            </text>
          </g>
        ))}
        {rows.map((r, i) => {
          const yMid = M.t + rh * i + rh / 2
          return (
            <g key={r.label}>
              <text x={M.l - 10} y={yMid + 4} textAnchor="end" fontSize="10.5" fill="var(--panel-dim)">
                {r.label}
              </text>
              <rect x={M.l} y={yMid - 8} width={x(r.value) - M.l} height={16} fill={r.ink} fillOpacity={0.85} />
              <text x={x(r.value) + 8} y={yMid + 4} fontSize="11" fill="var(--panel-mark)">
                {say(r.value)}
              </text>
            </g>
          )
        })}
      </svg>
    </Panel>
  )
}
