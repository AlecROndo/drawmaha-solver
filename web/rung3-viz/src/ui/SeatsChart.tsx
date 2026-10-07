import { FROZEN, WORKERS, perHundred, series } from '../research'
import { Panel } from './site'

/**
 * Two seats, two best responses. Exploitability is the mean of what a
 * perfect P0 wins against the strategy's P1 and what a perfect P1 wins
 * against its P0. The game is not symmetric — P1 acts last in both rounds
 * and at the draw — so the two halves are read against the seats' own
 * values in self-play, not against zero: a perfect P0 still LOSES 0.034 a
 * hand, because the seat is worth −0.091 to begin with.
 */

const W = 560
const H = 120
const M = { l: 30, r: 30 }
const X_MIN = -0.2
const X_MAX = 0.45
const x = (v: number) => M.l + ((v - X_MIN) / (X_MAX - X_MIN)) * (W - M.l - M.r)

const signed = (v: number, d = 3) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`

export function SeatsChart() {
  const v0 = FROZEN.value_p0
  const v1 = -FROZEN.value_p0
  const br0 = FROZEN.br0
  const br1 = FROZEN.br1
  const gain0 = br0 - v0
  const gain1 = br1 - v1
  const run = series('lcfr')

  return (
    <Panel
      n="03"
      k="Fig. 3 · two seats, two best responses"
      title="A perfect P0 still loses: the seat is worth a tenth of an ante to P1."
      say={
        <>
          Chips per hand for the frozen strategy. Each seat’s value when the strategy plays itself (hollow), and
          what a perfect adversary in that seat wins against it (filled). Exploitability is the mean of the two
          gains over the seats’ own values.
        </>
      }
      label="Best responses against the frozen strategy, by seat"
    >
      <svg className="chart seats" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Seat values and best responses on a number line">
        {[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4].map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={22} y2={H - 26} stroke={v === 0 ? 'var(--panel-dim)' : 'var(--panel-hair)'} />
            <text x={x(v)} y={H - 10} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
              {v === 0 ? '0' : signed(v, 1)}
            </text>
          </g>
        ))}
        {/* P0's row */}
        <text x={M.l} y={38} fontSize="10.5" fill="var(--panel-dim)">
          P0
        </text>
        <line x1={x(v0)} x2={x(br0)} y1={46} y2={46} stroke="var(--seat-p0)" strokeWidth={2} />
        <circle cx={x(v0)} cy={46} r={4.5} fill="var(--panel)" stroke="var(--seat-p0)" strokeWidth={1.6} />
        <circle cx={x(br0)} cy={46} r={4.5} fill="var(--seat-p0)" />
        <text x={x(v0)} y={35} textAnchor="middle" fontSize="10.5" fill="var(--seat-p0)">
          self-play {signed(v0)}
        </text>
        <text x={x(br0) + 8} y={50} fontSize="10.5" fill="var(--seat-p0)">
          best response {signed(br0)} · gains {signed(gain0)}
        </text>
        {/* P1's row */}
        <text x={M.l} y={72} fontSize="10.5" fill="var(--panel-dim)">
          P1
        </text>
        <line x1={x(v1)} x2={x(br1)} y1={80} y2={80} stroke="var(--seat-p1)" strokeWidth={2} />
        <circle cx={x(v1)} cy={80} r={4.5} fill="var(--panel)" stroke="var(--seat-p1)" strokeWidth={1.6} />
        <circle cx={x(br1)} cy={80} r={4.5} fill="var(--seat-p1)" />
        <text x={x(v1)} y={69} textAnchor="middle" fontSize="10.5" fill="var(--seat-p1)">
          self-play {signed(v1)}
        </text>
        <text x={x(br1) + 8} y={84} fontSize="10.5" fill="var(--seat-p1)">
          best response {signed(br1)} · gains {signed(gain1)}
        </text>
      </svg>
      <dl className="stat">
        <div>
          <dt>exploitability</dt>
          <dd>{FROZEN.exploitability.toFixed(4)}</dd>
        </div>
        <div>
          <dt>= (BR₀ + BR₁) / 2</dt>
          <dd>
            ({signed(br0)} {br1 >= 0 ? '+' : '−'} {Math.abs(br1).toFixed(3)}) / 2
          </dd>
        </div>
        <div>
          <dt>= mean gain over the seats</dt>
          <dd>
            ({signed(gain0)} + {signed(gain1)}) / 2
          </dd>
        </div>
      </dl>
      <table className="rows run" aria-label="The LCFR run, checkpoint by checkpoint">
        <thead>
          <tr>
            <th>hands/seat</th>
            <th>BR₀</th>
            <th>BR₁</th>
            <th>exploit. /100</th>
            <th>P0 value</th>
          </tr>
        </thead>
        <tbody>
          {run.map((p) => (
            <tr key={p.iteration}>
              <td>{(p.iteration * WORKERS).toLocaleString()}</td>
              <td>{signed(p.br0)}</td>
              <td>{signed(p.br1)}</td>
              <td>{perHundred(p.exploitability).toFixed(1)}</td>
              <td>{signed(p.value_p0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  )
}
