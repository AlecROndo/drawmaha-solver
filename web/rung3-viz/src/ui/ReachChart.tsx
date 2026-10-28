import { REACH, REACH_NAME, binLabel, perVisit, type ReachProfile } from '../reach'
import { linScale, logScale } from './scales'

/**
 * Where the exploitability lives: one row per reach band, commonest spots
 * first. Left, each strategy's share of its own exploitability that sits in
 * the band, with the band's share of the infosets as the quiet bar. Right,
 * on a log axis, what a spot in the band leaks each time it is reached: the
 * band's exploitability over its frequency, chips per hundred visits. The
 * left panel says where the total comes from; the right says how bad a row
 * is once play gets there, which is what the sampler's coverage decides.
 */

const W = 760
const M = { l: 214, t: 14 }
const SPLIT = 470
const RIGHT = { l: SPLIT + 40, r: 720 }
const BAR = 12
const INK: Record<keyof typeof REACH, string> = { tabular: 'var(--rule-lcfr)', 'deep-exact': 'var(--obj-exact)', 'deep-pi': 'var(--obj-pi)' }

const share = (p: ReachProfile, i: number) => (100 * p.bins[i].exploitability) / p.exploitability
const countShare = (p: ReachProfile, i: number) => (100 * p.bins[i].infosets) / p.infosets
const leak = (p: ReachProfile, i: number) => 100 * perVisit(p.bins[i])

export function ReachChart({ profiles }: { profiles: (keyof typeof REACH)[] }) {
  const base = REACH[profiles[0]]
  const rows = base.bins.map((bin, i) => ({ bin, i })).filter(({ i }) => profiles.some((p) => REACH[p].bins[i].infosets > 0 && REACH[p].bins[i].frequency > 0))
  const xmax = Math.max(40, ...rows.flatMap(({ i }) => [countShare(base, i), ...profiles.map((p) => share(REACH[p], i))])) * 1.12
  const x = linScale(0, xmax, M.l, SPLIT - 24)
  const leaks = rows.flatMap(({ i }) => profiles.map((p) => leak(REACH[p], i))).filter((v) => v > 0)
  const lo = 10 ** Math.floor(Math.log10(Math.min(...leaks)))
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...leaks)))
  const lx = logScale(lo, hi, RIGHT.l, RIGHT.r)
  const barH = BAR
  const ROW = BAR * (profiles.length + 1) + 12
  const H = M.t + rows.length * ROW + 44
  const ticks: number[] = []
  for (let v = lo; v <= hi; v *= 10) ticks.push(v)
  return (
    <>
      <div className="legend">
        {profiles.map((p) => (
          <span key={p} className="item">
            <span className="swatch" style={{ background: INK[p] }} />
            {REACH_NAME[p]}
          </span>
        ))}
        <span className="item">
          <span className="swatch" style={{ background: 'var(--iv-30)' }} />
          share of the infosets
        </span>
      </div>
      <svg className="chart reach" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Share of exploitability per reach band, and the leak per visit">
        <text x={M.l} y={M.t - 2} fontSize="10" fill="var(--panel-dim)">
          share of exploitability · share of infosets
        </text>
        <text x={RIGHT.l} y={M.t - 2} fontSize="10" fill="var(--panel-dim)">
          leak per 100 visits, chips, log
        </text>
        {[0, 10, 20, 30, 40, 50, 60, 70].filter((v) => v < xmax).map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={M.t + 6} y2={M.t + rows.length * ROW} stroke="var(--panel-hair)" />
            <text x={x(v)} y={M.t + rows.length * ROW + 16} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
              {v}%
            </text>
          </g>
        ))}
        {ticks.map((v) => (
          <g key={v}>
            <line x1={lx(v)} x2={lx(v)} y1={M.t + 6} y2={M.t + rows.length * ROW} stroke="var(--panel-hair)" />
            <text x={lx(v)} y={M.t + rows.length * ROW + 16} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
              {v >= 1 ? v : v.toFixed(String(v).length - 2)}
            </text>
          </g>
        ))}
        <text x={M.l} y={M.t + rows.length * ROW + 36} fontSize="10" fill="var(--panel-dim)">
          rows: how often the spot comes up when the strategy plays itself, per hand
        </text>
        {rows.map(({ bin, i }, r) => {
          const top = M.t + r * ROW + 8
          const mid = top + (ROW - 12) / 2
          return (
            <g key={bin.floor}>
              <text x={M.l - 10} y={mid + 4} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
                {binLabel(bin)}
              </text>
              {profiles.map((p, k) => {
                const v = share(REACH[p], i)
                const l = leak(REACH[p], i)
                return (
                  <g key={p}>
                    <rect x={M.l} y={top + k * barH} width={Math.max(0, x(v) - M.l)} height={barH - 2} fill={INK[p]} fillOpacity={0.9}>
                      <title>{`${REACH_NAME[p]}: ${v.toFixed(1)}% of its exploitability in ${REACH[p].bins[i].infosets.toLocaleString()} spots; ${l.toFixed(2)} chips per 100 visits`}</title>
                    </rect>
                    <text x={x(v) + 5} y={top + k * barH + barH - 3} fontSize="9.5" fill={INK[p]}>
                      {v.toFixed(1)}
                    </text>
                    {l > 0 && (
                      <g>
                        <circle cx={lx(l)} cy={top + k * barH + barH / 2 - 1} r={3.5} fill={INK[p]} />
                        <text x={lx(l) + 7} y={top + k * barH + barH / 2 + 3} fontSize="9.5" fill={INK[p]}>
                          {l >= 10 ? l.toFixed(0) : l.toFixed(1)}
                        </text>
                      </g>
                    )}
                  </g>
                )
              })}
              <rect x={M.l} y={top + profiles.length * barH} width={Math.max(0, x(countShare(base, i)) - M.l)} height={barH - 2} fill="var(--iv-30)">
                <title>{`${countShare(base, i).toFixed(1)}% of the infosets (${base.bins[i].infosets.toLocaleString()})`}</title>
              </rect>
              <text x={x(countShare(base, i)) + 5} y={top + profiles.length * barH + barH - 3} fontSize="9.5" fill="var(--iv-45)">
                {countShare(base, i).toFixed(1)}
              </text>
            </g>
          )
        })}
      </svg>
    </>
  )
}
