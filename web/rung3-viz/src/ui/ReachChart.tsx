import { useState } from 'react'

import { REACH, REACH_NAME, binLabel, hands, perVisit, type ReachProfile } from '../reach'
import { linScale, logScale } from './scales'

/**
 * Where the exploitability lives, read along one axis: how often a spot
 * comes up when the strategy plays itself, commonest band at the left,
 * rarest at the right. Two panels share that axis. Top, each strategy's
 * share of its own exploitability per band as bars, with the band's share of
 * the infosets as the hollow bar beside them, so the eye sees at once where
 * the total sits and how many rows it sits in. Bottom, on a log axis, what a
 * spot in the band leaks each time it is reached, drawn as a line, so "a row
 * rots as the sampler sees it less" is a curve rising to the right and the
 * net's flatter curve lies beside the table's. Hover or arrow a band to read
 * every number in it under the plot.
 *
 * Inks: the table alone (Figure 7, in rung 3's part) wears the race's LCFR
 * yellow; beside the net (Figure 9) the three strategies wear the inks of
 * the race at equal hands — the table dim, the exact average yellow, Π ivory.
 */

const W = 760
const M = { l: 54, r: 112 }
const TOP = { t: 26, b: 150 }
const BOTTOM = { t: 196, b: 330 }
const H = 392
const LABEL_H = 14

type Key = keyof typeof REACH

const inkOf = (profiles: Key[]): Record<Key, string> =>
  profiles.length === 1
    ? { tabular: 'var(--rule-lcfr)', 'deep-exact': 'var(--obj-exact)', 'deep-pi': 'var(--obj-pi)' }
    : { tabular: 'var(--table-ink)', 'deep-exact': 'var(--obj-exact)', 'deep-pi': 'var(--obj-pi)' }

const share = (p: ReachProfile, i: number) => (100 * p.bins[i].exploitability) / p.exploitability
const countShare = (p: ReachProfile, i: number) => (100 * p.bins[i].infosets) / p.infosets
const leak = (p: ReachProfile, i: number) => 100 * perVisit(p.bins[i])
/** "1 in 1k": the band edge as how often a spot comes up. */
const edgeWord = (frequency: number) => `1 in ${hands(Math.round(1 / frequency))}`

export function ReachChart({ profiles }: { profiles: Key[] }) {
  const INK = inkOf(profiles)
  const base = REACH[profiles[0]]
  // the bands any shown profile populates, commonest first; the band index into every profile's bins is the same
  const bands = base.bins.map((bin, i) => ({ bin, i })).filter(({ i }) => profiles.some((p) => REACH[p].bins[i].infosets > 0 && REACH[p].bins[i].frequency > 0))
  const n = bands.length
  const bw = (W - M.l - M.r) / n
  const cx = (k: number) => M.l + (k + 0.5) * bw
  const yMax = Math.ceil(Math.max(...bands.flatMap(({ i }) => [countShare(base, i), ...profiles.map((p) => share(REACH[p], i))])) / 10) * 10
  const y = linScale(0, yMax, TOP.b, TOP.t)
  const leaks = bands.flatMap(({ i }) => profiles.map((p) => leak(REACH[p], i))).filter((v) => v > 0)
  const lo = 10 ** Math.floor(Math.log10(Math.min(...leaks)))
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...leaks)))
  const ly = logScale(lo, hi, BOTTOM.b, BOTTOM.t)
  const lticks: number[] = []
  for (let v = lo; v <= hi; v *= 10) lticks.push(v)
  const series = profiles.length + 1
  const barW = (bw * 0.78) / series
  const barX = (k: number, s: number) => cx(k) - (bw * 0.78) / 2 + s * barW
  const [at, setAt] = useState(n - 1)
  const picked = bands[at]
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    setAt((k) => (e.key === 'ArrowLeft' ? Math.max(0, k - 1) : Math.min(n - 1, k + 1)))
  }
  // the leak lines' labels at both ends, stacked where they would collide
  const stacked = (band: number) => {
    const labels = profiles.map((p) => ({ p, v: leak(REACH[p], bands[band].i) })).map((e) => ({ ...e, ly: ly(e.v) + 4 })).sort((a, b) => a.ly - b.ly)
    for (let k = 1; k < labels.length; k++) labels[k].ly = Math.max(labels[k].ly, labels[k - 1].ly + LABEL_H)
    return labels
  }
  const ends = stacked(n - 1)
  const starts = stacked(0)
  const leakWord = (v: number) => (v >= 10 ? v.toFixed(0) : v.toFixed(1))
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
          <span className="swatch" style={{ background: 'transparent', boxShadow: 'inset 0 0 0 1px var(--iv-45)', height: 6 }} />
          share of the infosets
        </span>
      </div>
      <svg className="chart reach" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Share of exploitability per reach band, and the leak per visit" tabIndex={0} onKeyDown={onKey}>
        {/* the picked band */}
        <rect x={M.l + at * bw} y={TOP.t - 4} width={bw} height={BOTTOM.b - TOP.t + 4} fill="var(--panel-mark)" fillOpacity={0.05} />

        {/* top: shares */}
        <text x={4} y={TOP.t - 12} fontSize="10" fill="var(--panel-dim)">
          share of the exploitability · share of the infosets, %
        </text>
        {Array.from({ length: yMax / 10 + 1 }, (_, k) => k * 10).map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke="var(--panel-hair)" />
            <text x={M.l - 8} y={y(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {v}
            </text>
          </g>
        ))}
        {bands.map(({ i }, k) => (
          <g key={i}>
            {profiles.map((p, s) => {
              const v = share(REACH[p], i)
              return (
                <g key={p}>
                  <rect x={barX(k, s)} y={y(v)} width={barW - 2} height={Math.max(0, y(0) - y(v))} fill={INK[p]} fillOpacity={0.9} />
                  {profiles.length === 1 && v >= 1 && (
                    <text x={barX(k, s) + (barW - 2) / 2} y={y(v) - 4} textAnchor="middle" fontSize="9.5" fill={INK[p]}>
                      {v.toFixed(1)}
                    </text>
                  )}
                </g>
              )
            })}
            <rect x={barX(k, profiles.length)} y={y(countShare(base, i))} width={barW - 2} height={Math.max(0, y(0) - y(countShare(base, i)))} fill="none" stroke="var(--iv-45)" />
            {profiles.length === 1 && countShare(base, i) >= 1 && (
              <text x={barX(k, profiles.length) + (barW - 2) / 2} y={y(countShare(base, i)) - 4} textAnchor="middle" fontSize="9.5" fill="var(--iv-45)">
                {countShare(base, i).toFixed(1)}
              </text>
            )}
          </g>
        ))}

        {/* bottom: leak per visit */}
        <text x={4} y={BOTTOM.t - 12} fontSize="10" fill="var(--panel-dim)">
          leak per 100 visits: chips a best responder wins each time play reaches a spot in the band, log
        </text>
        {lticks.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={ly(v)} y2={ly(v)} stroke="var(--panel-hair)" />
            <text x={M.l - 8} y={ly(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {v >= 1 ? v : v.toFixed(String(v).length - 2)}
            </text>
          </g>
        ))}
        {profiles.map((p) => {
          const pts = bands.map(({ i }, k) => ({ k, v: leak(REACH[p], i) })).filter((d) => d.v > 0)
          return (
            <g key={p}>
              <path d={pts.map((d, j) => `${j ? 'L' : 'M'} ${cx(d.k).toFixed(1)} ${ly(d.v).toFixed(1)}`).join(' ')} fill="none" stroke={INK[p]} strokeWidth={p === 'tabular' && profiles.length > 1 ? 1.6 : 2} />
              {pts.map((d) => (
                <circle key={d.k} cx={cx(d.k)} cy={ly(d.v)} r={3.2} fill={INK[p]} />
              ))}
            </g>
          )
        })}
        {starts.map(({ p, v, ly: yy }) => (
          <text key={p} x={cx(0) - 8} y={yy} textAnchor="end" fontSize="10.5" fill={INK[p]}>
            {leakWord(v)}
          </text>
        ))}
        {ends.map(({ p, v, ly: yy }) => (
          <text key={p} x={cx(n - 1) + 9} y={yy} fontSize="10.5" fill={INK[p]}>
            {profiles.length > 1 ? `${REACH_NAME[p].replace('Deep CFR, ', '').replace(', the table', '')} ${leakWord(v)}` : leakWord(v)}
          </text>
        ))}

        {/* the shared axis: band edges, how often a spot comes up */}
        {bands.map(({ bin }, k) => (
          <g key={bin.floor}>
            <line x1={M.l + k * bw} x2={M.l + k * bw} y1={TOP.t} y2={BOTTOM.b} stroke="var(--panel-hair)" />
            <text x={M.l + k * bw} y={BOTTOM.b + 18} textAnchor="middle" fontSize="10" fill="var(--panel-dim)">
              {bin.ceiling === null ? '' : edgeWord(bin.ceiling)}
            </text>
          </g>
        ))}
        <line x1={W - M.r} x2={W - M.r} y1={TOP.t} y2={BOTTOM.b} stroke="var(--panel-hair)" />
        <text x={W - M.r} y={BOTTOM.b + 18} textAnchor="middle" fontSize="10" fill="var(--panel-dim)">
          {edgeWord(bands[n - 1].bin.floor)}
        </text>
        <text x={M.l} y={BOTTOM.b + 38} fontSize="10" fill="var(--panel-dim)">
          how often play reaches the spot, once in N hands: commonest bands at the left, rarest at the right
        </text>

        {bands.map((_, k) => (
          <rect key={k} x={M.l + k * bw} y={TOP.t - 4} width={bw} height={BOTTOM.b - TOP.t + 4} fill="transparent" onMouseEnter={() => setAt(k)} />
        ))}
      </svg>
      <dl className="stat">
        <div>
          <dt>band</dt>
          <dd>{binLabel(picked.bin)}</dd>
        </div>
        <div>
          <dt>infosets</dt>
          <dd>
            {base.bins[picked.i].infosets.toLocaleString()}
            <small>{countShare(base, picked.i).toFixed(1)}% of the rows</small>
          </dd>
        </div>
        {profiles.map((p) => (
          <div key={p}>
            <dt style={{ color: INK[p] }}>{REACH_NAME[p]}</dt>
            <dd>
              {share(REACH[p], picked.i).toFixed(1)}% of its exploitability
              <small>{leak(REACH[p], picked.i).toFixed(2)} chips per 100 visits</small>
            </dd>
          </div>
        ))}
      </dl>
    </>
  )
}
