/**
 * The strength ribbon, stood on end: every holding in the filtered range
 * sorted by one number (total equity, the inner half, the outer half, or
 * the scoop proxy inner × outer — a feature, not a scoop probability, since
 * the two halves share the hand's cards), weakest at the bottom, strongest
 * at the top, each
 * slice a thin horizontal bar whose bands are the mix. Polarisation is a
 * shape you can see: pink at both ends, yellow in the middle. When five
 * cards are placed, the held hand's percentile is a line across the strip.
 */

import type { Mix, RibbonKey } from '../engine/aggregate'
import { MixBar } from './bars'

export const RIBBON_KEYS: { key: RibbonKey; label: string }[] = [
  { key: 'total', label: 'total' },
  { key: 'inner', label: 'inner' },
  { key: 'outer', label: 'outer' },
  { key: 'scoop', label: 'scoop' },
]

export function Ribbon({
  bins,
  rkey,
  onKey,
  percentile,
  all,
}: {
  bins: Mix[]
  rkey: RibbonKey
  onKey: (k: RibbonKey) => void
  /** the held hand's percentile under this key, or null when fewer than five cards are placed */
  percentile: number | null
  all: Mix
}) {
  const W = 300
  const H = 360
  const n = bins.length
  const rowH = n ? H / n : 0
  return (
    <div className="ribbon">
      <div className="lbl">
        strength ribbon
        <span className="keys" role="tablist" aria-label="sort the ribbon by">
          {RIBBON_KEYS.map((k) => (
            <button key={k.key} type="button" role="tab" className={`key ${rkey === k.key ? 'on' : ''}`} aria-selected={rkey === k.key} onClick={() => onKey(k.key)}>
              {k.label}
            </button>
          ))}
        </span>
      </div>
      <div className="ribbon-body">
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="ribbon-svg" aria-label="every holding sorted by strength, weakest at the bottom; each slice's bands are its mix">
          {bins.map((m, i) => {
            const y = H - (i + 1) * rowH
            const fw = m.f * W
            const cw = m.c * W
            return (
              <g key={i}>
                <rect x={0} y={y} width={fw} height={rowH + 0.4} fill="var(--fold)" />
                <rect x={fw} y={y} width={cw} height={rowH + 0.4} fill="var(--call)" />
                <rect x={fw + cw} y={y} width={Math.max(0, W - fw - cw)} height={rowH + 0.4} fill="var(--pot)" />
              </g>
            )
          })}
          {percentile !== null && (
            <g>
              <line x1={-4} x2={W + 4} y1={H - percentile * H} y2={H - percentile * H} stroke="var(--ivory)" strokeWidth={2} />
              <text x={W - 2} y={H - percentile * H - 5} textAnchor="end" className="axis lit">
                your hand · {Math.round(percentile * 100)}th
              </text>
            </g>
          )}
        </svg>
        <div className="ribbon-axis">
          <span>weakest at the bottom</span>
          <span>strongest at the top</span>
        </div>
      </div>
      <div className="ribbon-foot">
        <MixBar mix={all} h={6} />
        <span>
          {n === 0 ? 'nothing left in the filter' : `${n} slices · x: fold | ${'call'} | pot`}
        </span>
      </div>
    </div>
  )
}
