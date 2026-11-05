/**
 * The strength ribbon: the filtered range stood on end, strongest at the
 * top, every band split left to right by its mix. It has two readings,
 * chosen by the keys over it.
 *
 * "class" (the default) is the band stack: one band per inner class, its
 * height that class's share of the range in view, its bands the class's
 * mix, the strongest class on top and each band named at its right. It is
 * the grid's row headers drawn to scale, so the shape of the range — where
 * the weight sits, and how it plays — is one glance.
 *
 * The other four keys sort every holding by one number (total equity, the
 * inner half, the outer half, or the scoop proxy inner × outer — a feature,
 * not a scoop probability, since the two halves share the hand's cards) and
 * draw it as `bins` equal slices, weakest at the bottom. Polarisation is a
 * shape you can see: the pot band at both ends, the passive band in the
 * middle. When a whole hand is placed, its percentile under the key is an
 * ink line across the strip.
 */

import type { Cell, Mix, RibbonKey } from '../engine/aggregate'
import { pct } from './bars'

/** What the ribbon is showing: the class stack, or holdings sorted by a key. */
export type RibbonView = 'class' | RibbonKey

export const RIBBON_KEYS: { key: RibbonView; label: string }[] = [
  { key: 'class', label: 'class' },
  { key: 'total', label: 'total' },
  { key: 'inner', label: 'inner' },
  { key: 'outer', label: 'outer' },
  { key: 'scoop', label: 'scoop' },
]

const W = 272
const H = 300
/** the width the bands take; the class names sit in the rest */
const BW = 136

export function Ribbon({
  bins,
  classes,
  rview,
  onView,
  percentile,
}: {
  /** the sorted slices, weakest first, for the four sorting keys */
  bins: Mix[]
  /** the inner classes, weakest first, with their cells in the filtered view */
  classes: { label: string; cell: Cell }[]
  rview: RibbonView
  onView: (k: RibbonView) => void
  /** the held hand's percentile under the key, or null when no whole hand is placed */
  percentile: number | null
}) {
  const fills = (m: Mix, y: number, h: number, w: number, key: string | number) => {
    const fw = m.f * w
    const cw = m.c * w
    return (
      <g key={key}>
        <rect x={0} y={y} width={fw} height={h} fill="var(--rx-f)" />
        <rect x={fw} y={y} width={cw} height={h} fill="var(--rx-c)" />
        <rect x={fw + cw} y={y} width={Math.max(0, w - fw - cw)} height={h} fill="var(--rx-p)" />
      </g>
    )
  }

  let body: React.ReactNode
  if (rview === 'class') {
    const total = classes.reduce((s, c) => s + c.cell.share, 0)
    let y = 0
    const bands: React.ReactNode[] = []
    // Strongest on top: walk the classes from the last.
    for (let i = classes.length - 1; i >= 0; i--) {
      const { label, cell } = classes[i]
      if (cell.share <= 0 || total <= 0) continue
      const h = (cell.share / total) * H
      const top = y
      y += h
      bands.push(
        <g key={label}>
          {fills(cell, top, Math.max(0, h - 1), BW, label)}
          {h >= 13 && (
            <text x={BW + 8} y={top + Math.min(h / 2, 12) + 4} className="rx-band-name">
              {label} <tspan className="rx-band-v">{pct(cell.share / total)}</tspan>
            </text>
          )}
        </g>,
      )
    }
    body = (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="ribbon-svg" role="img" aria-label="the range in view by inner class, strongest at the top; each band's height is its share and its fills are its mix">
        {bands}
        {total <= 0 && (
          <text x={0} y={16} className="rx-band-name">
            nothing left in the filter
          </text>
        )}
      </svg>
    )
  } else {
    const n = bins.length
    const rowH = n ? H / n : 0
    body = (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="ribbon-svg" role="img" aria-label="every holding sorted by strength, weakest at the bottom; each slice's fills are its mix">
        {bins.map((m, i) => fills(m, H - (i + 1) * rowH, rowH + 0.4, W, i))}
        {percentile !== null && (
          <g>
            <line x1={0} x2={W} y1={H - percentile * H} y2={H - percentile * H} stroke="var(--ink)" strokeWidth={2} />
            <text x={W - 4} y={H - percentile * H - 6} textAnchor="end" className="rx-band-name on-fill">
              your hand · {Math.round(percentile * 100)}th
            </text>
          </g>
        )}
      </svg>
    )
  }

  return (
    <div className="ribbon">
      <span className="keys" role="tablist" aria-label="what the ribbon shows">
        {RIBBON_KEYS.map((k) => (
          <button key={k.key} type="button" role="tab" className={`key ${rview === k.key ? 'on' : ''}`} aria-selected={rview === k.key} onClick={() => onView(k.key)}>
            {k.label}
          </button>
        ))}
      </span>
      {body}
      <p className="legend-line">
        {rview === 'class'
          ? 'Each band is one inner class at its share of the range in view, split by its mix; the strongest class is on top.'
          : bins.length === 0
            ? 'Nothing left in the filter.'
            : `Every holding sorted by ${rview === 'scoop' ? 'inner × outer' : `${rview} equity`}, in ${bins.length} slices, weakest at the bottom.`}
      </p>
    </div>
  )
}
