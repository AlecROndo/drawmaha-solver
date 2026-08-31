import { useRef, useState } from 'react'
import type { HistoryPoint } from '../sim/engine'
import { fmtIter } from './format'
import { decadeLabel, nearestPoint, type LogXScale } from './chartScale'

/**
 * Shared chrome for the two log-x time charts (ConvergenceChart, ExploitChart):
 * one canvas size, decade gridlines, hover tracking, crosshair, right-edge
 * direct labels, and the tooltip shell. The charts only draw their own series.
 */

export const CHART_W = 560
export const CHART_H = 250

export interface ChartMargins {
  l: number
  r: number
  t: number
  b: number
}

/** Vertical gridline + label at every decade of the log-x scale. */
export function DecadeGridlines({ scale, m }: { scale: LogXScale; m: ChartMargins }) {
  return (
    <>
      {scale.decades.map((d) => (
        <g key={d}>
          <line x1={scale.x(d)} x2={scale.x(d)} y1={m.t} y2={CHART_H - m.b} stroke="var(--grid)" />
          <text x={scale.x(d)} y={CHART_H - 8} textAnchor="middle" fontSize="10" fill="var(--muted)">
            {decadeLabel(d)}
          </text>
        </g>
      ))}
    </>
  )
}

/** Track the history point under the mouse, inverting the log-x scale. */
export function useChartHover(points: HistoryPoint[], scale: LogXScale, m: ChartMargins) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<HistoryPoint | null>(null)

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * CHART_W
    const frac = Math.min(Math.max((px - m.l) / (CHART_W - m.l - m.r), 0), 1)
    setHover(nearestPoint(points, 10 ** (frac * Math.log10(scale.maxT))))
  }

  return { wrapRef, hover, onMove, onLeave: () => setHover(null) }
}

/** Horizontal gridline + left-edge tick label. */
export function YGridline({ y, label, m }: { y: number; label: React.ReactNode; m: ChartMargins }) {
  return (
    <g>
      <line x1={m.l} x2={CHART_W - m.r} y1={y} y2={y} stroke="var(--grid)" />
      <text x={m.l - 6} y={y + 3} textAnchor="end" fontSize="10" fill="var(--muted)">
        {label}
      </text>
    </g>
  )
}

/** Vertical hairline at the hovered x. */
export function HoverCrosshair({ x, m }: { x: number; m: ChartMargins }) {
  return <line x1={x} x2={x} y1={m.t} y2={CHART_H - m.b} stroke="var(--axis)" />
}

/** Direct label at the right edge: colored tick + text ink, instead of a legend lookup. */
export function EdgeLabel({ y, color, text, m }: { y: number; color: string; text: string; m: ChartMargins }) {
  return (
    <g>
      <rect x={CHART_W - m.r + 4} y={y - 1.5} width={10} height={3} rx={1.5} fill={color} />
      <text x={CHART_W - m.r + 18} y={y + 3} fontSize="10" fill="var(--ink-2)">
        {text}
      </text>
    </g>
  )
}

/**
 * Tooltip shell pinned near the hovered x: always renders the iteration row,
 * then the chart's own value rows. `clamp` is the tooltip's width allowance so
 * it never overflows the panel's right edge.
 */
export function ChartTooltip({
  x,
  wrap,
  clamp,
  t,
  children,
}: {
  x: number
  wrap: HTMLDivElement
  clamp: number
  t: number
  children: React.ReactNode
}) {
  return (
    <div
      className="chart-tooltip"
      style={{ left: Math.min((x / CHART_W) * wrap.clientWidth + 10, wrap.clientWidth - clamp), top: 8 }}
    >
      <div className="row">
        <span>iteration</span>
        <span className="val">{fmtIter(t)}</span>
      </div>
      {children}
    </div>
  )
}
