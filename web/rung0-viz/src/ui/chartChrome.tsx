import { useRef, useState } from 'react'
import type { HistoryPoint } from '../sim/engine'
import { fmtIter } from './format'
import { decadeLabel, nearestPoint, type LogXScale } from './chartScale'

/**
 * Shared chrome for the log-x time charts (ConvergenceChart, ExploitChart):
 * decade gridlines, hover tracking, crosshair, right-edge direct labels, and
 * the tooltip shell. The charts only draw their own series.
 *
 * Every piece takes the same `box` — the viewBox plus its four margins —
 * because the page puts these charts at two widths: a half-column figure and
 * one that runs the full container. Passing the geometry rather than importing
 * a constant is what lets one chart be twice as wide without its axis labels
 * being scaled up with it.
 */

export interface ChartBox {
  w: number
  h: number
  l: number
  r: number
  t: number
  b: number
}

/** Half-column figures: two of these sit side by side. */
export const HALF: Omit<ChartBox, 'l' | 'r' | 't' | 'b'> = { w: 560, h: 250 }
/** One figure across the whole container. */
export const FULL: Omit<ChartBox, 'l' | 'r' | 't' | 'b'> = { w: 1180, h: 300 }

/** Vertical gridline + label at every decade of the log-x scale. */
export function DecadeGridlines({ scale, box }: { scale: LogXScale; box: ChartBox }) {
  return (
    <>
      {scale.decades.map((d) => (
        <g key={d}>
          <line x1={scale.x(d)} x2={scale.x(d)} y1={box.t} y2={box.h - box.b} stroke="var(--panel-hair)" />
          <text x={scale.x(d)} y={box.h - 8} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
            {decadeLabel(d)}
          </text>
        </g>
      ))}
    </>
  )
}

/** Track the history point under the mouse, inverting the log-x scale. */
export function useChartHover(points: HistoryPoint[], scale: LogXScale, box: ChartBox) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<HistoryPoint | null>(null)

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * box.w
    const frac = Math.min(Math.max((px - box.l) / (box.w - box.l - box.r), 0), 1)
    setHover(nearestPoint(points, 10 ** (frac * Math.log10(scale.maxT))))
  }

  return { wrapRef, hover, onMove, onLeave: () => setHover(null) }
}

/** Horizontal gridline + left-edge tick label. */
export function YGridline({
  y,
  label,
  box,
}: {
  y: number
  label: React.ReactNode
  box: ChartBox
}) {
  return (
    <g>
      <line x1={box.l} x2={box.w - box.r} y1={y} y2={y} stroke="var(--panel-hair)" />
      <text x={box.l - 8} y={y + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
        {label}
      </text>
    </g>
  )
}

/** Vertical hairline at the hovered x. */
export function HoverCrosshair({ x, box }: { x: number; box: ChartBox }) {
  return <line x1={x} x2={x} y1={box.t} y2={box.h - box.b} stroke="var(--panel-hair)" />
}

/** Direct label at the right edge: colored tick + text ink, instead of a legend lookup. */
export function EdgeLabel({
  y,
  color,
  text,
  box,
}: {
  y: number
  color: string
  text: string
  box: ChartBox
}) {
  return (
    <g>
      <rect x={box.w - box.r + 6} y={y - 1.5} width={10} height={3} rx={1.5} fill={color} />
      <text x={box.w - box.r + 20} y={y + 3.5} fontSize="11" fill="var(--panel-dim)">
        {text}
      </text>
    </g>
  )
}

/**
 * Tooltip shell pinned near the hovered x: always renders the iteration row,
 * then the chart's own value rows. `clamp` is the tooltip's width allowance so
 * it never overflows the figure's right edge.
 */
export function ChartTooltip({
  x,
  wrap,
  clamp,
  t,
  box,
  children,
}: {
  x: number
  wrap: HTMLDivElement
  clamp: number
  t: number
  box: ChartBox
  children: React.ReactNode
}) {
  return (
    <div
      className="chart-tooltip"
      style={{
        left: Math.min((x / box.w) * wrap.clientWidth + 10, wrap.clientWidth - clamp),
        top: 8,
      }}
    >
      <div className="row">
        <span>round</span>
        <span className="val">{fmtIter(t)}</span>
      </div>
      {children}
    </div>
  )
}
