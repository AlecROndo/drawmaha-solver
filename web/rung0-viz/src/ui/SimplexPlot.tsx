import { useEffect, useRef } from 'react'
import type { Engine } from '../sim/engine'
import type { Vec3 } from '../sim/game'
import { ACTIONS } from '../sim/game'
import { FigureHead } from './site'

const SIZE = 470
const PAD = 42

/** Barycentric → canvas: rock bottom-left, paper bottom-right, scissors top. */
function project(sigma: Vec3): [number, number] {
  const vr: [number, number] = [PAD, SIZE - PAD]
  const vp: [number, number] = [SIZE - PAD, SIZE - PAD]
  const vs: [number, number] = [SIZE / 2, PAD]
  return [
    sigma[0] * vr[0] + sigma[1] * vp[0] + sigma[2] * vs[0],
    sigma[0] * vr[1] + sigma[1] * vp[1] + sigma[2] * vs[1],
  ]
}

function cssVar(el: HTMLElement, name: string): string {
  return getComputedStyle(el).getPropertyValue(name).trim()
}

/**
 * The strategy triangle: the current strategy orbits forever (fading gray
 * trail); the average strategy spirals into the center (⅓, ⅓, ⅓). The
 * current-cycles / average-converges contrast in one picture.
 */
export function SimplexPlot({ engine, version }: { engine: Engine; version: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== SIZE * dpr) {
      canvas.width = SIZE * dpr
      canvas.height = SIZE * dpr
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, SIZE, SIZE)

    const axis = cssVar(canvas, '--axis')
    const muted = cssVar(canvas, '--muted')
    const ink2 = cssVar(canvas, '--ink-2')
    const curCol = cssVar(canvas, '--cur-obj')
    const avgCol = cssVar(canvas, '--avg-obj')
    const actionCols = [cssVar(canvas, '--rock'), cssVar(canvas, '--paper'), cssVar(canvas, '--scissors')]

    // triangle
    const [rx, ry] = project([1, 0, 0])
    const [px, py] = project([0, 1, 0])
    const [sx, sy] = project([0, 0, 1])
    ctx.strokeStyle = axis
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(rx, ry)
    ctx.lineTo(px, py)
    ctx.lineTo(sx, sy)
    ctx.closePath()
    ctx.stroke()

    // vertex labels: action-colored dot, then the name in text ink
    ctx.font = '12px "IBM Plex Mono", monospace'
    const labels: [string, number, number, number][] = [
      [ACTIONS[0], rx - 4, ry + 18, 0],
      [ACTIONS[1], px - 38, py + 18, 1],
      [ACTIONS[2], sx + 10, sy + 2, 2],
    ]
    ctx.textAlign = 'left'
    for (const [name, x, yy, i] of labels) {
      ctx.fillStyle = actionCols[i]
      ctx.beginPath()
      ctx.arc(x, yy - 3.5, 3.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = ink2
      ctx.fillText(name, x + 7, yy)
    }

    // center mark (⅓, ⅓, ⅓)
    const [cx, cy] = project([1 / 3, 1 / 3, 1 / 3])
    ctx.strokeStyle = muted
    ctx.beginPath()
    ctx.moveTo(cx - 5, cy)
    ctx.lineTo(cx + 5, cy)
    ctx.moveTo(cx, cy - 5)
    ctx.lineTo(cx, cy + 5)
    ctx.stroke()

    const history = engine.history
    if (history.length > 1) {
      // current-strategy orbit: last K points, fading toward the past
      const K = 300
      const start = Math.max(1, history.length - K)
      for (let i = start; i < history.length; i++) {
        const [x0, y0] = project(history[i - 1].cur)
        const [x1, y1] = project(history[i].cur)
        ctx.globalAlpha = 0.08 + 0.6 * ((i - start) / (history.length - start))
        ctx.strokeStyle = curCol
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(x0, y0)
        ctx.lineTo(x1, y1)
        ctx.stroke()
      }
      ctx.globalAlpha = 1

      // average-strategy spiral: full trail
      ctx.strokeStyle = avgCol
      ctx.lineWidth = 2
      ctx.globalAlpha = 0.8
      ctx.beginPath()
      const [ax0, ay0] = project(history[0].avg)
      ctx.moveTo(ax0, ay0)
      for (let i = 1; i < history.length; i++) {
        const [x, yy] = project(history[i].avg)
        ctx.lineTo(x, yy)
      }
      ctx.stroke()
      ctx.globalAlpha = 1

      // endpoints
      const [cx1, cy1] = project(history[history.length - 1].cur)
      ctx.fillStyle = curCol
      ctx.beginPath()
      ctx.arc(cx1, cy1, 4, 0, Math.PI * 2)
      ctx.fill()
      const [ax1, ay1] = project(history[history.length - 1].avg)
      ctx.fillStyle = avgCol
      ctx.beginPath()
      ctx.arc(ax1, ay1, 4.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = cssVar(canvas, '--surface')
      ctx.lineWidth = 2
      ctx.stroke()
    }
  }, [engine, version])

  return (
    <section className="h6" aria-label="Strategy simplex">
      <FigureHead n="Fig. 2" title="The average spirals in, the current strategy orbits">
        Each point is a strategy; the corners are pure rock, paper and scissors. The grey trail is
        the last 300 rounds of σ.
      </FigureHead>
      <div className="legend">
        <span className="item">
          <span className="swatch" style={{ background: 'var(--avg-obj)' }} />
          average S/n, every round so far
        </span>
        <span className="item">
          <span className="swatch" style={{ background: 'var(--cur-obj)' }} />
          current σ, last 300 rounds
        </span>
      </div>
      <canvas ref={canvasRef} style={{ width: '100%', maxWidth: SIZE, display: 'block' }} />
    </section>
  )
}
