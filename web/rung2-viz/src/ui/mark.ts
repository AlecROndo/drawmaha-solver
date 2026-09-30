/**
 * The Drawmaha chip: a dithered, tilted poker chip turning on its axis, drawn
 * as ASCII on a canvas. A port of the homepage's chip.js, so the mark on a rung
 * page is the same object as the mark on the cover — same ramp, same tilt,
 * same wobble, same speed-up under the mouse.
 *
 * The canvas is sized by CSS (square); the glyph grid follows its box. The
 * field is transparent, so the chip sits on whatever the page is.
 */

const FONT = "'IBM Plex Mono', ui-monospace, Menlo, monospace"
const RAMP = " .'`,:;~-=+*x%#@"
let CHARSET = ''
for (let c = 32; c < 127; c++) CHARSET += String.fromCharCode(c)
CHARSET += '♠♥♦♣·'
const CIDX: Record<string, number> = {}
for (let i = 0; i < CHARSET.length; i++) CIDX[CHARSET[i]] = i
const RAMPIDX = [...RAMP].map((c) => CIDX[c])
const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v)
const BAYER = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
]

type Vec3 = [number, number, number]
type Mat3 = [Vec3, Vec3, Vec3]

/** A character grid over a canvas: one glyph atlas per palette colour, blitted per cell. */
class Grid {
  canvas: HTMLCanvasElement
  ctx: CanvasRenderingContext2D
  palette: string[]
  fontPx: number
  mouse = { x: -1e6, y: -1e6, in: false }
  dirty = true
  dpr = 1
  cw = 1
  ch = 1
  cols = 0
  rows = 0
  ar = 1
  W = 0
  H = 0
  ox = 0
  oy = 0
  gw = 1
  gh = 1
  chb = new Uint8Array(0)
  cob = new Uint8Array(0)
  atlas: HTMLCanvasElement[] = []

  constructor(canvas: HTMLCanvasElement, palette: string[], fontPx: number) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.palette = palette
    this.fontPx = fontPx
    this.resize()
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const r = this.canvas.getBoundingClientRect()
    if (r.width === 0) return
    this.dpr = dpr
    this.canvas.width = Math.round(r.width * dpr)
    this.canvas.height = Math.round(r.height * dpr)
    const ctx = this.ctx
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.font = `${this.fontPx}px ${FONT}`
    this.cw = ctx.measureText('M').width
    this.ch = this.fontPx * 1.02
    this.cols = Math.floor(r.width / this.cw)
    this.rows = Math.floor(r.height / this.ch)
    this.ar = this.ch / this.cw
    this.W = this.cols
    this.H = this.rows * this.ar
    this.ox = (r.width - this.cols * this.cw) / 2
    this.oy = (r.height - this.rows * this.ch) / 2
    const n = this.cols * this.rows
    this.chb = new Uint8Array(n)
    this.cob = new Uint8Array(n)
    this.buildAtlas()
    this.dirty = true
  }

  buildAtlas() {
    const dpr = this.dpr
    const gw = Math.ceil(this.cw * dpr)
    const gh = Math.ceil(this.ch * dpr)
    this.gw = gw
    this.gh = gh
    this.atlas = this.palette.map((col) => {
      const c = document.createElement('canvas')
      c.width = gw * CHARSET.length
      c.height = gh
      const x = c.getContext('2d')!
      x.font = `${this.fontPx * dpr}px ${FONT}`
      x.textBaseline = 'middle'
      x.textAlign = 'center'
      x.fillStyle = col
      for (let i = 0; i < CHARSET.length; i++) x.fillText(CHARSET[i], i * gw + gw / 2, gh / 2 + 0.5)
      return c
    })
  }

  clear() {
    this.chb.fill(0)
    this.cob.fill(0)
  }

  draw() {
    const { ctx, gw, gh, cols, rows, dpr } = this
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    const ox = this.ox * dpr
    const oy = this.oy * dpr
    const pw = this.cw * dpr
    const ph = this.ch * dpr
    let i = 0
    for (let y = 0; y < rows; y++)
      for (let x = 0; x < cols; x++, i++) {
        const c = this.chb[i]
        if (!c) continue
        ctx.drawImage(this.atlas[this.cob[i]], c * gw, 0, gw, gh, ox + x * pw, oy + y * ph, gw, gh)
      }
  }

  ramp(v: number) {
    return RAMPIDX[clamp(Math.round(v * (RAMP.length - 1)), 0, RAMP.length - 1)]
  }
}

const rotX = (a: number): Mat3 => {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ]
}
const rotY = (a: number): Mat3 => {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ]
}
const rotZ = (a: number): Mat3 => {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ]
}
const mul = (A: Mat3, B: Mat3): Mat3 =>
  A.map((r) =>
    [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]),
  ) as Mat3
const norm3 = (v: Vec3): Vec3 => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / l, v[1] / l, v[2] / l]
}

interface ChipParams {
  cx: number
  cy: number
  R: number
  hz: number
  /** 3x3, columns are the chip's local axes in screen space (x right, y down, z away) */
  M: Mat3
  L: Vec3
  colBase: number
  colPattern: number
  colDim: number
  sectors: number
}

/** A ray-traced chip: two faces and a rim, sectored pattern, Bayer-dithered onto six glyphs. */
function rayChip(g: Grid, o: ChipParams) {
  const { cols, rows, ar } = g
  const m = o.M
  const toLocal = (x: number, y: number, z: number): Vec3 => [
    m[0][0] * x + m[1][0] * y + m[2][0] * z,
    m[0][1] * x + m[1][1] * y + m[2][1] * z,
    m[0][2] * x + m[1][2] * y + m[2][2] * z,
  ]
  const toWorld = (x: number, y: number, z: number): Vec3 => [
    m[0][0] * x + m[0][1] * y + m[0][2] * z,
    m[1][0] * x + m[1][1] * y + m[1][2] * z,
    m[2][0] * x + m[2][1] * y + m[2][2] * z,
  ]
  const { R, hz, L, sectors } = o
  const d = toLocal(0, 0, 1)
  const chars = [0, CIDX['.'], CIDX[':'], CIDX['+'], CIDX['%'], CIDX['@']]
  const ext = R * 1.05 + hz
  const cx0 = Math.max(0, Math.floor(o.cx - ext))
  const cx1 = Math.min(cols - 1, Math.ceil(o.cx + ext))
  const cy0 = Math.max(0, Math.floor((o.cy - ext) / ar))
  const cy1 = Math.min(rows - 1, Math.ceil((o.cy + ext) / ar))
  for (let cyy = cy0; cyy <= cy1; cyy++) {
    const py = (cyy + 0.5) * ar - o.cy
    for (let cxx = cx0; cxx <= cx1; cxx++) {
      const px = cxx + 0.5 - o.cx
      if (px * px + py * py > ext * ext) continue
      const oo = toLocal(px, py, -400)
      let best = 1e9
      let mat = 0
      let I = 0
      for (const zc of [hz, -hz]) {
        if (Math.abs(d[2]) < 1e-6) continue
        const tc = (zc - oo[2]) / d[2]
        if (tc < 0 || tc > best) continue
        const x = oo[0] + tc * d[0]
        const y = oo[1] + tc * d[1]
        const r = Math.hypot(x, y)
        if (r > R) continue
        best = tc
        const rn = r / R
        const ang = Math.atan2(y, x)
        const sector = Math.floor((ang / (2 * Math.PI) + 1) * sectors + 0.5) % 2
        mat = (rn > 0.8 && sector) || (rn > 0.5 && rn < 0.58) ? 2 : 1
        const n = toWorld(0, 0, zc > 0 ? 1 : -1)
        const nl = n[0] * L[0] + n[1] * L[1] + n[2] * L[2]
        const hl = Math.pow(nl * 0.5 + 0.5, 2)
        const lam = Math.max(0, nl)
        I = 0.12 + 0.62 * hl + 0.35 * Math.pow(lam, 16)
        if (rn > 0.96) I += 0.15
        if (rn > 0.8 && !sector) I *= 0.9
      }
      const a = d[0] * d[0] + d[1] * d[1]
      if (a > 1e-9) {
        const b = 2 * (oo[0] * d[0] + oo[1] * d[1])
        const c = oo[0] * oo[0] + oo[1] * oo[1] - R * R
        const disc = b * b - 4 * a * c
        if (disc > 0) {
          const ts = (-b - Math.sqrt(disc)) / (2 * a)
          if (ts > 0 && ts < best) {
            const z = oo[2] + ts * d[2]
            if (Math.abs(z) <= hz) {
              best = ts
              const x = oo[0] + ts * d[0]
              const y = oo[1] + ts * d[1]
              const sector = Math.floor((Math.atan2(y, x) / (2 * Math.PI) + 1) * sectors + 0.5) % 2
              mat = sector ? 2 : 1
              const n = toWorld(x / R, y / R, 0)
              const nl = n[0] * L[0] + n[1] * L[1] + n[2] * L[2]
              I = 0.1 + 0.55 * Math.pow(nl * 0.5 + 0.5, 2) + 0.3 * Math.pow(Math.max(0, nl), 10)
            }
          }
        }
      }
      if (mat === 0) continue
      const i = cyy * cols + cxx
      const col = mat === 1 ? o.colBase : o.colPattern
      const bay = (BAYER[cyy % 4][cxx % 4] + 0.5) / 16 - 0.5
      const lv = clamp(Math.floor(I * 5 + bay * 1.15), 0, 5)
      if (lv > 0) {
        g.chb[i] = chars[lv]
        g.cob[i] = col
      } else {
        g.chb[i] = chars[1]
        g.cob[i] = o.colDim
      }
    }
  }
}

const hexA = (hex: string, a: number) => {
  let h = hex.replace('#', '')
  if (h.length === 3)
    h = h
      .split('')
      .map((c) => c + c)
      .join('')
  const n = parseInt(h, 16)
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`
}

export interface ChipOptions {
  /** the glyph colour of the chip's body */
  mark?: string
  /** the glyph colour of its pattern: the sectors and the inner ring */
  ink?: string
  /** cell size in px; 3 for a 56px mark, 4 for the 132px one that closes a page */
  font?: number
  /** turn rate in rad/s when the mouse is elsewhere */
  speed?: number
  tilt?: number
}

/**
 * Draw the chip on a canvas and keep it turning. Returns the teardown. With
 * reduced motion the chip is drawn once and still answers a resize.
 */
export function mountChip(canvas: HTMLCanvasElement, opts: ChipOptions = {}): () => void {
  const mark = opts.mark ?? '#f5f0e6'
  const ink = opts.ink ?? '#8e2038'
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches
  const g = new Grid(canvas, ['transparent', mark, ink, hexA(mark, 0.35)], opts.font ?? 4)
  const s = { yaw: 0.6, t: 0, speed: opts.speed ?? 0.3, tilt: opts.tilt ?? 0.8, drawn: false }

  const move = (cx: number, cy: number) => {
    const r = canvas.getBoundingClientRect()
    g.mouse.x = (cx - r.left - g.ox) / g.cw
    g.mouse.y = ((cy - r.top - g.oy) / g.ch) * g.ar
    g.mouse.in = true
    g.dirty = true
  }
  const onMove = (e: MouseEvent) => move(e.clientX, e.clientY)
  const onLeave = () => {
    g.mouse.in = false
    g.dirty = true
  }
  canvas.addEventListener('mousemove', onMove)
  canvas.addEventListener('mouseleave', onLeave)

  const ro = new ResizeObserver(() => {
    g.resize()
    s.drawn = false
  })
  ro.observe(canvas)
  if (document.fonts?.ready)
    document.fonts.ready.then(() => {
      g.resize()
      s.drawn = false
    })

  let raf = 0
  let last = performance.now()
  const tick = (now: number) => {
    const dt = Math.min(0.05, (now - last) / 1000)
    last = now
    if (!document.hidden && g.cols && !(reduced && s.drawn && !g.dirty)) {
      g.dirty = false
      s.drawn = true
      if (!reduced) {
        s.t += dt
        s.yaw += dt * (g.mouse.in ? s.speed * 3.5 : s.speed)
      }
      const { W, H } = g
      const R = Math.min(W, H) * 0.46
      g.clear()
      rayChip(g, {
        cx: W / 2,
        cy: H / 2,
        R,
        hz: R * 0.13,
        M: mul(rotX(s.tilt + 0.12 * Math.sin(s.t * 0.37)), mul(rotY(0.25 * Math.sin(s.t * 0.29)), rotZ(s.yaw))),
        L: norm3([-0.55, -0.5, -0.67]),
        colBase: 1,
        colPattern: 2,
        colDim: 3,
        sectors: 12,
      })
      g.draw()
    }
    raf = requestAnimationFrame(tick)
  }
  raf = requestAnimationFrame(tick)

  return () => {
    cancelAnimationFrame(raf)
    ro.disconnect()
    canvas.removeEventListener('mousemove', onMove)
    canvas.removeEventListener('mouseleave', onLeave)
  }
}
