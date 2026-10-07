/**
 * The rung objects: a chip, a fan of cards, the oval table, the deck, the
 * stack. One small raymarcher over signed-distance fields, shaded into a
 * character ramp with a Bayer dither and blitted as glyphs. A port of the
 * homepage's ladder icons, so the object a rung page opens on is the object
 * that stands for that rung in the cover's ladder.
 */

const MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace"
const BAYER = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
]
const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v)
const dprOf = () => Math.min(2, window.devicePixelRatio || 1)

/** A character grid over a canvas, one glyph atlas per palette entry. */
interface CharGrid {
  canvas: HTMLCanvasElement
  ctx: CanvasRenderingContext2D
  fontPx: number
  palette: string[]
  cols: number
  rows: number
  cw: number
  ch: number
  ox: number
  oy: number
  dpr: number
  gw: number
  gh: number
  chb: Uint8Array
  cob: Uint8Array
  atlas: HTMLCanvasElement[]
  CH: string
  CIDX: Record<string, number>
  resize(): boolean
  draw(): void
}

function makeGrid(canvas: HTMLCanvasElement, fontPx: number, palette: string[]): CharGrid {
  const ctx = canvas.getContext('2d')!
  let CH = ''
  for (let c = 32; c < 127; c++) CH += String.fromCharCode(c)
  const CIDX: Record<string, number> = {}
  for (let i = 0; i < CH.length; i++) CIDX[CH[i]] = i
  const g: CharGrid = {
    canvas,
    ctx,
    fontPx,
    palette,
    cols: 0,
    rows: 0,
    cw: 1,
    ch: 1,
    ox: 0,
    oy: 0,
    dpr: 1,
    gw: 1,
    gh: 1,
    chb: new Uint8Array(0),
    cob: new Uint8Array(0),
    atlas: [],
    CH,
    CIDX,
    resize() {
      const dpr = dprOf()
      const r = canvas.getBoundingClientRect()
      if (!r.width) return false
      g.dpr = dpr
      canvas.width = Math.round(r.width * dpr)
      canvas.height = Math.round(r.height * dpr)
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.font = `${fontPx}px ${MONO}`
      g.cw = ctx.measureText('M').width
      g.ch = fontPx * 1.05
      g.cols = Math.floor(r.width / g.cw)
      g.rows = Math.floor(r.height / g.ch)
      g.ox = (r.width - g.cols * g.cw) / 2
      g.oy = (r.height - g.rows * g.ch) / 2
      g.chb = new Uint8Array(g.cols * g.rows)
      g.cob = new Uint8Array(g.cols * g.rows)
      const gw = Math.ceil(g.cw * dpr)
      const gh = Math.ceil(g.ch * dpr)
      g.gw = gw
      g.gh = gh
      g.atlas = palette.map((col) => {
        const c = document.createElement('canvas')
        c.width = gw * CH.length
        c.height = gh
        const x = c.getContext('2d')!
        x.font = `${fontPx * dpr}px ${MONO}`
        x.textBaseline = 'middle'
        x.textAlign = 'center'
        x.fillStyle = col
        for (let i = 0; i < CH.length; i++) x.fillText(CH[i], i * gw + gw / 2, gh / 2 + 0.5)
        return c
      })
      return true
    },
    draw() {
      const { gw, gh, cols, rows, dpr } = g
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const ox = g.ox * dpr
      const oy = g.oy * dpr
      const pw = g.cw * dpr
      const ph = g.ch * dpr
      let i = 0
      for (let y = 0; y < rows; y++)
        for (let x = 0; x < cols; x++, i++) {
          const c = g.chb[i]
          if (!c) continue
          ctx.drawImage(g.atlas[g.cob[i]], c * gw, 0, gw, gh, ox + x * pw, oy + y * ph, gw, gh)
        }
    },
  }
  return g
}

/* ---- the signed-distance primitives ---- */

const chipSD = (x: number, y: number, z: number, R: number, H: number, RB = 0.05) => {
  const r = Math.hypot(x, z)
  const dx = r - R + RB
  const dy = Math.abs(y) - H / 2 + RB
  return Math.min(Math.max(dx, dy), 0) + Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) - RB
}
const rbox = (x: number, y: number, z: number, hx: number, hy: number, hz: number, rr: number) => {
  const ax = Math.abs(x) - hx + rr
  const ay = Math.abs(y) - hy + rr
  const az = Math.abs(z) - hz + rr
  return (
    Math.min(Math.max(ax, ay, az), 0) + Math.hypot(Math.max(ax, 0), Math.max(ay, 0), Math.max(az, 0)) - rr
  )
}
const rot2 = (x: number, z: number, a: number): [number, number] => {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [c * x + s * z, -s * x + c * z]
}
const chipPat = (x: number, z: number, R: number) => {
  const rn = Math.hypot(x, z) / R
  const a = Math.atan2(z, x)
  const ph = (a / 6.2832 + 2) * 8
  const fr = ph - Math.floor(ph)
  return (fr < 0.36 && rn > 0.8) || (rn > 0.5 && rn < 0.56) ? 1 : 0
}
const cardPat = (lx: number, lz: number, hx: number, hz: number) => {
  const ax = Math.abs(lx)
  const az = Math.abs(lz)
  const bo = Math.max(ax - (hx - 0.09), az - (hz - 0.09))
  if (bo > -0.02 && bo < 0.02) return 1
  if (ax * 1.15 + az * 0.9 < 0.16) return 1
  return 0
}

/** The material the last sdf() call hit: 1 body, 2 pattern (the accent colour), 3/4 felt and rail, 5 the deck's edge. */
let im = 0

export type IconName = 'chip' | 'cards' | 'table' | 'deck' | 'stack'

interface Scene {
  elev: number
  dist: number
  look: [number, number, number]
  yaw0: number
  sdf(x: number, y: number, z: number): number
}

const ICONS: Record<IconName, Scene> = {
  chip: {
    elev: 0.6,
    dist: 3.3,
    look: [0.15, 0.1, -0.2],
    yaw0: 0.3,
    sdf(x, y, z) {
      /* one chip lying on the felt, a second leaning on it */
      let d = chipSD(x, y - 0.1, z, 1, 0.2)
      im = chipPat(x, z, 1) ? 2 : 1
      const [lx, lz] = rot2(x - 0.55, z + 0.95, 0.5)
      const ly = y - 0.55
      const th = 1.15
      const ry = Math.cos(th) * ly - Math.sin(th) * lz
      const rz = Math.sin(th) * ly + Math.cos(th) * lz
      const d2 = chipSD(lx, ry, rz, 1, 0.2)
      if (d2 < d) {
        d = d2
        im = chipPat(lx, rz, 1) ? 2 : 1
      }
      return d
    },
  },
  cards: {
    elev: 1.05,
    dist: 4.3,
    look: [0, 0, 0],
    yaw0: 0.25,
    sdf(x, y, z) {
      let d = 1e9
      for (let k = 0; k < 3; k++) {
        const a = -0.62 + k * 0.62
        const [lx, lz] = rot2(x - (k - 1) * 0.85, z + Math.abs(k - 1) * 0.22, a)
        const ly = y - 0.03 - k * 0.06
        const dk = rbox(lx, ly, lz, 0.72, 0.03, 1.0, 0.08)
        if (dk < d) {
          d = dk
          im = ly > 0.02 && cardPat(lx, lz, 0.72, 1.0) ? 2 : 1
        }
      }
      return d
    },
  },
  table: {
    elev: 0.78,
    dist: 3.35,
    look: [0, 0.1, 0],
    yaw0: 0.05,
    sdf(x, y, z) {
      /* a stadium of felt sunk inside a raised, rounded rail, a betting line inset from the edge,
         and one chip resting on the felt at each of four seats */
      const S = 0.72
      const A = 1.05 * S
      const R = 0.95 * S
      const H = 0.05 * S
      const RR = 0.02 * S
      const RW = 0.12 * S
      const RH = 0.16 * S
      const ly = y - RH
      const qx = Math.max(Math.abs(x) - A, 0)
      const de = Math.hypot(qx, z) - R
      const wz = Math.abs(ly) - H
      let d = (de > 0 && wz > 0 ? Math.hypot(de, wz) : Math.max(de, wz)) - RR
      im = ly > 0 && Math.abs(de + 0.3 * S) < 0.022 * S ? 2 : 3
      const dr = Math.abs(de - RW) - RW
      const wr = Math.abs(ly) - RH
      const rail = (dr > 0 && wr > 0 ? Math.hypot(dr, wr) : Math.max(dr, wr)) - 0.03 * S
      if (rail < d) {
        d = rail
        im = 1
      }
      const CR = 0.19 * S
      const CH = 0.028 * S
      const cy = RH + H + RR + CH + 0.003
      for (const s of [
        [-0.95, 0.15],
        [-0.4, -0.55],
        [0.55, 0.5],
        [0.95, -0.2],
      ]) {
        const cx = s[0] * S
        const cz = s[1] * S
        const q = chipSD(x - cx, y - cy, z - cz, CR, 2 * CH, 0.006)
        if (q < d) {
          d = q
          im = chipPat(x - cx, z - cz, CR) ? 2 : 1
        }
      }
      return d
    },
  },
  deck: {
    elev: 0.5,
    dist: 4.1,
    look: [0.1, 0.5, 0],
    yaw0: 0.35,
    sdf(x, y, z) {
      /* the deck as a block of cards, one card lifted through the gate */
      const [lx, lz] = rot2(x, z, 0.3)
      let d = rbox(lx, y - 0.4, lz, 0.72, 0.4, 1.0, 0.06)
      im =
        Math.abs(y - 0.4) < 0.39 &&
        (Math.abs(Math.abs(lx) - 0.72) < 0.03 || Math.abs(Math.abs(lz) - 1.0) < 0.03)
          ? 5
          : 1
      const th = 0.55
      const cx = x - 0.35
      const cy = y - 1.35
      const cz = z + 0.2
      const ry = Math.cos(th) * cy - Math.sin(th) * cz
      const rz = Math.sin(th) * cy + Math.cos(th) * cz
      const [ex, ez] = rot2(cx, rz, 0.15)
      const d2 = rbox(ex, ry, ez, 0.72, 0.03, 1.0, 0.08)
      if (d2 < d) {
        d = d2
        im = ry > 0.02 && cardPat(ex, ez, 0.72, 1.0) ? 2 : 1
      }
      return d
    },
  },
  stack: {
    elev: 0.36,
    dist: 4.9,
    look: [0.5, 0.85, 0.3],
    yaw0: 0.2,
    sdf(x, y, z) {
      let d = 1e9
      for (let i = 0; i < 8; i++) {
        const ox = Math.sin(i * 12.9) * 0.05
        const oz = Math.cos(i * 7.1) * 0.05
        const q = chipSD(x - ox, y - (i + 0.5) * 0.2, z - oz, 1, 0.2)
        if (q < d) {
          d = q
          im = 1
          if (Math.abs(y - (i + 0.5) * 0.2) > 0.085 && Math.hypot(x - ox, z - oz) < 0.99)
            im = chipPat(x - ox, z - oz, 1) ? 2 : 1
          else if (Math.hypot(x - ox, z - oz) > 0.9) {
            const a = Math.atan2(z - oz, x - ox) + i * 0.4
            const ph = (a / 6.2832 + 2) * 8
            const fr = ph - Math.floor(ph)
            im = fr < 0.36 ? 2 : 1
          }
        }
      }
      for (let i = 0; i < 3; i++) {
        const q = chipSD(x - 1.9, y - (i + 0.5) * 0.2, z - 0.9, 1, 0.2)
        if (q < d) {
          d = q
          im = 1
          if (Math.hypot(x - 1.9, z - 0.9) > 0.9) {
            const a = Math.atan2(z - 0.9, x - 1.9) + i
            const ph = (a / 6.2832 + 2) * 8
            const fr = ph - Math.floor(ph)
            im = fr < 0.36 ? 2 : 1
          } else if (Math.abs(y - (i + 0.5) * 0.2) > 0.085) im = chipPat(x - 1.9, z - 0.9, 1) ? 2 : 1
        }
      }
      return d
    },
  },
}

const IRAMP = ' .:-+*%@'

function renderIcon(g: CharGrid, sc: Scene, yaw: number) {
  const { cols, rows } = g
  if (!cols) return
  g.chb.fill(0)
  g.cob.fill(0)
  const ce = Math.cos(sc.elev)
  const se = Math.sin(sc.elev)
  const eye = [
    sc.look[0] + sc.dist * ce * Math.sin(yaw),
    sc.look[1] + sc.dist * se,
    sc.look[2] + sc.dist * ce * Math.cos(yaw),
  ]
  let f = [sc.look[0] - eye[0], sc.look[1] - eye[1], sc.look[2] - eye[2]]
  const fl = Math.hypot(f[0], f[1], f[2])
  f = f.map((v) => v / fl)
  let r = [-f[2], 0, f[0]]
  const rl = Math.hypot(r[0], r[2])
  r = [r[0] / rl, 0, r[2] / rl]
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
  const cellAsp = g.ch / g.cw
  const aspect = cols / (rows * cellAsp)
  const focal = 2.6
  /* key light from the camera's upper left, rim from behind right */
  const kx = -0.55
  const ky = 0.8
  const kz = 0.5
  let L = [
    r[0] * kx + u[0] * ky - f[0] * kz,
    r[1] * kx + u[1] * ky - f[1] * kz,
    r[2] * kx + u[2] * ky - f[2] * kz,
  ]
  const Ll = Math.hypot(L[0], L[1], L[2])
  L = L.map((v) => v / Ll)
  let L2 = [
    r[0] * 0.6 + u[0] * 0.4 + f[0] * 0.7,
    r[1] * 0.6 + u[1] * 0.4 + f[1] * 0.7,
    r[2] * 0.6 + u[2] * 0.4 + f[2] * 0.7,
  ]
  const L2l = Math.hypot(L2[0], L2[1], L2[2])
  L2 = L2.map((v) => v / L2l)
  for (let cy = 0; cy < rows; cy++)
    for (let cx = 0; cx < cols; cx++) {
      const px0 = (((cx + 0.5) / cols) * 2 - 1) * aspect
      const py0 = 1 - ((cy + 0.5) / rows) * 2
      let dx = f[0] * focal + r[0] * px0 + u[0] * py0
      let dy = f[1] * focal + r[1] * px0 + u[1] * py0
      let dz = f[2] * focal + r[2] * px0 + u[2] * py0
      const dl = Math.hypot(dx, dy, dz)
      dx /= dl
      dy /= dl
      dz /= dl
      let t = 0.5
      let hit = false
      let px = 0
      let py = 0
      let pz = 0
      const tg = dy < -1e-6 ? -eye[1] / dy : 1e9
      for (let i = 0; i < 64; i++) {
        px = eye[0] + dx * t
        py = eye[1] + dy * t
        pz = eye[2] + dz * t
        const d = sc.sdf(px, py, pz)
        if (d < 0.004) {
          hit = true
          break
        }
        t += d * 0.9
        if (t > 12 || t > tg + 0.05) break
      }
      let I = 0
      let cls = 0
      if (hit) {
        const m = im
        const e = 0.006
        const nx = sc.sdf(px + e, py, pz) - sc.sdf(px - e, py, pz)
        const ny = sc.sdf(px, py + e, pz) - sc.sdf(px, py - e, pz)
        const nz = sc.sdf(px, py, pz + e) - sc.sdf(px, py, pz - e)
        const nl = Math.hypot(nx, ny, nz) || 1
        const n = [nx / nl, ny / nl, nz / nl]
        const ao = clamp(sc.sdf(px + n[0] * 0.12, py + n[1] * 0.12, pz + n[2] * 0.12) / 0.12, 0.3, 1)
        const diff = Math.max(0, n[0] * L[0] + n[1] * L[1] + n[2] * L[2])
        let sh = 1
        {
          let tt = 0.04
          for (let k = 0; k < 18; k++) {
            const d = sc.sdf(px + L[0] * tt, py + L[1] * tt, pz + L[2] * tt)
            if (d < 0.003) {
              sh = 0.25
              break
            }
            tt += Math.max(0.03, d)
            if (tt > 3) break
          }
        }
        let hx = L[0] - dx
        let hy = L[1] - dy
        let hz = L[2] - dz
        const hl = Math.hypot(hx, hy, hz)
        hx /= hl
        hy /= hl
        hz /= hl
        const spec = Math.pow(Math.max(0, n[0] * hx + n[1] * hy + n[2] * hz), 26)
        const ndv = -(n[0] * dx + n[1] * dy + n[2] * dz)
        const rim =
          Math.pow(1 - Math.max(0, ndv), 3) * Math.max(0, n[0] * L2[0] + n[1] * L2[1] + n[2] * L2[2])
        let alb = 1.0
        if (m === 3) alb = 0.42
        else if (m === 4) alb = 0.22
        else if (m === 5) alb = 0.7
        I =
          alb * (0.07 * ao + 0.1 * Math.max(0, ndv) * ao + 1.0 * diff * sh + 0.35 * spec * sh) +
          0.5 * rim * ao
        I = 1 - Math.exp(-2.1 * I)
        cls = m === 2 ? 2 : 1
      } else if (tg < 1e8) {
        const gx = eye[0] + dx * tg
        const gz = eye[2] + dz * tg
        const rr = Math.hypot(gx - sc.look[0], gz - sc.look[2]) / 2.4
        if (rr < 1) {
          let sh = 1
          {
            let tt = 0.05
            for (let k = 0; k < 18; k++) {
              const d = sc.sdf(gx + L[0] * tt, L[1] * tt, gz + L[2] * tt)
              if (d < 0.003) {
                sh = 0.1
                break
              }
              tt += Math.max(0.03, d)
              if (tt > 4) break
            }
          }
          const fall = 1 - rr
          I = 0.11 * fall * fall * sh
          cls = 3
        }
      }
      if (!cls) continue
      const bay = (BAYER[cy % 4][cx % 4] + 0.5) / 16 - 0.5
      let lv = clamp(Math.round(I * (IRAMP.length - 1) + bay * 1.15), 0, IRAMP.length - 1)
      if (lv === 0) {
        if (cls === 3) continue
        lv = 1
      }
      if (cls === 2 && lv < 2) lv = 2
      g.chb[cy * cols + cx] = g.CIDX[IRAMP[lv]]
      g.cob[cy * cols + cx] = cls === 2 ? 1 : cls === 3 ? 2 : 0
    }
  g.draw()
}

export interface ObjectOptions {
  /** cell size in px; the homepage draws the ladder icons at 4 */
  font?: number
  /** rad/s while the page is quiet; the object spins ~3x faster under the mouse */
  speed?: number
  /** glyph colours: the body, the pattern, the floor shadow */
  palette?: [string, string, string]
}

/**
 * Draw one rung object on a canvas and keep it turning slowly while it is on
 * screen. Returns the teardown. Rendering is a full raymarch per frame, so
 * the quiet rate is held to ~12 fps and the object only draws while the
 * canvas intersects the viewport; with reduced motion it is drawn once.
 */
export function mountObject(canvas: HTMLCanvasElement, icon: IconName, opts: ObjectOptions = {}): () => void {
  const sc = ICONS[icon]
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches
  const g = makeGrid(canvas, opts.font ?? 4, opts.palette ?? ['#f5f0e6', '#c23553', 'rgba(245,240,230,.42)'])
  const st = { yaw: sc.yaw0, hover: false, visible: true, last: 0, drawn: false }
  const speed = opts.speed ?? 0.35

  const onEnter = () => {
    st.hover = true
  }
  const onLeave = () => {
    st.hover = false
  }
  canvas.addEventListener('mouseenter', onEnter)
  canvas.addEventListener('mouseleave', onLeave)

  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) st.visible = e.isIntersecting
    },
    { rootMargin: '80px' },
  )
  io.observe(canvas)
  const ro = new ResizeObserver(() => {
    if (g.resize()) {
      renderIcon(g, sc, st.yaw)
      st.drawn = true
    }
  })
  ro.observe(canvas)
  if (document.fonts?.ready)
    document.fonts.ready.then(() => {
      if (g.resize()) renderIcon(g, sc, st.yaw)
    })

  let timer = 0
  let raf = 0
  const frame = (now: number) => {
    raf = 0
    if (!g.cols) g.resize()
    if (g.cols && st.visible && !document.hidden) {
      const dt = st.last ? Math.min(0.1, (now - st.last) / 1000) : 0
      st.last = now
      if (!reduced) {
        st.yaw += dt * (st.hover ? speed * 3 : speed)
        renderIcon(g, sc, st.yaw)
      } else if (!st.drawn) {
        renderIcon(g, sc, st.yaw)
        st.drawn = true
      }
    } else st.last = 0
    if (!reduced) timer = window.setTimeout(() => (raf = requestAnimationFrame(frame)), st.hover ? 40 : 80)
  }
  raf = requestAnimationFrame(frame)

  return () => {
    clearTimeout(timer)
    cancelAnimationFrame(raf)
    io.disconnect()
    ro.disconnect()
    canvas.removeEventListener('mouseenter', onEnter)
    canvas.removeEventListener('mouseleave', onLeave)
  }
}
