/* chip.js — the Drawmaha chip: a dithered, tilted poker chip turning on its axis, drawn as ASCII on a canvas.
   The plain-JS twin of the apps' src/ui/mark.ts, served at /chip.js and loaded by every static page. The mark on
   paper: the face in ink (#16140f), the edge sectors and the inner ring in the chip's oxblood (#8e2038). Transparent
   field. On paper how dark a cell reads is how much of it the glyph covers, so — as in mark.ts — the grid is set in
   Plex Mono 600, every glyph is emboldened by a stroke of its own colour (EMBOLDEN of the cell size), the face's ramp
   starts at ':' and the pattern is always '@'. Every glyph is drawn at full opacity.
   Two ways in:
     mountChip(canvas, { mark, ink, font, speed, tilt })  — one canvas, the same options as mark.ts; returns a teardown.
     mountChips({ mark, ink })                            — every <canvas data-chip> on the page, options per canvas via
                                                            data-font="1.5" (cell px), data-speed="0.3" (rad/s), data-tilt="0.8". */
(function () {
const FONT = "600 {px}px 'IBM Plex Mono', ui-monospace, Menlo, monospace";
const font = (px) => FONT.replace('{px}', String(px));
/* the stroke laid over each glyph, as a fraction of the cell size: the coverage that makes 1.5 px cells read */
const EMBOLDEN = 0.16;
/* the face's glyphs from shadow to light; the pattern is always the last */
const FACE = ':+%#@@';
const RAMP = " .'`,:;~-=+*x%#@";
let CHARSET = '';
for (let c = 32; c < 127; c++) CHARSET += String.fromCharCode(c);
CHARSET += '♠♥♦♣·';
const CIDX = {}; for (let i = 0; i < CHARSET.length; i++) CIDX[CHARSET[i]] = i;
const RAMPIDX = [...RAMP].map((c) => CIDX[c]);
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
class Grid {
  constructor(canvas, palette, fontPx) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.palette = palette; this.fontPx = fontPx || 11;
    this.mouse = { x: -1e6, y: -1e6, in: false };
    this.c = 1;
    this.resize();
    const move = (cx, cy) => {
      const r = canvas.getBoundingClientRect();
      this.mouse.x = ((cx - r.left) - this.ox) / this.cw;
      this.mouse.y = (((cy - r.top) - this.oy) / this.ch) * this.ar;
      this.mouse.in = true;
      this.dirty = true;
    };
    this.onMove = (e) => move(e.clientX, e.clientY);
    this.onLeave = () => { this.mouse.in = false; this.dirty = true; };
    this.onTouch = (e) => { const t = e.touches[0]; move(t.clientX, t.clientY); };
    canvas.addEventListener('mousemove', this.onMove);
    canvas.addEventListener('mouseleave', this.onLeave);
    canvas.addEventListener('touchmove', this.onTouch, { passive: true });
  }
  /* the teardown's half: the listeners the constructor attached, removed by name */
  detach() {
    this.canvas.removeEventListener('mousemove', this.onMove);
    this.canvas.removeEventListener('mouseleave', this.onLeave);
    this.canvas.removeEventListener('touchmove', this.onTouch);
  }
  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const r = this.canvas.getBoundingClientRect();
    if (r.width === 0) return;
    this.dpr = dpr;
    this.canvas.width = Math.round(r.width * dpr); this.canvas.height = Math.round(r.height * dpr);
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.font = font(this.fontPx);
    this.cw = ctx.measureText('M').width;
    this.ch = this.fontPx * 1.02;
    this.cols = Math.floor(r.width / this.cw); this.rows = Math.floor(r.height / this.ch);
    this.ar = this.ch / this.cw;
    this.W = this.cols; this.H = this.rows * this.ar;
    this.ox = (r.width - this.cols * this.cw) / 2; this.oy = (r.height - this.rows * this.ch) / 2;
    this.n = this.cols * this.rows;
    this.chb = new Uint8Array(this.n); this.cob = new Uint8Array(this.n);
    this.buildAtlas();
    this.dirty = true;
  }
  buildAtlas() {
    const dpr = this.dpr, gw = Math.ceil(this.cw * dpr), gh = Math.ceil(this.ch * dpr);
    this.gw = gw; this.gh = gh;
    this.atlas = this.palette.map((col) => {
      const c = document.createElement('canvas');
      c.width = gw * CHARSET.length; c.height = gh;
      const x = c.getContext('2d');
      x.font = font(this.fontPx * dpr); x.textBaseline = 'middle'; x.textAlign = 'center'; x.fillStyle = col;
      x.strokeStyle = col; x.lineWidth = this.fontPx * dpr * EMBOLDEN; x.lineJoin = 'round';
      for (let i = 0; i < CHARSET.length; i++) {
        x.fillText(CHARSET[i], i * gw + gw / 2, gh / 2 + 0.5);
        x.strokeText(CHARSET[i], i * gw + gw / 2, gh / 2 + 0.5);
      }
      return c;
    });
  }
  clear() { this.chb.fill(0); this.cob.fill(0); }
  draw() {
    const ctx = this.ctx, { gw, gh, cols, rows, dpr } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (this.palette[0] === 'transparent') ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    else { ctx.fillStyle = this.palette[0]; ctx.fillRect(0, 0, this.canvas.width, this.canvas.height); }
    const ox = this.ox * dpr, oy = this.oy * dpr, pw = this.cw * dpr, ph = this.ch * dpr;
    let i = 0;
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++, i++) {
      const c = this.chb[i]; if (!c) continue;
      ctx.drawImage(this.atlas[this.cob[i]], c * gw, 0, gw, gh, ox + x * pw, oy + y * ph, gw, gh);
    }
  }
  /* set a cell by cell coordinates */
  set(cx, cy, ci, col) {
    if (cx < 0 || cy < 0 || cx >= this.cols || cy >= this.rows) return;
    const i = cy * this.cols + cx; this.chb[i] = ci; this.cob[i] = col;
  }
  ramp(v) { return RAMPIDX[clamp(Math.round(v * (RAMP.length - 1)), 0, RAMP.length - 1)]; }
  /* fill a bbox (scene units) with fn(px,py) -> brightness; fn may reassign g.c for colour */
  fill(x0, y0, x1, y1, col, fn) {
    const cx0 = Math.max(0, Math.floor(x0)), cx1 = Math.min(this.cols - 1, Math.ceil(x1));
    const cy0 = Math.max(0, Math.floor(y0 / this.ar)), cy1 = Math.min(this.rows - 1, Math.ceil(y1 / this.ar));
    for (let cy = cy0; cy <= cy1; cy++) {
      const py = (cy + 0.5) * this.ar;
      for (let cx = cx0; cx <= cx1; cx++) {
        this.c = col;
        const v = fn(cx + 0.5, py);
        if (v > 0) { const i = cy * this.cols + cx; this.chb[i] = this.ramp(v); this.cob[i] = this.c; }
        else if (v < -1) { const i = cy * this.cols + cx; this.chb[i] = 0; }
      }
    }
  }
  /* a stroke between two scene points; the glyph follows the slope unless given */
  line(x0, y0, x1, y1, col, ch) {
    const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy / this.ar);
    const n = Math.max(1, Math.ceil(len * 2));
    const ci = CIDX[ch || slopeChar(Math.atan2(dy, dx))];
    for (let k = 0; k <= n; k++) {
      const s = k / n;
      this.set(Math.floor(x0 + dx * s), Math.floor((y0 + dy * s) / this.ar), ci, col);
    }
  }
  text(cx, cy, str, col) {
    for (let i = 0; i < str.length; i++) { const ci = CIDX[str[i]]; if (ci) this.set(cx + i, cy, ci, col); }
  }
  /* where a scene point lands, in cells */
  cellOf(x, y) { return [Math.floor(x), Math.floor(y / this.ar)]; }
}
const BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
/* ================================================================
   SHARED PIECES FOR THE SECOND ROUND
   ================================================================ */
const rotX = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[1, 0, 0], [0, c, -s], [0, s, c]]; };
const rotY = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; };
const rotZ = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };
const mul = (A, B) => A.map((r, i) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const norm3 = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
/* a ray-traced chip. o.M = 3x3, columns are the chip's local axes in screen space (x right, y down, z away) */
function rayChip(g, o) {
  const { cols, rows, ar } = g, m = o.M;
  const toLocal = (x, y, z) => [m[0][0] * x + m[1][0] * y + m[2][0] * z, m[0][1] * x + m[1][1] * y + m[2][1] * z, m[0][2] * x + m[1][2] * y + m[2][2] * z];
  const toWorld = (x, y, z) => [m[0][0] * x + m[0][1] * y + m[0][2] * z, m[1][0] * x + m[1][1] * y + m[1][2] * z, m[2][0] * x + m[2][1] * y + m[2][2] * z];
  const R = o.R, hz = o.hz, L = o.L, sectors = o.sectors || 12;
  const d = toLocal(0, 0, 1);
  const chars = [...FACE].map((c) => CIDX[c]);
  const ext = R * 1.05 + hz;
  const cx0 = Math.max(0, Math.floor(o.cx - ext)), cx1 = Math.min(cols - 1, Math.ceil(o.cx + ext));
  const cy0 = Math.max(0, Math.floor((o.cy - ext) / ar)), cy1 = Math.min(rows - 1, Math.ceil((o.cy + ext) / ar));
  for (let cyy = cy0; cyy <= cy1; cyy++) {
    const py = (cyy + 0.5) * ar - o.cy;
    for (let cxx = cx0; cxx <= cx1; cxx++) {
      const px = cxx + 0.5 - o.cx;
      if (px * px + py * py > ext * ext) continue;
      const oo = toLocal(px, py, -400);
      let best = 1e9, mat = 0, I = 0;
      for (const zc of [hz, -hz]) {
        if (Math.abs(d[2]) < 1e-6) continue;
        const tc = (zc - oo[2]) / d[2];
        if (tc < 0 || tc > best) continue;
        const x = oo[0] + tc * d[0], y = oo[1] + tc * d[1], r = Math.hypot(x, y);
        if (r > R) continue;
        best = tc;
        const rn = r / R, ang = Math.atan2(y, x);
        const sector = Math.floor(((ang / (2 * Math.PI)) + 1) * sectors + 0.5) % 2;
        mat = (rn > 0.8 && sector) || (rn > 0.5 && rn < 0.58) ? 2 : 1;
        const n = toWorld(0, 0, zc > 0 ? 1 : -1);
        const nl = n[0] * L[0] + n[1] * L[1] + n[2] * L[2];
        const hl = Math.pow(nl * 0.5 + 0.5, 2), lam = Math.max(0, nl);
        I = 0.4 + 0.7 * hl + 0.35 * Math.pow(lam, 16);
        if (rn > 0.96) I += 0.15;
        if (rn > 0.8 && !sector) I *= 0.9;
      }
      const a = d[0] * d[0] + d[1] * d[1];
      if (a > 1e-9) {
        const b = 2 * (oo[0] * d[0] + oo[1] * d[1]), c = oo[0] * oo[0] + oo[1] * oo[1] - R * R;
        const disc = b * b - 4 * a * c;
        if (disc > 0) {
          const ts = (-b - Math.sqrt(disc)) / (2 * a);
          if (ts > 0 && ts < best) {
            const z = oo[2] + ts * d[2];
            if (Math.abs(z) <= hz) {
              best = ts;
              const x = oo[0] + ts * d[0], y = oo[1] + ts * d[1];
              const sector = Math.floor(((Math.atan2(y, x) / (2 * Math.PI)) + 1) * sectors + 0.5) % 2;
              mat = sector ? 2 : 1;
              const n = toWorld(x / R, y / R, 0);
              const nl = n[0] * L[0] + n[1] * L[1] + n[2] * L[2];
              I = 0.3 + 0.7 * Math.pow(nl * 0.5 + 0.5, 2) + 0.35 * Math.pow(Math.max(0, nl), 10);
            }
          }
        }
      }
      if (mat === 0) continue;
      const i = cyy * cols + cxx;
      const col = mat === 1 ? o.colBase : o.colPattern;
      if (o.dither) {
        const bay = (BAYER[cyy % 4][cxx % 4] + 0.5) / 16 - 0.5;
        /* the pattern is always the densest glyph: the sectors are what make the disc read as a chip, even in shadow */
        const lv = mat === 2 ? 5 : clamp(Math.floor(I * 5 + bay * 1.15), 0, 5);
        g.chb[i] = chars[lv]; g.cob[i] = col;
      } else {
        g.chb[i] = g.ramp(clamp(I, 0.08, 1)); g.cob[i] = col;
      }
    }
  }
}

const CHIP_SCENES = [];
function mountChips(opts) {
  const mark = (opts && opts.mark) || '#16140f', ink = (opts && opts.ink) || '#8e2038';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  for (const canvas of document.querySelectorAll('canvas[data-chip]')) {
    const g = new Grid(canvas, ['transparent', mark, ink], parseFloat(canvas.dataset.font || '1.5'));
    CHIP_SCENES.push({ g, canvas, yaw: 0.6, t: 0, speed: parseFloat(canvas.dataset.speed || '0.3'), tilt: parseFloat(canvas.dataset.tilt || '0.8'), drawn: false });
  }
  const ro = new ResizeObserver(() => CHIP_SCENES.forEach((s) => { s.g.resize(); s.drawn = false; }));
  CHIP_SCENES.forEach((s) => ro.observe(s.canvas));
  if (document.fonts && document.fonts.load) document.fonts.load(font(12)).then(() => CHIP_SCENES.forEach((s) => { s.g.resize(); s.drawn = false; }));
  let last = performance.now();
  function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (!document.hidden) for (const s of CHIP_SCENES) {
      if (reduced && s.drawn && !s.g.dirty) continue;
      s.g.dirty = false; s.drawn = true;
      if (!reduced) { s.t += dt; s.yaw += dt * (s.g.mouse.in ? s.speed * 3.5 : s.speed); }
      const g = s.g, { W, H } = g; const R = Math.min(W, H) * 0.46;
      g.clear();
      rayChip(g, { cx: W / 2, cy: H / 2, R, hz: R * 0.13, M: mul(rotX(s.tilt + 0.12 * Math.sin(s.t * 0.37)), mul(rotY(0.25 * Math.sin(s.t * 0.29)), rotZ(s.yaw))), L: norm3([-0.55, -0.5, -0.67]), colBase: 1, colPattern: 2, dither: true, sectors: 12 });
      g.draw();
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
window.mountChips = mountChips;

/* One canvas, the way mark.ts mounts it: draw the chip and keep it turning; returns the teardown. With reduced motion
   the chip is drawn once and still answers a resize. */
function mountChip(canvas, opts) {
  opts = opts || {};
  const mark = opts.mark || '#16140f', ink = opts.ink || '#8e2038';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const g = new Grid(canvas, ['transparent', mark, ink], opts.font || 1.5);
  const s = { yaw: 0.6, t: 0, speed: opts.speed == null ? 0.3 : opts.speed, tilt: opts.tilt == null ? 0.8 : opts.tilt, drawn: false };
  const ro = new ResizeObserver(() => { g.resize(); s.drawn = false; });
  ro.observe(canvas);
  if (document.fonts && document.fonts.load) document.fonts.load(font(12)).then(() => { g.resize(); s.drawn = false; });
  let raf = 0, last = performance.now();
  function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (!document.hidden && g.cols && !(reduced && s.drawn && !g.dirty)) {
      g.dirty = false; s.drawn = true;
      if (!reduced) { s.t += dt; s.yaw += dt * (g.mouse.in ? s.speed * 3.5 : s.speed); }
      const { W, H } = g; const R = Math.min(W, H) * 0.46;
      g.clear();
      rayChip(g, { cx: W / 2, cy: H / 2, R, hz: R * 0.13, M: mul(rotX(s.tilt + 0.12 * Math.sin(s.t * 0.37)), mul(rotY(0.25 * Math.sin(s.t * 0.29)), rotZ(s.yaw))), L: norm3([-0.55, -0.5, -0.67]), colBase: 1, colPattern: 2, dither: true, sectors: 12 });
      g.draw();
    }
    raf = requestAnimationFrame(tick);
  }
  raf = requestAnimationFrame(tick);
  return () => { cancelAnimationFrame(raf); ro.disconnect(); g.detach(); };
}
window.mountChip = mountChip;
})();
