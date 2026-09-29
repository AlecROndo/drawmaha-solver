"""Vercel entrypoint: the cover page for the drawmaha-solver validation ladder.

The repo is a solver library and CLI; this page states where the ladder
stands and links each completed rung's live artifact (rung 0: the
regret-matching visualizer at /rung0; rung 1: the Kuhn CFR visualizer at
/rung1). The GTOWizard-style dashboard planned for rung 4 replaces this page.

Stdlib only on purpose — the page must never depend on the solver's numeric
stack, so a heavy dependency can't break the deploy.

Visual language is the site's duotone system, shared with both visualizers:
one hue and one paper that swap for the light colour scheme, the validation
ladder drawn as the nav, a persistent identity rail, three type voices
(Instrument Serif display / IBM Plex Mono body / Kalam annotation), and
monoline illustration at a single stroke weight.

The page opens on two ASCII windows drawn on canvas by HERO_JS: a whole hand
of Drawmaha seen from above, and five men at a bar who watch you. The rail's
mark is a third, small one: a dithered chip turning on its axis. The windows
keep the oxblood field in both colour schemes (they are rooms you look into);
the chip sits on the field and follows it. The script reads the colour tokens
off :root so the canvases stay inside the duotone.
"""

from http.server import BaseHTTPRequestHandler

# The ASCII engine and the three scenes. A raw string: the JS has its own
# backslashes. Extracted from the studies page at web/hero-studies/studies.html.
HERO_JS = r"""
const FONT = "'IBM Plex Mono', ui-monospace, Menlo, monospace";
const RAMP = " .'`,:;~-=+*x%#@";
let CHARSET = '';
for (let c = 32; c < 127; c++) CHARSET += String.fromCharCode(c);
CHARSET += '♠♥♦♣·';
const CIDX = {}; for (let i = 0; i < CHARSET.length; i++) CIDX[CHARSET[i]] = i;
const RAMPIDX = [...RAMP].map((c) => CIDX[c]);
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, s) => a + (b - a) * s;
const ease = (s) => (s < 0 ? 0 : s > 1 ? 1 : s * s * (3 - 2 * s));
const easeOut = (s) => 1 - Math.pow(1 - clamp(s, 0, 1), 3);
/* value noise: enough for felt, currents and range fields */
function hash3(i, j, k) {
  let n = Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(k, 1440671221);
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
}
function noise3(x, y, z) {
  const i = Math.floor(x), j = Math.floor(y), k = Math.floor(z);
  let fx = x - i, fy = y - j, fz = z - k;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
  const a = lerp(hash3(i, j, k), hash3(i + 1, j, k), fx);
  const b = lerp(hash3(i, j + 1, k), hash3(i + 1, j + 1, k), fx);
  const c = lerp(hash3(i, j, k + 1), hash3(i + 1, j, k + 1), fx);
  const d = lerp(hash3(i, j + 1, k + 1), hash3(i + 1, j + 1, k + 1), fx);
  return lerp(lerp(a, b, fy), lerp(c, d, fy), fz);
}
/* signed distances, all in scene units (isotropic) */
const sdBox = (x, y, hw, hh, r = 0) => {
  const qx = Math.abs(x) - hw + r, qy = Math.abs(y) - hh + r;
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
};
const sdSeg = (px, py, ax, ay, bx, by) => {
  const dx = bx - ax, dy = by - ay;
  const t = clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1), 0, 1);
  return Math.hypot(px - ax - t * dx, py - ay - t * dy);
};
const inEllipse = (x, y, rx, ry) => (x * x) / (rx * rx) + (y * y) / (ry * ry);
/* which glyph draws a stroke at this angle (screen y points down) */
function slopeChar(a) {
  a = ((a % Math.PI) + Math.PI) % Math.PI;
  if (a < Math.PI / 8 || a > (7 * Math.PI) / 8) return '-';
  if (a < (3 * Math.PI) / 8) return '\\';
  if (a < (5 * Math.PI) / 8) return '|';
  return '/';
}
/* suit pips as inside-tests on a [-1,1] square, y down */
function inSuit(s, x, y) {
  if (s === 'h') {
    return Math.hypot(x - 0.5, y + 0.35) < 0.55 || Math.hypot(x + 0.5, y + 0.35) < 0.55 ||
      (y > -0.35 && y < 0.95 && Math.abs(x) < 1.02 * (0.95 - y) / 1.3);
  }
  if (s === 's') {
    const yy = -y + 0.15;
    const body = Math.hypot(x - 0.5, yy + 0.35) < 0.55 || Math.hypot(x + 0.5, yy + 0.35) < 0.55 ||
      (yy > -0.35 && yy < 0.95 && Math.abs(x) < 1.02 * (0.95 - yy) / 1.3);
    const stem = Math.abs(x) < 0.13 && y > 0.3 && y < 0.98;
    const base = y > 0.8 && y < 1.0 && Math.abs(x) < (y - 0.55) * 1.2;
    return body || stem || base;
  }
  if (s === 'd') return Math.abs(x) / 0.72 + Math.abs(y) / 1.0 < 1;
  /* club */
  const stem = Math.abs(x) < 0.13 && y > 0.1 && y < 0.98;
  const base = y > 0.8 && y < 1.0 && Math.abs(x) < (y - 0.55) * 1.2;
  return Math.hypot(x, y + 0.45) < 0.44 || Math.hypot(x - 0.45, y - 0.15) < 0.44 ||
    Math.hypot(x + 0.45, y - 0.15) < 0.44 || stem || base;
}
const SUITCH = { s: '♠', h: '♥', d: '♦', c: '♣' };
const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
function dealCards(n, seed) {
  const seen = {}, out = [];
  let k = 0;
  while (out.length < n) {
    const r = RANKS[Math.floor(hash3(seed, k, 1) * 13)], s = 'shdc'[Math.floor(hash3(seed, k, 2) * 4)];
    k++;
    if (seen[r + s]) continue;
    seen[r + s] = true; out.push({ r, s });
  }
  return out;
}
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
    canvas.addEventListener('mousemove', (e) => move(e.clientX, e.clientY));
    canvas.addEventListener('mouseleave', () => { this.mouse.in = false; this.dirty = true; });
    canvas.addEventListener('touchmove', (e) => { const t = e.touches[0]; move(t.clientX, t.clientY); }, { passive: true });
  }
  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const r = this.canvas.getBoundingClientRect();
    if (r.width === 0) return;
    this.dpr = dpr;
    this.canvas.width = Math.round(r.width * dpr); this.canvas.height = Math.round(r.height * dpr);
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.font = `${this.fontPx}px ${FONT}`;
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
      x.font = `${this.fontPx * dpr}px ${FONT}`; x.textBaseline = 'middle'; x.textAlign = 'center'; x.fillStyle = col;
      for (let i = 0; i < CHARSET.length; i++) x.fillText(CHARSET[i], i * gw + gw / 2, gh / 2 + 0.5);
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
/* ---------- a card, drawn back-to-front safe (later calls overwrite) ---------- */
function drawCard(g, cx, cy, w, h, ang, sx, face, cols) {
  /* cols: {edge, fill, black, red, back} palette indices */
  if (sx < 0.04) sx = 0.04;
  const ca = Math.cos(ang), sa = Math.sin(ang);
  const ex = (w / 2) * sx * Math.abs(ca) + (h / 2) * Math.abs(sa) + 2, ey = (w / 2) * sx * Math.abs(sa) + (h / 2) * Math.abs(ca) + 2;
  const pip = w * 0.19;
  g.fill(cx - ex, cy - ey, cx + ex, cy + ey, cols.edge, (px, py) => {
    const dx = px - cx, dy = py - cy;
    const u = (dx * ca + dy * sa) / sx, v = -dx * sa + dy * ca;
    const d = sdBox(u, v, w / 2, h / 2, 1.4);
    if (d > 0.35) return 0;
    if (d > -0.75) {
      /* which edge: pick a slope glyph for it */
      const onSide = Math.abs(u) > w / 2 - 1.4 && Math.abs(v) < h / 2 - 1.4;
      const onTop = Math.abs(v) > h / 2 - 1.4 && Math.abs(u) < w / 2 - 1.4;
      if (sx < 0.35) return 0.95;
      if (onSide || onTop) return 0; /* straight edges get slope glyphs in the second pass */
      return 0.55;
    }
    if (!face) {
      const a = Math.floor(u / 2.2) + Math.floor(v / 1.5);
      g.c = cols.back;
      return (a % 2 === 0) ? 0.32 : 0.14;
    }
    /* face */
    if (sx < 0.35) { g.c = cols.fill; return 0.08; }
    if (inSuit(face.s, u / pip, v / pip)) { g.c = (face.s === 'h' || face.s === 'd') ? cols.red : cols.black; return 1; }
    g.c = cols.fill;
    return 0.07;
  });
  /* second pass for the slope glyphs on straight edges: cheap re-walk of the outline */
  const cornerU = w / 2 - 1.2, cornerV = h / 2 - 1.2;
  const P = (u, v) => [cx + (u * sx * ca - v * sa), cy + (u * sx * sa + v * ca)];
  const strokes = [[-cornerU, -cornerV, cornerU, -cornerV], [cornerU, cornerV, -cornerU, cornerV],
    [-cornerU, -cornerV, -cornerU, cornerV], [cornerU, -cornerV, cornerU, cornerV]];
  if (sx > 0.35) for (const s of strokes) { const a = P(s[0], s[1]), b = P(s[2], s[3]); g.line(a[0], a[1], b[0], b[1], cols.edge); }
  if (face && sx > 0.6) {
    const col = (face.s === 'h' || face.s === 'd') ? cols.red : cols.black;
    const a = P(-w / 2 + 2.2, -h / 2 + 2.6); const c1 = g.cellOf(a[0], a[1]);
    g.text(c1[0], c1[1], face.r, col); g.text(c1[0], c1[1] + 1, SUITCH[face.s], col);
    const b = P(w / 2 - 2.6, h / 2 - 4.2); const c2 = g.cellOf(b[0], b[1]);
    g.text(c2[0], c2[1], face.r, col); g.text(c2[0], c2[1] + 1, SUITCH[face.s], col);
  }
}
function ellipseOutline(g, cx, cy, rx, ry, col, a0 = 0, a1 = Math.PI * 2) {
  const n = Math.max(12, Math.ceil((rx + ry) * 1.2 * (a1 - a0) / (Math.PI * 2)));
  for (let k = 0; k < n; k++) {
    const t0 = a0 + ((a1 - a0) * k) / n, t1 = a0 + ((a1 - a0) * (k + 1)) / n;
    g.line(cx + Math.cos(t0) * rx, cy + Math.sin(t0) * ry, cx + Math.cos(t1) * rx, cy + Math.sin(t1) * ry, col);
  }
}
const BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
/* ================================================================
   SHARED PIECES FOR THE SECOND ROUND
   ================================================================ */
const flight = (a, b, s, arc) => [lerp(a[0], b[0], s), lerp(a[1], b[1], s) - Math.sin(Math.PI * s) * (arc || 0)];
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
  const chars = [0, CIDX['.'], CIDX[':'], CIDX['+'], CIDX['%'], CIDX['@']];
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
        I = 0.12 + 0.62 * hl + 0.35 * Math.pow(lam, 16);
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
              I = 0.1 + 0.55 * Math.pow(nl * 0.5 + 0.5, 2) + 0.3 * Math.pow(Math.max(0, nl), 10);
            }
          }
        }
      }
      if (mat === 0) continue;
      const i = cyy * cols + cxx;
      const col = mat === 1 ? o.colBase : o.colPattern;
      if (o.dither) {
        const bay = (BAYER[cyy % 4][cxx % 4] + 0.5) / 16 - 0.5;
        const lv = clamp(Math.floor(I * 5 + bay * 1.15), 0, 5);
        if (lv > 0) { g.chb[i] = chars[lv]; g.cob[i] = col; } else { g.chb[i] = chars[1]; g.cob[i] = o.colDim; }
      } else {
        g.chb[i] = g.ramp(clamp(I, 0.08, 1)); g.cob[i] = col;
      }
    }
  }
}
/* ---------- the table hand: one timeline, three ways of drawing it ---------- */
const SEATS = 6;
function tableTimeline(t, st) {
  /* returns everything in table coordinates: X in [-1,1] across, Z in [-1,1] with +1 = far (top) */
  const P = 12.0, loop = Math.floor(t / P), tt = t - loop * P;
  const seats = [];
  for (let k = 0; k < SEATS; k++) { const a = -Math.PI / 2 + k * Math.PI / 3; seats.push({ a, X: Math.cos(a) * 0.86, Z: Math.sin(a) * 0.82, pX: Math.cos(a) * 1.16, pZ: Math.sin(a) * 1.16 }); }
  /* seat 3 is far top = dealer's side; you sit at seat 0 (bottom, Z = -0.82) */
  const winner = Math.floor(hash3(loop, 3, 3) * SEATS);
  const you = dealCards(5, loop + 41);
  const dealer = { X: 0, Z: 1.22 };
  const cards = [], chips = [], heads = [];
  let arm = null;
  const dealAt = (r, k) => 0.5 + (r * SEATS + k) * 0.1, dealDur = 0.36;
  const order = [0, 1, 2, 3, 4, 5].map((k) => (k + 1) % SEATS); /* deal starts left of you */
  for (let r = 0; r < 5; r++) for (let ki = 0; ki < SEATS; ki++) {
    const k = order[ki], s = seats[k], t0 = dealAt(r, ki), fs = (tt - t0) / dealDur; if (fs < 0) continue;
    const tX = s.X + Math.cos(s.a + Math.PI / 2) * (r - 2) * 0.075, tZ = s.Z + Math.sin(s.a + Math.PI / 2) * (r - 2) * 0.075;
    let X = tX, Z = tZ, flying = false, spin = 0, h = 0;
    if (fs < 1) { const e = easeOut(fs); X = lerp(dealer.X, tX, e); Z = lerp(dealer.Z * 0.9, tZ, e); flying = true; spin = fs * 7 + k; h = Math.sin(Math.PI * fs); if (!arm) arm = [tX, tZ]; }
    const ms = (tt - 9.6 - ki * 0.06) / 0.5; let gone = false;
    if (ms > 0) { const e = ease(ms); X = lerp(tX, 0, e); Z = lerp(tZ, 0, e); if (ms > 1) gone = true; }
    if (!gone) cards.push({ X, Z, seat: k, idx: r, faceUp: k === 0 && fs >= 1 && ms <= 0, card: you[r], flying, spin, h });
  }
  /* betting: each seat acts in turn, its chip arcs into the pot; then the pot goes to the winner */
  const betAt = 4.2, betStep = 0.55, potAt = betAt + SEATS * betStep + 0.9;
  let actor = -1;
  for (let ki = 0; ki < SEATS; ki++) {
    const k = order[ki], s = seats[k], t0 = betAt + ki * betStep, fs = (tt - t0) / 0.5;
    const from = [s.X * 0.72, s.Z * 0.72], to = [(hash3(loop, k, 7) - 0.5) * 0.3, (hash3(loop, k, 8) - 0.5) * 0.22];
    if (tt >= t0 - 0.35 && tt < t0 + 0.5) actor = k;
    if (fs < 0) { chips.push({ X: from[0], Z: from[1], h: 0, seat: k }); continue; }
    let X = to[0], Z = to[1], h = 0;
    if (fs < 1) { const e = easeOut(fs); X = lerp(from[0], to[0], e); Z = lerp(from[1], to[1], e); h = Math.sin(Math.PI * fs); }
    const ws = (tt - potAt) / 0.7;
    if (ws > 0) { const w = seats[winner]; const e = ease(ws); X = lerp(to[0], w.X * 0.72 + (k - 2.5) * 0.04, e); Z = lerp(to[1], w.Z * 0.72, e); }
    if (tt > 9.6 + 1.2) continue;
    chips.push({ X, Z, h, seat: k });
  }
  /* where each head looks: the dealer while dealing, the actor while betting, the winner at showdown */
  let focus;
  if (tt < betAt - 0.3) focus = [dealer.X, dealer.Z];
  else if (tt < potAt) { const a = actor >= 0 ? seats[actor] : null; focus = a ? [a.X, a.Z] : [0, 0]; }
  else if (tt < 9.6) { const w = seats[winner]; focus = [w.X, w.Z]; }
  else focus = [seats[0].pX, seats[0].pZ];
  for (let k = 0; k < SEATS; k++) {
    const s = seats[k];
    const want = Math.atan2(focus[1] - s.pZ, focus[0] - s.pX);
    let cur = st.look[k]; if (cur === undefined) cur = want;
    let dlt = want - cur; while (dlt > Math.PI) dlt -= 2 * Math.PI; while (dlt < -Math.PI) dlt += 2 * Math.PI;
    cur += dlt * 0.08; st.look[k] = cur;
    /* a little idle sway */
    heads.push({ X: s.pX, Z: s.pZ, look: cur + 0.12 * Math.sin(t * 0.9 + k * 1.7), seat: k, talking: k === actor });
  }
  const stage = tt < betAt - 0.3 ? 'DEAL' : tt < potAt ? 'BETS' : tt < 9.6 ? 'SHOWDOWN' : 'MUCK';
  return { seats, cards, chips, heads, arm, dealer, winner, stage, loop, tt, potAt, actor, focus };
}
function barRoom(g, t, o) {
  const { W, H, ar } = g;
  const D = o.D;
  const shelves = [H * 0.17, H * 0.33], counter = H * 0.58, lampY = H * 0.075;
  const glowAt = (px, py) => {
    let glow = 0;
    for (const lx of o.lamps) { const dx = (px - lx) / (W * 0.07), dy = (py - lampY) / (H * 0.6); if (dy > 0) glow += Math.exp(-dx * dx / (0.3 + dy * 2.2)) * Math.exp(-dy * 1.8); }
    return Math.min(glow, 1);
  };
  g.fill(0, 0, W, counter + 2, D, (px, py) => {
    const cx = Math.floor(px), cy = Math.floor(py / ar);
    const glow = glowAt(px, py);
    if (Math.abs(py - counter) < ar * 0.6) return 0.4;
    for (let si = 0; si < shelves.length; si++) {
      const sy = shelves[si];
      if (Math.abs(py - sy) < ar * 0.55) return 0.45;
      const slot = 8, k = Math.floor(px / slot), u = px - k * slot - slot / 2;
      if (hash3(k, si, 11) < 0.3) continue;
      const bh = (13 + 11 * hash3(k, si, 12)) * (H / 150), bw = 1.4 + 1.1 * hash3(k, si, 13);
      const h = sy - py; if (h < 0 || h > bh) continue;
      const neck = h > bh * 0.6, hw = neck ? bw * 0.4 : bw;
      if (Math.abs(u) > hw) continue;
      if (h > bh * 0.93) { g.c = o.F; return 0.7; }
      if (Math.abs(u) > hw - 0.55) return 0.5;                    /* the outline */
      if (u < -hw * 0.25 && u > -hw * 0.55 && !neck) return 0.45 + 0.2 * glow; /* the highlight */
      return 0;
    }
    const bay = (BAYER[cy % 4][cx % 4] + 0.5) / 16;
    if (glow * 0.55 > bay) return 0.14 + 0.3 * glow;
    return 0;
  });
  { const x0 = W * 0.05, x1 = W * 0.95, y0 = H * 0.03, y1 = shelves[1] + 3 * ar;
    g.line(x0, y0, x1, y0, D, '='); g.line(x0, y0 + ar, x1, y0 + ar, D, '-');
    g.line(x0, y0, x0, y1, D, '|'); g.line(x1, y0, x1, y1, D, '|'); g.line(x0 + 1, y0, x0 + 1, y1, D, ':'); g.line(x1 - 1, y0, x1 - 1, y1, D, ':'); }
  for (const lx of o.lamps) {
    g.line(lx, 0, lx, lampY - 2 * ar, D, '|');
    g.fill(lx - 7, lampY - 2 * ar, lx + 7, lampY + 1.6 * ar, o.F, (px, py) => {
      const yy = (py - (lampY - 2 * ar)) / (3.6 * ar); if (yy < 0 || yy > 1) return 0;
      const hw = 1.5 + 5 * yy; if (Math.abs(px - lx) > hw) return 0;
      return yy > 0.7 ? 0.95 : 0.55;
    });
  }
}
/* smoke(), with an amplitude: the same plume at a fraction of the density */
function thinSmoke(g, sx, sy, t, col, wind, S, amp) {
  const top = Math.max(0, sy - 70 * S);
  g.fill(sx - 30 * S, top, sx + 45 * S, sy, col, (px, py) => {
    const h = (sy - py) / S;
    if (h < 0) return 0;
    const curl = 5 * S * Math.sin(h * 0.09 + t * 0.7) + 2.5 * S * Math.sin(h * 0.23 - t * 1.1) + 1.5 * S * Math.sin(h * 0.5 + t * 2.3);
    const xc = sx + wind * h * 0.22 * S + curl * Math.min(1, h / 10);
    const w = (1.2 + h * 0.5) * S;
    const gauss = Math.exp(-Math.pow((px - xc) / w, 2));
    const n = 0.6 * noise3(px * 0.07 / S, py * 0.045 / S + t * 0.9, t * 0.15) + 0.4 * noise3(px * 0.18 / S, py * 0.11 / S + t * 1.6, t * 0.3);
    const v = Math.pow(gauss, 1.4) * Math.pow(Math.max(0, n - 0.22) * 1.5, 1.3) * (1 - h / 62) * 1.1 * amp;
    return v > 0.1 * amp ? Math.min(v, 0.75) : 0;
  });
}

/* a giant: the boss's portrait model, parameterised so each seat is a different man.
   (hx, hy) is the head centre in scene units, S the scale (head half-width 21 S, so 42 S wide).
   o: {t, look (-1..1), glint, lean, seed, F, K, A, E, D,
       build, jaw, headW, hatH, glassW, hat, bald, glasses, reflect, brows, scar, beard (hash threshold; 1.1 = clean), cig, cigar} */
function giant(g, hx, hy, S, o) {
  const { W, H, ar } = g;
  const F = o.F, K = o.K, A = o.A, E = o.E, D = o.D;
  const build = o.build || 1, jawF = o.jaw || 1, headW = o.headW || 1, hatH = o.hatH || 1, glassW = o.glassW || 1;
  const beard = o.beard === undefined ? 1.1 : o.beard;
  const look = clamp(o.look || 0, -1, 1);
  const Lx = -0.12, Ly = -0.92;
  const lamFor = (nx, ny, nz) => clamp(nx * Lx + ny * Ly + nz * 0.62, 0, 1);
  /* three tones: ink dither in the shadow, the field itself as the mid-tone (the cell is cleared), bone on the lit planes */
  const hard = (v) => { if (v < 0.27) { g.c = K; return 0.3 + 0.45 * (1 - v / 0.27); } if (v < 0.48) return -2; g.c = F; return 0.2 + 0.8 * (v - 0.48) / 0.52; };
  const U = (x, y) => [hx + x * S, hy + (y - 44) * S];
  /* jacket: shoulders wider than the head by a lot, lit from above, pinstriped; lapels */
  {
    const [jx, jy] = U(0, 108); const rx = 72 * S * build, ry = 44 * S * Math.sqrt(build), y0 = U(0, 70)[1];
    g.fill(jx - rx, y0, jx + rx, Math.min(H, jy + ry), K, (px, py) => {
      const nx = (px - jx) / rx, ny = (py - jy) / ry; const e = nx * nx + ny * ny; if (e > 1) return 0;
      if (e > 0.95) { g.c = D; return 0.5; }
      const ridge = -ny - Math.abs(nx) * 0.25;
      if (ridge > 0.9) { g.c = F; return 0.15 + 0.3 * (ridge - 0.9) / 0.1; }
      return hash3(Math.floor(px), Math.floor(py / ar), 23) < 0.1 ? 0.35 : -2;
    });
    const [sx0, sy0] = U(0, 70); const vh = 14 * S, vw = 7.5 * S;
    g.fill(sx0 - vw, sy0, sx0 + vw, sy0 + vh, F, (px, py) => { const yy = (py - sy0) / vh; if (yy < 0 || yy > 1) return 0; return Math.abs(px - sx0) < vw * (1 - yy) ? 0.62 - 0.3 * yy : 0; });
    for (const s of [-1, 1]) { const a = U(s * 8, 70), b = U(s * 28 * build, 104); g.line(a[0], a[1], b[0], b[1], F); const a2 = U(s * 8, 70), b2 = U(s * 16, 75); g.line(a2[0], a2[1], b2[0], b2[1], F); }
  }
  /* neck */
  { const [nx0, ny0] = U(0, 71); const hw = 14 * S * Math.sqrt(build), hh = 8 * S;
    g.fill(nx0 - hw, ny0 - hh, nx0 + hw, ny0 + hh, F, (px, py) => { const u = (px - nx0) / hw; if (Math.abs(u) > 1) return 0; return hard(0.15 + 0.6 * lamFor(u, 0.3, Math.sqrt(1 - u * u))); }); }
  /* head: an ellipse fused with a broad jaw */
  const erx = 21 * S * headW, ery = 23.5 * S, jw = 19 * S * jawF, jh = 10.5 * S, jr = 7.5 * S, jcy = hy + 14.5 * S;
  const sdHead = (px, py) => {
    const k = Math.sqrt(inEllipse(px - hx, py - hy, erx, ery)); const dE = (k - 1) * Math.min(erx, ery);
    return Math.min(dE, sdBox(px - hx, py - jcy, jw, jh, jr));
  };
  const brimY = U(0, 23.5)[1];
  g.fill(hx - erx - 2, hy - ery - 2, hx + erx + 2, jcy + jh + 2, F, (px, py) => {
    const d = sdHead(px, py); if (d > 0.3) return 0;
    if (d > -1.0) { g.c = F; return 0.42; }
    const eps = 0.9;
    const gx = (sdHead(px + eps, py) - sdHead(px - eps, py)) / (2 * eps), gy = (sdHead(px, py + eps) - sdHead(px, py - eps)) / (2 * eps);
    const e = clamp(-d / (13 * S), 0, 1), nz = Math.sqrt(1 - (1 - e) * (1 - e));
    let v = 0.12 + 0.8 * lamFor(gx * (1 - e), gy * (1 - e), nz);
    const ux = (px - hx) / S, uy = (py - hy) / S;
    v -= 0.18 * Math.exp(-Math.pow((Math.abs(ux) - 12 * headW) / 4.5, 2) - Math.pow((uy - 7) / 4.5, 2));
    v += 0.1 * Math.exp(-Math.pow((uy + 8.5) / 1.6, 2)) * (Math.abs(ux) < 13 ? 1 : 0);
    v -= 0.08 * Math.exp(-Math.pow(ux / 1.2, 2) - Math.pow((uy - 21) / 2, 2));
    if (o.hat && py > brimY && py < brimY + 14 * S) v *= 0.5;
    if (!o.hat && o.bald && uy < -10) v = clamp(v * 1.12, 0.3, 0.9);
    v = clamp(v, 0.05, 1);
    if (beard < 1 && uy > 9 && Math.abs(ux) < 16 * jawF && v >= 0.3 && hash3(Math.floor(px), Math.floor(py / ar), 7) > beard) { g.c = K; return beard < 0.7 ? 0.7 : 0.5; }
    return hard(v);
  });
  /* ears */
  for (const s of [-1, 1]) { const [ex, ey] = U(s * 22.2 * headW, 45); g.fill(ex - 3 * S, ey - 4.5 * S, ex + 3 * S, ey + 4.5 * S, F, (px, py) => { const e = inEllipse(px - ex, py - ey, 2.7 * S, 4.3 * S); if (e > 1) return 0; return 0.28 + 0.25 * (1 - e); }); }
  /* nose: a wedge lit from the left */
  { const [nx0, ny0] = U(0, 40.5); const top = ny0, bot = ny0 + 13 * S;
    g.fill(nx0 - 4 * S, top, nx0 + 4 * S, bot + S, F, (px, py) => {
      const yy = (py - top) / (bot - top); if (yy < 0 || yy > 1) return 0;
      const hw = 0.6 * S + 2.6 * S * yy; const u = (px - nx0) / hw; if (Math.abs(u) > 1) return 0;
      if (yy > 0.85 && Math.abs(Math.abs(u) - 0.55) < 0.25) { g.c = K; return 0.7; }
      if (u < 0) return 0.5 + 0.4 * (1 - yy);
      g.c = K; return 0.45;
    }); }
  /* mouth: flat, a little down at the corners */
  { const a = U(-8, 58.6), b = U(6, 58.3); g.line(a[0], a[1], b[0], b[1], K, '-'); const c = U(-9.1, 57.8); g.set(...g.cellOf(c[0], c[1]), CIDX['`'], K); }
  if (o.scar) { const a = U(16, 46), b = U(11.5, 57); g.line(a[0], a[1], b[0], b[1], K); g.line(a[0] + 1, a[1], b[0] + 1, b[1], K); for (let k = 0; k < 3; k++) { const p = U(16 - 1.4 * k - 0.7, 48.5 + 3.4 * k); g.line(p[0] - 1.2 * S, p[1], p[0] + 1.2 * S, p[1], K, '-'); } }
  /* a cigarette or a cigar, with its ember and smoke */
  if (o.cig) {
    const [cga, cgb] = o.cigar ? [U(5, 58.8), U(21, 53.2)] : [U(5, 58.6), U(18, 54.5)];
    const r = (o.cigar ? 1.75 : 0.7) * S;
    g.fill(cga[0] - r, cgb[1] - r, cgb[0] + r, cga[1] + r, o.cigar ? A : F, (px, py) => {
      const d = sdSeg(px, py, cga[0], cga[1], cgb[0], cgb[1]) - r; if (d > 0.2) return 0;
      const n = 1 + d / r; const along = (px - cga[0]) / (cgb[0] - cga[0]);
      if (o.cigar && along > 0.86) { g.c = D; return 0.35; }
      if (o.cigar && along > 0.6 && along < 0.68) return 0.9;
      return o.cigar ? 0.3 + 0.55 * (1 - n * n) : 0.8;
    });
    const ember = 0.75 + 0.25 * noise3(o.t * 6 + (o.seed || 0), 1, 1);
    const [ex, ey] = [cgb[0] + 0.8 * S, cgb[1] - 0.3 * S];
    const er = o.cigar ? 1.4 : 0.9;
    g.fill(ex - 1.6 * S, ey - 1.6 * S, ex + 1.6 * S, ey + 1.6 * S, E, (px, py) => inEllipse(px - ex, py - ey, er * S, er * 0.9 * S) <= 1 ? ember : 0);
  }
  /* hat: crown, band, brim; or a bare head */
  if (o.hat) {
    const top = U(0, 23.5 - 22.5 * hatH)[1], bot = U(0, 22.5)[1];
    g.fill(hx - 26 * S, Math.max(0, top), hx + 26 * S, bot, K, (px, py) => {
      const yy = (py - top) / (bot - top); if (yy < 0 || yy > 1) return 0;
      const hw = (19.5 + 4.6 * yy) * S * headW; const u = (px - hx) / hw; if (Math.abs(u) > 1) return 0;
      if (yy > 0.8) { g.c = A; return 0.3 + 0.35 * Math.sqrt(1 - u * u); }
      if (Math.abs(u) > 0.93 || yy < 0.06) { g.c = F; return 0.55; }
      const dent = Math.exp(-u * u * 4) * Math.exp(-Math.pow((yy - 0.12) / 0.16, 2));
      return hash3(Math.floor(px), Math.floor(py / ar), 21) < 0.45 + 0.4 * dent ? 0.4 : -2;
    });
    const brx = 34 * S * headW, bry = 4.6 * S;
    g.fill(hx - brx, brimY - bry, hx + brx, brimY + bry, K, (px, py) => {
      const e = inEllipse(px - hx, py - brimY, brx, bry); if (e > 1) return 0;
      const under = py > brimY; const edge = e > 0.8;
      if (under) return edge ? 0.5 : -2;
      return edge ? 0.95 : 0.85;
    });
  }
  /* eyes: dark glasses with a glint, or a bare stare from under the brows */
  const gy = U(0, 40)[1];
  if (o.glasses) {
    const gl = (o.glint || 0) * S;
    for (const s of [-1, 1]) {
      const ex = hx + s * 10.6 * S * headW; const hw = 9.6 * S * glassW, hh = 5.4 * S;
      g.fill(ex - hw - 1, gy - hh - 1, ex + hw + 1, gy + hh + 1, K, (px, py) => {
        const u = px - ex, v = py - gy;
        const d = sdBox(u, v + 0.4 * S * (Math.abs(u) / hw), hw, hh, 2.6 * S); if (d > 0.25) return 0;
        if (d > -0.7) { g.c = F; return 0.85; }
        const band = (u - v * 0.75) - gl;
        if (Math.abs(band) < 1.3 * S) { g.c = F; return 0.95; }
        if (Math.abs(band - 3.6 * S) < 0.7 * S) { g.c = F; return 0.45; }
        if (o.reflect) {
          for (let k = 0; k < 5; k++) { const cu = (k - 2) * 3.1 * S; if (Math.abs(u - cu) < 1.2 * S && v > -0.6 * S && v < 3.4 * S) { g.c = F; return 0.4; } }
          if (v > 3.2 * S && v < 3.6 * S) { g.c = F; return 0.2; }
        }
        if (v < -hh * 0.35) { g.c = F; return 0.18 + 0.12 * (-v / hh); }
        return 0.95;
      });
      const a = [ex + s * hw, gy - 0.5 * S], b = U(s * 21.8 * headW, 43); g.line(a[0], a[1], b[0], b[1], F);
    }
    const a = U(-1.2, 39.6), b = U(1.2, 39.6); g.line(a[0], a[1], b[0], b[1], F, '-');
  } else {
    for (const s of [-1, 1]) {
      const ex = hx + s * 10.2 * S * headW;
      const rx = 6.2 * S, ry = 2.6 * S;
      g.fill(ex - rx - 1, gy - ry - 1, ex + rx + 1, gy + ry + 1, F, (px, py) => {
        const e = inEllipse(px - ex, py - gy, rx, ry); if (e > 1) return 0;
        const ix = ex + look * 2.6 * S, ir = 2.3 * S;
        if (inEllipse(px - ix, py - gy, ir, ir * 1.05) <= 1) { g.c = K; return inEllipse(px - ix, py - gy, ir * 0.45, ir * 0.5) <= 1 ? 0.98 : 0.8; }
        return 0.55;                                            /* the white of the eye */
      });
      const a = [ex - rx * 0.9, gy - ry * 0.8], b = [ex + rx * 0.9, gy - ry * 1.05]; g.line(a[0], a[1], b[0], b[1], K, '='); /* the upper lid, heavy */
    }
  }
  if (o.brows || !o.glasses) for (const s of [-1, 1]) { const a = U(s * 3.5, 33.5), b = U(s * 16 * headW, 31.5 - (o.brows ? 0 : 0.8)); g.line(a[0], a[1], b[0], b[1], K, '='); g.line(a[0], a[1] + ar, b[0], b[1] + ar, K, '-'); }
}
/* the plume from a giant's cigarette or cigar, drawn before any of the men so it never crosses a face */
function giantSmoke(g, hx, hy, S, o) {
  if (!o.cig) return;
  const U = (x, y) => [hx + x * S, hy + (y - 44) * S];
  const cgb = o.cigar ? U(21, 53.2) : U(18, 54.5);
  const [ex, ey] = [cgb[0] + 0.8 * S, cgb[1] - 0.3 * S];
  thinSmoke(g, ex + 0.5 * S, ey - 1.5 * S, o.t + (o.seed || 0), o.smokeCol, o.wind, S * (o.cigar ? 0.8 : 0.6), 0.5);
}
/* the hand and the cards, drawn after the table since they rest above it. Same (hx, hy, S) as the giant. */
function giantHand(g, hx, hy, S, o) {
  const F = o.F, K = o.K, D = o.D, side = o.lefty ? -1 : 1;
  const U = (x, y) => [hx + x * S, hy + (y - 44) * S];
  const hard = (v) => { if (v < 0.27) { g.c = K; return 0.35 + 0.55 * (1 - v / 0.27); } if (v < 0.48) return -2; g.c = F; return 0.2 + 0.8 * (v - 0.48) / 0.52; };
  const ccols = { edge: F, fill: D, black: F, red: F, back: D };
  const [cxh, cyh] = U(side * 24 * (o.build || 1), 82);
  const cw = 8.5 * S * (o.cardScale || 1), chh = 12 * S * (o.cardScale || 1);
  for (let i = 0; i < 3; i++) drawCard(g, cxh + (i - 1) * 2.2 * S * (o.cardScale || 1), cyh, cw, chh, (i - 1) * 0.2 * side, 1, null, ccols);
  /* the hand over the lower half of the cards, knuckles as dark seams */
  const hxh = cxh + side * 1.2 * S, hyh = cyh + 4.5 * S * (o.cardScale || 1);
  const rx = 8 * S * (o.cardScale || 1), ry = 5 * S * (o.cardScale || 1);
  g.fill(hxh - rx, hyh - ry, hxh + rx, hyh + ry, F, (px, py) => { const e = inEllipse(px - hxh, py - hyh, rx, ry); if (e > 1) return 0; if (e > 0.86) { g.c = F; return 0.42; } const v = 0.2 + 0.6 * (1 - e) - 0.25 * ((py - hyh) / ry); if (Math.abs(((px - hxh) / (2.4 * S)) % 1 - 0.5) < 0.1 && py < hyh) { g.c = K; return 0.5; } return hard(v); });
}
/* a small round table between you and them, the pot on it */
function smallTable(g, cx, cy, rx, ry, o) {
  const { ar } = g;
  g.fill(cx - rx - 1, cy - ry - 1, cx + rx + 1, cy + ry + 1, o.dim, (px, py) => {
    const e = inEllipse(px - cx, py - cy, rx, ry); if (e > 1) return 0;
    if (e > 0.9) { g.c = o.rail; return 0.9; }
    if (e > 0.8) { g.c = o.K; return 0.55; }
    const on = Math.floor(px) % 3 === 0 && Math.floor(py / ar) % 2 === 0;
    const lamp = Math.exp(-((px - cx) * (px - cx) / (rx * rx) + (py - cy) * (py - cy) / (ry * ry)) * 1.4);
    return on ? 0.1 + 0.16 * lamp : -2;
  });
}

/* the company. o: {seats: [{X (fraction of W from centre), eyeY (fraction of H), Sdiv (S = H / Sdiv), kit}],
   lamps (fractions of W), tableY, tableR, cardScale, label} */
function giants(g, t, st, o) {
  const { W, H, ar } = g;
  const T = tableTimeline(t, st);
  const P = 11, loop = Math.floor(t / P), tt = t - loop * P, cx = W / 2;
  const deck = dealCards(7, loop + 13), hand = deck.slice(0, 5), fresh = deck.slice(5, 7), swap = [2, 4];
  const cols = { edge: 1, fill: 4, black: 1, red: 2, back: 3 };
  /* the room breathes: a slow creep toward you and back */
  const zoom = 1 + 0.03 * Math.sin(t * 0.21);
  barRoom(g, t, { D: 4, F: 3, lamps: o.lamps.map((f) => cx + f * W) });
  /* seats, far to near (small to big) so the near ones overwrite */
  const seats = o.seats.map((sd, i) => ({ ...sd, seat: i + 1 })).sort((a, b) => b.Sdiv - a.Sdiv);
  if (!st.lean) st.lean = {};
  const you = [0, -1.6];
  const drawn = [];
  for (const p of seats) {
    const S = (H / p.Sdiv) * zoom, hx = cx + p.X * W * zoom, hy = H * 0.5 + (p.eyeY - 0.5) * H * zoom;
    const side = p.X < -0.05 ? -1 : p.X > 0.05 ? 1 : Math.sin(t * 0.3 + p.seat);
    giantSmoke(g, hx, hy, S, { ...p.kit, t, seed: p.seat * 2.7, smokeCol: 4, wind: side * (1.6 + 1.0 * Math.sin(t * 0.3 + p.seat)) });
  }
  for (const p of seats) {
    const talking = T.stage === 'BETS' && T.actor === p.seat;
    const cur = st.lean[p.seat] || 0; st.lean[p.seat] = cur + ((talking ? 1 : 0) - cur) * 0.06;
    const lean = st.lean[p.seat];
    const S = (H / p.Sdiv) * zoom * (1 + 0.08 * lean);
    const hx = cx + p.X * W * zoom, hy = H * 0.5 + (p.eyeY - 0.5) * H * zoom + lean * 3 * S;
    let look = 0;
    if (T.stage === 'BETS' && T.actor >= 0 && T.actor !== p.seat) {
      const a = T.actor === 0 ? you : (o.seats[T.actor - 1] ? [o.seats[T.actor - 1].X * 3.2, 1.0] : you);
      const dxA = a[0] - p.X * 3.2, dzA = a[1] - 1.0, dxY = you[0] - p.X * 3.2, dzY = you[1] - 1.0;
      look = clamp((dxA / Math.hypot(dxA, dzA) - dxY / Math.hypot(dxY, dzY)) * 1.2, -1, 1);
    }
    giant(g, hx, hy, S, { t, look, ...p.kit, F: 1, K: 5, A: 6, E: 2, D: 3, smokeCol: 4, seed: p.seat * 2.7, glint: Math.sin(t * 0.5 + p.seat) * 8 });
    drawn.push({ hx, hy, S, kit: p.kit });
  }
  /* the table: small. Its far rim sits under their hands */
  const tx = cx, ty = H * o.tableY, trx = W * o.tableR, try_ = H * o.tableR * 0.62;
  smallTable(g, tx, ty, trx, try_, { rail: 1, dim: 4, K: 5 });
  for (const d of drawn) if (d.kit.cards) giantHand(g, d.hx, d.hy, d.S, { ...d.kit, F: 1, K: 5, D: 3, cardScale: o.cardScale || 0.8 });
  /* chips: bets arc into the pool */
  for (const ch of T.chips) { const x = tx + ch.X * trx * 0.9, y = ty - ch.Z * try_ * 0.9; const c = g.cellOf(x, y - ch.h * 6); g.text(c[0] - 1, c[1], ch.h > 0.4 ? '(@)' : ' @ ', 6); }
  /* your fan, and the draw */
  const wantS = g.mouse.in ? 0.06 + 0.22 * clamp(g.mouse.x / W, 0, 1) : 0.13 + 0.05 * Math.sin(t * 0.6);
  if (st.spread === undefined) st.spread = 0.14;
  st.spread += (wantS - st.spread) * 0.07;
  const w = Math.min(W * 0.07, H * 0.17), h = w * 1.42, Rr = H * 0.32, py = H * 1.16, px = cx;
  const draws = [];
  for (let i = 0; i < 5; i++) {
    const th = (i - 2) * st.spread; const base = [px + Math.sin(th) * Rr, py - Math.cos(th) * Rr];
    const k = swap.indexOf(i); let pos = base, ang = th, face = hand[i], sx = 1, scale = 1;
    if (k >= 0) {
      const lift = ease((tt - 2.2 - k * 0.25) / 0.6), fly = ease((tt - 3.3 - k * 0.2) / 0.9), come = ease((tt - 5.0 - k * 0.4) / 1.0), flip = (tt - 6.6 - k * 0.3) / 0.5;
      if (fly >= 1 && come <= 0) continue;
      if (come > 0) { const farP = [tx + (k - 0.5) * trx * 0.5, ty - try_ * 0.3]; pos = flight(farP, base, come, 0); scale = lerp(0.3, 1, come); ang = lerp(0, th, come); face = null; if (flip >= 0 && flip < 1) { sx = Math.abs(Math.cos(flip * Math.PI)); if (flip > 0.5) face = fresh[k]; } else if (flip >= 1) face = fresh[k]; }
      else if (fly > 0) { const farP = [tx + (k - 0.5) * trx * 0.6, ty + try_ * 0.1]; pos = flight([base[0], base[1] - 8], farP, fly, -H * 0.05); scale = lerp(1, 0.3, fly); ang = th * (1 - fly); face = fly > 0.3 ? null : face; }
      else if (lift > 0) { pos = [base[0] + Math.sin(th) * 8 * lift, base[1] - Math.cos(th) * 8 * lift]; }
    }
    draws.push({ pos, ang, face, sx, scale, i });
  }
  draws.sort((a, b) => a.scale - b.scale || a.i - b.i);
  for (const d of draws) drawCard(g, d.pos[0], d.pos[1], w * d.scale, h * d.scale, d.ang, Math.max(d.sx, 0.05), d.face, cols);
  const shadeCap = (a, b, r, col) => g.fill(Math.min(a[0], b[0]) - r - 1, Math.min(a[1], b[1]) - r - 1, Math.max(a[0], b[0]) + r + 1, H, col, (x, y) => { const d = sdSeg(x, y, a[0], a[1], b[0], b[1]) - r; if (d > 0.2) return 0; const n = 1 + d / r; if (d > -0.9) return 0.8; return 0.22 + 0.55 * (1 - n * n) * (0.7 + 0.3 * ((x - a[0]) < 0 ? 1 : 0.6)); });
  shadeCap([cx - w * 0.05, H * 1.02], [cx + w * 0.2, H * 0.88], w * 0.2, 3);
  shadeCap([cx - w * 0.75, H * 1.05], [cx - w * 0.62, H * 0.95], w * 0.1, 3);
  shadeCap([cx - w * 0.5, H * 1.08], [cx - w * 0.42, H * 0.97], w * 0.1, 3);
  const stage = tt < 2.2 ? 'YOUR FIVE' : tt < 3.3 ? 'PICK TWO' : tt < 5.0 ? 'DISCARD' : tt < 7.3 ? 'DRAW TWO' : 'NEW HAND';
  g.text(2, 1, 'INTERNAL  ' + stage, 3);
  const s2 = (o.label || '') + T.stage; g.text(g.cols - s2.length - 2, 1, s2, 3);
}
/* the five men. Seat order is left to right; the two on the ends sit nearest. */
const COMPANY = [
  { X: -0.42, eyeY: 0.37, Sdiv: 100, kit: { build: 1.15, jaw: 1.12, headW: 1.05, hat: true, hatH: 1.0, glasses: true, glassW: 1.05, cig: true, cigar: true, beard: 0.62, cards: true } },
  { X: -0.21, eyeY: 0.43, Sdiv: 125, kit: { build: 1.0, jaw: 0.95, headW: 0.95, hat: false, bald: true, glasses: true, glassW: 0.9, reflect: true, cig: false, beard: 1.1, scar: true, cards: true, lefty: true } },
  { X: 0.0, eyeY: 0.46, Sdiv: 140, kit: { build: 0.95, jaw: 1.0, headW: 1.0, hat: false, glasses: false, brows: true, cig: true, beard: 0.8, cards: true } },
  { X: 0.21, eyeY: 0.43, Sdiv: 125, kit: { build: 1.1, jaw: 1.2, headW: 1.08, hat: true, hatH: 0.75, glasses: false, brows: true, cig: true, cigar: true, beard: 0.55, cards: true, lefty: true } },
  { X: 0.42, eyeY: 0.37, Sdiv: 100, kit: { build: 1.25, jaw: 1.05, headW: 0.98, hat: true, hatH: 1.1, glasses: true, glassW: 1.15, cig: false, beard: 0.92, scar: true, cards: true } },
];
/* keyframes: each key says "at time t start moving to (X, Z, ang), taking dur, with arc". Between keys the thing sits. */
function kfAt(keys, tt) {
  let prev = null, cur = null;
  for (const k of keys) { if (k.t <= tt) { prev = cur; cur = k; } else break; }
  if (!cur) return null;
  if (cur.gone && tt >= cur.t + (cur.dur || 0)) return null;
  const s = cur.dur ? clamp((tt - cur.t) / cur.dur, 0, 1) : 1;
  if (s >= 1 || !prev) return { X: cur.X, Z: cur.Z, ang: cur.ang || 0, h: 0, moving: false, face: cur.face, s: 1, key: cur };
  const e = cur.flick ? easeOut(s) : ease(s);
  const spin = cur.spin ? (cur.spin * (1 - e)) : 0;
  return { X: lerp(prev.X, cur.X, e), Z: lerp(prev.Z, cur.Z, e), ang: lerp(prev.ang || 0, cur.ang || 0, e) + spin, h: Math.sin(Math.PI * s) * (cur.arc || 0), moving: true, face: cur.flipTo && s > 0.5 ? cur.flipTo : (cur.flip ? prev.face : cur.face), flip: cur.flip ? s : -1, s, key: cur };
}
const HAND_P = 17.0;
function gameTimeline(t, st) {
  const loop = Math.floor(t / HAND_P), tt = t - loop * HAND_P;
  const R = (id, n) => hash3(loop, id, n), RS = (id, n) => R(id, n) * 2 - 1;
  const N = 6;
  const seats = [];
  for (let k = 0; k < N; k++) { const a = -Math.PI / 2 + (k + 0.5) * (2 * Math.PI / N); seats.push({ a, X: Math.cos(a) * 0.72, Z: Math.sin(a) * 0.7, pX: Math.cos(a) * 1.3, pZ: Math.sin(a) * 1.3 }); }
  const dealer = { X: 0, Z: 1.2 }, deck = { X: 0.13, Z: 0.98 }, muck = { X: -0.3, Z: 0.95 }, pot = { X: 0, Z: -0.16 };
  /* the fan in front of a seat: n cards along a short arc, facing the seat */
  const fanPos = (k, i, n) => { const s = seats[k]; const tx = Math.cos(s.a + Math.PI / 2), tz = Math.sin(s.a + Math.PI / 2); const off = (i - (n - 1) / 2) * 0.085; return { X: s.X + tx * off, Z: s.Z + tz * off, ang: -s.a + Math.PI / 2 + (i - (n - 1) / 2) * 0.12 }; };
  /* --- the schedule --- */
  const T = {};
  T.deal0 = 0.5; const dealStep = 0.105, dealDur = 0.42;
  T.tidy = T.deal0 + 30 * dealStep + 0.3;
  T.bet1 = T.tidy + 0.9; const betStep = 0.24, betDur = 0.4;
  T.sweep1 = T.bet1 + N * betStep + 0.4;
  T.flop = T.sweep1 + 0.7;
  T.bet2 = T.flop + 1.5;
  T.sweep2 = T.bet2 + N * betStep + 0.4;
  T.draw = T.sweep2 + 0.6; const drawSeat = 0.62;
  T.bet3 = T.draw + N * drawSeat + 0.3;
  T.sweep3 = T.bet3 + N * betStep + 0.4;
  T.turn = T.sweep3 + 0.5;
  T.bet4 = T.turn + 1.0;
  T.sweep4 = T.bet4 + N * betStep + 0.4;
  T.river = T.sweep4 + 0.5;
  T.show = T.river + 1.0;
  T.muck = T.show + 1.4;
  const winner = Math.floor(R(3, 3) * N);
  const cards = [], chips = [];
  /* hole cards */
  const order = []; for (let k = 0; k < N; k++) order.push(k);
  const discards = seats.map((_, k) => Math.floor(R(10 + k, 1) * 3.4)); /* 0..3 */
  for (let k = 0; k < N; k++) {
    const nd = discards[k];
    const keep = [];
    for (let r = 0; r < 5; r++) {
      const id = k * 10 + r;
      const fan5 = fanPos(k, r, 5);
      const land = { X: fan5.X + RS(id, 2) * 0.07, Z: fan5.Z + RS(id, 3) * 0.07, ang: fan5.ang + RS(id, 4) * 0.9 };
      const keys = [{ t: -1, ...deck, ang: 0 }, { t: T.deal0 + (r * N + k) * dealStep, dur: dealDur, ...land, arc: 0.35, spin: 6 + R(id, 5) * 5, flick: true }, { t: T.tidy + k * 0.06, dur: 0.35, ...fan5 }];
      const isDiscard = r < nd;
      if (isDiscard) {
        const t0 = T.draw + k * drawSeat + r * 0.09;
        const scat = { X: muck.X + RS(id, 6) * 0.16, Z: muck.Z + RS(id, 7) * 0.12 - 0.06, ang: RS(id, 8) * 2.5 };
        keys.push({ t: t0, dur: 0.45, ...scat, arc: 0.25, spin: 5 + R(id, 9) * 6, flick: true });
        keys.push({ t: T.draw + k * drawSeat + 0.36 + r * 0.05, dur: 0.28, X: muck.X + RS(id, 6) * 0.02, Z: muck.Z + RS(id, 7) * 0.02, ang: RS(id, 8) * 0.3 });
      } else keep.push(r);
      keys.push({ t: T.muck + k * 0.05 + r * 0.03, dur: 0.5, X: muck.X, Z: muck.Z, ang: 0, gone: true });
      cards.push({ id, seat: k, keys, hole: true });
    }
    /* the replacements: dealt into the gaps, then the fan closes up */
    for (let j = 0; j < nd; j++) {
      const id = 100 + k * 10 + j;
      const fan5 = fanPos(k, j, 5);
      const land = { X: fan5.X + RS(id, 2) * 0.06, Z: fan5.Z + RS(id, 3) * 0.06, ang: fan5.ang + RS(id, 4) * 0.8 };
      const keys = [{ t: -1, ...deck, ang: 0 }, { t: T.draw + k * drawSeat + 0.5 + j * 0.11, dur: 0.4, ...land, arc: 0.3, spin: 6 + R(id, 5) * 5, flick: true }, { t: T.draw + (k + 1) * drawSeat + 0.05, dur: 0.3, ...fan5 }];
      keys.push({ t: T.muck + k * 0.05 + j * 0.03, dur: 0.5, X: muck.X, Z: muck.Z, ang: 0, gone: true });
      cards.push({ id, seat: k, keys, hole: true });
    }
    /* the winner's cards turn over at showdown */
    if (k === winner) for (const c of cards) if (c.seat === k) { const f = c.keys[c.keys.length - 2]; c.keys.splice(c.keys.length - 1, 0, { t: T.show + 0.1 + (c.id % 10) * 0.08, dur: 0.35, X: f.X, Z: f.Z - Math.sin(seats[k].a) * 0.06, ang: f.ang, flip: true, flipTo: dealCards(1, loop * 31 + c.id)[0] }); }
  }
  /* the board and the burns */
  const board = dealCards(5, loop + 11);
  const boardAt = [T.flop + 0.35, T.flop + 0.5, T.flop + 0.65, T.turn + 0.3, T.river + 0.3], burnAt = [T.flop, T.turn, T.river];
  for (let i = 0; i < 5; i++) {
    const slot = { X: (i - 2) * 0.16, Z: 0.16, ang: 0 };
    cards.push({ id: 200 + i, board: true, keys: [{ t: -1, ...deck, ang: 0 }, { t: boardAt[i], dur: 0.45, ...slot, arc: 0.12, flick: true, flip: true, flipTo: board[i] }, { t: T.muck + 0.4 + i * 0.05, dur: 0.45, X: muck.X, Z: muck.Z, ang: 0, gone: true }] });
  }
  for (let i = 0; i < 3; i++) cards.push({ id: 210 + i, keys: [{ t: -1, ...deck, ang: 0 }, { t: burnAt[i], dur: 0.3, X: muck.X + 0.02 * i, Z: muck.Z + 0.01 * i, ang: 0.2 * i, arc: 0.05 }, { t: T.muck, dur: 0.5, X: muck.X, Z: muck.Z, ang: 0, gone: true }] });
  /* the bets: a chip or two lands in front of the bettor, then the dealer sweeps the round into the pot */
  const rounds = [[T.bet1, T.sweep1], [T.bet2, T.sweep2], [T.bet3, T.sweep3], [T.bet4, T.sweep4]];
  let actor = -1, potN = 0;
  rounds.forEach(([b0, sw], ri) => {
    for (let k = 0; k < N; k++) {
      const s = seats[k], t0 = b0 + k * betStep;
      if (tt >= t0 - 0.2 && tt < t0 + betDur + 0.1) actor = k;
      const n = 1 + Math.floor(R(40 + ri * 10 + k, 1) * 2);
      if (R(40 + ri * 10 + k, 2) < 0.18 && ri > 0) continue; /* checks */
      for (let j = 0; j < n; j++) {
        const id = 300 + ri * 20 + k * 3 + j;
        const from = { X: s.X * 0.86, Z: s.Z * 0.86 }, land = { X: s.X * 0.52 + RS(id, 1) * 0.07, Z: s.Z * 0.5 + RS(id, 2) * 0.06 };
        const inPot = { X: pot.X + RS(id, 3) * 0.1, Z: pot.Z + RS(id, 4) * 0.07 };
        const keys = [{ t: -1, ...from }, { t: t0 + j * 0.12, dur: betDur, ...land, arc: 0.3, flick: true }, { t: sw + k * 0.05, dur: 0.4, ...inPot }];
        const w = seats[winner];
        keys.push({ t: T.show + 0.6 + (potN % 7) * 0.05, dur: 0.5, X: w.X * 0.8 + RS(id, 5) * 0.08, Z: w.Z * 0.8 + RS(id, 6) * 0.06 });
        keys.push({ t: T.muck + 0.2, dur: 0.01, X: w.X, Z: w.Z, gone: true });
        chips.push({ id, seat: k, keys, first: t0 + j * 0.12 });
        potN++;
      }
    }
  });
  /* the dealer's arm: toward whatever the dealer is throwing or dragging right now */
  let arm = null;
  const live = [], liveChips = [];
  for (const c of cards) { const p = kfAt(c.keys, tt); if (!p) continue; live.push({ ...c, ...p }); if (p.moving && (p.key.spin || p.key.flick) && c.keys[0].X === deck.X) arm = [p.X, p.Z]; if (p.moving && p.key.X === muck.X + RS(c.id, 6) * 0.02) arm = [p.X, p.Z]; }
  for (const ch of chips) { const p = kfAt(ch.keys, tt); if (!p) continue; if (tt < ch.first) continue; liveChips.push({ ...ch, ...p }); if (p.moving && Math.abs(p.key.X - (pot.X + RS(ch.id, 3) * 0.1)) < 1e-9) arm = [p.X, p.Z]; }
  /* where the heads look */
  const stage = tt < T.tidy ? 'DEAL' : tt < T.sweep1 + 0.3 ? 'BETS' : tt < T.flop + 1.2 ? 'FLOP' : tt < T.sweep2 + 0.3 ? 'BETS' : tt < T.bet3 ? 'DRAW' : tt < T.sweep3 + 0.3 ? 'BETS' : tt < T.turn + 0.9 ? 'TURN' : tt < T.sweep4 + 0.3 ? 'BETS' : tt < T.show ? 'RIVER' : tt < T.muck ? 'SHOWDOWN' : 'MUCK';
  let focus;
  if (stage === 'BETS' && actor >= 0) focus = [seats[actor].X, seats[actor].Z];
  else if (stage === 'DRAW') { const k = clamp(Math.floor((tt - T.draw) / drawSeat), 0, N - 1); focus = [seats[k].X, seats[k].Z]; }
  else if (stage === 'FLOP' || stage === 'TURN' || stage === 'RIVER') focus = [0, 0.16];
  else if (stage === 'SHOWDOWN') focus = [seats[winner].X, seats[winner].Z];
  else focus = [dealer.X, dealer.Z];
  const heads = [];
  for (let k = 0; k < N; k++) {
    const s = seats[k];
    const want = Math.atan2(focus[1] - s.pZ, focus[0] - s.pX);
    let cur = st.look[k]; if (cur === undefined) cur = want;
    let dlt = want - cur; while (dlt > Math.PI) dlt -= 2 * Math.PI; while (dlt < -Math.PI) dlt += 2 * Math.PI;
    cur += dlt * 0.08; st.look[k] = cur;
    heads.push({ X: s.pX, Z: s.pZ, look: cur + 0.1 * Math.sin(t * 0.9 + k * 1.7), seat: k, talking: stage === 'BETS' && k === actor });
  }
  const potShown = liveChips.filter((c) => !c.moving && Math.abs(c.X - pot.X) < 0.2 && Math.abs(c.Z - pot.Z) < 0.15).length;
  return { seats, cards: live, chips: liveChips, heads, arm, dealer, deck, muck, pot, winner, stage, loop, tt, discards, T, potShown };
}
/* a chip seen straight from above */
function drawChipTop(g, x, y, R, col, spotCol) {
  g.fill(x - R - 1, y - R - 1, x + R + 1, y + R + 1, col, (px, py) => {
    const dx = px - x, dy = py - y; const d = Math.hypot(dx, dy) / R; if (d > 1) return 0;
    if (d > 0.72) { const sector = Math.floor(((Math.atan2(dy, dx) / (2 * Math.PI)) + 1) * 8 + 0.5) % 2; if (sector) { g.c = spotCol; return 0.9; } return 0.75; }
    if (d > 0.55) return 0.3;
    return 0.5;
  });
}
/* the sidelines view: a smaller table, bigger people, real cards and chips */
function drawGameTop(g, T, o) {
  const { W, H, ar } = g;
  const cx = W / 2, cy = H * 0.5, rx = W * 0.25, ry = H * 0.29;
  const P = (X, Z) => [cx + X * rx, cy - Z * ry];
  const ccols = { edge: o.face, fill: o.dimmer, black: o.face, red: o.red, back: o.dim };
  const cw = o.cardW, chh = o.cardH;
  /* felt with a lamp pool, and the rail */
  g.fill(cx - rx - 3, cy - ry - 3, cx + rx + 3, cy + ry + 3, o.dim, (px, py) => {
    const e = inEllipse(px - cx, py - cy, rx, ry); const de = (Math.sqrt(e) - 1) * Math.min(rx, ry); if (de > -0.6) return 0;
    const lampv = Math.exp(-inEllipse(px - cx, py - cy, rx * 0.6, ry * 0.6) * 1.2) * (0.95 + 0.05 * noise3(T.tt * 2, 0, 0));
    const on = Math.floor(px) % 4 === 0 && Math.floor(py / ar) % 2 === 0;
    return on ? 0.05 + 0.3 * lampv : (lampv > 0.6 ? 0.04 : 0);
  });
  ellipseOutline(g, cx, cy, rx, ry, o.rail);
  ellipseOutline(g, cx, cy, rx + 2.6, ry + 2.4, o.dim);
  /* the deck and the muck: small stacks by the dealer */
  { const [dx, dy] = P(T.deck.X, T.deck.Z); for (let i = 2; i >= 0; i--) drawCard(g, dx + i * 0.5, dy + i * 0.5, cw, chh, 0, 1, null, ccols); }
  /* chips on the table (the flying ones come later) */
  for (const ch of T.chips) if (!ch.moving) { const [x, y] = P(ch.X, ch.Z); drawChipTop(g, x, y, o.chipR, o.sand, o.face); }
  /* cards on the table */
  const flying = [];
  for (const c of T.cards) {
    if (c.moving && (c.h > 0.001)) { flying.push(c); continue; }
    const [x, y] = P(c.X, c.Z);
    let sx = 1, face = c.face || null;
    if (c.flip >= 0 && c.flip < 1) { sx = Math.abs(Math.cos(c.flip * Math.PI)); }
    drawCard(g, x, y, cw, chh, c.ang, Math.max(sx, 0.06), face, ccols);
  }
  /* the dealer: head, shoulders, and an arm out to whatever is in the air */
  const [dx0, dy0] = P(T.dealer.X, T.dealer.Z);
  ellipseOutline(g, dx0, dy0 - 2, 15, 7.5, o.head, Math.PI, 2 * Math.PI);
  g.fill(dx0 - 6, dy0 - 5, dx0 + 6, dy0 + 5, o.head, (px, py) => { const d = Math.hypot(px - dx0, py - dy0); if (d > 4.6) return 0; return d > 3.7 ? 0.9 : 0.35; });
  { let hx = dx0 + 9, hy = dy0 + 3;
    if (T.arm) { const [ax, ay] = P(T.arm[0], T.arm[1]); const L = Math.hypot(ax - dx0, (ay - dy0)) || 1; const reach = Math.min(L, 15); hx = dx0 + ((ax - dx0) / L) * reach; hy = dy0 + ((ay - dy0) / L) * reach * 0.8; g.line(dx0 + ((ax - dx0) / L) * 4.5, dy0 + ((ay - dy0) / L) * 4, hx, hy, o.head); }
    const c = g.cellOf(hx, hy); g.text(c[0] - 1, c[1], '(=)', o.head); }
  g.text(Math.floor(dx0) - 3, Math.floor(dy0 / ar) - 5, 'DEALER', o.dim);
  /* the players: big heads from above, a hat brim on some, shoulders, a cigarette on some */
  for (const hd of T.heads) {
    const [hx, hy] = P(hd.X, hd.Z); const kit = o.kits[hd.seat];
    const a = Math.atan2(-(hd.Z), hd.X);
    ellipseOutline(g, hx, hy, 16 * kit.build, 8.5 * kit.build, o.head, a - Math.PI / 2 - 0.2, a + Math.PI / 2 + 0.2);
    const r = 5.4 * kit.size;
    g.fill(hx - r - 1, hy - r - 1, hx + r + 1, hy + r + 1, o.head, (px, py) => { const d = Math.hypot(px - hx, py - hy) / r; if (d > 1) return 0; if (kit.hat) { if (d > 0.9) return 0.9; const crease = Math.abs((px - hx) * Math.cos(hd.look) + (py - hy) * Math.sin(hd.look)) < 0.6; return crease ? 0.85 : (d < 0.6 ? 0.25 : 0.45); } if (d > 0.86) return 0.9; const hair = hash3(Math.floor(px), Math.floor(py / ar), 3) < (kit.bald ? 0.1 : 0.55); return hair ? (hd.talking ? 0.6 : 0.4) : 0.15; });
    if (kit.hat) ellipseOutline(g, hx, hy, r * 1.45, r * 1.45, o.head);
    const nx = hx + Math.cos(hd.look) * r * 1.05, ny = hy - Math.sin(hd.look) * r * 1.05;
    g.set(...g.cellOf(nx, ny), CIDX[kit.hat ? '.' : 'o'], o.head);
    if (kit.cig) { const ex = hx + Math.cos(hd.look + 0.35) * r * 1.6, ey = hy - Math.sin(hd.look + 0.35) * r * 1.6; g.line(nx, ny, ex, ey, o.head); g.set(...g.cellOf(ex, ey), CIDX['@'], o.red); thinSmoke(g, ex, ey - 1, T.tt * 1.0 + hd.seat * 3, o.dimmer, 0.6 + 0.6 * Math.sin(T.tt * 0.4 + hd.seat), 0.32, 0.55); }
  }
  /* the cards and chips in the air, bigger the higher they are */
  for (const c of flying) { const [x, y] = P(c.X, c.Z); const sc = 1 + 0.5 * c.h; drawCard(g, x, y - c.h * 9, cw * sc, chh * sc, c.ang, 1, null, ccols); }
  for (const ch of T.chips) if (ch.moving && ch.h > 0.001) { const [x, y] = P(ch.X, ch.Z); drawChipTop(g, x, y - ch.h * 9, o.chipR * (1 + 0.6 * ch.h), o.sand, o.face); }
  else if (ch.moving) { const [x, y] = P(ch.X, ch.Z); drawChipTop(g, x, y, o.chipR, o.sand, o.face); }
  /* labels */
  g.text(2, 1, 'INTERNAL  HAND ' + String(T.loop + 1).padStart(3, '0') + '  ' + T.stage, o.dim);
  if (T.stage === 'SHOWDOWN') { const w = T.seats[T.winner]; const [wx, wy] = P(w.pX, w.pZ); g.text(Math.floor(wx) - 2, Math.floor(wy / ar) + (w.Z > 0 ? -6 : 6), 'WINS', o.sand); }
  if (T.stage === 'DRAW') { const k = clamp(Math.floor((T.tt - T.T.draw) / 0.62), 0, 5); const s = T.seats[k]; const [sx, sy] = P(s.pX, s.pZ); const n = T.discards[k]; g.text(Math.floor(sx) - 4, Math.floor(sy / ar) + (s.Z > 0 ? -6 : 6), n === 0 ? 'STANDS PAT' : 'DRAWS ' + n, o.sand); }
  if (T.potShown > 0) { const [px, py] = P(T.pot.X, T.pot.Z); const s = 'POT ' + (T.potShown * 25); g.text(Math.floor(px - s.length / 2), Math.floor(py / ar) + 3, s, o.dimmer); }
}

/* ================================================================
   COLOURS — read from the page's tokens so the canvases stay in the
   duotone. The two windows keep the oxblood field in both schemes (they
   are rooms you look into); the logo sits on the field and follows it.
   ================================================================ */
const CSSV = getComputedStyle(document.documentElement);
const tok = (n, fb) => (CSSV.getPropertyValue(n).trim() || fb);
const hexA = (hex, a) => { let h = hex.replace('#', ''); if (h.length === 3) h = h.split('').map((c) => c + c).join(''); const n = parseInt(h, 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; };
const OX = tok('--oxblood', '#8e2038'), BONE = tok('--bone', '#f5f0e6'), INK = '#12070a';
const BONE_DIM = hexA(BONE, 0.5), BONE_34 = hexA(BONE, 0.34), SAND = hexA(BONE, 0.85), CORAL = BONE;
const logoPalette = () => { const mark = tok('--mark', BONE); return ['transparent', mark, INK, hexA(mark, 0.35)]; };

const SCENES = {};

/* the logo: the chip, tilted, turning slowly; faster under the mouse */
SCENES.logo = {
  palette: logoPalette(), fontPx: 4, still: 0,
  /* init runs again on resize and once the fonts land; keep the yaw so the chip does not jump */
  init() { if (this.yaw === undefined) this.yaw = 0.6; },
  frame(g, t, dt) {
    const { W, H } = g;
    const R = Math.min(W, H) * 0.46, cx = W / 2, cy = H / 2;
    this.yaw += dt * (g.mouse.in ? 1.1 : 0.3);
    const tilt = 0.8 + 0.12 * Math.sin(t * 0.37), wobble = 0.25 * Math.sin(t * 0.29);
    rayChip(g, { cx, cy, R, hz: R * 0.13, M: mul(rotX(tilt), mul(rotY(wobble), rotZ(this.yaw))), L: norm3([-0.55, -0.5, -0.67]), colBase: 1, colPattern: 2, colDim: 3, dither: true, sectors: 12 });
  },
};

/* a whole hand, from the sidelines */
SCENES.D1 = {
  palette: [OX, BONE, SAND, BONE_DIM, CORAL, BONE_34], fontPx: 7, still: 12.6,
  init() { this.st = { look: [] }; },
  frame(g, t) {
    const T = gameTimeline(t, this.st);
    drawGameTop(g, T, { rail: 1, dim: 3, dimmer: 5, sand: 2, red: 4, face: 1, head: 3, cardW: 10, cardH: 13.5, chipR: 3.2,
      kits: [{ size: 1.0, build: 1.0, hat: true, cig: false }, { size: 1.1, build: 1.15, hat: false, cig: true }, { size: 0.95, build: 0.95, hat: true, cig: true }, { size: 1.05, build: 1.1, hat: false, cig: false, bald: true }, { size: 1.0, build: 1.05, hat: true, cig: false }, { size: 1.12, build: 1.2, hat: false, cig: true }] });
  },
};

/* five giants at the bar */
SCENES.G5 = {
  palette: [OX, BONE, CORAL, BONE_DIM, BONE_34, INK, SAND], fontPx: 7, still: 5.8,
  init() { this.st = { look: [] }; },
  frame(g, t) {
    giants(g, t, this.st, { lamps: [-0.3, 0.02, 0.34], tableY: 0.86, tableR: 0.12, cardScale: 0.75, label: 'FIVE ON ONE  ', seats: COMPANY });
  },
};

/* ================================================================
   MOUNT — one grid per canvas[data-scene]; draws only while in view and the
   tab is visible; with reduced motion, one still frame that still answers
   the mouse.
   ================================================================ */
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;
const live = [];
for (const canvas of document.querySelectorAll('canvas[data-scene]')) {
  const scene = SCENES[canvas.dataset.scene]; if (!scene) continue;
  const g = new Grid(canvas, scene.palette, scene.fontPx);
  scene.init(g);
  live.push({ g, scene, canvas, t: REDUCED_MOTION ? scene.still : 0, visible: true, drawn: false });
}
const io = new IntersectionObserver((entries) => { for (const e of entries) { const l = live.find((x) => x.canvas === e.target); if (l) l.visible = e.isIntersecting; } }, { rootMargin: '60px' });
const ro = new ResizeObserver(() => live.forEach((l) => { l.g.resize(); l.scene.init(l.g); l.drawn = false; }));
live.forEach((l) => { io.observe(l.canvas); ro.observe(l.canvas); });
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => live.forEach((l) => { l.g.resize(); l.scene.init(l.g); l.drawn = false; }));
matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
  for (const l of live) if (l.scene === SCENES.logo) { l.g.palette = logoPalette(); l.g.buildAtlas(); l.drawn = false; }
});
let last = performance.now();
function tick(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (!document.hidden) for (const l of live) {
    if (!l.visible) continue;
    if (REDUCED_MOTION) { if (l.drawn && !l.g.dirty) continue; l.g.dirty = false; l.drawn = true; }
    else l.t += dt;
    l.g.clear(); l.scene.frame(l.g, l.t, dt); l.g.draw();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

"""

PAGE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drawmaha Solver — rung 1 complete: CFR solves Kuhn poker exactly</title>
<style>
  /* Self-hosted @fontsource files, copied to /fonts by the Vercel
     buildCommand. No third-party font CDN. */
  @font-face {
    font-family: "Instrument Serif"; font-style: normal; font-weight: 400;
    font-display: swap; src: url(/fonts/instrument-serif-latin-400-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "IBM Plex Mono"; font-style: normal; font-weight: 400;
    font-display: swap; src: url(/fonts/ibm-plex-mono-latin-400-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "IBM Plex Mono"; font-style: normal; font-weight: 500;
    font-display: swap; src: url(/fonts/ibm-plex-mono-latin-500-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "IBM Plex Mono"; font-style: normal; font-weight: 600;
    font-display: swap; src: url(/fonts/ibm-plex-mono-latin-600-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "Kalam"; font-style: normal; font-weight: 400;
    font-display: swap; src: url(/fonts/kalam-latin-400-normal.woff2) format("woff2");
  }

  :root {
    color-scheme: dark light;
    --oxblood: #8e2038;
    --bone: #f5f0e6;

    --field: var(--oxblood);
    --mark: var(--bone);
    --mark-dim: rgba(245, 240, 230, 0.72);
    --hair: rgba(245, 240, 230, 0.28);

    --panel: var(--bone);
    --panel-mark: var(--oxblood);
    --panel-dim: #b04d65;
    --panel-hair: rgba(142, 32, 56, 0.22);
    --panel-border: transparent;

    --serif: "Instrument Serif", Georgia, serif;
    --mono: "IBM Plex Mono", ui-monospace, monospace;
    --script: "Kalam", cursive;

    --rail-w: 232px;
    --gutter: clamp(20px, 3.4vw, 56px);
  }

  @media (prefers-color-scheme: light) {
    :root {
      /* Same two colours, swapped. Panels deepen a step so they still read as
         islands once the field is paper too. */
      --field: var(--bone);
      --mark: var(--oxblood);
      --mark-dim: rgba(142, 32, 56, 0.76);
      --hair: rgba(142, 32, 56, 0.22);
      --panel: #eae0cb;
      --panel-border: rgba(142, 32, 56, 0.26);
      /* The panel deepens a step in this scheme, so the ink on it has to come
         down with it or it loses the contrast it had on bone. */
      --panel-dim: #a1465c;
    }
  }

  * { box-sizing: border-box; }

  body {
    margin: 0; background: var(--field); color: var(--mark);
    font: 400 14px/1.6 var(--mono); -webkit-font-smoothing: antialiased;
  }
  a { color: inherit; }
  :focus-visible { outline: 2px solid var(--mark); outline-offset: 3px; }
  /* Inside a paper panel the ring flips with everything else: --mark is bone
     and so is the panel, so the field ring would draw bone-on-bone. */
  .panel :focus-visible { outline-color: var(--panel-mark); }

  /* ---------- the ladder, drawn as the nav ---------- */

  nav.line {
    position: sticky; top: 0; z-index: 20; background: var(--field);
    border-bottom: 1px solid var(--hair); padding: 20px 0 16px;
  }
  /* No fixed height: the track must grow with its labels, or the sub-lines
     hang below the bar's background and float transparent over whatever
     scrolls under them. */
  nav.line .track { position: relative; margin: 0 var(--gutter); }
  nav.line .bar {
    position: absolute; left: 10%; right: 10%; top: 11px; height: 1px;
    background: var(--hair);
  }
  /* Stations sit at 10/30/50/70/90% — the centres of five equal columns — so
     the climbed segment runs from the first station to the third. */
  nav.line .bar.done {
    left: 10%; right: 50%; top: 10.5px; height: 2px; background: var(--mark);
  }
  nav.line ol {
    position: relative; display: grid; grid-template-columns: repeat(5, 1fr);
    list-style: none; margin: 0; padding: 0;
  }
  nav.line li { text-align: center; min-width: 0; }
  nav.line .dot {
    display: block; width: 13px; height: 13px; margin: 5px auto 0;
    border-radius: 50%; border: 2px solid var(--mark); background: var(--field);
  }
  nav.line li.done .dot { background: var(--mark); }
  nav.line li.todo .dot { border-color: var(--mark-dim); }
  nav.line a, nav.line span.stop-name {
    display: block; margin-top: 10px; font: 500 13px/1 var(--mono);
    letter-spacing: 0.06em; text-transform: uppercase; text-decoration: none;
  }
  nav.line li.todo span.stop-name { color: var(--mark-dim); }
  nav.line a:hover { text-decoration: underline; text-underline-offset: 0.3em; }
  nav.line .sub {
    display: block; margin-top: 5px; font: 400 12px/1.3 var(--mono);
    color: var(--mark-dim); text-transform: none; letter-spacing: 0;
  }

  /* ---------- the identity rail ---------- */

  .shell { display: grid; grid-template-columns: var(--rail-w) minmax(0, 1fr); }
  aside.rail {
    position: sticky; top: 98px; align-self: start; height: calc(100vh - 98px);
    padding: 34px 26px 26px var(--gutter); border-right: 1px solid var(--hair);
    display: flex; flex-direction: column; gap: 20px;
  }
  aside.rail .mark { display: block; width: 92px; height: 92px; margin-left: -6px; }
  aside.rail .mark canvas { display: block; width: 100%; height: 100%; }
  aside.rail h1 { font: 400 40px/0.98 var(--serif); letter-spacing: -0.01em; margin: 0; }
  aside.rail .quote { font: 400 13px/1.55 var(--mono); color: var(--mark-dim); margin: 0; }
  aside.rail dl { margin: 0; display: flex; flex-direction: column; gap: 12px; }
  aside.rail dt { font: 600 13px/1.3 var(--mono); }
  aside.rail dd { margin: 2px 0 0; font: 400 12.5px/1.35 var(--mono); color: var(--mark-dim); }
  aside.rail .spacer { flex: 1; min-height: 20px; }
  aside.rail .sketch { width: 100%; height: auto; }

  .btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 8px;
    text-decoration: none; background: transparent; color: var(--mark);
    border: 1px solid var(--mark); border-radius: 3px; padding: 10px 14px;
    font: 500 11.5px/1 var(--mono); letter-spacing: 0.1em; text-transform: uppercase;
    cursor: pointer;
  }
  .btn:hover { background: var(--mark); color: var(--field); }

  /* ---------- type voices ---------- */

  main { padding: 34px var(--gutter) 96px 40px; }
  .stop {
    font: 500 11.5px/1 var(--mono); letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--mark-dim); margin: 0;
  }
  h2.big {
    font: 400 clamp(34px, 4.6vw, 58px)/1.03 var(--serif); letter-spacing: -0.015em;
    margin: 14px 0 0; max-width: 18ch;
  }
  .squiggle { display: block; width: 232px; height: 9px; margin: 7px 0 0; }
  .lede { font: 400 14.5px/1.75 var(--mono); max-width: 62ch; margin: 26px 0 0; }
  .links { margin: 20px 0 0; display: flex; flex-direction: column; gap: 9px; }
  .links a {
    font: 400 13.5px/1.4 var(--mono); text-decoration: underline;
    text-underline-offset: 0.32em; width: fit-content;
  }
  .links a:hover { color: var(--mark-dim); }
  section { margin-top: 74px; }

  /* ---------- paper panels ---------- */

  /* The two windows stack at full width: five men across a half-width
     window are too small to be anyone in particular. */
  .panels { display: grid; grid-template-columns: minmax(0, 1fr); gap: 20px; margin-top: 26px; }
  .panel {
    position: relative; background: var(--panel); color: var(--panel-mark);
    border: 1px solid var(--panel-border); border-radius: 13px;
    padding: 26px; min-width: 0; display: flex; flex-direction: column;
  }
  .panel .no {
    position: absolute; top: 15px; right: 15px; width: 28px; height: 28px;
    border-radius: 50%; border: 1.5px solid var(--panel-mark);
    font: 500 12px/25px var(--mono); text-align: center;
  }
  .panel .k {
    font: 400 9px/1 var(--mono); letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--panel-dim);
  }
  .panel h3 {
    font: 400 26px/1.12 var(--serif); letter-spacing: -0.01em; margin: 9px 0 0;
    max-width: 26ch;
  }
  .panel .say {
    font: 400 12.5px/1.65 var(--mono); color: var(--panel-dim); margin: 10px 0 0;
  }
  .panel svg { display: block; width: 100%; height: auto; }
  .fields { list-style: none; margin: 15px 0 0; padding: 0; }
  .fields li {
    display: flex; justify-content: space-between; gap: 14px; padding: 8px 0;
    border-top: 1px solid var(--panel-hair);
    font: 400 12px/1.35 var(--mono); font-variant-numeric: tabular-nums;
  }
  .fields li span:first-child { color: var(--panel-dim); }

  /* ---------- the two windows ---------- */

  /* Not paper panels: rooms you look into. They keep the oxblood field in
     both schemes, so in the light scheme they invert against the page the
     way the paper panels do in the dark one. */
  .hero {
    position: relative; margin: 0; aspect-ratio: 2 / 1; min-width: 0;
    background: var(--oxblood); color: var(--bone);
    border: 1px solid var(--hair); border-radius: 13px; overflow: hidden;
  }
  .hero canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  .hero.wide { aspect-ratio: 2.2 / 1; }
  .hero figcaption {
    position: absolute; left: 14px; bottom: 11px; z-index: 1; margin: 0;
    font: 500 9.5px/1 var(--mono); letter-spacing: 0.13em; text-transform: uppercase;
    color: rgba(245, 240, 230, 0.6); pointer-events: none;
  }

  @media (max-width: 1080px) {
    .shell { grid-template-columns: minmax(0, 1fr); }
    aside.rail {
      position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--hair);
    }
    aside.rail .spacer { display: none; }
    main { padding: 30px var(--gutter) 70px; }
    .panels, .records { grid-template-columns: minmax(0, 1fr); }
    nav.line .sub { display: none; }
  }
</style>
</head>
<body>

<nav class="line" aria-label="The validation ladder">
  <div class="track">
    <div class="bar"></div>
    <div class="bar done"></div>
    <ol>
      <li class="done"><span class="dot"></span><a href="/rung0">Rung 0<span class="sub">rock-paper-scissors</span></a></li>
      <li class="done"><span class="dot"></span><a href="/rung1">Rung 1<span class="sub">kuhn poker</span></a></li>
      <li class="done"><span class="dot"></span><a href="/rung2">Rung 2<span class="sub">leduc</span></a></li>
      <li class="todo"><span class="dot"></span><span class="stop-name">Rung 3<span class="sub">mini-drawmaha</span></span></li>
      <li class="todo"><span class="dot"></span><span class="stop-name">Rung 4<span class="sub">full drawmaha</span></span></li>
    </ol>
  </div>
</nav>

<div class="shell">

  <aside class="rail">
    <a class="mark" href="https://drawmaha.app" aria-label="Drawmaha Solver, home"><canvas data-scene="logo" aria-hidden="true"></canvas></a>
    <h1>Drawmaha<br>Solver.</h1>
    <p class="quote">&ldquo;Each rung is checked against a known answer before we climb.&rdquo;</p>
    <dl>
      <div><dt>now</dt><dd>Rung 2, complete</dd></div>
      <div><dt>next</dt><dd>Rung 3 &middot; mini-Drawmaha</dd></div>
      <div><dt>method</dt><dd>Deep CFR</dd></div>
    </dl>
    <span class="spacer"></span>
    <svg class="sketch" viewBox="0 0 150 62" aria-hidden="true">
      <g fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
        <ellipse cx="34" cy="46" rx="20" ry="7"/><path d="M14 46v-7M54 46v-7"/>
        <ellipse cx="34" cy="39" rx="20" ry="7"/><path d="M14 39v-7M54 39v-7"/>
        <ellipse cx="34" cy="32" rx="20" ry="7"/>
        <path d="M22 31.4a13 6 0 0 0 24 0" stroke-opacity=".55"/>
        <ellipse cx="96" cy="50" rx="15" ry="5.4"/><path d="M81 50v-5.5M111 50v-5.5"/>
        <ellipse cx="96" cy="44.5" rx="15" ry="5.4"/>
        <rect x="112" y="16" width="24" height="33" rx="3" transform="rotate(11 124 32)"/>
        <rect x="104" y="14" width="24" height="33" rx="3" transform="rotate(-4 116 30)"/>
        <path d="M113 27.5l3.6-4 3.6 4-3.6 4.2z"/>
        <path d="M6 57h138" stroke-opacity=".4"/>
      </g>
    </svg>
    <a class="btn" href="/rung2#play">Play the solver &rarr;</a>
  </aside>

  <main>

    <div class="panels">
      <figure class="hero">
        <canvas data-scene="D1" role="img" aria-label="A whole hand of Drawmaha seen from above: the deal, the bets, the flop, the draw, the turn, the river, the showdown"></canvas>
        <figcaption>a whole hand, from the sidelines</figcaption>
      </figure>
      <figure class="hero wide">
        <canvas data-scene="G5" role="img" aria-label="Five men at a bar, hats and dark glasses, watching you across a small table; your five cards fan at the bottom"></canvas>
        <figcaption>five on one &middot; move the mouse</figcaption>
      </figure>
    </div>

    <section>
      <p class="stop">Stop 00 &middot; the ladder</p>
      <h2 class="big">Five games between here and a Drawmaha solver.</h2>
      <svg class="squiggle" viewBox="0 0 232 9" aria-hidden="true">
        <path d="M2 6.2c14-5 28 3.4 42-.6s28-4.6 42 .4 28 4 42-.8 28-4 42 1 28 3.4 60-1"
              fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
      </svg>

      <p class="lede">Drawmaha is a split-pot draw/Omaha hybrid with no existing
      solver. So it gets built rung by rung: every rung is a smaller game whose
      answer is already known, and nothing moves up until the solver
      reproduces it.</p>

      <div class="links">
        <a href="/rung0">watch regret matching find Nash &rarr;</a>
        <a href="/rung1">watch CFR discover the bluff &rarr;</a>
        <a href="/rung2">walk the betting line, range against range &rarr;</a>
      </div>

      <table class="rows">
        <tr>
          <td class="n">0</td><td class="g">Rock-paper-scissors</td>
          <td class="p">regret-matching ledger math</td>
          <td class="s">complete &middot; <a href="/rung0">live demo &rarr;</a></td>
        </tr>
        <tr>
          <td class="n">1</td><td class="g">Kuhn poker</td>
          <td class="p">tabular CFR vs. the known exact equilibrium</td>
          <td class="s">complete &middot; <a href="/rung1">live demo &rarr;</a></td>
        </tr>
        <tr>
          <td class="n">2</td><td class="g">Leduc poker</td>
          <td class="p">CFR with a board, vs. published benchmarks</td>
          <td class="s">complete &middot; <a href="/rung2">live demo &rarr;</a></td>
        </tr>
        <tr class="pending">
          <td class="n">3</td><td class="g">Mini-drawmaha</td>
          <td class="p">split pots, and a draw the deck answers differently</td>
          <td class="s">pending</td>
        </tr>
        <tr class="pending">
          <td class="n">4</td><td class="g">Full drawmaha</td>
          <td class="p">Deep CFR: nets replace the regret tables</td>
          <td class="s">pending</td>
        </tr>
      </table>
    </section>

    <section>
      <p class="stop">Stop 01 &middot; measured so far</p>
      <h2 class="big">Every number is the solver's own.</h2>
      <p class="lede">Exported from the run that produced it. The browser draws
      these and never recomputes them.</p>

      <div class="records">
        <div class="panel">
          <span class="k">Rung 0 &middot; rock-paper-scissors</span>
          <h3>0.0009 chips per round from Nash</h3>
          <p class="say">Self-play lands on (0.334, 0.333, 0.333). Against a
          50%-rock opponent the ledger converges to pure paper and earns +0.24 a
          round, against a best response of +0.25.</p>
          <ul class="fields">
            <li><span>iterations</span><span>100,000</span></li>
            <li><span>exploitability</span><span>0.0009 / round</span></li>
            <li><span>vs 50% rock</span><span>+0.24 / round</span></li>
          </ul>
        </div>

        <figure class="polaroid">
          <svg viewBox="0 0 300 190" role="img" aria-label="The jack's bluff frequency settling at one third of the king's value bet">
            <g fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round">
              <path d="M34 158h240M34 158V22" stroke-opacity=".5"/>
              <g stroke-opacity=".2"><path d="M34 118h240M34 78h240M34 38h240"/></g>
              <path d="M34 150c26 0 34-64 52-64s28 26 46 26 30-14 52-14 40 4 90 3" stroke-width="1.9"/>
              <path d="M34 156c30 2 44-22 62-22s30 8 48 8 34-4 56-4 38 1 74 1"
                    stroke-width="1.3" stroke-dasharray="4 3.5"/>
              <circle cx="274" cy="101" r="3.4" fill="currentColor"/>
              <circle cx="274" cy="139" r="3.4" fill="currentColor"/>
            </g>
            <!-- Direct labels sit ABOVE their end point: beside it, they ran
                 straight through the dot. -->
            <text x="34" y="16" font-family="IBM Plex Mono, monospace" font-size="9"
                  fill="currentColor" opacity=".65" letter-spacing="1.2">P(BET) BY CARD</text>
            <text x="276" y="92" font-family="IBM Plex Mono, monospace" font-size="9"
                  fill="currentColor" text-anchor="end">king .663</text>
            <text x="276" y="130" font-family="IBM Plex Mono, monospace" font-size="9"
                  fill="currentColor" text-anchor="end">jack .220</text>
          </svg>
          <figcaption>the bluff nobody taught it &mdash; exactly &#8531; of the value bet</figcaption>
        </figure>
      </div>

      <div class="panel" style="margin-top:20px;max-width:52%;">
          <span class="k">Rung 1 &middot; kuhn poker</span>
          <h3>0.00063 chips per hand from Nash</h3>
          <p class="say">Vanilla CFR reproduces Kuhn's closed form: the jack
          bluffs 0.220 of the time and the king value-bets 0.663 &mdash; the 1:3
          ratio the equilibrium requires, found from nothing.</p>
          <ul class="fields">
            <li><span>iterations</span><span>100,000</span></li>
            <li><span>game value</span><span>&minus;0.05555 vs &minus;1/18</span></li>
            <li><span>best response</span><span>exact, over 2&#8310; strategies</span></li>
          </ul>
      </div>
    </section>

  </main>
</div>

<script>
""" + HERO_JS + """
</script>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
