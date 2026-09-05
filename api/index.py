"""Vercel entrypoint: the cover page for the drawmaha-solver validation ladder.

The repo is a solver library and CLI; this page states where the ladder
stands and links each completed rung's live artifact (rung 0: the
regret-matching visualizer at /rung0; rung 1: the Kuhn CFR visualizer at
/rung1). The GTOWizard-style dashboard planned for rung 4 replaces this page.

Stdlib only on purpose — the page must never depend on the solver's numeric
stack, so a heavy dependency can't break the deploy. Visual language matches
both visualizers: a 1272px 12-column grid, IBM Plex Sans for headings and UI,
Plex Serif for reading copy, Plex Mono for numbers and field labels, and
hairline rows instead of cards.
"""

from http.server import BaseHTTPRequestHandler

PAGE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drawmaha Solver — rung 1 complete: CFR solves Kuhn poker exactly</title>
<style>
  /* Self-hosted IBM Plex (the same @fontsource files the visualizers
     bundle); copied to /fonts by the Vercel buildCommand. No third-party
     font CDN. */
  @font-face {
    font-family: "IBM Plex Sans"; font-style: normal; font-weight: 400;
    font-display: swap; src: url(/fonts/ibm-plex-sans-latin-400-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "IBM Plex Sans"; font-style: normal; font-weight: 500;
    font-display: swap; src: url(/fonts/ibm-plex-sans-latin-500-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "IBM Plex Sans"; font-style: normal; font-weight: 600;
    font-display: swap; src: url(/fonts/ibm-plex-sans-latin-600-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "IBM Plex Sans"; font-style: normal; font-weight: 700;
    font-display: swap; src: url(/fonts/ibm-plex-sans-latin-700-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "IBM Plex Serif"; font-style: normal; font-weight: 400;
    font-display: swap; src: url(/fonts/ibm-plex-serif-latin-400-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "IBM Plex Mono"; font-style: normal; font-weight: 400;
    font-display: swap; src: url(/fonts/ibm-plex-mono-latin-400-normal.woff2) format("woff2");
  }
  @font-face {
    font-family: "IBM Plex Mono"; font-style: normal; font-weight: 500;
    font-display: swap; src: url(/fonts/ibm-plex-mono-latin-500-normal.woff2) format("woff2");
  }

  :root {
    color-scheme: light dark;
    /* Two inks and a paper. Greys are the ink tinted toward the paper,
       never a neutral #888. */
    --paper: #f7f7f3;
    --paper-2: #ecebe4;
    --ink: #141412;
    --ink-2: #52514e;
    --ink-3: #8a8880;
    --hair: rgba(20, 20, 18, 0.12);
    --hair-2: rgba(20, 20, 18, 0.24);
    /* The dark band at the foot of the page: the ink used as ground, with
       the paper colour as its text. */
    --band: #141412;
    --band-ink: #f7f7f3;
    --band-ink-2: rgba(247, 247, 243, 0.55);
    --sans: "IBM Plex Sans", system-ui, sans-serif;
    --serif: "IBM Plex Serif", Georgia, serif;
    --mono: "IBM Plex Mono", ui-monospace, monospace;
    --site-w: 1272px;
    --margin: clamp(20px, 4vw, 84px);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      /* The same relationship inverted: the ink becomes the ground and the
         paper becomes the text, rather than black-on-white flipped to
         white-on-black. */
      --paper: #141412;
      --paper-2: #1e1e1b;
      --ink: #f2f1ea;
      --ink-2: #a8a69c;
      --ink-3: #78766e;
      --hair: rgba(242, 241, 234, 0.14);
      --hair-2: rgba(242, 241, 234, 0.26);
      /* On a dark page the band cannot be darker still; it becomes the
         flat second surface, marked off by a hairline instead. */
      --band: #1e1e1b;
      --band-ink: #f2f1ea;
      --band-ink-2: rgba(242, 241, 234, 0.5);
    }
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font: 16px/1.5 var(--sans);
    -webkit-font-smoothing: antialiased;
  }

  a { color: var(--ink); }
  a:hover { color: var(--ink-2); }

  :focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }

  .wrap { max-width: var(--site-w); margin: 0 auto; padding: 0 var(--margin); }

  /* ---------- nav: no bottom border, one button, four text links ---------- */

  nav.top {
    min-height: 64px; display: flex; align-items: center;
    justify-content: space-between; gap: 24px; flex-wrap: wrap;
    padding: 12px 0;
  }
  .wordmark {
    font: 700 16px/1 var(--sans); color: var(--ink); text-decoration: none;
  }
  nav.top .links {
    display: flex; align-items: center; gap: 32px; flex-wrap: wrap;
  }
  nav.top ul {
    display: flex; gap: 28px; list-style: none; margin: 0; padding: 0;
    align-items: center;
  }
  nav.top ul a { font: 400 15px/1 var(--sans); text-decoration: none; }
  nav.top ul a.here {
    text-decoration: underline; text-underline-offset: 0.35em;
    text-decoration-thickness: 1px;
  }
  .btn {
    display: inline-flex; align-items: center; gap: 10px;
    background: var(--ink); color: var(--paper); border-radius: 8px;
    padding: 9px 16px; font: 500 14px/1.2 var(--sans);
    text-decoration: none; white-space: nowrap;
  }
  .btn:hover { color: var(--paper); opacity: 0.86; }

  /* ---------- the 12-column grid everything sits on ---------- */

  .g12 {
    display: grid; grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: 32px;
  }
  .l4 { grid-column: 1 / 5; }
  .r8 { grid-column: 5 / 13; }

  /* ---------- hero: headline left, serif dek offset to column 9 ---------- */

  header.hero {
    display: grid; grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: 32px; padding: 132px 0 88px; align-items: start;
  }
  header.hero h1 {
    grid-column: 1 / 9; margin: 0;
    font: 700 clamp(34px, 4.5vw, 64px)/1.05 var(--sans);
    letter-spacing: 0; text-wrap: balance;
  }
  header.hero .dek {
    grid-column: 9 / 13; margin: 8px 0 0;
    font: 400 17px/1.55 var(--serif); color: var(--ink);
  }
  header.hero .dek a {
    text-decoration: underline; text-underline-offset: 0.2em;
  }

  /* ---------- type roles ---------- */

  .label {
    font: 500 12px/1.4 var(--mono); text-transform: uppercase;
    letter-spacing: 0.03em; color: var(--ink-3);
  }
  h2.sec { font: 600 24px/1.3 var(--sans); margin: 0; letter-spacing: 0; }
  h3.rec { font: 600 17px/1.35 var(--sans); margin: 0; letter-spacing: 0; }
  .serif { font: 400 16px/1.5 var(--serif); color: var(--ink); margin: 0; }
  .serif.dim { color: var(--ink-2); }
  .num {
    font: 400 15px/1.4 var(--mono); font-variant-numeric: tabular-nums;
  }

  section.sect { padding: 96px 0 0; }

  /* ---------- the ladder: hairline rows, not cards ---------- */

  .sec-head {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 24px; margin-bottom: 24px; flex-wrap: wrap;
  }
  .ladder { border-bottom: 1px solid var(--hair); }
  .ladder .row {
    display: grid; grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: 0 32px; align-items: baseline;
    padding: 16px 0; border-top: 1px solid var(--hair);
  }
  .ladder .head { padding: 0 0 10px; border-top: 0; }
  .ladder .c-rung { grid-column: 1 / 2; }
  .ladder .c-game { grid-column: 2 / 5; font-weight: 600; }
  .ladder .c-proves { grid-column: 5 / 10; }
  .ladder .c-status {
    grid-column: 10 / 13; text-align: right;
    font-size: 15px; color: var(--ink-2);
  }
  .ladder .c-status a {
    text-decoration: underline; text-underline-offset: 0.2em;
  }

  /* ---------- measured: two flat record cards ---------- */

  .records {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 24px;
  }
  .card {
    background: var(--paper-2); border-radius: 16px;
    padding: 28px 28px 24px;
    display: flex; flex-direction: column; gap: 14px;
  }
  .card .serif { font-size: 15px; }
  .fields { list-style: none; margin: 0; padding: 0; }
  .fields li {
    display: flex; justify-content: space-between; gap: 16px;
    padding: 10px 0; border-top: 1px solid var(--hair);
  }
  .fields .v {
    font: 400 15px/1.4 var(--mono); font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .card .cta { margin-top: 4px; }

  /* ---------- writeups: rows again, different columns ---------- */

  .rows { list-style: none; margin: 0; padding: 0; }
  .rows li {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 24px; padding: 16px 0; border-bottom: 1px solid var(--hair);
  }
  .rows li:first-child { border-top: 1px solid var(--hair); }
  .rows .t { font: 600 15px/1.4 var(--sans); }
  .rows .m {
    font: 400 15px/1.4 var(--sans); color: var(--ink-2); white-space: nowrap;
  }

  /* ---------- the band at the foot ---------- */

  footer.band {
    background: var(--band); color: var(--band-ink);
    padding: 72px 0 48px; margin-top: 120px;
  }
  footer.band .wordmark, footer.band a { color: var(--band-ink); }
  footer.band a { text-decoration: none; font: 400 13px/1.4 var(--sans); }
  footer.band a:hover { text-decoration: underline; text-underline-offset: 0.2em; }
  footer.band h3 {
    font: 500 12px/1.4 var(--mono); text-transform: uppercase;
    letter-spacing: 0.03em; color: var(--band-ink-2); margin: 0 0 14px;
  }
  footer.band ul {
    list-style: none; margin: 0; padding: 0;
    display: flex; flex-direction: column; gap: 9px;
  }
  footer.band li.pending {
    font: 400 13px/1.4 var(--sans); color: var(--band-ink-2);
  }
  footer.band .tiny {
    font: 400 12px/1.5 var(--mono); color: var(--band-ink-2); margin: 56px 0 0;
  }
  footer.band .about { grid-column: 1 / 5; }
  footer.band .rungs { grid-column: 6 / 9; }
  footer.band .about .tiny { margin-top: 20px; max-width: 34ch; }

  @media (max-width: 900px) {
    header.hero { padding: 56px 0 56px; }
    header.hero h1, header.hero .dek { grid-column: 1 / 13; }
    section.sect { padding-top: 64px; }
    .g12 { gap: 24px; }
    .l4, .r8 { grid-column: 1 / 13; }
    .records { grid-template-columns: minmax(0, 1fr); }
    .ladder .row { gap: 2px 0; }
    .ladder .c-rung, .ladder .c-game, .ladder .c-proves, .ladder .c-status {
      grid-column: 1 / 13; text-align: left;
    }
    .ladder .head { display: none; }
    footer.band { margin-top: 72px; }
    footer.band .about, footer.band .rungs { grid-column: 1 / 13; }
    footer.band .rungs { margin-top: 40px; }
  }
</style>
</head>
<body>
<div class="wrap">

<nav class="top">
  <a class="wordmark" href="/">drawmaha solver</a>
  <div class="links">
    <ul>
      <li><a class="here" href="/">Ladder</a></li>
      <li><a href="/rung0">Rung 0</a></li>
      <li><a href="/rung1">Rung 1</a></li>
    </ul>
    <a class="btn" href="/rung1#play">Play against CFR <span aria-hidden="true">&rarr;</span></a>
  </div>
</nav>

<header class="hero">
  <h1>Five games between here and a Drawmaha solver</h1>
  <p class="dek">A Deep-CFR solver for heads-up pot-limit Drawmaha, a split-pot
  draw/Omaha hybrid with no existing solver. It is built rung by rung on a
  <a href="#ladder">validation ladder</a>: each rung checked against a known
  answer before climbing.</p>
</header>

<section id="ladder">
  <div class="sec-head">
    <h2 class="sec">The validation ladder</h2>
    <span class="label">5 rungs &middot; 2 complete</span>
  </div>
  <div class="ladder">
    <div class="row head">
      <span class="label c-rung">Rung</span>
      <span class="label c-game">Game</span>
      <span class="label c-proves">What it proves</span>
      <span class="label c-status">Status</span>
    </div>
    <div class="row">
      <span class="num c-rung">0</span>
      <span class="c-game">Rock-paper-scissors</span>
      <span class="serif dim c-proves">regret-matching ledger math</span>
      <span class="c-status">complete &middot; <a href="/rung0">Live demo &rarr;</a></span>
    </div>
    <div class="row">
      <span class="num c-rung">1</span>
      <span class="c-game">Kuhn poker</span>
      <span class="serif dim c-proves">tabular CFR vs. the known exact equilibrium</span>
      <span class="c-status">complete &middot; <a href="/rung1">Live demo &rarr;</a></span>
    </div>
    <div class="row">
      <span class="num c-rung">2</span>
      <span class="c-game">Leduc poker</span>
      <span class="serif dim c-proves">CFR with a board, vs. published benchmarks</span>
      <span class="c-status">next</span>
    </div>
    <div class="row">
      <span class="num c-rung">3</span>
      <span class="c-game">Mini-drawmaha</span>
      <span class="serif dim c-proves">split pots, draws, the face-up draw-1 rule</span>
      <span class="c-status">pending</span>
    </div>
    <div class="row">
      <span class="num c-rung">4</span>
      <span class="c-game">Full drawmaha</span>
      <span class="serif dim c-proves">Deep CFR: nets replace the regret tables</span>
      <span class="c-status">pending</span>
    </div>
  </div>
</section>

<section class="sect g12">
  <div class="l4">
    <h2 class="sec">Measured so far</h2>
    <p class="serif dim" style="margin-top:14px;max-width:32ch;">Every number here
    is exported from the solver's own run. The browser draws it and never
    recomputes it.</p>
  </div>
  <div class="r8 records">
    <div class="card">
      <span class="label">Rung 0 &middot; Rock-paper-scissors</span>
      <h3 class="rec">0.0009 chips per round from Nash</h3>
      <p class="serif dim">Self-play average strategy (0.334, 0.333, 0.333)
      against the uniform Nash. Against a 50%-rock opponent the ledger
      converges to pure paper and earns +0.24 a round, against a best response
      of +0.25.</p>
      <ul class="fields">
        <li><span class="label">Iterations</span><span class="v">100,000</span></li>
        <li><span class="label">Exploitability</span><span class="v">0.0009 chips / round</span></li>
        <li><span class="label">Vs 50% rock</span><span class="v">+0.24 / round</span></li>
      </ul>
      <div class="cta"><a class="btn" href="/rung0">Watch the ledger run <span aria-hidden="true">&rarr;</span></a></div>
    </div>
    <div class="card">
      <span class="label">Rung 1 &middot; Kuhn poker</span>
      <h3 class="rec">0.00063 chips per hand from Nash</h3>
      <p class="serif dim">Vanilla CFR reproduces Kuhn's closed form: the jack
      bluffs 0.220 of the time, the king value-bets 0.663, the 1:3 ratio found
      from nothing.</p>
      <ul class="fields">
        <li><span class="label">Iterations</span><span class="v">100,000</span></li>
        <li><span class="label">Game value</span><span class="v">&minus;0.05555 vs &minus;1/18</span></li>
        <li><span class="label">Best response</span><span class="v">exact, over 2<sup>6</sup> strategies</span></li>
      </ul>
      <div class="cta"><a class="btn" href="/rung1">Scrub the solve <span aria-hidden="true">&rarr;</span></a></div>
    </div>
  </div>
</section>

<section class="sect g12">
  <div class="l4">
    <h2 class="sec">Writeups</h2>
    <p class="serif dim" style="margin-top:14px;max-width:32ch;">Written as each
    rung landed. They live in the repo until this site hosts them.</p>
  </div>
  <ul class="rows r8">
    <li><span class="t">Regret matching, one ledger at a time</span><span class="m">Rung 0</span></li>
    <li><span class="t">How CFR finds Kuhn's bluff</span><span class="m">Rung 1</span></li>
    <li><span class="t">From Kuhn to Leduc: what a board changes</span><span class="m">Rung 2 delta</span></li>
    <li><span class="t">How poker solvers are trained</span><span class="m">Survey &middot; 55 references</span></li>
    <li><span class="t">Pre-registered evaluation protocol</span><span class="m">README</span></li>
  </ul>
</section>

</div>

<footer class="band">
  <div class="wrap g12">
    <div class="about">
      <a class="wordmark" href="/">drawmaha solver</a>
      <p class="tiny">A Deep-CFR solver for heads-up pot-limit Drawmaha, built on
      a validation ladder. This page is served until the rung-4 dashboard
      exists.</p>
    </div>
    <div class="rungs">
      <h3>Rungs</h3>
      <ul>
        <li><a href="/rung0">0 &middot; Rock-paper-scissors</a></li>
        <li><a href="/rung1">1 &middot; Kuhn poker</a></li>
        <li class="pending">2 &middot; Leduc poker &mdash; next</li>
        <li class="pending">3 &middot; Mini-drawmaha &mdash; pending</li>
        <li class="pending">4 &middot; Full drawmaha &mdash; pending</li>
      </ul>
    </div>
  </div>
  <div class="wrap">
    <p class="tiny">Code, figures and writeups live in AlecROndo/drawmaha-solver (private).</p>
  </div>
</footer>
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
