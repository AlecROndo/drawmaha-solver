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
    --mark-dim: rgba(245, 240, 230, 0.62);
    --hair: rgba(245, 240, 230, 0.28);

    --panel: var(--bone);
    --panel-mark: var(--oxblood);
    --panel-dim: #b4536a;
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
      --mark-dim: rgba(142, 32, 56, 0.62);
      --hair: rgba(142, 32, 56, 0.22);
      --panel: #eae0cb;
      --panel-border: rgba(142, 32, 56, 0.26);
    }
  }

  * { box-sizing: border-box; }

  body {
    margin: 0; background: var(--field); color: var(--mark);
    font: 400 14px/1.6 var(--mono); -webkit-font-smoothing: antialiased;
  }
  a { color: inherit; }
  :focus-visible { outline: 2px solid var(--mark); outline-offset: 3px; }

  /* ---------- the ladder, drawn as the nav ---------- */

  nav.line {
    position: sticky; top: 0; z-index: 20; background: var(--field);
    border-bottom: 1px solid var(--hair); padding: 22px 0 12px;
  }
  nav.line .track { position: relative; height: 34px; margin: 0 var(--gutter); }
  nav.line .bar {
    position: absolute; left: 10%; right: 10%; top: 11px; height: 1px;
    background: var(--hair);
  }
  /* Stations sit at 10/30/50/70/90% — the centres of five equal columns — so
     the climbed segment runs from the first station to the second. */
  nav.line .bar.done {
    left: 10%; right: 70%; top: 10.5px; height: 2px; background: var(--mark);
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
    display: block; margin-top: 10px; font: 500 11.5px/1 var(--mono);
    letter-spacing: 0.06em; text-transform: uppercase; text-decoration: none;
  }
  nav.line li.todo span.stop-name { color: var(--mark-dim); }
  nav.line a:hover { text-decoration: underline; text-underline-offset: 0.3em; }
  nav.line .sub {
    display: block; margin-top: 5px; font: 400 11px/1.3 var(--mono);
    color: var(--mark-dim); text-transform: none; letter-spacing: 0;
  }

  /* ---------- the identity rail ---------- */

  .shell { display: grid; grid-template-columns: var(--rail-w) minmax(0, 1fr); }
  aside.rail {
    position: sticky; top: 92px; align-self: start; height: calc(100vh - 92px);
    padding: 34px 26px 26px var(--gutter); border-right: 1px solid var(--hair);
    display: flex; flex-direction: column; gap: 20px;
  }
  aside.rail .mark { width: 46px; height: 46px; }
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

  .panels {
    display: grid; grid-template-columns: 1.05fr 1fr; gap: 20px; margin-top: 26px;
  }
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

  /* ---------- the playful object: a hand you deal ---------- */

  .deal { align-items: center; justify-content: center; flex: 1; }
  .deal .cue {
    font: 400 22px/1 var(--mono); letter-spacing: 0.22em; text-align: center;
    margin: 10px 0 22px; user-select: none;
  }
  .ticket {
    position: relative; width: 100%; max-width: 372px; cursor: pointer;
    border: 1.5px solid var(--panel-mark); border-radius: 8px;
    padding: 15px 18px 13px; background: var(--panel);
  }
  .ticket:active { transform: translateY(1px); }
  .ticket .perf {
    position: relative; border-top: 1.5px dashed var(--panel-mark);
    margin: 13px -18px 11px;
  }
  .ticket .perf i {
    position: absolute; top: -8px; width: 15px; height: 15px; border-radius: 50%;
    background: var(--field);
  }
  .ticket .perf i.l { left: -8px; }
  .ticket .perf i.r { right: -8px; }
  .ticket .row { display: flex; gap: 22px; }
  .ticket .f { flex: 1; min-width: 0; }
  .ticket .tk {
    font: 400 8.5px/1 var(--mono); letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--panel-dim);
  }
  .ticket .v {
    font: 600 15px/1.25 var(--mono); margin-top: 4px;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  .ticket .v.big { font-size: 19px; letter-spacing: 0.02em; }
  .ticket .hand { font-size: 20px; letter-spacing: 0.06em; }
  .ticket .foot {
    display: flex; justify-content: space-between; align-items: baseline; margin-top: 3px;
  }
  .ticket .note { font: 400 15px/1 var(--script); }
  .ticket .serial {
    font: 400 8.5px/1 var(--mono); color: var(--panel-dim); letter-spacing: 0.1em;
  }

  /* ---------- hairline rows ---------- */

  .rows { width: 100%; border-collapse: collapse; margin: 22px 0 0; }
  .rows td { padding: 15px 0; border-bottom: 1px solid var(--hair); vertical-align: baseline; }
  .rows tr:first-child td { border-top: 1px solid var(--hair); }
  .rows .g { font: 600 14.5px/1.35 var(--mono); width: 210px; }
  .rows .n { width: 34px; font: 400 13px/1.3 var(--mono); color: var(--mark-dim); }
  .rows .p {
    font: 400 13.5px/1.5 var(--mono); color: var(--mark-dim); padding-left: 26px;
  }
  .rows .s {
    text-align: right; font: 400 13px/1.3 var(--mono); white-space: nowrap; padding-left: 20px;
  }
  .rows .s a { text-decoration: underline; text-underline-offset: 0.3em; }
  .rows tr.pending .g, .rows tr.pending .n { color: var(--mark-dim); }

  /* ---------- records, and the one off-axis element ---------- */

  .records { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 26px; }
  .polaroid {
    background: var(--panel); color: var(--panel-mark);
    border: 1px solid var(--panel-border); border-radius: 3px;
    padding: 11px 11px 0; transform: rotate(-2.2deg); align-self: start; margin: 30px 0 0;
  }
  .polaroid svg { display: block; width: 100%; height: auto; }
  .polaroid figcaption {
    font: 400 16px/1.5 var(--script); color: var(--panel-mark); text-align: center;
    padding: 7px 4px 11px;
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
      <li class="todo"><span class="dot"></span><span class="stop-name">Rung 2<span class="sub">leduc</span></span></li>
      <li class="todo"><span class="dot"></span><span class="stop-name">Rung 3<span class="sub">mini-drawmaha</span></span></li>
      <li class="todo"><span class="dot"></span><span class="stop-name">Rung 4<span class="sub">full drawmaha</span></span></li>
    </ol>
  </div>
</nav>

<div class="shell">

  <aside class="rail">
    <svg class="mark" viewBox="0 0 46 46" aria-hidden="true">
      <g fill="none" stroke="currentColor" stroke-width="1.5">
        <circle cx="23" cy="23" r="22"/>
        <path d="M23 11c-4.4 4.6-8 8-8 11.6a8 8 0 0 0 16 0C31 19 27.4 15.6 23 11Z"/>
        <path d="M23 30.6V35M19.4 35h7.2"/>
      </g>
    </svg>
    <h1>Drawmaha<br>Solver.</h1>
    <p class="quote">&ldquo;Each rung is checked against a known answer before we climb.&rdquo;</p>
    <dl>
      <div><dt>now</dt><dd>Rung 1, complete</dd></div>
      <div><dt>next</dt><dd>Rung 2 &middot; Leduc poker</dd></div>
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
    <a class="btn" href="/rung1#play">Play the solver &rarr;</a>
  </aside>

  <main>

    <div class="panels">
      <div class="panel">
        <span class="no">01</span>
        <div class="deal" style="display:flex;flex-direction:column;">
          <p class="cue">CLICK TO DEAL</p>
          <div class="ticket" id="ticket" role="button" tabindex="0" aria-label="Deal a new hand">
            <div class="row">
              <div class="f"><div class="tk">Game</div><div class="v big">DRAWMAHA</div></div>
              <div class="f"><div class="tk">Variant</div><div class="v">pot-limit &middot; split</div></div>
            </div>
            <div class="perf"><i class="l"></i><i class="r"></i></div>
            <div class="row">
              <div class="f"><div class="tk">Your five</div><div class="v hand" id="hand">A&spades; K&spades; 7&hearts; 4&diams; 2&clubs;</div></div>
            </div>
            <div class="row" style="margin-top:11px;">
              <div class="f"><div class="tk">Seat</div><div class="v">you</div></div>
              <div class="f"><div class="tk">Rung</div><div class="v">1 of 5</div></div>
              <div class="f"><div class="tk">Dealt</div><div class="v" id="clock">&mdash;</div></div>
            </div>
            <div class="foot">
              <span class="note">a solver, built rung by rung</span>
              <span class="serial">DMH-04</span>
            </div>
          </div>
        </div>
      </div>

      <div class="panel">
        <span class="no">02</span>
        <figure style="margin:0;flex:1;display:flex;align-items:center;">
          <svg viewBox="0 0 420 380" role="img" aria-label="A dealt five-card hand fanned on a card table, under a hanging lamp">
            <g fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
              <path d="M210 4v36"/>
              <path d="M180 74l14-34h32l14 34z"/>
              <path d="M180 74h60" stroke-opacity=".5"/>
              <g stroke-opacity=".3"><path d="M186 86l-14 26M234 86l14 26M210 86v26"/></g>
              <ellipse cx="210" cy="248" rx="172" ry="76"/>
              <ellipse cx="210" cy="240" rx="172" ry="76" stroke-opacity=".45"/>
              <path d="M38 248c0 42 77 76 172 76s172-34 172-76"/>
              <g stroke-opacity=".28">
                <path d="M60 292l-9 11M86 308l-9 11M118 320l-9 11M154 329l-9 11
                         M266 340l9-11M302 329l9-11M334 314l9-11M360 294l9-11"/>
              </g>
              <g>
                <g transform="rotate(-26 210 318)"><rect x="184" y="176" width="52" height="74" rx="5"/><path d="M210 206l7-9 7 9-7 9z" stroke-opacity=".55"/></g>
                <g transform="rotate(-13 210 318)"><rect x="184" y="172" width="52" height="74" rx="5"/><path d="M210 202l7-9 7 9-7 9z" stroke-opacity=".55"/></g>
                <g><rect x="184" y="170" width="52" height="74" rx="5"/><path d="M210 200l7-9 7 9-7 9z" stroke-opacity=".55"/></g>
                <g transform="rotate(13 210 318)"><rect x="184" y="172" width="52" height="74" rx="5"/><path d="M210 202l7-9 7 9-7 9z" stroke-opacity=".55"/></g>
                <g transform="rotate(26 210 318)"><rect x="184" y="176" width="52" height="74" rx="5"/><path d="M210 206l7-9 7 9-7 9z" stroke-opacity=".55"/></g>
              </g>
              <g transform="translate(88 254)">
                <ellipse cx="0" cy="24" rx="26" ry="9"/><path d="M-26 24v-8M26 24v-8"/>
                <ellipse cx="0" cy="16" rx="26" ry="9"/><path d="M-26 16v-8M26 16v-8"/>
                <ellipse cx="0" cy="8" rx="26" ry="9"/>
                <path d="M-15 7a17 8 0 0 0 30 0" stroke-opacity=".5"/>
              </g>
              <g transform="translate(332 264)">
                <ellipse cx="0" cy="20" rx="22" ry="8"/><path d="M-22 20v-7M22 20v-7"/>
                <ellipse cx="0" cy="13" rx="22" ry="8"/>
                <path d="M-12 12a14 7 0 0 0 24 0" stroke-opacity=".5"/>
              </g>
            </g>
          </svg>
        </figure>
      </div>
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
        <tr class="pending">
          <td class="n">2</td><td class="g">Leduc poker</td>
          <td class="p">CFR with a board, vs. published benchmarks</td>
          <td class="s">next</td>
        </tr>
        <tr class="pending">
          <td class="n">3</td><td class="g">Mini-drawmaha</td>
          <td class="p">split pots, draws, the face-up draw-1 rule</td>
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
  // The one interactive object. A Drawmaha hand is five cards; clicking the
  // ticket deals a fresh one and re-stamps the time.
  var RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2'];
  var SUITS = ['\\u2660','\\u2665','\\u2666','\\u2663'];

  function deal() {
    var seen = {}, out = [];
    while (out.length < 5) {
      var card = RANKS[Math.floor(Math.random() * RANKS.length)] +
                 SUITS[Math.floor(Math.random() * SUITS.length)];
      if (seen[card]) continue;
      seen[card] = true;
      out.push(card);
    }
    document.getElementById('hand').textContent = out.join(' ');
    stamp();
  }

  function stamp() {
    var d = new Date();
    var pad = function (n) { return String(n).padStart(2, '0'); };
    document.getElementById('clock').textContent =
      pad(d.getHours() % 12 || 12) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds()) +
      (d.getHours() < 12 ? ' AM' : ' PM');
  }

  var ticket = document.getElementById('ticket');
  ticket.addEventListener('click', deal);
  ticket.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); deal(); }
  });
  stamp();
  setInterval(stamp, 1000);
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
