"""Vercel entrypoint: the cover page for the drawmaha-solver validation ladder.

The repo is a solver library and CLI; this page states where the ladder
stands and links each completed rung's live artifact (rung 0: the
regret-matching visualizer at /rung0; rung 1: the Kuhn CFR visualizer at
/rung1). The GTOWizard-style dashboard planned for rung 4 replaces this page.

Stdlib only on purpose — the page must never depend on the solver's numeric
stack, so a heavy dependency can't break the deploy. Visual language matches
web/rung0-viz: IBM Plex Sans/Mono, paper surface, hairline-ruled figures.
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
  /* Self-hosted IBM Plex (same @fontsource files the /rung0 visualizer
     bundles); copied to /fonts by the Vercel buildCommand. No third-party
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
  :root {
    color-scheme: light dark;
    --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --rule: #e1e0d9; --rule-strong: #c3c2b7;
    --done: #006300; --next: #2a78d6;
    --sans: "IBM Plex Sans", system-ui, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
      --rule: #2c2c2a; --rule-strong: #383835;
      --done: #0ca30c; --next: #3987e5;
    }
  }
  * { box-sizing: border-box; }
  body {
    background: var(--surface); color: var(--ink);
    font: 15px/1.55 var(--sans);
    max-width: 46rem; margin: 0 auto; padding: 34px 28px 64px;
  }
  .eyebrow {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 10px;
  }
  h1 {
    font-size: clamp(26px, 4vw, 32px); font-weight: 600;
    letter-spacing: -0.015em; line-height: 1.15; margin: 0 0 8px;
  }
  .dek { color: var(--ink-2); margin: 0 0 30px; max-width: 62ch; }
  section { border-top: 1px solid var(--rule-strong); padding: 16px 0 26px; }
  .fig {
    font-family: var(--mono); font-size: 10.5px; font-weight: 500;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
    margin: 0 0 6px;
  }
  h2 { font-size: 15px; font-weight: 600; margin: 0 0 14px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 9px 14px 9px 0; border-bottom: 1px solid var(--rule); }
  tr:last-child td { border-bottom: none; }
  th {
    font-family: var(--mono); font-size: 10.5px; font-weight: 500;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted);
  }
  td.n { font-family: var(--mono); color: var(--ink-2); width: 1%; padding-right: 22px; }
  td.status { font-family: var(--mono); font-size: 12.5px; white-space: nowrap; }
  .done { color: var(--done); font-weight: 600; }
  .next { color: var(--next); font-weight: 500; }
  .pending { color: var(--muted); }
  a.demo { color: var(--ink); text-decoration-color: var(--rule-strong); text-underline-offset: 3px; }
  a.demo:hover { text-decoration-color: var(--ink); }
  .results p { margin: 0; max-width: 68ch; color: var(--ink-2); }
  .results code {
    font-family: var(--mono); font-size: 0.92em; color: var(--ink);
    background: none; padding: 0;
  }
  footer {
    border-top: 1px solid var(--rule); padding-top: 14px;
    font-family: var(--mono); font-size: 11.5px; color: var(--muted);
  }
  footer code { font-family: var(--mono); }
</style>
</head>
<body>
<p class="eyebrow">drawmaha solver &middot; validation ladder</p>
<h1>Five games between here and a Drawmaha solver</h1>
<p class="dek">A Deep-CFR solver for heads-up pot-limit Drawmaha (split-pot
draw/Omaha hybrid), built rung by rung on a validation ladder &mdash; each rung
checked against a known answer before climbing.</p>

<section>
  <p class="fig">Fig. 1</p>
  <h2>The ladder &mdash; rungs 0 and 1 complete, Leduc next</h2>
  <table>
    <tr><th>Rung</th><th>Game</th><th>What it proves</th><th>Status</th></tr>
    <tr>
      <td class="n">0</td><td>Rock-paper-scissors</td>
      <td>regret-matching ledger math</td>
      <td class="status"><span class="done">complete</span> &middot; <a class="demo" href="/rung0">live demo &rarr;</a></td>
    </tr>
    <tr>
      <td class="n">1</td><td>Kuhn poker</td>
      <td>tabular CFR vs. the known exact equilibrium</td>
      <td class="status"><span class="done">complete</span> &middot; <a class="demo" href="/rung1">live demo &rarr;</a></td>
    </tr>
    <tr>
      <td class="n">2</td><td>Leduc poker</td>
      <td>CFR with a board, vs. published benchmarks</td>
      <td class="status"><span class="next">next</span></td>
    </tr>
    <tr>
      <td class="n">3</td><td>Mini-drawmaha</td>
      <td>split pots, draws, face-up draw-1 rule</td>
      <td class="status"><span class="pending">pending</span></td>
    </tr>
    <tr>
      <td class="n">4</td><td>Full drawmaha</td>
      <td>Deep CFR &mdash; nets replace the regret tables</td>
      <td class="status"><span class="pending">pending</span></td>
    </tr>
  </table>
</section>

<section class="results">
  <p class="fig">Fig. 2</p>
  <h2>Rung 0 measured results &mdash; 0.0009 chips/round from Nash at 100k iterations</h2>
  <p>Average strategy <code>(0.334, 0.333, 0.333)</code> vs. the uniform Nash
  <code>(1/3, 1/3, 1/3)</code>, exploitable for <code>0.0009</code>
  chips/round; against a 50%-rock opponent the ledger converges to pure
  paper and earns <code>+0.24</code>/round (best response: <code>+0.25</code>).
  Watch the ledger run live in the <a class="demo" href="/rung0">rung-0 demo</a>.</p>
</section>

<section class="results">
  <p class="fig">Fig. 3</p>
  <h2>Rung 1 measured results &mdash; 0.00063 chips/hand from Nash at 100k iterations</h2>
  <p>Vanilla CFR reproduces Kuhn's closed form: the jack bluffs
  <code>0.220</code> of the time and the king value-bets <code>0.663</code>
  &mdash; the 1:3 ratio the equilibrium requires, found from nothing. Game
  value to the first player is <code>&minus;0.05555</code> against the exact
  <code>&minus;1/18</code>, and exploitability is measured by an exact
  best response over all 2<sup>6</sup> pure strategies per seat. Scrub the
  solve and play a hand against it in the
  <a class="demo" href="/rung1">rung-1 demo</a>.</p>
</section>

<footer>This page is served until the rung-4 solver dashboard exists.
Code, figures, and the full writeup live in the GitHub repo
(<code>AlecROndo/drawmaha-solver</code>, private).</footer>
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
