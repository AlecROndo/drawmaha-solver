"""Vercel entrypoint: a minimal status page for the drawmaha-solver project.

The repo is a solver library and CLI, not a web app (yet). Vercel's Python
builder requires an entrypoint, so this handler gives deployments something
honest to serve: what the project is and where the validation ladder stands.
The GTOWizard-style dashboard planned for rung 4 replaces this page.

Stdlib only on purpose — the page must never depend on the solver's numeric
stack, so a heavy dependency can't break the deploy.
"""

from http.server import BaseHTTPRequestHandler

PAGE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drawmaha Solver — rung 0 complete: regret matching verified on RPS</title>
<style>
  :root {
    --surface: #fcfcfb; --ink: #0b0b0b; --secondary: #52514e;
    --muted: #898781; --grid: #e1e0d9; --blue: #2a78d6; --green: #1baf7a;
  }
  body {
    background: var(--surface); color: var(--ink);
    font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 44rem; margin: 3rem auto; padding: 0 1.25rem;
  }
  h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
  .sub { color: var(--secondary); margin-top: 0; }
  table { border-collapse: collapse; width: 100%; margin: 1.25rem 0; }
  th, td { text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--grid); }
  th { color: var(--muted); font-weight: 600; font-size: 0.85rem; }
  .done { color: var(--green); font-weight: 600; }
  .next { color: var(--blue); }
  .pending { color: var(--muted); }
  .results { background: #fff; border: 1px solid var(--grid); border-radius: 8px; padding: 0.9rem 1.1rem; }
  .results code { font-size: 0.92em; }
  footer { color: var(--muted); font-size: 0.85rem; margin-top: 2rem; }
</style>
</head>
<body>
<h1>Drawmaha Solver</h1>
<p class="sub">A Deep-CFR solver for heads-up pot-limit Drawmaha (split-pot
draw/Omaha hybrid), built rung by rung on a validation ladder — each rung
checked against a known answer before climbing.</p>

<table>
  <tr><th>Rung</th><th>Game</th><th>What it proves</th><th>Status</th></tr>
  <tr><td>0</td><td>Rock-paper-scissors</td><td>regret-matching ledger math</td><td class="done">complete</td></tr>
  <tr><td>1</td><td>Kuhn poker</td><td>tabular CFR vs. the known exact equilibrium</td><td class="next">next</td></tr>
  <tr><td>2</td><td>Leduc poker</td><td>CFR with a board, vs. published benchmarks</td><td class="pending">pending</td></tr>
  <tr><td>3</td><td>Mini-drawmaha</td><td>split pots, draws, face-up draw-1 rule</td><td class="pending">pending</td></tr>
  <tr><td>4</td><td>Full drawmaha</td><td>Deep CFR — nets replace the regret tables</td><td class="pending">pending</td></tr>
</table>

<div class="results">
  <strong>Rung 0 measured results</strong> (100k self-play iterations):
  average strategy <code>(0.334, 0.333, 0.333)</code> vs. the uniform Nash
  <code>(1/3, 1/3, 1/3)</code>, exploitable for <code>0.0009</code>
  chips/round; against a 50%-rock opponent the ledger converges to pure
  paper and earns <code>+0.24</code>/round (best response: <code>+0.25</code>).
</div>

<footer>This placeholder is served until the rung-4 solver dashboard exists.
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
