"""Diagrams for the rung0-rps writeup — house card style (see skill assets)."""
import os
import shutil
import subprocess

os.makedirs("diagrams", exist_ok=True)

INK    = "#1F2A33"
MUTED  = "#6B7078"
EDGE   = "#AEB9C7"
EDGELB = "#8A95A3"

NEUTRAL = ("#F2F6FA", "#C3D2E0")
BLUE    = ("#DEEAF4", "#8FB2D4")
GREEN   = ("#DCEDE6", "#8DBFB2")
GREEN_S = ("#C7E2D8", "#5FA08F")
AMBER   = ("#F5E7DA", "#DBB08C")
GRAY    = ("#F4F6F8", "#D5DCE3")
PEACH   = ("#F4E1D6", "#DAA588")

FONT = "Helvetica"

def _header(rankdir):
    return f'''
  graph [rankdir={rankdir}, bgcolor="transparent", fontname="{FONT}",
         nodesep=0.5, ranksep=0.7, pad=0.35, splines=spline];
  node  [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=11,
         fontcolor="{INK}", penwidth=1.1, margin="0.22,0.13"];
  edge  [fontname="{FONT}", fontsize=9, color="{EDGE}", fontcolor="{EDGELB}",
         arrowsize=0.7, arrowhead=vee, penwidth=1.1];
'''

def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def card(node_id, title, subtitle=None, kind=NEUTRAL, shape=None, dashed=False):
    fill, border = kind
    title_color = MUTED if kind is GRAY else INK
    label = f'<<B>{_esc(title)}</B>'
    if subtitle:
        label += f'<BR/><FONT POINT-SIZE="9" COLOR="{MUTED}">{_esc(subtitle)}</FONT>'
    label += ">"
    style = "rounded,filled,dashed" if dashed else "rounded,filled"
    extra = f', shape={shape}' if shape else ""
    return (f'  {node_id} [label={label}, fillcolor="{fill}", color="{border}", '
            f'fontcolor="{title_color}", style="{style}"{extra}];\n')

def cluster(cid, label, body, fill="#FBFCFE", border="#E3E9F0"):
    return (f'''  subgraph cluster_{cid} {{
    label=<<FONT POINT-SIZE="10" COLOR="{MUTED}"><B>{_esc(label)}</B></FONT>>;
    labeljust="l"; style="rounded,filled"; fillcolor="{fill}";
    color="{border}"; penwidth=1.0; margin=14;
{body}  }}
''')

def render(name, body, rankdir="TB"):
    if shutil.which("dot") is None:
        raise SystemExit("graphviz `dot` not found on PATH — install graphviz.")
    dot = f"digraph {{\n{_header(rankdir)}\n{body}\n}}\n"
    with open(f"diagrams/{name}.dot", "w") as f:
        f.write(dot)
    subprocess.run(["dot", "-Tpng", "-Gdpi=220", f"diagrams/{name}.dot",
                    "-o", f"diagrams/{name}.png"], check=True)
    print(f"wrote diagrams/{name}.png")

def as_built_architecture():
    """Everything in the diagram is new code (green); blue marks the two
    ways in and the artifacts that come out. Two entry points branch into a
    shared core: the analysis CLI drives the ledger directly, the play CLI
    goes through the player layer; both converge on the ledger + rules."""
    entries = ""
    entries += card("analysis", "analysis.py  (uv run rps-analysis)",
                    "runs self-play + vs-biased-opponent experiments, draws 4 figures", BLUE)
    entries += card("play", "play.py  (uv run rps-play)",
                    "terminal loop: a human plays rounds against the learner", BLUE)
    core = ""
    core += card("players", "players.py",
                 "Player interface: human input, fixed strategy, learner; match runner", GREEN)
    core += card("ledger", "regret_matching.py",
                 "the ledger: play ∝ positive regret; the running average converges", GREEN_S)
    core += card("game", "game.py",
                 "RPS rules: payoff matrix, winner, exploitability metric", GREEN)
    body = cluster("entry", "entry points — src/drawmaha_solver/rps/", entries)
    body += cluster("core", "core — src/drawmaha_solver/rps/", core)
    body += card("figs", "figures/rung0/*.png",
                 "4 committed convergence figures (embedded in the README)", BLUE, shape="folder")
    body += card("term", "terminal session",
                 "round-by-round score + what the bot learned about you", BLUE)
    body += '''
  analysis -> ledger  [label="steps the ledger each round"];
  analysis -> game    [label="payoffs + exploitability"];
  analysis -> figs    [label="writes"];
  play     -> players [label="human vs learner"];
  players  -> ledger  [label="feeds each finished round"];
  players  -> game    [label="payoffs"];
  play     -> term    [label="prints"];
  ledger -> figs [style=invis];
  game   -> term [style=invis];
'''
    render("as_built_architecture", body, rankdir="TB")

if __name__ == "__main__":
    as_built_architecture()
