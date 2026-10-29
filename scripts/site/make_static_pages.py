"""Write the site's four static pages — home, Rules, Benchmarks, Releases — from committed data.

The pages under `web/cover/` are plain HTML served at the root of the site.
Their charts are inline SVG and their tables are numbers, and every one of
those numbers is read here from a file in the repository, never typed:

* `README.md` — the results table (one row per game, with its method, what it
  is graded against and its grade) and the rock-paper-scissors run's line;
* `figures/rung3/grades.json` — the four regret rules' race, every mark;
* `figures/rung3/answer_sheet.json` — the frozen table's round-one mixes;
* `figures/rung4/*.json` — the Deep CFR grades, the ÷ √pot arm among them;
* `web/rung1-viz/src/data/solve.json`, `web/rung2-viz/src/data/solve.json` —
  the Kuhn and Leduc solves the live demos draw;
* `web/solver-viz/public/mini/` — the frozen table, cut at the opening spot
  (the range matrix on the homepage is rebuilt from its key table and chunk);
* `web/rung3-viz/public/deals/pack-00.json` — the dealt hand on the homepage;
* `src/drawmaha_solver/minidrawmaha/answer/releases.py` — the release
  catalogue, read with `ast` rather than imported, so this script needs
  nothing beyond the standard library.

The two facts with no committed home are the release tags' publication dates
(`PUBLISHED`, from `gh release list`) and the dates in the research index
(`RESEARCH_DATES`, from the merge dates in `git log`); both are named where
they are set.

Run it after any of those files changes; the pages are its output and are
never edited by hand (`tests/test_cover_page.py` regenerates them in memory
and fails on any difference):

    uv run python scripts/site/make_static_pages.py

The chrome — the top bar, the footer — is the markup `site.tsx` renders in
the apps, written out as HTML with the same classes so the linked
`/theme.css` styles both the same way. Each page links `/fonts.css`,
`/theme.css` and `/chip.js`, carries one inline `<style>` of page layout, and
mounts the chip twice.
"""

from __future__ import annotations

import ast
import gzip
import html
import itertools
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "cover"
REPO = "https://github.com/AlecROndo/drawmaha-solver"
ME = "scripts/site/make_static_pages.py"


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


# ===================================================================== data

# ------------------------------------------------------------ the README

# The README's results table names the method in its own words; three cells
# say "rung" or "ladder", which the product's pages leave to the README. Each
# is reworded here by its exact text, so a README edit that changes one fails
# loudly instead of slipping the old word through.
REWORD = {
    "rung 3's grader and rung 3's answer": "the same exact walk and the tabular answer",
    "a head-to-head checkpoint ladder and a trained exploiter": (
        "head-to-head checkpoint matches and a trained exploiter"
    ),
}

# Where each results row is told on the Benchmarks page, in the README's order.
BENCH_ANCHORS = ("rps", "kuhn", "leduc", "mini", "deep", "full")


def readme_results() -> list[dict[str, str]]:
    """The README's results table: game, method, graded against, the grade and what it is in."""
    lines = (ROOT / "README.md").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Rung | Game |"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        _, game, method, against, result, _ = cells
        against = REWORD.get(against, against)
        number = re.search(r"\d+\.\d+", result)
        if number:
            grade = number.group(0)
            note = result.replace(grade, "", 1)
            # "LCFR wins: 0.061 chips/hand …" — the label before the colon is the method's
            note = re.sub(r"^[^:]*:\s*", "", note) if ":" in result[: number.start()] else note
            note = re.sub(r"\s+", " ", note).strip()
        else:
            grade, note = "—", result
        rows.append({"game": game, "method": method, "against": against, "grade": grade, "note": note})
    assert len(rows) == len(BENCH_ANCHORS), f"the README's results table has {len(rows)} rows"
    for row, anchor in zip(rows, BENCH_ANCHORS):
        row["anchor"] = anchor
        for cell in row.values():
            assert not re.search(r"\b(rungs?|ladder)\b", cell, re.IGNORECASE), f"README cell {cell!r} needs a REWORD"
    return rows


def readme_rps() -> dict[str, object]:
    """Rock-paper-scissors' self-play average, its grade, and what it earns against a 50%-rock opponent."""
    text = (ROOT / "README.md").read_text()
    m = re.search(
        r"Self-play average strategy reaches \(([\d.]+), ([\d.]+), ([\d.]+)\) after (\S+) iterations, "
        r"exploitable for ([\d.]+) chips/round; against a 50%-rock opponent the ledger converges to pure "
        r"paper and earns \+([\d.]+)/round \(best response: \+([\d.]+)\)",
        text,
    )
    assert m, "the README's rock-paper-scissors line changed; update readme_rps"
    rock, paper, scissors, iterations, grade, earned, best = m.groups()
    return {
        "average": (float(rock), float(paper), float(scissors)),
        "iterations": iterations,
        "grade": grade,
        "earned": earned,
        "best": best,
    }


def readme_kuhn_table() -> list[tuple[str, str, str, str]]:
    """The README's Kuhn table: iterations, average and current exploitability, game value."""
    lines = (ROOT / "README.md").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| iterations | exploitability (average)"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(tuple(c.strip() for c in line.strip("|").split("|")))
    assert len(rows) == 5
    return rows


# ------------------------------------------------------------ grades

RUNG3 = load("figures/rung3/grades.json")
SHEET = load("figures/rung3/answer_sheet.json")
POT_HALF = load("figures/rung4/pot_0.5-256x4-s16000-lr5e-4.json")
POT_ONE = load("figures/rung4/pot_1-256x4-s16000-lr5e-4.json")
RAW = load("figures/rung4/sweep_256x4-s16000-lr5e-4.json")

RULES = (("lcfr", "LCFR"), ("vanilla", "Vanilla"), ("cfrplus", "CFR+"), ("dcfr", "DCFR"))


def race() -> dict[str, list[tuple[float, float]]]:
    """Each regret rule's linear average by hands per seat: iteration × the run's workers."""
    out: dict[str, list[tuple[float, float]]] = {}
    for rule, name in RULES:
        pts = []
        for key, entry in RUNG3.items():
            parts = key.split("/")
            if len(parts) == 3 and parts[0] == rule and parts[2] == "linear":
                pts.append((int(parts[1]) * entry["workers"], entry["exploitability"]))
        out[name] = sorted(pts)
    return out


def deep_marks(grades: dict, column: str) -> list[tuple[float, float]]:
    return sorted(
        (entry["hands_per_seat"], entry["exploitability"])
        for key, entry in grades.items()
        if key.endswith(f"/{column}")
    )


def at(points: list[tuple[float, float]], hands: float) -> float:
    return next(v for x, v in points if x == hands)


RACE = race()
LCFR = RACE["LCFR"]
DEEP = deep_marks(POT_HALF, "exact")  # the ÷ √pot arm's exact average: the best graded Deep CFR object
DEEP_HANDS = DEEP[-1][0]
UNIFORM = RUNG3["uniform_random"]["exploitability"]


def g3(v: float) -> str:
    """A grade as the pages print it: three decimals, or two significant figures under 0.01."""
    return f"{v:.3f}" if v >= 0.01 else f"{v:.2g}"


def hands(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:g} million"
    return f"{v / 1e3:g} thousand"


def short_hands(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


# ------------------------------------------------------------ the opening spot

CATS = ("high card", "pair", "straight", "flush", "straight flush", "trips")
COLS = ("nothing", "two to the straight", "two to the flush", "pairs the board", "trips with the board")
# The same columns as what a hand does, for a sentence: "… when it pairs the board".
REACHING = ("holds nothing", "holds two to the straight", "holds two to the flush", "pairs the board", "makes trips with the board")


def _rank(card: int) -> int:
    return card // 3


def _suit(card: int) -> int:
    return card % 3


def inner_category(hole: tuple[int, ...]) -> str:
    """`hands.inner_score`'s category of three held cards (no wrap-around straight in five ranks)."""
    ranks = sorted(_rank(c) for c in hole)
    top = max(ranks.count(r) for r in ranks)
    straight = len(set(ranks)) == 3 and ranks[2] - ranks[0] == 2
    flush = len({_suit(c) for c in hole}) == 1
    if top == 3:
        return "trips"
    if straight and flush:
        return "straight flush"
    if flush:
        return "flush"
    if straight:
        return "straight"
    return "pair" if top == 2 else "high card"


def round_one_column(hole: tuple[int, ...], board: int) -> int:
    """solver-viz's `r1Col`: what the hole reaches for with the one board card, strongest reading first."""
    paired = sum(_rank(c) == _rank(board) for c in hole)
    if paired >= 2:
        return 4
    if paired == 1:
        return 3
    if sum(_suit(c) == _suit(board) for c in hole) >= 2:
        return 2
    for a, b in itertools.combinations(hole, 2):
        ranks = (_rank(a), _rank(b), _rank(board))
        if len(set(ranks)) == 3 and max(ranks) - min(ranks) <= 3:
            return 1
    return 0


def opening_spot() -> dict[str, object]:
    """Seat 0's range at the first decision, over every deal, from the frozen table's export.

    The export's opening point holds one row per canonical picture (hole and
    board, suits relabelled jointly). A picture stands for every deal its
    suits can be permuted into, so each row is weighted by the size of that
    orbit, which makes the whole an average over all 15 × 364 deals with equal
    weight — the same average the answer sheet's round-one mixes take, which
    the rows are checked against below.
    """
    base = ROOT / "web" / "solver-viz" / "public" / "mini"
    manifest = json.loads((base / "index.json").read_text())
    point = manifest["points"][0]
    assert (point["player"], point["boardCards"], point["lines"], point["actions"]) == (0, 1, [""], ["c", "p"])
    shape = manifest["shapes"][point["shape"]]
    keys = (base / shape["file"]).read_bytes()
    rows = gzip.decompress((base / point["file"]).read_bytes())
    n, width, stride, scale = point["n"], len(point["actions"]), shape["stride"], manifest["scale"]
    assert len(keys) == n * stride and len(rows) == n * width

    perms = list(itertools.permutations(range(3)))
    cells: dict[tuple[str, int], list[float]] = {}
    for k in range(n):
        hole, board = tuple(keys[k * stride : k * stride + 3]), keys[k * stride + 3]
        orbit = {
            (frozenset(_rank(c) * 3 + p[_suit(c)] for c in hole), _rank(board) * 3 + p[_suit(board)])
            for p in perms
        }
        weight = len(orbit)
        bet = rows[k * width + 1] / scale
        cell = cells.setdefault((inner_category(hole), round_one_column(hole, board)), [0.0, 0.0])
        cell[0] += weight
        cell[1] += weight * bet
    deals = int(sum(c[0] for c in cells.values()))
    assert deals == 15 * math.comb(14, 3), deals

    sheet = SHEET["round_one"]["p0_open"]
    for cat in CATS:
        mass = sum(c[0] for (r, _), c in cells.items() if r == cat)
        bet = sum(c[1] for (r, _), c in cells.items() if r == cat) / mass
        assert abs(bet - sheet[cat.replace(" ", "_")][1]) < 0.005, f"{cat}: the export and the answer sheet disagree"

    def share(pred) -> float:
        return sum(c[0] for key, c in cells.items() if pred(key)) / deals

    def bet_of(pred) -> float:
        mass = sum(c[0] for key, c in cells.items() if pred(key))
        return sum(c[1] for key, c in cells.items() if pred(key)) / mass

    return {
        "deals": deals,
        "bet": bet_of(lambda key: True),
        "rows": {cat: (share(lambda key, c=cat: key[0] == c), sheet[cat.replace(" ", "_")]) for cat in CATS},
        "cols": [(share(lambda key, j=j: key[1] == j), bet_of(lambda key, j=j: key[1] == j)) for j in range(5)],
        "cells": {key: (c[0] / deals, c[1] / c[0]) for key, c in cells.items()},
    }


SPOT = opening_spot()


# ------------------------------------------------------------ a dealt hand

def card_face(card: int) -> tuple[str, bool]:
    """A mini-drawmaha card as rank and suit symbol, and whether the suit is red."""
    return "23456"[_rank(card)] + "♣♦♥"[_suit(card)], _suit(card) != 0


def dealt_hand() -> dict[str, object]:
    """The deal in the first chunk of the Play tab's pack whose opening mix is the most mixed."""
    pack = load("web/rung3-viz/public/deals/index.json")
    deals = load(f"web/rung3-viz/public/deals/{pack['chunks'][0]}")
    best = min(range(len(deals)), key=lambda i: max(deals[i]["r1"][""]))
    deal = deals[best]
    hole = tuple(deal["order"][0])
    board = deal["deck"][6]
    check, bet = (v / 1000 for v in deal["r1"][""])
    return {
        "n": best + 1,
        "of": pack["deals"],
        "hole": hole,
        "board": board,
        "inner": inner_category(hole),
        "reach": REACHING[round_one_column(hole, board)],
        "check": check,
        "bet": bet,
        "release": pack["strategy"]["release"],
        "grade": pack["grade"]["exploitability"],
        "grade_hands": pack["grade"]["handsPerSeat"],
    }


DEALT = dealt_hand()


# ------------------------------------------------------------ releases

def _module_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


# When each release tag was published on GitHub (`gh release list`, read 2026-10-29).
PUBLISHED = {
    "rung3-lcfr-2m": "2026-10-02",
    "rung4-deep-t500": "2026-10-26",
    "rung4-deep-pot0.5-t200": "2026-10-28",
}

ANSWER = ROOT / "src" / "drawmaha_solver" / "minidrawmaha" / "answer"


def releases() -> list[dict[str, object]]:
    """`RELEASES` in releases.py, each with its grade read from the file and key the catalogue names."""
    names = _module_constants(ANSWER / "strategy.py") | _module_constants(ANSWER / "releases.py")
    pack_release = _module_constants(ANSWER / "deal_pack.py")["PACK_RELEASE"]
    out = []
    for node in ast.walk(ast.parse((ANSWER / "releases.py").read_text())):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Release"):
            continue
        fields = {}
        for kw in node.keywords:
            fields[kw.arg] = names[kw.value.id] if isinstance(kw.value, ast.Name) else ast.literal_eval(kw.value)
        entry = load(fields["grade_file"])[fields["grade_key"]]
        hands_per_seat = entry.get("hands_per_seat") or entry["iteration"] * entry["workers"]
        if "config" in entry:
            cfg = entry["config"]
            column = fields["grade_key"].split("/")[2]
            mark = int(fields["grade_key"].split("/")[1])
            what = (
                f"Deep CFR, {cfg['depth']} × {cfg['width']} net, "
                f"{'the policy net Π' if column == 'pi' else 'the exact average'} at iteration {mark}"
            )
            if cfg.get("pot_power") == 0.5:
                what += ", regret targets ÷ √pot"
        else:
            what = f"Tabular {entry['rule'].upper()}, the {entry['column']} average at {entry['iteration']:,} iterations"
        if fields["directory"] is not None:
            shipped = [("/solver", "Range explorer"), ("/learn", "Learn")]
            command = f"uv run minidraw-ranges --release {fields['name']}"
        elif fields["name"] == pack_release:
            shipped = [("/play", "Play")]
            command = "uv run minidraw-deals"
        else:
            shipped = []
            command = None
        out.append(
            {
                "name": fields["name"],
                "tag": fields["tag"],
                "asset": fields["asset"],
                "published": PUBLISHED[fields["tag"]],
                "what": what,
                "grade": entry["exploitability"],
                "hands": hands_per_seat,
                "source": f"{fields['grade_file']} · {fields['grade_key']}",
                "shipped": shipped,
                "command": command,
            }
        )
    assert len(out) == 5, f"releases.py names {len(out)} releases"
    return sorted(out, key=lambda r: (r["published"], r["name"]), reverse=True)


# ===================================================================== charts

def line_chart(
    series: list[dict],
    *,
    w: int,
    h: int,
    xticks: list[tuple[str, float]],
    yticks: list[tuple[str, float]],
    xlab: str,
    ylab: str,
    pad: tuple[int, int, int, int] = (30, 150, 44, 44),
    label: str,
) -> str:
    """Lines on log–log axes, each ending in a direct label (its name, then its last value in muted ink).

    `series`: dicts of `name`, `pts` [(x, y)], `tone` (`acc` / `ink` / `mut`),
    optional `dash` and `end` (the text after the name). The winning series is
    the accent and drawn heavier; the domain is the ticks and the data, so no
    mark falls outside the axes.
    """
    top, right, bottom, left = pad
    # room for the longest end label (Geist at 12 px runs about 6.6 px a character)
    right = max(right, *(int(len(s["name"] + " " + s.get("end", "")) * 6.6) + 18 for s in series if not s.get("below")))
    pw, ph = w - left - right, h - top - bottom
    xs = [x for s in series for x, _ in s["pts"]] + [v for _, v in xticks]
    ys = [y for s in series for _, y in s["pts"]] + [v for _, v in yticks]
    x0, x1 = math.log10(min(xs)), math.log10(max(xs))
    y0, y1 = math.log10(min(ys)), math.log10(max(ys))

    def X(v: float) -> float:
        return left + (math.log10(v) - x0) / (x1 - x0) * pw

    def Y(v: float) -> float:
        return top + (1 - (math.log10(v) - y0) / (y1 - y0)) * ph

    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">']
    for text, v in yticks:
        y = Y(v)
        o.append(f'<line class="ch-grid" x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="ch-ax" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(text)}</text>')
    o.append(f'<line class="ch-base" x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}"/>')
    for text, v in xticks:
        x = X(v)
        o.append(f'<line class="ch-base" x1="{x:.1f}" x2="{x:.1f}" y1="{top + ph}" y2="{top + ph + 4}"/>')
        o.append(f'<text class="ch-ax" x="{x:.1f}" y="{top + ph + 18}" text-anchor="middle">{esc(text)}</text>')
    o.append(f'<text class="ch-ax" x="{left + pw}" y="{top + ph + 36}" text-anchor="end">{esc(xlab)}</text>')
    o.append(f'<text class="ch-ax" x="{left - 8}" y="{top - 14}">{esc(ylab)}</text>')

    ends = []
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in s["pts"])
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        heavy = " ch-heavy" if s["tone"] == "acc" else ""
        o.append(f'<polyline class="ch-line t-{s["tone"]}{heavy}" points="{pts}"{dash}/>')
        # a mark on every measured point, unless the line is a dense curve; then only its end
        marks = s["pts"] if len(s["pts"]) <= 12 else s["pts"][-1:]
        for x, y in marks:
            o.append(f'<circle class="ch-pt f-{s["tone"]}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3"/>')
        lx, ly = s["pts"][-1]
        if s.get("below"):
            # a line that ends early, under another: its label sits beneath its own end, read leftwards
            end = f'<tspan class="ch-end"> {esc(s["end"])}</tspan>' if s.get("end") else ""
            o.append(
                f'<text class="ch-lbl f-{s["tone"]}" x="{X(lx) + 4:.1f}" y="{Y(ly) + 20:.1f}" text-anchor="end">'
                f'{esc(s["name"])}{end}</text>'
            )
        else:
            ends.append([Y(ly), X(lx), s])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 15:
            ends[i][0] = ends[i - 1][0] + 15
    for y, x, s in ends:
        end = f'<tspan class="ch-end"> {esc(s["end"])}</tspan>' if s.get("end") else ""
        o.append(f'<text class="ch-lbl f-{s["tone"]}" x="{x + 10:.1f}" y="{y + 4:.1f}">{esc(s["name"])}{end}</text>')
    o.append("</svg>")
    return "".join(o)


def hbars(rows: list[tuple[str, float, str, str]], *, w: int, label: str, labw: int = 180, maxv: float | None = None) -> str:
    """Magnitude bars on a linear axis from zero, one row each: (name, value, printed value, tone)."""
    barh, gap = 16, 12
    h = len(rows) * (barh + gap) - gap + 2
    pw = w - labw - 80
    maxv = maxv or max(v for _, v, _, _ in rows)
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">']
    y = 0
    for name, v, printed, tone in rows:
        bw = max(v / maxv * pw, 1.5)
        o.append(f'<text class="ch-row" x="{labw - 12}" y="{y + barh / 2 + 4:.1f}" text-anchor="end">{esc(name)}</text>')
        o.append(f'<rect class="f-{tone}" x="{labw}" y="{y}" width="{bw:.1f}" height="{barh}"/>')
        o.append(f'<text class="ch-val" x="{labw + bw + 8:.1f}" y="{y + barh / 2 + 4:.1f}">{esc(printed)}</text>')
        y += barh + gap
    o.append("</svg>")
    return "".join(o)


def figure(n: str | None, title: str, sub: str, body: str, caption: str, *, fid: str | None = None) -> str:
    head = f'<span class="fig-n">{esc(n)}</span>' if n else ""
    ident = f' id="{fid}"' if fid else ""
    return (
        f'<figure class="fig"{ident}><div class="fig-head">{head}<div class="fig-title">{title}</div>'
        f'<div class="fig-sub">{sub}</div></div><div class="fig-body">{body}</div>'
        f'<figcaption class="fig-cap">{caption}</figcaption></figure>'
    )


HAND_TICKS = [(short_hands(v), v) for v in (1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7)]
EXPL_TICKS = [(t, float(t)) for t in ("0.05", "0.1", "0.2", "0.5", "1", "2", "4")]


def equal_hands_chart(w: int, h: int) -> str:
    deep_end = f"{g3(DEEP[-1][1])} at {short_hands(DEEP[-1][0])}"
    lcfr_end = f"{g3(LCFR[-1][1])} at {short_hands(LCFR[-1][0])}"
    return line_chart(
        [
            {"name": "Tabular LCFR", "pts": LCFR, "tone": "ink", "end": lcfr_end},
            {"name": "Deep CFR", "pts": DEEP, "tone": "acc", "end": deep_end, "below": True},
        ],
        w=w,
        h=h,
        xticks=HAND_TICKS,
        yticks=EXPL_TICKS,
        xlab="hands per seat",
        ylab="exploitability, chips per hand",
        pad=(30, 128, 44, 40),
        label="Exploitability by hands per seat: Deep CFR's exact average against tabular LCFR",
    )


def race_chart(w: int, h: int) -> str:
    tones = {"LCFR": ("acc", False), "Vanilla": ("ink", False), "CFR+": ("mut", False), "DCFR": ("mut", True)}
    return line_chart(
        [
            {"name": name, "pts": RACE[name], "tone": tones[name][0], "dash": tones[name][1], "end": g3(RACE[name][-1][1])}
            for name in ("DCFR", "CFR+", "Vanilla", "LCFR")
        ],
        w=w,
        h=h,
        xticks=HAND_TICKS,
        yticks=EXPL_TICKS,
        xlab="hands per seat",
        ylab="exploitability, chips per hand",
        pad=(30, 110, 44, 40),
        label="Exploitability by hands per seat for four regret rules on mini-drawmaha",
    )


# ===================================================================== chrome

NAV = (("research", "Research", "/research"), ("solver", "Solver", "/solver"),
       ("benchmarks", "Benchmarks", "/benchmarks"), ("play", "Play", "/play"), ("rules", "Rules", "/rules"))


def top_bar(here: str) -> str:
    items = "\n".join(
        f'      <a class="navbtn" href="{href}"{" aria-current=\"page\"" if key == here else ""}>{label}</a>'
        for key, label, href in NAV
    )
    return f"""<header class="top">
  <div class="wrap">
    <a class="chip-home" href="/" aria-label="Drawmaha home"><canvas aria-hidden="true"></canvas><span class="word">Drawmaha <small>Solver</small></span></a>
    <nav class="nav" aria-label="Site">
{items}
    </nav>
    <a class="btn primary" href="/solver">Open the solver</a>
  </div>
</header>"""


FOOTER = f"""<footer>
  <div class="wrap">
    <div class="foot-cols">
      <div class="foot-about">
        <a class="chip-end" href="/" aria-label="Drawmaha home"><canvas aria-hidden="true"></canvas></a>
        <p>A research project by Alec Ondo. A heads-up pot-limit Drawmaha solver, built one game at a time, each graded against a known answer. Python, NumPy and torch; training on Modal; the site on Vercel. The code, the graders and the data behind every figure are public.</p>
      </div>
      <div>
        <h4>Product</h4>
        <ul>
          <li><a href="/solver">Range explorer</a></li>
          <li><a href="/play">Play the solver</a></li>
          <li><a href="/rules">Rules of Drawmaha</a></li>
          <li><a href="/releases">Releases</a></li>
        </ul>
      </div>
      <div>
        <h4>Research</h4>
        <ul>
          <li><a href="/research">Mini-drawmaha, solved</a></li>
          <li><a href="/learn">Learn</a></li>
          <li><a href="/benchmarks">Benchmarks</a></li>
          <li><a href="{REPO}/blob/main/paper/drawmaha-ml-survey.pdf">ML survey (PDF)</a></li>
        </ul>
      </div>
      <div>
        <h4>Resources</h4>
        <ul>
          <li><a href="{REPO}">GitHub</a></li>
          <li><a href="{REPO}/tree/main/figures">Figure data</a></li>
          <li><a href="{REPO}/releases">Strategy packs</a></li>
          <li><a href="/cover">The old cover</a></li>
        </ul>
      </div>
    </div>
    <div class="foot-line">
      <span>© 2026 Alec Ondo · Built in public</span>
      <a href="{REPO}">github.com/AlecROndo/drawmaha-solver</a>
    </div>
  </div>
</footer>"""

CHIP = """<script src="/chip.js"></script>
<script>
(function () {
  var colours = { mark: '#16140f', ink: '#8e2038' };
  mountChip(document.querySelector('.chip-home canvas'), { font: 1.5, mark: colours.mark, ink: colours.ink });
  mountChip(document.querySelector('.chip-end canvas'), { font: 1.5, speed: 0.32, mark: colours.mark, ink: colours.ink });
})();
</script>"""

# Shared by every page's inline style: the charts, the links, the cards. Page
# layout follows it per page. Nothing here moves but a link's colour.
BASE_CSS = """
/* Page layout only; the chrome, the tokens and the type come from /theme.css. */
main { padding-bottom: 0; }
.num { font-variant-numeric: tabular-nums; }
.muted { color: var(--ink-3); }
.more { color: var(--ink); font-weight: 500; text-decoration: none; border-bottom: 1px solid var(--rule-2); transition: color 0.15s, border-color 0.15s; }
.more:hover { color: var(--accent); border-color: var(--accent); }
.prose a, .fig-cap a, td a.in { color: var(--accent); text-decoration: underline; text-decoration-color: var(--accent-tint); text-underline-offset: 3px; transition: text-decoration-color 0.15s; }
.prose a:hover, .fig-cap a:hover, td a.in:hover { text-decoration-color: var(--accent); }
code { font-family: var(--mono); font-size: 0.86em; background: var(--paper-2); padding: 1px 5px; border-radius: 3px; color: var(--ink); }
.band { margin-top: 112px; }
.split-aside h2 { margin-top: 0; }
.split-aside p.more-row { margin-top: 20px; }
/* charts: ink, the accent for the finding, muted ink; a dash for a fourth series */
.ch-grid { stroke: var(--rule); stroke-width: 1; }
.ch-base { stroke: var(--rule-2); stroke-width: 1; }
.ch-ax { fill: var(--ink-3); font-size: 11px; }
.ch-row { fill: var(--ink); font-size: 13px; }
.ch-val { fill: var(--ink); font-size: 12px; font-variant-numeric: tabular-nums; }
.ch-line { fill: none; stroke-width: 1.75; stroke-linejoin: round; stroke-linecap: round; }
.ch-line.ch-heavy { stroke-width: 2.5; }
.ch-pt { stroke: var(--paper); stroke-width: 1.5; }
.ch-lbl { font-size: 12px; font-weight: 600; }
.ch-end { fill: var(--ink-3); font-weight: 400; font-variant-numeric: tabular-nums; }
.ch-guide { stroke: var(--ink-3); stroke-width: 1; stroke-dasharray: 3 4; }
.t-acc { stroke: var(--accent); } .t-ink { stroke: var(--ink); } .t-mut { stroke: var(--ink-3); }
.f-acc { fill: var(--accent); } .f-ink { fill: var(--ink); } .f-mut { fill: var(--ink-3); } .f-chk { fill: var(--act-check); } .f-rule { fill: var(--rule-2); }
/* a grade in a table: the number in the serif, what it is in beneath */
.grade b { display: block; font-family: var(--serif); font-weight: 400; font-size: 22px; line-height: 1.1; color: var(--ink); }
.grade span { display: block; margin-top: 5px; font-size: 13px; line-height: 1.4; color: var(--ink-3); }
table.hair tr.dim .grade b { color: var(--ink-3); }
table.hair td.k a { text-decoration: none; }
table.hair td.k a:hover { color: var(--accent); }
/* cards */
.cards { display: flex; gap: 8px; align-items: center; }
.pc { width: 44px; height: 62px; border: 1px solid var(--rule-2); border-radius: 4px; background: #fffdf8; display: flex; align-items: center; justify-content: center; font-size: 16px; font-weight: 500; color: var(--ink); font-variant-numeric: tabular-nums; }
.pc.red { color: var(--accent); }
.pc.out { opacity: 0.4; text-decoration: line-through; }
.pc.blank { background: var(--paper-2); border-style: dashed; color: var(--ink-3); }
/* a mix bar: check / fold in blue, bet in the accent */
.mix { display: flex; height: 8px; gap: 2px; }
.mix i { display: block; height: 100%; }
.mix .c { background: var(--act-check); }
.mix .b { background: var(--accent); }
.key { display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; }
.key::before { content: ''; width: 10px; height: 10px; border-radius: 2px; background: currentColor; }
.key.c { color: var(--act-check); } .key.b { color: var(--accent); }
.key span { color: var(--ink-2); }
@media (max-width: 1024px) {
  .band { margin-top: 72px; }
}
"""


def page(*, title: str, description: str, here: str, css: str, body: str, sources: str, script: str = "") -> str:
    return f"""<!doctype html>
<!-- Written by {ME} from {sources}.
     Do not edit by hand: change the generator and run `uv run python {ME}`. -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="stylesheet" href="/fonts.css">
<link rel="stylesheet" href="/theme.css">
<style>{BASE_CSS}{css}</style>
</head>
<body>
{top_bar(here)}
<main class="wrap">
{body}
</main>
{FOOTER}
{CHIP}{script}
</body>
</html>
"""


def cards(labels: list[int | None], *, out: set[int] = frozenset()) -> str:
    o = []
    for card in labels:
        if card is None:
            o.append('<span class="pc blank" aria-label="not yet dealt">?</span>')
            continue
        face, red = card_face(card)
        cls = "pc" + (" red" if red else "") + (" out" if card in out else "")
        o.append(f'<span class="{cls}">{face}</span>')
    return f'<span class="cards">{"".join(o)}</span>'


def mix_bar(check: float, bet: float) -> str:
    return f'<span class="mix" aria-hidden="true"><i class="c" style="flex:{check:.4f}"></i><i class="b" style="flex:{bet:.4f}"></i></span>'


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def pct1(v: float) -> str:
    return f"{v * 100:.1f}%" if v < 0.1 else f"{v * 100:.0f}%"


# ===================================================================== tables

RESULTS = readme_results()


def results_table(link: str) -> str:
    """The README's results, one row a game. `link` is `bench` for the Benchmarks page's own anchors."""
    rows = []
    for r in RESULTS:
        href = f"#{r['anchor']}" if link == "bench" else f"/benchmarks#{r['anchor']}"
        dim = ' class="dim"' if r["grade"] == "—" else ""
        # The archived Cover (and any old link to /#rung4) points at the old
        # homepage's Deep CFR step; on the homepage that anchor is this row.
        ident = ' id="rung4"' if link == "home" and r["anchor"] == "deep" else ""
        rows.append(
            f'<tr{ident}{dim}><td class="k"><a href="{href}">{esc(r["game"])}</a></td><td>{esc(r["method"])}</td>'
            f'<td>{esc(r["against"])}</td><td class="grade num"><b>{esc(r["grade"])}</b><span>{esc(r["note"])}</span></td></tr>'
        )
    return (
        '<table class="hair results"><thead><tr><th>Game</th><th>Method</th><th>Graded against</th>'
        f'<th>Exploitability</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


# ===================================================================== home

def matrix() -> str:
    max_bet = max(bet for _, bet in SPOT["cells"].values())
    o = ['<div class="mx" role="table" aria-label="Seat 0\'s opening range: inner hand by what the hole reaches for with the board card">']
    o.append('<div class="mx-corner" role="columnheader"><span>held ↓ · with the board →</span></div>')
    for name, (share, bet) in zip(COLS, SPOT["cols"]):
        o.append(f'<div class="mx-ch" role="columnheader"><span>{esc(name)}</span><span class="num muted">{pct1(share)} · bet {pct(bet)}</span></div>')
    for cat in CATS:
        share, (check, bet) = SPOT["rows"][cat]
        o.append(
            f'<div class="mx-rh" role="rowheader"><div class="mx-rt"><span>{esc(cat)}</span><span class="num muted">{pct1(share)}</span></div>'
            f'{mix_bar(check, bet)}<div class="mx-rm num">check {pct(check)} · bet {pct(bet)}</div></div>'
        )
        for j in range(5):
            cell = SPOT["cells"].get((cat, j))
            if cell is None:
                o.append('<div class="mx-c empty" role="cell" aria-label="no hands"></div>')
                continue
            cshare, cbet = cell
            tint = round(cbet / max_bet * 100)
            o.append(
                f'<div class="mx-c" role="cell" style="background:color-mix(in srgb, var(--accent-tint) {tint}%, var(--paper))">'
                f'<span class="mx-v num">{pct1(cshare)}</span><span class="mx-b num">bet {pct(cbet)}</span></div>'
            )
    o.append("</div>")
    return "".join(o)


RESEARCH_DATES = {  # merge dates from `git log`: #67 and #68 on 2026-10-28; the survey added 2026-08-29
    "write-up": "Oct 2026",
    "pot": "Oct 2026",
    "survey": "Aug 2026",
    "benchmarks": "Oct 2026",
}


def home() -> str:
    lcfr_at_deep = at(LCFR, DEEP_HANDS)
    raw_at_deep = at(deep_marks(RAW, "exact"), DEEP_HANDS)
    pot_one_at_deep = at(deep_marks(POT_ONE, "exact"), DEEP_HANDS)
    cfrplus, dcfr = RACE["CFR+"][-1][1], RACE["DCFR"][-1][1]
    best_col = max(range(5), key=lambda j: SPOT["cols"][j][1])
    hero_fig = figure(
        None,
        "Deep CFR is below the table at every hand count tried",
        "Exploitability of the average strategy on mini-drawmaha, log scales",
        equal_hands_chart(568, 330),
        f"At {hands(DEEP_HANDS)} hands per seat the Deep CFR exact average grades {g3(DEEP[-1][1])} chips per hand; "
        f"tabular LCFR grades {g3(lcfr_at_deep)} and needs {hands(LCFR[-1][0])} to reach {g3(LCFR[-1][1])}. "
        "One exact best-response walk grades both.",
    )
    hole_face = " ".join(card_face(c)[0] for c in DEALT["hole"])
    body = f"""
<section class="hero">
  <div class="hero-copy">
    <p class="eyebrow">A research project · Deep CFR on mini-drawmaha</p>
    <h1>A solver for a poker game nobody has solved.</h1>
    <p class="lede">Drawmaha is a split-pot draw and Omaha hybrid. This solver is built one game at a time: four smaller games, each solved and graded against a known answer before the next is started.</p>
    <div class="cta"><a class="btn primary" href="/solver">Open the range explorer</a><a class="more" href="/research">Read the research →</a></div>
  </div>
  <div class="hero-fig">{hero_fig}</div>
</section>

<section class="band split" id="results">
  <div class="split-aside">
    <p class="eyebrow">Results</p>
    <h2>Four games solved, each graded by exploitability</h2>
    <p>Exploitability is what a best response wins per hand against the solver's average strategy. Zero is a Nash equilibrium. Every grader reads the game's rules and nothing from the solver it grades.</p>
    <p class="more-row"><a class="more" href="/benchmarks">All benchmarks →</a></p>
  </div>
  <div class="split-main">{results_table("home")}</div>
</section>

<section class="band split" id="explorer">
  <div class="split-aside">
    <p class="eyebrow">Range explorer</p>
    <h2>Seat 0 opens with a bet {pct(SPOT["bet"])} of the time, most often when it {REACHING[best_col]}</h2>
    <p>Rows are what the three held cards make on their own; columns are what they reach for with the board card. Each cell is its share of every opening hand and how often that share bets. The explorer opens any cell down to a single hand, at every spot of the game.</p>
    <p class="more-row"><a class="more" href="/solver">Open the range explorer →</a></p>
  </div>
  <div class="split-main">
    <figure class="fig">
      <div class="fig-head"><div class="fig-title">Seat 0 to act, round one, pot 2 · tabular LCFR, exploitability {g3(SHEET["frozen_strategy"]["exploitability"])}</div>
      <div class="fig-sub">All {SPOT["deals"]:,} deals of a hand and a board card, equally weighted</div></div>
      <div class="fig-body">
        <div class="spot-mix">{mix_bar(1 - SPOT["bet"], SPOT["bet"])}<div class="spot-keys"><span class="key c"><span>check {pct(1 - SPOT["bet"])}</span></span><span class="key b"><span>bet the pot {pct(SPOT["bet"])}</span></span></div></div>
        {matrix()}
      </div>
      <figcaption class="fig-cap">Read from the frozen table's export at the opening spot; the row mixes are the answer sheet's. A deeper tint bets more; a dashed cell holds no hands.</figcaption>
    </figure>
  </div>
</section>

<section class="band split" id="research">
  <div class="split-aside">
    <p class="eyebrow">Research</p>
    <h2>Keeping negative regret wins the race, and a network passes the table at equal hands</h2>
    <p>One long write-up with its data and figures, a survey of the machine learning behind it, and the benchmarks every method was checked on first.</p>
  </div>
  <div class="split-main">
    <table class="hair index"><tbody>
      <tr><td class="when num">{RESEARCH_DATES["write-up"]}</td><td><a class="title" href="/research">Mini-drawmaha, solved</a><span>Linear CFR grades {g3(LCFR[-1][1])} at {hands(LCFR[-1][0])} hands per seat, against {g3(cfrplus)} and {g3(dcfr)} for the two rules that drop negative regret; a Deep CFR run passes the table at every hand count tried.</span></td><td class="kind">Write-up</td></tr>
      <tr><td class="when num">{RESEARCH_DATES["pot"]}</td><td><a class="title" href="/research#ch-06">Regret targets divided by √pot train a better network</a><span>{g3(DEEP[-1][1])} against {g3(raw_at_deep)} undivided at {hands(DEEP_HANDS)} hands per seat, same network, same steps; dividing by the whole pot gives {g3(pot_one_at_deep)}.</span></td><td class="kind">Chapter</td></tr>
      <tr><td class="when num">{RESEARCH_DATES["survey"]}</td><td><a class="title" href="{REPO}/blob/main/paper/drawmaha-ml-survey.pdf">The machine learning behind the solver</a><span>A survey: regret matching, CFR and its sampled and discounted variants, Deep CFR, and how each is graded.</span></td><td class="kind">PDF</td></tr>
      <tr><td class="when num">{RESEARCH_DATES["benchmarks"]}</td><td><a class="title" href="/benchmarks">Rock-paper-scissors, Kuhn and Leduc, graded</a><span>The three small games against a closed form and an OpenSpiel solve, each with a live demo.</span></td><td class="kind">Benchmarks</td></tr>
    </tbody></table>
  </div>
</section>

<section class="band split" id="play">
  <div class="split-aside">
    <p class="eyebrow">Play</p>
    <h2>Act against the solver, with its answer beside yours</h2>
    <p>A hand is dealt from a pack of {DEALT["of"]}. You act; the coach shows how the solver mixes at the same spot with the same cards. Settled hands go to a ledger.</p>
    <p class="more-row"><a class="more" href="/play">Play the solver →</a></p>
  </div>
  <div class="split-main">
    <figure class="fig deal">
      <div class="fig-head"><div class="fig-title">Deal {DEALT["n"]} of {DEALT["of"]}: with {hole_face} the solver checks {pct(DEALT["check"])} and bets {pct(DEALT["bet"])}</div>
      <div class="fig-sub">Seat 0 to act, pot 2 · Deep CFR exact average, release <code>{esc(DEALT["release"])}</code>, exploitability {g3(DEALT["grade"])}</div></div>
      <div class="fig-body deal-body">
        <div>
          <div class="deal-row"><div><p class="eyebrow">Your hand</p>{cards(list(DEALT["hole"]))}</div><div><p class="eyebrow">Board</p>{cards([DEALT["board"], None])}</div></div>
          <p class="deal-read">{esc(DEALT["inner"]).capitalize()} in hand; with the board card it {esc(DEALT["reach"])}.</p>
          <div class="deal-acts"><a class="btn" href="/play">Check</a><a class="btn" href="/play">Bet the pot</a></div>
        </div>
        <div class="coach">
          <p class="eyebrow">Coach · the solver's mix here</p>
          <div class="band-row"><span>check</span><span class="track"><i class="c" style="width:{DEALT["check"] * 100:.1f}%"></i></span><span class="num">{pct(DEALT["check"])}</span></div>
          <div class="band-row"><span>bet</span><span class="track"><i class="b" style="width:{DEALT["bet"] * 100:.1f}%"></i></span><span class="num">{pct(DEALT["bet"])}</span></div>
        </div>
      </div>
      <figcaption class="fig-cap">Read from the Play tab's deal pack: the strategy's mix at this node of this deal. The three held cards are scored at showdown as they stand; two of them with both board cards make the other half.</figcaption>
    </figure>
  </div>
</section>

<section class="band" id="method">
  <p class="eyebrow">Method</p>
  <div class="method">
    <div><h3>Graded by an exact best response</h3><p>Each game is solved, then a best response is computed exactly against the average strategy. What it wins per hand is the grade. A game is not called solved on a learning curve.</p></div>
    <div><h3>A grader shares no code with its solver</h3><p>Every grader imports the game's rules and nothing else, so a bug cannot sit in both and cancel. Hidden cards are summed over before the responder takes a maximum.</p></div>
    <div><h3>Every figure is regenerated</h3><p>Python, NumPy and torch; training runs on Modal and the site on Vercel. Every chart and table here is written from committed data by a named command, never edited by hand.</p></div>
  </div>
</section>
"""
    css = """
.hero { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); column-gap: 24px; align-items: end; padding-top: 72px; }
.hero-copy { grid-column: 1 / 7; padding-right: 24px; }
.hero-fig { grid-column: 7 / 13; }
.hero h1 { font-size: 64px; line-height: 1.02; letter-spacing: -0.015em; max-width: none; }
.hero .lede { margin-top: 28px; max-width: 520px; }
.cta { display: flex; gap: 24px; align-items: center; margin-top: 36px; }
.hero .fig { border-top: 0; padding-top: 0; }
table.results th:nth-child(1) { width: 22%; } table.results th:nth-child(2) { width: 20%; } table.results th:nth-child(4) { width: 30%; }
.spot-mix { margin-bottom: 18px; }
.spot-mix .mix { height: 10px; }
.spot-keys { margin-top: 8px; font-size: 13px; }
.mx { display: grid; grid-template-columns: 150px repeat(5, minmax(0, 1fr)); gap: 4px; }
.mx-corner, .mx-ch { border-bottom: 1px solid var(--ink); padding: 0 0 8px 4px; display: flex; flex-direction: column; justify-content: flex-end; gap: 2px; font-size: 12px; line-height: 1.3; color: var(--ink-2); }
.mx-corner { padding-left: 0; color: var(--ink-3); }
.mx-ch .muted { font-size: 11px; }
.mx-rh { padding: 8px 12px 8px 0; font-size: 13px; }
.mx-rt { display: flex; justify-content: space-between; margin-bottom: 6px; }
.mx-rh .mix { height: 5px; }
.mx-rm { margin-top: 4px; font-size: 11px; color: var(--ink-3); }
.mx-c { min-height: 64px; border: 1px solid var(--rule); padding: 8px; display: flex; flex-direction: column; justify-content: space-between; }
.mx-c.empty { border-style: dashed; background: transparent; }
.mx-v { font-size: 16px; font-weight: 500; color: var(--ink); }
.mx-b { font-size: 11px; color: var(--ink-2); }
table.index td { padding: 18px 12px 18px 0; }
table.index td.when { width: 96px; color: var(--ink-3); }
table.index td.kind { width: 100px; text-align: right; padding-right: 0; color: var(--ink-3); font-size: 13px; }
table.index .title { display: block; font-family: var(--serif); font-size: 20px; line-height: 1.25; color: var(--ink); text-decoration: none; transition: color 0.15s; }
table.index .title:hover { color: var(--accent); }
table.index td span { display: block; margin-top: 6px; max-width: 56ch; color: var(--ink-3); font-size: 14px; }
.deal-body { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 40px; }
.deal-row { display: flex; gap: 32px; }
.deal-row .eyebrow { margin-bottom: 10px; }
.deal-read { margin: 12px 0 0; font-size: 13px; color: var(--ink-3); }
.deal-acts { display: flex; gap: 10px; margin-top: 20px; }
.coach .eyebrow { margin-bottom: 14px; }
.band-row { display: grid; grid-template-columns: 52px minmax(0, 1fr) 44px; align-items: center; gap: 10px; font-size: 13px; margin-bottom: 10px; }
.band-row .num { text-align: right; }
.track { height: 10px; background: var(--paper-2); display: block; }
.track i { display: block; height: 100%; }
.track .c { background: var(--act-check); } .track .b { background: var(--accent); }
.method { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-top: 1px solid var(--ink); padding-top: 24px; }
.method > div { padding: 0 24px; }
.method > div:first-child { padding-left: 0; }
.method > div + div { border-left: 1px solid var(--rule); }
.method p { margin: 10px 0 0; color: var(--ink-2); }
@media (max-width: 1024px) {
  .hero { grid-template-columns: minmax(0, 1fr); row-gap: 40px; padding-top: 48px; }
  .hero-copy, .hero-fig { grid-column: 1 / -1; }
  .hero h1 { font-size: 48px; }
  .method { grid-template-columns: minmax(0, 1fr); row-gap: 24px; }
  .method > div, .method > div + div { padding: 0; border-left: 0; }
}
@media (max-width: 768px) {
  .mx { grid-template-columns: 110px repeat(5, minmax(0, 1fr)); font-size: 12px; }
  .deal-body { grid-template-columns: minmax(0, 1fr); }
}
"""
    return page(
        title="Drawmaha Solver",
        description="A solver for heads-up pot-limit Drawmaha, the split-pot draw and Omaha hybrid nobody has solved, built one game at a time and graded by exploitability.",
        here="home",
        css=css,
        body=body,
        sources="README.md, figures/rung3/, figures/rung4/, web/solver-viz/public/mini/ and web/rung3-viz/public/deals/",
    )


# ===================================================================== rules

def rules() -> str:
    holdings = math.comb(52, 5)
    holdem, omaha = math.comb(52, 2), math.comb(52, 4)
    discards = 2**5

    def big(faces: str, out: str = "") -> str:
        o = []
        for face in faces.split():
            red = face[-1] in "♥♦"
            o.append(f'<span class="pc{" red" if red else ""}{" out" if face == out else ""}">{face}</span>')
        return f'<span class="cards">{"".join(o)}</span>'

    streets = [
        ("Deal", "Five cards to each player, face down. Both ante: a bomb pot, so there is no betting before the flop.", "no betting"),
        ("Flop", "Three cards to the board, face up and shared.", "a betting round"),
        ("Draw", "Each player discards none to all five and takes as many back. One card drawn comes face up; keep it, or take the next one face down.", "no betting"),
        ("Turn", "A fourth card to the board.", "a betting round"),
        ("River", "The fifth and last board card.", "a betting round"),
    ]
    ruler = "".join(
        f'<li><span class="st-n num">{i}</span><span class="st-t">{name}</span><span class="st-d">{text}</span>'
        f'<span class="st-bet{" on" if bet != "no betting" else ""}">{bet}</span></li>'
        for i, (name, text, bet) in enumerate(streets, 1)
    )
    body = f"""
<header class="page-head">
  <div class="page-head-copy">
    <p class="eyebrow">Rules</p>
    <h1>Five cards, one draw, and a pot that pays twice.</h1>
    <p class="lede">Drawmaha is a split-pot draw and Omaha hybrid, played here heads-up and pot-limit. Each player holds five cards, a board is dealt as in Omaha with one draw after the flop, and at showdown the same five cards are read twice: half the pot to the best Omaha hand, half to the best five-card draw hand.</p>
  </div>
</header>

<section class="band first" id="streets">
  <figure class="fig">
    <div class="fig-head"><span class="fig-n">Figure 1</span><div class="fig-title">Five streets and three betting rounds, then a showdown read twice</div>
    <div class="fig-sub">The order of one hand; the accent marks a betting round</div></div>
    <div class="fig-body"><ol class="ruler">{ruler}<li class="show"><span class="st-n num" aria-hidden="true">&#160;</span><span class="st-t">Showdown</span><span class="st-d">Half the pot to the best Omaha hand, half to the best draw hand. A tie for a half splits that half.</span></li></ol></div>
    <figcaption class="fig-cap">Pot-limit throughout: a bet is at most the pot. The draw is the street Omaha never has, and it is where the tree grows.</figcaption>
  </figure>
</section>

<section class="band split" id="hand">
  <div class="split-aside">
    <p class="eyebrow">One hand, read twice</p>
    <h2>The same five cards make a king-high straight flush one way and a pair of sevens the other</h2>
    <p>Dealt A♠ K♠ 7♥ 7♦ 3♣. The flop comes Q♠ J♠ 7♣; the 3♣ is thrown and the 10♠ drawn; the turn is the 4♦ and the river the 9♠.</p>
  </div>
  <div class="split-main">
    <figure class="fig">
      <div class="fig-head"><span class="fig-n">Figure 2</span><div class="fig-title">One holding takes the Omaha half and loses the draw half</div>
      <div class="fig-sub">The hand after the draw and the full board, scored by both rules</div></div>
      <div class="fig-body">
        <div class="hand-row"><div><p class="eyebrow">Hand after the draw</p>{big("A♠ K♠ 10♠ 7♥ 7♦")}</div><div><p class="eyebrow">Board</p>{big("Q♠ J♠ 7♣ 4♦ 9♠")}</div></div>
        <div class="reads">
          <table class="hair"><thead><tr><th>Omaha half</th><th class="r">Result</th></tr></thead><tbody>
            <tr><td class="k">K♠ 10♠ + Q♠ J♠ 9♠</td><td class="r n">King-high straight flush</td></tr>
            <tr><td colspan="2">Exactly two from the hand and exactly three from the board. The A♠ cannot join: only two held cards may play.</td></tr>
          </tbody></table>
          <table class="hair"><thead><tr><th>Draw half</th><th class="r">Result</th></tr></thead><tbody>
            <tr><td class="k">A♠ K♠ 10♠ 7♥ 7♦</td><td class="r n">Pair of sevens</td></tr>
            <tr><td colspan="2">The five held cards as they stand after the draw, no board. Against 9♦ 8♥ 7♠ 6♣ 5♦, a nine-high straight, this half is lost.</td></tr>
          </tbody></table>
        </div>
      </div>
      <figcaption class="fig-cap">Every bet in the hand was made with both readings in mind. A holding right for one half and wrong for the other is the ordinary case.</figcaption>
    </figure>
  </div>
</section>

<section class="band" id="size">
  <p class="eyebrow">Why it is unsolved</p>
  <h2 class="size-h">Two rankings on one pot, and a draw that multiplies the tree</h2>
  <div class="tiles">
    <div><span class="v num">{holdings:,}</span><span class="l">five-card holdings a player can be dealt, against {holdem:,} in hold'em and {omaha:,} in Omaha</span></div>
    <div><span class="v num">{discards}</span><span class="l">ways to discard from each holding at the draw, each opening onto fresh cards off the deck</span></div>
    <div><span class="v num">2</span><span class="l">rankings on one pot: an Omaha hand and a draw hand, read from the same five cards</span></div>
  </div>
  <p class="close prose">The solver works up to the full game through smaller ones with known answers. Mini-drawmaha keeps the draw and the split pot on three cards and a fifteen-card deck, and it is solved: <a href="/research">read the research</a>, <a href="/benchmarks">see the benchmarks</a>, or <a href="/play">play the solver</a>.</p>
</section>
"""
    css = """
.band.first { margin-top: 48px; }
.band.first > .fig { border-top: 0; padding-top: 0; }
.ruler { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); border-top: 1px solid var(--ink); }
.ruler li { display: flex; flex-direction: column; padding: 18px 20px 0 0; margin-right: 20px; border-right: 1px solid var(--rule); }
.ruler li:last-child { border-right: 0; margin-right: 0; }
.st-n { font-family: var(--serif); font-size: 30px; line-height: 1; margin-bottom: 10px; color: var(--ink); }
.st-t { font-weight: 500; margin-bottom: 4px; }
.st-d { font-size: 13px; line-height: 1.45; color: var(--ink-3); flex: 1; }
.st-bet { margin-top: 16px; padding: 10px 0 14px; border-top: 1px solid var(--rule); font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3); }
.st-bet.on { color: var(--accent); border-top: 2px solid var(--accent); padding-top: 9px; }
.ruler li.show .st-n { color: var(--ink-3); }
.hand-row { display: flex; gap: 40px; align-items: flex-end; }
.hand-row .eyebrow { margin-bottom: 10px; }
.reads { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 32px; margin-top: 28px; }
.reads td.r, .reads td.k { white-space: nowrap; }
.size-h { margin-top: 0; max-width: 24ch; }
.tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-top: 1px solid var(--ink); margin-top: 28px; }
.tiles > div { padding: 22px 24px 22px 0; }
.tiles > div + div { border-left: 1px solid var(--rule); padding-left: 24px; }
.tiles .v { display: block; font-family: var(--serif); font-size: 44px; line-height: 1; letter-spacing: -0.02em; color: var(--ink); }
.tiles .l { display: block; margin-top: 10px; font-size: 13px; line-height: 1.45; color: var(--ink-3); max-width: 32ch; }
.close { margin: 28px 0 0; max-width: 640px; font-family: var(--serif); font-size: 18px; line-height: 1.6; color: var(--ink-2); }
@media (max-width: 1024px) {
  .ruler { grid-template-columns: repeat(3, minmax(0, 1fr)); row-gap: 20px; }
  .ruler li:nth-child(3n) { border-right: 0; }
  .reads { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 768px) {
  .ruler { grid-template-columns: minmax(0, 1fr); }
  .ruler li { border-right: 0; margin-right: 0; }
  .tiles { grid-template-columns: minmax(0, 1fr); }
  .tiles > div + div { border-left: 0; padding-left: 0; border-top: 1px solid var(--rule); }
  .hand-row { flex-direction: column; align-items: flex-start; gap: 20px; }
}
"""
    return page(
        title="Rules of Drawmaha · Drawmaha Solver",
        description="Drawmaha is a split-pot draw and Omaha hybrid: five cards each, an Omaha board with one draw after the flop, and a pot paid half to the best Omaha hand and half to the best draw hand.",
        here="rules",
        css=css,
        body=body,
        sources="README.md (the target game)",
    )


# ===================================================================== benchmarks

def benchmarks() -> str:
    rps = readme_rps()
    kuhn = load("web/rung1-viz/src/data/solve.json")
    leduc = load("web/rung2-viz/src/data/solve.json")
    kuhn_rows = readme_kuhn_table()

    # Rock-paper-scissors: the self-play average against the uniform Nash.
    throws = ("Rock", "Paper", "Scissors")
    rps_chart = hbars(
        [(t, v, f"{v:.3f}", "acc") for t, v in zip(throws, rps["average"])],
        w=420,
        labw=90,
        maxv=0.5,
        label="Self-play average strategy of rock-paper-scissors",
    )

    # Kuhn: the committed solve's two curves, every checkpoint.
    k_iter = kuhn["iterations"]
    avg = list(zip(k_iter, kuhn["exploitabilityAverage"]))
    cur = list(zip(k_iter, kuhn["exploitabilityCurrent"]))
    kuhn_chart = line_chart(
        [
            {"name": "Current strategy", "pts": cur, "tone": "mut", "end": f"{cur[-1][1]:.3f}"},
            {"name": "Average strategy", "pts": avg, "tone": "acc", "end": f"{avg[-1][1]:.5f}"},
        ],
        w=520,
        h=300,
        xticks=[(short_hands(v), v) for v in (1, 10, 100, 1e3, 1e4, 1e5)],
        yticks=[(t, float(t)) for t in ("0.001", "0.01", "0.1", "0.5")],
        xlab="iterations",
        ylab="exploitability, chips per hand",
        pad=(30, 150, 44, 48),
        label="Kuhn poker: exploitability of the average and current strategies by iteration",
    )
    kuhn_table = "".join(
        f'<tr><td class="n num">{esc(i)}</td><td class="r n num">{esc(a)}</td><td class="r num">{esc(c)}</td><td class="r num">{esc(v)}</td></tr>'
        for i, a, c, v in kuhn_rows
    )

    # Leduc: the committed solve's two strategies after its run.
    l_avg, l_cur = leduc["exploitabilityAverage"], leduc["exploitabilityCurrent"]
    leduc_chart = hbars(
        [("Current strategy", l_cur, f"{l_cur:.3f}", "mut"), ("Average strategy", l_avg, f"{l_avg:.4f}", "acc")],
        w=520,
        labw=130,
        label="Leduc poker: exploitability of the current and average strategies",
    )
    ratio = l_cur / l_avg

    lcfr_at_deep = at(LCFR, DEEP_HANDS)
    cfrplus, dcfr, vanilla = RACE["CFR+"][-1][1], RACE["DCFR"][-1][1], RACE["Vanilla"][-1][1]
    pi_deep = at(deep_marks(POT_HALF, "pi"), DEEP_HANDS)

    rail = "".join(
        f'<li><a href="#{a}"{" aria-current=\"true\"" if i == 0 else ""}>{esc(t)}</a></li>'
        for i, (a, t) in enumerate(
            [("rps", "Rock-paper-scissors"), ("kuhn", "Kuhn poker"), ("leduc", "Leduc poker"),
             ("mini", "Mini-drawmaha"), ("deep", "Deep CFR"), ("full", "Full Drawmaha")]
        )
    )
    body = f"""
<header class="page-head">
  <div class="page-head-copy">
    <p class="eyebrow">Benchmarks</p>
    <h1>Four games with known answers, and how far each solver is from them.</h1>
    <p class="lede">Each solver is graded by exploitability: the chips per hand a best response wins against its average strategy. Zero is a Nash equilibrium. Each game was solved and graded before the next was started.</p>
  </div>
</header>

<section class="results-top">{results_table("bench")}</section>

<div class="bench">
  <nav class="rail" aria-label="On this page"><p class="k">On this page</p><ol>{rail}</ol></nav>
  <div class="bench-main">

  <section class="b" id="rps">
    <p class="eyebrow">Benchmark 1 · Rock-paper-scissors</p>
    <h2>Regret matching finds the uniform Nash of rock-paper-scissors</h2>
    <p class="sub">Two regret matchers play each other. Each keeps a ledger of how much better every throw would have done, and mixes in proportion to the positive part. The current strategy wanders; the average converges.</p>
    <div class="two">
      {figure(None, f"After {rps['iterations']} rounds the average throws each a third of the time", "Self-play average strategy, share of throws; the dashed line is the uniform Nash", rps_chart.replace("</svg>", _third_line(420, 90, 0.5, 3) + "</svg>"), f"Exploitable for {rps['grade']} chips a round. Against a 50%-rock opponent the same ledger settles on pure paper and earns +{rps['earned']} a round, where the best response earns +{rps['best']}.")}
      <div class="kv">
        <div><b class="num">{rps["grade"]}</b><span>chips a round exploitable, after {rps["iterations"]} rounds</span></div>
        <div><b class="num">+{rps["earned"]}</b><span>a round against 50% rock, of the +{rps["best"]} a best response earns</span></div>
        <p><a class="more" href="/rung0">Run it live →</a></p>
      </div>
    </div>
  </section>

  <section class="b" id="kuhn">
    <p class="eyebrow">Benchmark 2 · Kuhn poker</p>
    <h2>Vanilla CFR on Kuhn poker reaches {avg[-1][1]:.5f} of the closed-form equilibrium</h2>
    <p class="sub">Three cards, one bet, twelve information sets. The equilibrium is a one-parameter family known in closed form, and the game is small enough that all 64 pure strategies are enumerated for the grade. The game value is −1/18 for the first player.</p>
    <div class="two">
      {figure(None, "The average strategy falls steadily; the current one never settles", "Exploitability by iteration, vanilla CFR, log scales", kuhn_chart, "The current iterate cycles around 0.2 and is worse at 1,000 iterations than at 10. Only the average converges, which is why every strategy on this site is an average.")}
      <div>
        <table class="hair"><thead><tr><th>Iterations</th><th class="r">Average</th><th class="r">Current</th><th class="r">Game value</th></tr></thead><tbody>{kuhn_table}</tbody></table>
        <p class="note">Exploitability in chips per hand. At 100 iterations the game value is within 0.0009 of −1/18 and looks solved, while the strategy is still 15 times more exploitable than at 10,000.</p>
        <p class="go"><a class="more" href="/rung1">Run it live →</a></p>
      </div>
    </div>
  </section>

  <section class="b" id="leduc">
    <p class="eyebrow">Benchmark 3 · Leduc poker</p>
    <h2>Vanilla CFR on Leduc poker grades {l_avg:.3f} against an OpenSpiel solve</h2>
    <p class="sub">Six cards, two rounds, one community card, 288 information sets. There is no closed form, so the reference is OpenSpiel's solve of the same game, frozen in the repository as a fixture.</p>
    <div class="two">
      {figure(None, f"After {leduc['iterations']:,} iterations the average is {ratio:.0f} times less exploitable than the current strategy", "Exploitability in chips per hand, vanilla CFR, the committed solve", leduc_chart, "Both strategies come from the same run; only the average is the answer.")}
      <div>
        <table class="hair"><thead><tr><th>Measure</th><th class="r">This solver</th><th class="r">OpenSpiel</th></tr></thead><tbody>
          <tr><td class="k">Exploitability, chips per hand</td><td class="r n num">{l_avg:.4f}</td><td class="r num">—</td></tr>
          <tr><td class="k">Game value, seat 0</td><td class="r n num">{leduc["gameValue"]:.4f}</td><td class="r num">{leduc["gameValueExact"]:.4f}</td></tr>
          <tr><td class="k">Information sets</td><td class="r n num">288</td><td class="r num">288</td></tr>
        </tbody></table>
        <p class="note">The best response sums over the hidden hands a spot could be, then takes its maximum: a maximum inside the sum would be an opponent who sees face-down cards.</p>
        <p class="go"><a class="more" href="/rung2">Run it live →</a></p>
      </div>
    </div>
  </section>

  <section class="b" id="mini">
    <p class="eyebrow">Benchmark 4 · Mini-drawmaha</p>
    <h2>Linear CFR on mini-drawmaha grades {g3(LCFR[-1][1])} at {hands(LCFR[-1][0])} hands per seat</h2>
    <p class="sub">6,220,050 information sets, sampled by lockstep MCCFR, four regret rules raced on one seed. The grader is an exact best-response walk; no outside engine plays this game.</p>
    {figure(None, f"Linear CFR is four times less exploitable than CFR+ and DCFR at {hands(LCFR[-1][0])} hands per seat", "Exploitability of the linear average by hands per seat, seed 0, log scales", race_chart(896, 360), f"Vanilla ends at {g3(vanilla)}, CFR+ at {g3(cfrplus)} and DCFR at {g3(dcfr)}; uniform random play is {UNIFORM:.2f}. The full write-up is <a href=\"/research\">on the research page</a>.")}
  </section>

  <section class="b" id="deep">
    <p class="eyebrow">Benchmark 5 · Deep CFR</p>
    <h2>Deep CFR on mini-drawmaha grades {g3(DEEP[-1][1])} at {hands(DEEP_HANDS)} hands per seat</h2>
    <p class="sub">The same sampler and the same grader, with the regret table replaced by a 4 × 256 network per seat and its regret targets divided by √pot. Graded at equal hands per seat against the table.</p>
    {figure(None, "Deep CFR is below the table at every hand count tried", "Exploitability at equal hands per seat, exact average against tabular LCFR, log scales", equal_hands_chart(896, 340), f"At {hands(DEEP_HANDS)} hands per seat the table grades {g3(lcfr_at_deep)}. The run's policy network Π grades {g3(pi_deep)} at the same mark; the chapter on the network is <a href=\"/research#ch-06\">on the research page</a>.")}
  </section>

  <section class="b" id="full">
    <p class="eyebrow">Next · Full Drawmaha</p>
    <h2>Full Drawmaha has not been started</h2>
    <p class="sub">2,598,960 holdings, 32 discards from each, two rankings on one pot. The plan is Deep CFR at scale, graded by head-to-head checkpoint matches and a trained exploiter, since no exact walk crosses the full tree. The range explorer runs the full game already, with an illustrative strategy that it labels as such.</p>
    <p><a class="more" href="/solver">Open the range explorer →</a></p>
  </section>

  </div>
</div>
"""
    css = """
.results-top { margin-top: 40px; }
table.results th:nth-child(1) { width: 20%; } table.results th:nth-child(2) { width: 20%; } table.results th:nth-child(4) { width: 28%; }
.bench { display: grid; grid-template-columns: 200px minmax(0, 1fr); gap: 64px; margin-top: 88px; }
.bench-main { min-width: 0; }
.b { margin-bottom: 96px; scroll-margin-top: 24px; }
.b:last-child { margin-bottom: 0; }
.b h2 { margin: 0 0 12px; max-width: 30ch; }
.b .sub { margin: 0 0 28px; max-width: 640px; color: var(--ink-2); }
.two { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 40px; align-items: start; }
.two > figure.fig { border-top: 1px solid var(--ink); }
.kv { border-top: 1px solid var(--ink); }
.kv > div { padding: 16px 0; border-bottom: 1px solid var(--rule); }
.kv b { display: block; font-family: var(--serif); font-weight: 400; font-size: 32px; line-height: 1; color: var(--ink); }
.kv span { display: block; margin-top: 6px; font-size: 13px; color: var(--ink-3); }
.kv p, .go { margin: 18px 0 0; }
.b figure.fig { border-top-color: var(--ink); }
@media (max-width: 1024px) {
  .bench { grid-template-columns: minmax(0, 1fr); gap: 32px; margin-top: 56px; }
  .rail { display: none; }
  .two { grid-template-columns: minmax(0, 1fr); }
}
"""
    script = """
<script>
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.rail a'));
  var byId = {};
  links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
  var seen = {};
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { seen[e.target.id] = e.isIntersecting; });
    var here = links.map(function (a) { return a.getAttribute('href').slice(1); }).filter(function (id) { return seen[id]; })[0];
    if (!here) return;
    links.forEach(function (a) { if (a === byId[here]) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
  }, { rootMargin: '-20% 0px -60% 0px' });
  Object.keys(byId).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
})();
</script>"""
    return page(
        title="Benchmarks · Drawmaha Solver",
        description="Rock-paper-scissors, Kuhn poker, Leduc poker and mini-drawmaha, each solved and graded by exploitability against a known answer.",
        here="benchmarks",
        css=css,
        body=body,
        sources="README.md, figures/rung3/, figures/rung4/ and web/rung{1,2}-viz/src/data/solve.json",
        script=script,
    )


def _third_line(w: int, labw: int, maxv: float, rows: int) -> str:
    """The uniform Nash's ⅓ as a dashed guide across `hbars`' rows."""
    pw = w - labw - 80
    x = labw + (1 / 3) / maxv * pw
    h = rows * 28 - 12 + 2
    return (
        f'<line class="ch-guide" x1="{x:.1f}" x2="{x:.1f}" y1="-4" y2="{h + 2}"/>'
    )


# ===================================================================== releases

def releases_page() -> str:
    rows = []
    for r in releases():
        shipped = "".join(f'<a class="in ship" href="{href}">{esc(label)}</a>' for href, label in r["shipped"]) or '<span class="muted">no page; published only</span>'
        command = f'<code>{esc(r["command"])}</code>' if r["command"] else '<span class="muted">—</span>'
        rows.append(
            f'<tr><td class="k"><code>{esc(r["name"])}</code><span class="sub">{esc(r["what"])}</span><span class="asset">{esc(r["asset"])}</span></td>'
            f'<td class="date num">{esc(r["published"])}</td>'
            f'<td class="tag"><a class="in" href="{REPO}/releases/tag/{esc(r["tag"])}">{esc(r["tag"])}</a></td>'
            f'<td class="grade num"><b>{g3(r["grade"])}</b><span>chips per hand at {hands(r["hands"])} hands per seat</span></td>'
            f'<td>{shipped}</td><td>{command}</td></tr>'
        )
    body = f"""
<header class="page-head">
  <div class="page-head-copy">
    <p class="eyebrow">Releases</p>
    <h1>Every strategy the site ships, with its grade.</h1>
    <p class="lede">Each release is a frozen strategy published as a GitHub release asset and fetched by its SHA-256. Its grade is the exact best response's verdict, read from the committed file the catalogue names.</p>
  </div>
</header>
<section class="rel">
  <table class="hair releases"><thead><tr><th>Release</th><th>Published</th><th>Tag</th><th>Exploitability</th><th>Shipped to</th><th>Exported by</th></tr></thead>
  <tbody>{"".join(rows)}</tbody></table>
  <p class="note">The catalogue is <code>RELEASES</code> in <code>src/drawmaha_solver/minidrawmaha/answer/releases.py</code>; each grade is read from the file and key it names (<code>figures/rung3/answer_sheet.json</code>, <code>figures/rung4/</code>). An export copies the grade into its manifest, and the tests hold the copy to the file.</p>
</section>
"""
    css = """
.rel { margin-top: 40px; }
table.releases td { font-size: 14px; }
table.releases td .sub { display: block; margin-top: 4px; font-size: 13px; color: var(--ink-3); max-width: 34ch; }
table.releases td.k .sub { font-weight: 400; }
table.releases td .asset { display: block; margin-top: 4px; font-family: var(--mono); font-size: 11px; font-weight: 400; color: var(--ink-3); }
table.releases code { white-space: nowrap; }
table.releases td.tag, table.releases td.date { white-space: nowrap; }
table.releases .ship { display: block; width: max-content; margin-bottom: 4px; }
table.releases th:nth-child(1) { width: 30%; } table.releases th:nth-child(4) { width: 15%; }
.rel .note { margin-top: 24px; max-width: 760px; }
@media (max-width: 1024px) {
  table.releases { display: block; overflow-x: auto; }
}
"""
    return page(
        title="Releases · Drawmaha Solver",
        description="The frozen strategies drawmaha.app ships, each with its release tag, its grade and the command that exports it.",
        here="releases",
        css=css,
        body=body,
        sources="src/drawmaha_solver/minidrawmaha/answer/releases.py and the grade files it names",
    )


# ===================================================================== main

def render() -> dict[str, str]:
    return {
        "index.html": home(),
        "rules.html": rules(),
        "benchmarks.html": benchmarks(),
        "releases.html": releases_page(),
    }


def main() -> None:
    for name, text in render().items():
        (OUT / name).write_text(text)
        print(f"wrote web/cover/{name}")


if __name__ == "__main__":
    main()
