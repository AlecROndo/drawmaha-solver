"""The rung-4 grades quoted in prose are the committed grades.

`README.md`'s rung-4 table and the scoreboard in `docs/notes/rung4-run-log.md`
are typed by hand from `figures/rung4/*.json` (and `figures/rung3/grades.json`
for LCFR). These tests read every number in those two tables back from its
file, so a re-grade or a typo cannot leave the prose quoting a stale value.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNG4 = ROOT / "figures" / "rung4"

# Hands per seat at mark t are 10,000·t; LCFR's lockstep iteration is 1,000·t.
HANDS = {"100k": 10, "200k": 20, "500k": 50, "1M": 100, "2M": 200}


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def lcfr(t: int) -> float:
    return load("figures/rung3/grades.json")[f"lcfr/{1_000 * t:09d}/linear"]["exploitability"]


def mark(file: str, t: int) -> float:
    return load(file)[f"deep/{t:05d}/exact"]["exploitability"]


def sweep(size: str) -> float:
    return load("figures/rung4/sweep.json")["sizes"][size]["exact"]["exploitability"]


def rows(text: str, header_start: str) -> list[list[str]]:
    """The body rows of the Markdown table whose header line starts with `header_start`."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    body = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        body.append([cell.strip().strip("*") for cell in line.strip("|").split("|")])
    return body


def quoted(cell: str) -> float | None:
    return float(cell) if re.fullmatch(r"\d+\.\d{3}", cell) else None


def test_the_readme_rung4_table_quotes_the_committed_grades():
    readme = (ROOT / "README.md").read_text()
    table = rows(readme, "| hands per seat | LCFR | 3 × 64 |")
    columns = (
        lambda t: lcfr(t),
        lambda t: mark("figures/rung4/grades.json", t) if t <= 50 else sweep("64x3"),
        lambda t: sweep("128x3") if t == 100 else mark("figures/rung4/sweep_128x3_t200.json", t),
        lambda t: mark("figures/rung4/sweep_256x4.json", t),
        lambda t: mark("figures/rung4/sweep_256x4-s8000.json", t),
    )
    checked = 0
    for hands, *cells in table:
        t = HANDS[hands]
        for source, cell in zip(columns, cells, strict=True):
            if (value := quoted(cell)) is not None:
                assert value == round(source(t), 3), (hands, cell)
                checked += 1
    assert checked >= 14


def test_the_run_log_scoreboard_quotes_the_committed_grades():
    log = (ROOT / "docs" / "notes" / "rung4-run-log.md").read_text()
    table = rows(log, "| arm | where | net |")
    marks = (10, 20, 50, 100, 200)
    checked = 0
    for arm, _where, net, _steps, _lr, *cells, file in table:
        file = file.strip("`")
        assert (ROOT / file).exists(), file
        for t, cell in zip(marks, cells, strict=True):
            if (value := quoted(cell)) is None:
                continue
            if file == "figures/rung3/grades.json":
                graded = lcfr(t)
            elif file == "figures/rung4/sweep.json":
                width_x_depth = "x".join(reversed(net.replace(" ", "").split("×")))
                graded = sweep(width_x_depth)
            else:
                graded = mark(file, t)
            assert value == round(graded, 3), (arm, t, cell)
            checked += 1
    assert checked >= 25
