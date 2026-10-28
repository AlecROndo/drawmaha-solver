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
    starts = [i for i, line in enumerate(lines) if line.startswith(header_start)]
    assert len(starts) == 1, f"expected one table headed {header_start!r}, found {len(starts)}"
    start = starts[0]
    assert re.fullmatch(r"\|(\s*-+\s*\|)+", lines[start + 1]), f"no separator row under {header_start!r}"
    body = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        body.append([cell.strip().strip("*") for cell in line.strip("|").split("|")])
    return body


def quoted(cell: str) -> float | None:
    """A grade cell's value, or None for a cell with no digit in it; any other number fails."""
    if not re.search(r"\d", cell):
        return None
    assert re.fullmatch(r"\d+\.\d{3}", cell), f"grades are quoted to three decimals: {cell!r}"
    return float(cell)


def test_the_readme_rung4_table_quotes_the_committed_grades():
    readme = (ROOT / "README.md").read_text()
    table = rows(readme, "| hands per seat | LCFR | 3 × 64 |")
    columns = (
        lambda t: lcfr(t),
        # The Mac 3 × 64 run was graded to t = 50; its 1M point is the Modal sweep's.
        lambda t: mark("figures/rung4/grades.json", t) if t <= 50 else sweep("64x3"),
        lambda t: sweep("128x3") if t == 100 else mark("figures/rung4/sweep_128x3_t200.json", t),
        lambda t: mark("figures/rung4/sweep_256x4.json", t),
        lambda t: mark("figures/rung4/sweep_256x4-s8000.json", t),
    )
    checked = 0
    for hands, *cells in table:
        t = HANDS[hands]
        for column, (source, cell) in enumerate(zip(columns, cells, strict=True), start=1):
            if (value := quoted(cell)) is not None:
                assert value == round(source(t), 3), (hands, column, cell)
                checked += 1
    # Every cell with a digit is checked or fails in quoted(), so this guards only an empty table.
    assert checked > 0


def test_the_readme_rung4_columns_are_the_refits_the_caption_says():
    # The caption: Adam at 1e-3 and a 4,000-step refit unless the header says otherwise.
    def refit(file: str, key: str) -> tuple[int, float]:
        config = load(file)[key]["config"]
        return config["steps"], config["lr"]

    spec = (4_000, 1e-3)
    assert refit("figures/rung4/grades.json", "deep/00010/exact") == spec
    sizes = load("figures/rung4/sweep.json")["sizes"]
    for size in ("64x3", "128x3"):
        assert (sizes[size]["exact"]["config"]["steps"], sizes[size]["exact"]["config"]["lr"]) == spec
    assert refit("figures/rung4/sweep_128x3_t200.json", "deep/00200/exact") == spec
    assert refit("figures/rung4/sweep_256x4.json", "deep/00100/exact") == spec
    assert refit("figures/rung4/sweep_256x4-s8000.json", "deep/00100/exact") == (8_000, 1e-3)


def test_the_run_log_scoreboard_quotes_the_committed_grades():
    log = (ROOT / "docs" / "notes" / "rung4-run-log.md").read_text()
    table = rows(log, "| arm | where | net |")
    marks = (10, 20, 50, 100, 200)
    checked = 0
    for arm, _where, net, _steps, _lr, _pot_power, *cells, file in table:
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
    # Every cell with a digit is checked or fails in quoted(), so this guards only an empty table.
    assert checked > 0
