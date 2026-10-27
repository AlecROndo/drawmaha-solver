"""Grade the visit-clock DCFR checkpoints, linear and quadratic averages, into one grades file.

    uv run python scripts/minidrawmaha/grade_visit_clock.py figures/rung3/grades_visit_clock.json \
        runs/dcfr-visit/iter-000010000.npz runs/dcfr-visit/iter-000020000.npz \
        runs/dcfr-visit/iter-000050000.npz runs/dcfr-visit/iter-000100000.npz

That wrote `figures/rung3/grades_visit_clock.json` on 2026-10-07. The
checkpoints come from the one commit of the unpushed branch
`rung3-dcfr-visit-clock` (fe49de4), which keeps DCFR(1.5, 0) but clocks it
on the row: the stamp counts the row's own updates, and update n pays
factor n's discount at update n+1, so an idle row is never discounted. The
arm was run on the Mac to 100,000 lockstep iterations at W = 20 (2M hands
per seat), so its points land on the race's hands grid at 200k, 400k, 1M
and 2M hands; the race itself ran at W = 10.

Keys are `dcfr-visit/<iteration>/<column>` with `workers` beside each
grade, as `figures/rung3/grades.json` keys the race, so the page reads both
files the same way. A key already in the file is not regraded.
"""

import json
import sys
from pathlib import Path

from drawmaha_solver.minidrawmaha.answer.analysis import grade
from drawmaha_solver.minidrawmaha.answer.strategy import from_checkpoint
from drawmaha_solver.minidrawmaha.solver.lockstep import read_lockstep
from drawmaha_solver.minidrawmaha.solver.regret_rules import Average


def main() -> None:
    out = Path(sys.argv[1])
    grades = json.loads(out.read_text()) if out.exists() else {}
    for checkpoint in sys.argv[2:]:
        run = read_lockstep(Path(checkpoint))
        for column in ("linear", "quadratic"):
            key = f"dcfr-visit/{int(run['iteration']):09d}/{column}"
            if key in grades:
                continue
            g = grade(from_checkpoint(Path(checkpoint), Average(column)))
            grades[key] = {**g, "workers": int(run["workers"]), "seed": int(run["seed"])}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(grades, indent=2, sort_keys=True) + "\n")
            print(key, round(g["exploitability"], 5), flush=True)


if __name__ == "__main__":
    main()
