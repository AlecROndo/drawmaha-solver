"""One arm of the friction mini experiment: lockstep MCCFR, with or without RTCFR+-style friction.

    uv run python scripts/minidrawmaha/friction_experiment.py --rule cfr+ --workers 10 --mu 0.1 \
        --out runs/friction-mini/cfrplus-w10-mu0.1.jsonl

Trains one run in this process (`train_lockstep`; friction is not parallelised) to
`--hands` hands per seat. Every `--reset-hands` hands per seat it moves the friction's
reference to the current strategy (RTCFR+'s r ← x̂; see `solver/friction.py`). At each
mark it grades BOTH the current strategy (what the friction is meant to make converge)
and the linear average (what plain CFR ships) with the exact meter, and appends one
JSON line: hands per seat, iteration, the two exploitabilities in chips per hand, and
the seconds so far. `--mu 0` trains plain MCCFR, the arm's no-friction control.

Costs: about 0.3 GB for the table and as much again for the reference, plus the
meter's 1.5 GB; about 200 hands per seat per second at W = 10 on one core, and two
grades of about two minutes each per mark. No checkpoint is written: a friction
run's reference is not saved, so an interrupted run is rerun.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from drawmaha_solver.minidrawmaha.exploitability import exploitability, minidrawmaha_game
from drawmaha_solver.minidrawmaha.solver.friction import Friction
from drawmaha_solver.minidrawmaha.solver.infoset_table import (
    average_strategy,
    current_strategy,
)
from drawmaha_solver.minidrawmaha.solver.lockstep import new_lockstep, train_lockstep

DEFAULT_MARKS = (50_000, 100_000, 250_000, 500_000, 1_000_000)

def main() -> None:
    """Train one arm to its last mark, resetting the reference and grading on the way."""
    args = _parse()
    reset_every = _iterations(args.reset_hands, args.workers, "--reset-hands")
    marks = sorted({_iterations(m, args.workers, "--marks") for m in args.marks})
    solve = new_lockstep(args.seed, workers=args.workers, rule=args.rule)
    if args.mu > 0:
        solve.friction = Friction.new(solve.table, args.mu)
    game = minidrawmaha_game()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for t in _events(reset_every, marks):
        train_lockstep(solve, t - solve.iteration)
        if solve.friction is not None and t % reset_every == 0:
            solve.friction.reset()
        if t in marks:
            line = {
                "rule": args.rule,
                "workers": args.workers,
                "mu": args.mu,
                "reset_hands": args.reset_hands,
                "seed": args.seed,
                "iteration": t,
                "hands_per_seat": t * args.workers,
                "current": exploitability(current_strategy(solve.table), game=game),
                "average": exploitability(average_strategy(solve.table), game=game),
                "seconds": round(time.perf_counter() - started, 1),
            }
            with args.out.open("a") as file:
                file.write(json.dumps(line) + "\n")
            print(json.dumps(line), flush=True)

def _events(reset_every: int, marks: list[int]) -> list[int]:
    """Every iteration at which training must pause: each reset and each mark, in order.

    In: the reset cadence and the mark iterations. Out: the sorted union up to the
    last mark. Purpose: `main` trains from one pause to the next, so a reset lands
    exactly on its iteration and a grade sees the table as of its mark.
    """
    last = marks[-1]
    return sorted(set(range(reset_every, last + 1, reset_every)) | set(marks))

def _iterations(hands: int, workers: int, flag: str) -> int:
    """Hands per seat as lockstep iterations (W hands per seat each); refused unless exact.

    Purpose: arms at different W are compared at equal hands, so a mark or a
    reset that does not land on a whole iteration would compare unequal amounts.
    """
    if hands <= 0 or hands % workers:
        raise SystemExit(f"{flag}: {hands:,} hands is not a positive multiple of W = {workers}")
    return hands // workers

def _parse() -> argparse.Namespace:
    """The command line: rule, W, μ, reset cadence, budget, seed and output file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rule", choices=["lcfr", "cfr+"], required=True)
    parser.add_argument("--workers", type=int, default=10, help="W: hands per seat per iteration")
    parser.add_argument("--mu", type=float, default=0.0, help="friction strength in chips; 0 = none")
    parser.add_argument("--reset-hands", type=int, default=10_000,
                        help="hands per seat between reference resets")
    parser.add_argument("--marks", type=int, nargs="+", default=list(DEFAULT_MARKS),
                        help="hands per seat at which to grade")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    main()
