"""Grade a mini-drawmaha checkpoint with the exact meter, and time each part.

    uv run python scripts/grade_minidrawmaha_checkpoint.py runs/mccfr-seed0/solve.npz
    uv run python scripts/grade_minidrawmaha_checkpoint.py runs/dcfr/solve.npz --column quadratic

Grades one of the checkpoint's averages: the primary linear one by default, or
any extra column the run banked. The table is sized from what the checkpoint
records — its rule and its columns — so any run's checkpoint loads.

Prints both seats' best-response values, the exploitability, the expected
value, and how long the table, the compiled game and each measurement took.
Units are chips per hand with the ante as 1. Each public call snapshots the
profile itself, so the per-call times include a read of the whole table.
Budget about 2 GB of memory: the packed table is about 0.3 GB and the grader
holds 1.5 GB of showdown tables for the life of the process. A checkpoint saved
before the key kept the board order is refused on load: its ledgers belong to
a table of 3,142,290 keys, not this one.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from drawmaha_solver.minidrawmaha.exploitability import (
    best_response_value,
    expected_value,
    minidrawmaha_game,
)
from drawmaha_solver.minidrawmaha.infoset_table import StrategyView, new_infoset_table
from drawmaha_solver.minidrawmaha.lockstep import is_lockstep, load_lockstep, read_lockstep
from drawmaha_solver.minidrawmaha.mccfr import load_solve, read_run
from drawmaha_solver.minidrawmaha.regret_rules import (
    Average,
    RegretRule,
    column_average,
    validate_averages,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--column", type=Average, default=Average.LINEAR, choices=list(Average)
    )
    args = parser.parse_args()
    clock = time.perf_counter

    t = clock()
    if is_lockstep(args.checkpoint):
        run = read_lockstep(args.checkpoint)
        rule, averages = RegretRule(run["rule"]), validate_averages(run["averages"])
        table = new_infoset_table(extra_averages=len(averages))
        solve = load_lockstep(
            args.checkpoint, table=table, workers=run["workers"], rule=rule, averages=averages
        )
    else:
        rule, averages = read_run(args.checkpoint)
        table = new_infoset_table(extra_averages=len(averages))
        solve = load_solve(args.checkpoint, table=table, rule=rule, averages=averages)
    print(
        f"table + checkpoint: {clock() - t:.1f} s "
        f"(iteration {solve.iteration:,}, seed {solve.seed}, {rule.value}, "
        f"grading the {args.column.value} average)",
        flush=True,
    )

    t = clock()
    minidrawmaha_game()
    print(f"compiled game: {clock() - t:.1f} s", flush=True)

    strategies = StrategyView(
        table, lambda ledger: column_average(ledger, args.column, averages)
    )
    values = {}
    for responder in (0, 1):
        t = clock()
        values[responder] = best_response_value(strategies, responder=responder)
        print(f"BR_{responder} = {values[responder]:+.6f}   ({clock() - t:.1f} s)", flush=True)
    # The mean of the two best responses IS `exploitability(strategies)`;
    # calling it would re-read the table and redo both walks for the same number.
    print(f"exploitability = {(values[0] + values[1]) / 2:.6f} chips/hand", flush=True)

    t = clock()
    value = expected_value(strategies)
    print(
        f"expected value = ({value[0]:+.6f}, {value[1]:+.6f})   "
        f"sum {value[0] + value[1]:+.2e}   ({clock() - t:.1f} s)",
        flush=True,
    )

if __name__ == "__main__":
    main()
