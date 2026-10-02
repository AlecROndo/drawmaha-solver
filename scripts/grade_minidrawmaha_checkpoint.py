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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from drawmaha_solver.minidrawmaha.exploitability import (
    best_response_value,
    expected_value,
    minidrawmaha_game,
)
from drawmaha_solver.minidrawmaha.infoset_table import (
    InfoSetTable,
    StrategyView,
    new_infoset_table,
)
from drawmaha_solver.minidrawmaha.lockstep import (
    LockstepSolve,
    is_lockstep,
    load_lockstep,
    read_lockstep,
)
from drawmaha_solver.minidrawmaha.mccfr import Solve, load_solve, read_run
from drawmaha_solver.minidrawmaha.regret_rules import (
    Average,
    RegretRule,
    column_average,
    validate_averages,
)

def main() -> None:
    """Grade one checkpoint's chosen average and print each measurement with its time.

    Steps:
    1. Load the checkpoint into a table sized for its rule and columns.
    2. Compile the game the grader walks.
    3. Both seats' best-response values, and the exploitability they average to.
    4. The profile's expected value for both seats.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--column", type=Average, default=Average.LINEAR, choices=list(Average)
    )
    args = parser.parse_args()

    # Step 1: the checkpoint
    (table, solve, rule, averages), seconds = _timed(_load_checkpoint, args.checkpoint)
    print(
        f"table + checkpoint: {seconds:.1f} s "
        f"(iteration {solve.iteration:,}, seed {solve.seed}, {rule.value}, "
        f"grading the {args.column.value} average)",
        flush=True,
    )

    # Step 2: the compiled game
    _, seconds = _timed(minidrawmaha_game)
    print(f"compiled game: {seconds:.1f} s", flush=True)

    # Step 3: best responses
    strategies = StrategyView(
        table, lambda ledger: column_average(ledger, args.column, averages)
    )
    values = {}
    for responder in (0, 1):
        values[responder], seconds = _timed(best_response_value, strategies, responder=responder)
        print(f"BR_{responder} = {values[responder]:+.6f}   ({seconds:.1f} s)", flush=True)
    # The mean of the two best responses IS `exploitability(strategies)`;
    # calling it would re-read the table and redo both walks for the same number.
    print(f"exploitability = {(values[0] + values[1]) / 2:.6f} chips/hand", flush=True)

    # Step 4: expected value
    value, seconds = _timed(expected_value, strategies)
    print(
        f"expected value = ({value[0]:+.6f}, {value[1]:+.6f})   "
        f"sum {value[0] + value[1]:+.2e}   ({seconds:.1f} s)",
        flush=True,
    )

def _load_checkpoint(
    checkpoint: Path,
) -> tuple[InfoSetTable, LockstepSolve | Solve, RegretRule, tuple[Average, ...]]:
    """The checkpoint loaded into a fresh table, with the rule and columns it records.

    A lockstep checkpoint (it records `workers`) and a single-process one keep
    their run description differently; both come back in the same shape.
    """
    if is_lockstep(checkpoint):
        run = read_lockstep(checkpoint)
        rule, averages = RegretRule(run["rule"]), validate_averages(run["averages"])
        table = new_infoset_table(extra_averages=len(averages))
        solve = load_lockstep(
            checkpoint, table=table, workers=run["workers"], rule=rule, averages=averages
        )
    else:
        rule, averages = read_run(checkpoint)
        table = new_infoset_table(extra_averages=len(averages))
        solve = load_solve(checkpoint, table=table, rule=rule, averages=averages)
    return table, solve, rule, averages

def _timed[T](measure: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """`measure`'s result and the wall-clock seconds it took."""
    started = time.perf_counter()
    result = measure(*args, **kwargs)
    return result, time.perf_counter() - started

if __name__ == "__main__":
    main()
