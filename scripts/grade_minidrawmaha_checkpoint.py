"""Grade a mini-drawmaha checkpoint with the exact meter, and time each part.

    uv run python scripts/grade_minidrawmaha_checkpoint.py runs/mccfr-seed0/solve.npz

Prints both seats' best-response values, the exploitability, the expected
value, and how long the table, the compiled game and each measurement took.
Units are chips per hand with the ante as 1. Each public call snapshots the
profile itself, so the per-call times include a read of the whole table.
Budget about 3.5 GB of memory: the table is 1.77 GB and the grader holds
1.5 GB of showdown tables for the life of the process.
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
from drawmaha_solver.minidrawmaha.infoset_table import (
    average_strategy,
    new_infoset_table,
)
from drawmaha_solver.minidrawmaha.mccfr import load_solve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    clock = time.perf_counter

    t = clock()
    table = new_infoset_table()
    solve = load_solve(args.checkpoint, table=table)
    print(
        f"table + checkpoint: {clock() - t:.1f} s "
        f"(iteration {solve.iteration:,}, seed {solve.seed})",
        flush=True,
    )

    t = clock()
    minidrawmaha_game()
    print(f"compiled game: {clock() - t:.1f} s", flush=True)

    strategies = average_strategy(table)
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
