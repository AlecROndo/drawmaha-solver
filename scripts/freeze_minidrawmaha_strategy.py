"""Freeze one averaging column of a lockstep checkpoint into a strategy file.

    uv run python scripts/freeze_minidrawmaha_strategy.py \
        runs/modal/lcfr/iter-002000000.npz runs/strategy/strategy-lcfr-2m.npz

Prints the file's size and SHA-256, which is what `strategy.STRATEGY_SHA256`
pins once the file is uploaded as a release asset. Grade the result with
`--grade` (about two minutes, 2 GB): a faithful freeze matches the
checkpoint's own exploitability to the sixth decimal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from drawmaha_solver.minidrawmaha.exploitability import best_response_value
from drawmaha_solver.minidrawmaha.regret_rules import Average
from drawmaha_solver.minidrawmaha.strategy import (
    from_checkpoint,
    load_strategy,
    save_strategy,
    sha256_of,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--column", type=Average, default=Average.LINEAR, choices=list(Average)
    )
    parser.add_argument("--grade", action="store_true")
    args = parser.parse_args()

    save_strategy(from_checkpoint(args.checkpoint, args.column), args.output)
    print(f"{args.output}: {args.output.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {sha256_of(args.output)}")
    if args.grade:
        frozen = load_strategy(args.output)
        values = [best_response_value(frozen, responder=seat) for seat in (0, 1)]
        print(f"BR_0 = {values[0]:+.6f}   BR_1 = {values[1]:+.6f}")
        print(f"exploitability = {(values[0] + values[1]) / 2:.6f} chips/hand")


if __name__ == "__main__":
    main()
