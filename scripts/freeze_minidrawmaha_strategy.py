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
    """Freeze, print the size and digest, and grade if asked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="a lockstep checkpoint .npz")
    parser.add_argument("output", type=Path, help="where to write the strategy .npz")
    parser.add_argument(
        "--column", type=Average, default=Average.LINEAR, choices=list(Average),
        help="which averaging column to freeze (default: linear, the one the race graded)",
    )
    parser.add_argument(
        "--grade", action="store_true", help="grade the written file exactly (~2 min, 2 GB)"
    )
    args = parser.parse_args()

    save_strategy(from_checkpoint(args.checkpoint, args.column), args.output)
    print(f"{args.output}: {args.output.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {sha256_of(args.output)}")
    if args.grade:
        # Graded from the file just written, not the in-memory strategy, so
        # the number printed is the one a player loading it would get.
        frozen = load_strategy(args.output)
        values = [best_response_value(frozen, responder=seat) for seat in (0, 1)]
        print(f"BR_0 = {values[0]:+.6f}   BR_1 = {values[1]:+.6f}")
        # Exploitability is the mean of the two best responses; computed here
        # from the pair rather than by `exploitability`, which would walk both again.
        print(f"exploitability = {(values[0] + values[1]) / 2:.6f} chips/hand")

if __name__ == "__main__":
    main()
