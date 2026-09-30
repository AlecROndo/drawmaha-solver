"""Train one rule's lockstep run on this machine, resumably.

    caffeinate -i uv run python scripts/run_lockstep.py lcfr --out runs/lockstep-lcfr

Re-running the same command resumes from `<out>/resume.npz`. Ctrl-C (or
SIGTERM) finishes the iteration and saves. See `lockstep_run` for what is kept.
"""

import argparse
from pathlib import Path

from drawmaha_solver.minidrawmaha.lockstep_run import run_to

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rule", choices=["vanilla", "cfr+", "lcfr", "dcfr"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_to(args.out, rule=args.rule, target=args.target, workers=args.workers, seed=args.seed)

if __name__ == "__main__":
    main()
