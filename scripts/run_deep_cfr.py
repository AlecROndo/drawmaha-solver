"""Train Deep CFR on mini-drawmaha on this machine, resumably.

    caffeinate -i uv run --extra deep python scripts/run_deep_cfr.py --out runs/deep-smoke --target 3 --walks 1000

Re-running the same command resumes from `<out>/resume`. Ctrl-C (or SIGTERM)
finishes the iteration and saves. See `deepcfr.train` for what is kept and
which flags a resume may change (`--workers` yes; the seed, the walks, the
capacity and the net's size no).
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from drawmaha_solver.deepcfr.train import DeepConfig, run_deep

def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command line: `--out` and `--target`, then one flag per run parameter."""
    defaults = DeepConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=2_000, help="iterations to reach")
    parser.add_argument("--walks", type=int, default=defaults.walks, help="K walks per seat per iteration")
    parser.add_argument("--workers", type=int, default=defaults.workers, help="W walk processes; 0 walks in-process")
    parser.add_argument("--capacity", type=int, default=defaults.capacity, help="C rows per memory")
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--depth", type=int, default=defaults.depth)
    parser.add_argument("--steps", type=int, default=defaults.steps, help="Adam steps per refit")
    parser.add_argument("--batch", type=int, default=defaults.batch, help="rows per step")
    parser.add_argument("--threads", type=int, default=defaults.threads, help="torch threads for the refit")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args(argv)

def config_from_args(args: argparse.Namespace) -> DeepConfig:
    """The run the flags describe; refused by name if a value could not be used."""
    return DeepConfig(
        seed=args.seed,
        walks=args.walks,
        workers=args.workers,
        capacity=args.capacity,
        width=args.width,
        depth=args.depth,
        steps=args.steps,
        batch=args.batch,
        threads=args.threads,
    )

def main() -> None:
    args = parse()
    run_deep(args.out, config_from_args(args), target=args.target)

if __name__ == "__main__":
    main()
