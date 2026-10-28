"""The reach-weighted error of each policy-net refit against the exact average.

    uv run --extra deep python scripts/deepcfr/pi_reach_weighted.py \
        runs/sweep-256x4-s16000-lr5e-4 figures/rung4/pi_sweep/reach_weighted.json

That wrote `figures/rung4/pi_sweep/reach_weighted.json` on 2026-10-26, after
`pi_loss_sweep.py` had left one frozen refit per arm under the run's
`pi-sweep/` directory.

**Why reach-weighted.** A mean absolute difference over every column counts
a round-2 row nobody reaches the same as the opening spot. The best
response does not: it exploits what play actually arrives at. So each
column is weighted by its seat's own reach under the exact average, pushed
through `deepcfr.average.predecessors` the way the exact average itself
is accumulated, and four numbers are taken per refit: the reach-weighted
mean absolute difference over all columns (`rw_mad`), the same over the
small columns (exact probability in [0.01, 0.10]), the reach-weighted share
of small columns the refit zeroes (below 0.005), and the share of the small
columns' probability mass the refit loses. `rw_mad` is the one that tracks
exploitability across the sweep (0.033 for the run's Π at 0.136, 0.017 for
the cross-entropy refit at 0.109); the zeroed count does not, which is
what settles that the stall is a fit-quality effect at reached spots, not a
count of vanished bluffs.

The run's own Π (`frozen/t00500-pi.npz`) is included under the key
`run-pi (frozen t500)`; the `@t100` / `@t200` refits are skipped because
their exact average is a different file.
"""

import json
import sys
from pathlib import Path

import numpy as np

from drawmaha_solver.deepcfr.average import predecessors
from drawmaha_solver.minidrawmaha.answer.strategy import load_strategy
from drawmaha_solver.minidrawmaha.rules.features import layout


def own_reach(sigma: np.ndarray) -> np.ndarray:
    """Each row's own reach under `sigma`, one pass over the public points in walk order."""
    built = layout()
    reach = np.zeros(int(built.row_offset[-1]))
    for p in range(len(built.points)):
        rows = slice(built.row_offset[p], built.row_offset[p + 1])
        pred = predecessors()[p]
        if pred is None:
            reach[rows] = 1.0
            continue
        before = reach[built.row_offset[pred.point] + pred.positions]
        cols = built.column_offset[pred.point] + pred.positions * built.width[pred.point] + pred.columns
        reach[rows] = (before * sigma[cols] * pred.weights).sum(axis=1)
    return np.repeat(reach, np.repeat(built.width, np.diff(built.row_offset)))


def main() -> None:
    run = Path(sys.argv[1])
    out = Path(sys.argv[2])
    sweep = run / "pi-sweep"
    exact = load_strategy(run / "frozen" / "t00500-exact.npz")
    sigma = exact.probabilities.astype(np.float64)
    w = own_reach(sigma)
    small = (sigma >= 0.01) & (sigma <= 0.10)

    def stats(path: Path) -> dict:
        a = load_strategy(path).probabilities.astype(np.float64)
        d = np.abs(a - sigma)
        return {
            "rw_mad": float((w * d).sum() / w.sum()),
            "rw_small_mad": float((w * d)[small].sum() / w[small].sum()),
            "rw_small_zeroed": float(w[small & (a < 0.005)].sum() / w[small].sum()),
            "rw_small_mass_lost": float((w * np.maximum(sigma - a, 0))[small].sum() / (w * sigma)[small].sum()),
        }

    result = {"run-pi (frozen t500)": stats(run / "frozen" / "t00500-pi.npz")}
    for f in sorted(sweep.glob("pi-*.npz")):
        if "@" in f.name:
            continue
        result[f.stem[3:]] = stats(f)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for k, v in result.items():
        print(f"{k:28s} rw_mad {v['rw_mad']:.4f}  rw_small_mad {v['rw_small_mad']:.4f}  "
              f"rw_zeroed {v['rw_small_zeroed']:.3f}  small_mass_lost {v['rw_small_mass_lost']:.3f}")


if __name__ == "__main__":
    main()
