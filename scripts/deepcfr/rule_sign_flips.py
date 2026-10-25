"""How much a per-sample regret rule bends the regret memory: the measurement behind the DCFR / CFR+ screen.

    uv run --extra deep python scripts/deepcfr/rule_sign_flips.py runs/rule-dcfr-256x4:dcfr runs/rule-cfrplus-256x4:cfr+

For each `RUN_DIR:RULE` and each kept grid mark, each seat's advantage
memory is read as saved and `train.rule_targets` applied to it at that
mark's iteration, the way the refit applies it. Two things are counted.

**Mass kept**: the rule's share of the memory's negative and of its
positive regret, |Σ min(rule(r), 0)| ÷ |Σ min(r, 0)| and likewise for the
positive side.

**Bad actions turned good**: among the (spot, action) cells whose regret is
negative under the linear rule's weights (Σ t·r < 0, the mean the LCFR
refit fits), the share whose rescaled sum is positive. Regret matching plays
every action with positive regret, so a flipped cell is a bad action the
net is taught to play. Cells whose samples are all zero (actions the spot
does not offer) are not counted.

The result is written to `figures/rung4/rule_sign_flips.json`, keyed
`<run name>/<t>/p<seat>`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from drawmaha_solver.deepcfr.train import MARKS, mark_name, rule_targets

OUT = Path("figures/rung4/rule_sign_flips.json")

def seat_counts(memory: Path, *, now: int, rule: str) -> dict:
    """Mass kept and bad actions turned good for one seat's saved advantage memory."""
    saved = np.load(memory)
    rows, t = saved["rows"], saved["t"]
    targets = saved["targets"].astype(np.float64)
    ruled = rule_targets(targets, t, now=now, rule=rule).astype(np.float64)
    spots = int(rows.max()) + 1
    negative = flipped = 0
    for action in range(targets.shape[1]):
        offered = np.bincount(rows, weights=(targets[:, action] != 0), minlength=spots) > 0
        linear = np.bincount(rows, weights=targets[:, action] * t, minlength=spots)
        rescaled = np.bincount(rows, weights=ruled[:, action], minlength=spots)
        bad = offered & (linear < 0)
        negative += int(bad.sum())
        flipped += int((bad & (rescaled > 0)).sum())
    return {
        "samples": len(t),
        "negative_kept": float(np.minimum(ruled, 0).sum() / np.minimum(targets, 0).sum()),
        "positive_kept": float(np.maximum(ruled, 0).sum() / np.maximum(targets, 0).sum()),
        "bad_cells": negative,
        "flipped_cells": flipped,
        "flipped_share": flipped / negative,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="RUN_DIR:RULE, the rule one of dcfr, cfr+")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    held = {}
    for spec in args.runs:
        run, rule = spec.rsplit(":", 1)
        run = Path(run)
        for mark in sorted((run / MARKS).iterdir()):
            t = int(mark.name[1:])
            for seat in (0, 1):
                counts = seat_counts(mark / f"advantage-p{seat}-{mark_name(t)}.npz", now=t, rule=rule)
                held[f"{run.name}/{t:05d}/p{seat}"] = {"rule": rule, **counts}
                print(f"{run.name} t={t} p{seat}: flipped {counts['flipped_share']:.3f}, "
                      f"negative kept {counts['negative_kept']:.3f}", flush=True)
    args.out.write_text(json.dumps(held, indent=2) + "\n")
    print(f"wrote {args.out.resolve()}")

if __name__ == "__main__":
    main()
