"""The grader-backed refresher: the exact meter's leak, handed to the trainer's targeting.

    uv run python scripts/deepcfr/targeting_refresh.py --granularity line          # inspect one refresh
    uv run python scripts/deepcfr/targeting_refresh.py --granularity random --out /tmp/t00001.npz

`deepcfr/targeting.py` decides which table rows the boost stage computes
exactly, from a leak, a frequency and a draw weight per row, and never runs
the meter that measures the leak: the trainer must not share code with what
grades it. This script is the one place on the trainer's side where the two
meet. `GraderRefresher` is a `targeting.Refresher`: `run_deep_cfr.py` builds
one and hands it to the loop (`deepcfr.boost.Boost`), which calls it every R
iterations:

1. **The profile it grades.** `spec.leak == "average"` (the default) grades
   the frozen exact average, `ExactAverage.frozen` — the object the run is
   graded on, so the budget goes where the graded number leaks; the
   `ExactAverage` is handed in when the refresher is built, since the loop
   does not know which of its observers it is. `"current"` grades the
   table's current strategy instead, the iterate whose regrets the walks
   are sampling now, and needs no average. Which one aims better is open;
   both cost the same.
2. **The leak and frequency.** `analysis.gains_by_row`: two walks of
   `gains_by_infoset`, one per seat, each infoset's one-step best-response
   gain and play frequency laid on the table's rows. About 100 s on this Mac.
3. **The draw weight.** `BoostTargets.arrivals` under `BoostTargets.sigma`
   of the table: how often one walk with the table's current strategy
   arrives at each pool row, asked only for the pool and never for
   `infoset`, which draws nothing. Measured on this Mac under the LCFR-2M
   release's σ: a few seconds for a line pool (about 45,000 rows a seat),
   about eight minutes a seat for the random arm, whose pool is the whole
   seat (16 min 52 s for the whole refresh).
4. **The set.** `targeting.target_set`, which reads the two advantage
   memories for the noise score.

The random arm aims at nothing, so it skips steps 1 and 2.

Run as a script, it is an inspection tool: it loads a released strategy
into a table (so the table's σ IS the strategy), refreshes once with
`leak="current"` and empty memories (every noise at the cap), and prints
what each seat's pool is made of and what one iteration's draw takes from
each stage.
"""

from __future__ import annotations

import argparse
import numbers
from pathlib import Path

import numpy as np

from drawmaha_solver.deepcfr.average import ExactAverage
from drawmaha_solver.deepcfr.reservoir import Reservoir
from drawmaha_solver.deepcfr.targeting import (
    GRANULARITIES,
    TargetSet,
    TargetSpec,
    row_offsets,
    target_set,
)
from drawmaha_solver.deepcfr.train import current_iterate
from drawmaha_solver.minidrawmaha.answer.analysis import gains_by_row
from drawmaha_solver.minidrawmaha.answer.strategy import FrozenStrategy
from drawmaha_solver.minidrawmaha.exploitability import minidrawmaha_game
from drawmaha_solver.minidrawmaha.rules.enumeration import public_decision_points
from drawmaha_solver.minidrawmaha.solver.boost_target import BoostTargets
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable

class GraderRefresher:
    """A `targeting.Refresher` that measures the leak with the exact grader.

    `walks` is the run's K, walks per seat per iteration, which the noise
    score needs to turn the memory's row counts into visits per walk.
    `average` is the run's `ExactAverage`, the observer the loop accumulates;
    it is required when `spec.leak` is `"average"` and unused otherwise.
    `seed` only labels the frozen profile. `exact` is the boost oracle the
    arrivals come from; the default builds the whole game's, cached per
    process.
    """

    def __init__(
        self,
        spec: TargetSpec,
        *,
        walks: int,
        average: ExactAverage | None = None,
        seed: int = 0,
        exact: BoostTargets | None = None,
    ) -> None:
        if not isinstance(spec, TargetSpec):
            raise ValueError(f"spec must be a TargetSpec, got {type(spec).__name__}")
        # Checked here, not first inside the noise score after two minutes of grader walks.
        if isinstance(walks, bool) or not isinstance(walks, numbers.Integral) or walks < 1:
            raise ValueError(f"walks must be a positive int (walks per seat per iteration, K), got {walks!r}")
        # The random arm grades nothing, and "current" grades the table: only
        # "average" reads the average, and asking for it at the first refresh,
        # after the first iteration's walks, would be a refusal paid for.
        if spec.granularity != "random" and spec.leak == "average" and average is None:
            raise ValueError(
                "an aim at the exact average's leak needs the run's ExactAverage: pass average=; "
                "leak='current' grades the table and needs none"
            )
        self.spec = spec
        self.walks = walks
        self.average = average
        self.seed = seed
        self._exact = exact
        self._game = None

    def profile(self, *, t: int, table: PackedTable) -> FrozenStrategy:
        """The strategy whose leak is measured: the exact average so far, or the table's current one."""
        if self.spec.leak == "average":
            return self.average.frozen(iteration=t, seed=self.seed)
        return current_iterate(table, iteration=t, seed=self.seed)

    def __call__(self, *, t: int, table: PackedTable, memories: tuple[Reservoir, Reservoir]) -> TargetSet:
        sigma: list = []  # read from the table on the first arrival asked for; `infoset` asks none

        def arrival(seat: int, rows: np.ndarray) -> np.ndarray:
            if self._exact is None:
                self._exact = BoostTargets.build()
            if not sigma:
                sigma.append(self._exact.sigma(table))
            return self._exact.arrivals(sigma[0], seat=seat, rows=rows)

        leak = frequency = None
        if self.spec.granularity != "random":
            if self._game is None:
                self._game = minidrawmaha_game()
            leak, frequency = gains_by_row(self.profile(t=t, table=table), game=self._game)
        return target_set(
            self.spec, t=t, leak=leak, frequency=frequency, memories=memories, walks=self.walks, arrival=arrival
        )

# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def _stage_of_rows(rows: np.ndarray) -> np.ndarray:
    """Each row's stage name, from its public point."""
    offsets = row_offsets()
    points = public_decision_points()
    stage = np.array(
        ["round 2" if p.board_cards == 2 else "draw" if p.is_draw_decision else "round 1" for p in points]
    )
    return stage[np.searchsorted(offsets, rows, side="right") - 1]

def main(argv: list[str] | None = None) -> None:
    from drawmaha_solver.minidrawmaha.answer.releases import RELEASES
    from drawmaha_solver.minidrawmaha.answer.strategy import fetch_strategy, load_strategy

    parser = argparse.ArgumentParser(description="Refresh one target set from a released strategy and say what it holds")
    parser.add_argument("--release", choices=sorted(RELEASES), default="tabular")
    parser.add_argument("--granularity", choices=GRANULARITIES, default="line")
    parser.add_argument("--budget", type=int, default=TargetSpec.__dataclass_fields__["budget"].default)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round-two-only", action="store_true", help="leave round-1 and draw rows out of the arm")
    parser.add_argument("--out", type=Path, help="save the set here (.npz)")
    args = parser.parse_args(argv)

    release = RELEASES[args.release]
    strategy = load_strategy(fetch_strategy(url=release.url, sha256=release.sha256))
    table = PackedTable.whole_game()
    table.cumulative_regret[:] = strategy.probabilities
    # α only tells the walk's rows from boosted ones in a memory; these memories are empty.
    spec = TargetSpec(
        granularity=args.granularity, alpha=0.0, budget=args.budget, round_two_only=args.round_two_only, leak="current"
    )
    refresher = GraderRefresher(spec, walks=1, seed=args.seed)
    targets = refresher(t=1, table=table, memories=(Reservoir(1), Reservoir(1)))
    for seat in (0, 1):
        rows, weight = targets.rows[seat], targets.weight[seat]
        stage = _stage_of_rows(rows)
        drawn = _stage_of_rows(targets.rows_for(1, seat, args.seed))
        print(f"seat {seat}: pool {rows.size:,} rows from {targets.groups[seat].size:,} groups")
        for name in ("round 1", "draw", "round 2"):
            mass = weight[stage == name].sum() / weight.sum() if weight.sum() > 0 else 0.0
            print(f"  {name:>8}: {np.count_nonzero(stage == name):>9,} pool rows, "
                  f"{100 * mass:5.1f}% of the draw weight, {np.count_nonzero(drawn == name):>5,} drawn at t = 1")
    if args.out is not None:
        targets.save(args.out)
        print(f"wrote {args.out.resolve()}")

if __name__ == "__main__":
    main()
