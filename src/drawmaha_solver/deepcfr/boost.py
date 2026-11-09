"""The boost stage: one sampled boosted row per targeted infoset, at the walk's own weight there.

Deep CFR learns each infoset's regret from the samples the walks bank
there, one noisy draw per visit. The targeted-sampling plan's boost adds,
at B chosen infosets per seat per iteration, one extra row that is another
sample of the same regret with the opponent's hidden hand summed out — a
play-out from the row against the opponent's whole range
(`boost_target.BoostTargets.targets` under a `Sampling`) — and turns the
walk's own rows at those infosets down to weight α. `targeting` decides
WHICH rows; `boost_target` computes WHAT a boosted row says; this module is
the stage between the walks and the memory that joins the two inside
`train.step`.

**The weight, and why it is (1 − α) · K · arrival.** Plain sampling
reaches an infoset K · p times per iteration in expectation, p its arrival
rate under σ_t, and each visit banks one row of weight 1 that the refit
weighs by t. The boosted row stands in for those rows: at weight
(1 − α) · K · arrival, with the natural rows turned down to α, the
infoset's expected banked weight is α · K · p + (1 − α) · K · p = K · p,
what it was, and its expected banked regret is unchanged too, because the
boosted row is an unbiased sample of the same conditional mean the walk's
rows are drawn from. That is the whole contract: the boost never adds
emphasis, so the target list may change between refreshes and the run
stays an unbiased estimate of the same thing. Whether it REMOVES noise is
a measured fact, not a guarantee: the boosted row enters the fit at weight
W = (1 − α) · K · p with variance W² · σ_b², against the K · p walk rows'
K · p · σ_w², so the infoset is quieter only where σ_b² < σ_w² / (K · p)
(σ_b² the spread of one play-out, σ_w² of one walk sample) — a condition
hardest to meet at the busiest rows, and one the race (the plan's PR 5)
measures rather than this module assuming it. The fit's t is left to the
fit, as every row's is.

**Why the natural rows are turned down, not dropped.** The reservoir keeps
a uniform sample of the rows ever OFFERED (Algorithm R), and its draws
depend on the count of offers. Dropping a row would shift every later
offer's draw within the phase. So the natural rows are all offered, in
their order, at weight α — 0 means inert, kept for the count — and the
boosted rows are offered AFTER them, so within a phase the natural rows
keep their stream positions. Across phases a boosted run still differs
from a plain one: each phase adds up to B offers, so later phases' draws
and the resident set differ; the two memories agree in expectation, not
row for row (and to t = 200 at the run's capacity nothing is evicted, so
every offer of either run is resident). Only a run without an aim is
today's run bit for bit. The inert rows have a price once eviction starts:
at α = 0 a targeted infoset's natural rows hold reservoir slots at weight
0, so the memory's effective capacity is smaller by their share. How many
there are is the walk's visits to the drawn rows, about K · Σp over them
per phase, which depends on where the aim lands; `Injected.turned_down`
counts them and the progress line reports them (`rows.turned_down`), so
the race reads the share rather than this docstring guessing it.

**Why a row is treated all or nothing.** A targeted row's natural rows at α
and its boosted row at (1 − α) · K · arrival are one unit: either alone
would bias the infoset's expected weight. So a drawn row that cannot be
boosted is left untreated — its natural rows weigh 1, no row is added —
which is simply "not targeted this iteration", unbiased by construction.
Two cases are left untreated: a row the walk cannot reach (arrival 0, so
nothing to stand in for), and a row whose float32 weight would land exactly
on 1 or on α. The second exists because `targeting.noise` tells the walk's
rows from boosted rows by weight alone (1 or α is a walk row): a boosted
row at either weight would be read as a sample and spoil the noise score.
The skip is deterministic, not random — the same row under the same σ is
skipped every iteration until σ moves it off the value — which is
harmless at a measure-zero coincidence in float32 but worth knowing when a
test draws a row deliberately at K = 1. The skipped count is reported,
never hidden. α = 1 is refused outright: every boosted row would weigh 0,
B play-outs paid for rows the fit ignores.

**Why EXACT is refused.** `Sampling` has an exact form, the sum over every
branch below the row, which the tests use as the oracle for the sampled
forms. The boost stage never runs it: boosting is extra sampling at
high-variance nodes, never brute-forcing the tree from a node, because the
exact subtree sum is a hundred times a sampled play-out's cost at a round-1
row and, as a census of the tree, sets aside what the network is for —
learning a row from samples. `DeepConfig` refuses an exact sampling and
`inject` refuses it again, so the rule holds whoever builds the config. The
default, `Sampling()`, is the walk's own sampling below the row against the
whole opponent range: one second board card, one replacement per throw,
one opponent action line. The opponent-range count stays a knob, so one
dealt hand per play-out — the walk's sampling exactly, M play-outs
averaged — is one argument away.

**Where σ comes from.** The boosted row must be computed under the σ the
walks of this phase read, or its sample is of a different regret. `Boost.phase`
reads `BoostTargets.sigma(table)` from the collector's table right after
this phase's `learner.write`, the same array the walks read. The boost's own
randomness is PR 1's stream per row (`[seed, t, seat, row, "BOOST"]`) and
the target draw PR 3's (`"TARG"`); neither touches the walk's or the
reservoir's stream, so a boosted run's walks are the plain run's walks.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from drawmaha_solver.deepcfr.targeting import Refresher, TargetSet, TargetSpec
from drawmaha_solver.minidrawmaha.solver.boost_target import BoostTargets, Sampling

# ---------------------------------------------------------------------------
# What one phase's boost adds
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Injected:
    """One phase's advantage offer with the boost in it: the natural rows, then the boosted rows.

    `rows`, `targets` and `weight` are what `Reservoir.merge` takes, in the
    order it must see them — the walk's a rows first, in walk order, then
    the b boosted rows in the order they were drawn. `weight` is α at a
    natural row of a treated infoset, 1 at every other natural row, and
    (1 − α) · K · arrival at a boosted row. `boosted` is b; `skipped` counts
    the drawn rows left untreated (zero arrival, or a weight that would read
    as a walk row's); `turned_down` the natural rows at treated infosets,
    the ones now at weight α — memory slots made inert at α = 0, merely
    down-weighted at α > 0.
    """

    rows: np.ndarray  # int32 (a + b,)
    targets: np.ndarray  # float32 (a + b, TARGET_WIDTH)
    weight: np.ndarray  # float32 (a + b,)
    boosted: int
    skipped: int
    turned_down: int

# ---------------------------------------------------------------------------
# The weight
# ---------------------------------------------------------------------------

def boosted_weight(arrival: np.ndarray, *, alpha: float, walks: int) -> np.ndarray:
    """(1 − α) · K · arrival per row, float64: the sampled weight each boosted row stands in for.

    `arrival` is the expected visits one walk pays the row (`BoostTargets`),
    `walks` the run's K, so K · arrival is the weight plain sampling would
    bank there per iteration; (1 − α) is the share the boost takes over.
    """
    return (1.0 - float(alpha)) * int(walks) * np.asarray(arrival, dtype=np.float64)

# ---------------------------------------------------------------------------
# The injection
# ---------------------------------------------------------------------------

def inject(
    rows: np.ndarray,
    targets: np.ndarray,
    *,
    drawn: np.ndarray,
    exact,
    sigma,
    seat: int,
    t: int,
    seed: int,
    sampling: Sampling,
    alpha: float,
    walks: int,
) -> Injected:
    """The phase's advantage offer with one boosted row per treatable drawn row appended.

    `rows` and `targets` are the walk's own offer (`Collected.advantage_*`);
    `drawn` the sorted, distinct table rows of `seat` this iteration boosts
    (`TargetSet.rows_for`); `exact` the boost oracle (`BoostTargets`, or
    anything with its `targets` signature) and `sigma` its σ of the table
    the walks read; `sampling` what one boosted row samples, refused if
    exact; `alpha` in [0, 1) and `walks` ≥ 1 the run's. Row r's boosted
    target draws from `[seed, t, seat, r, "BOOST"]`, so it depends on
    nothing but those. A drawn row with zero arrival (`BoostTargets.targets`
    skips its play-out, so it costs only its reach), or whose float32 weight
    would equal 1 or α, is left untreated and counted in `skipped`.
    """
    if not isinstance(sampling, Sampling):
        raise ValueError(f"sampling must be a Sampling, got {type(sampling).__name__}")
    if sampling.is_exact:
        raise ValueError(
            "the boost is extra sampling at high-variance nodes, never the exact sum over the tree "
            f"below a row: {sampling!r} draws nothing; EXACT is the tests' oracle, not a run's"
        )
    if not (_is_number(alpha) and 0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be a number in [0, 1) — at 1 every boosted row would weigh 0 — got {alpha!r}")
    if not _is_int(walks) or walks < 1:
        raise ValueError(f"walks must be a positive int (K, walks per seat per iteration), got {walks!r}")
    drawn = _drawn_rows(drawn)
    rows = np.asarray(rows, dtype=np.int32)
    targets = np.asarray(targets, dtype=np.float32)

    regret, arrival = exact.targets(sigma, seat=seat, rows=drawn, sampling=sampling, seed=seed, t=t)
    arrival = np.asarray(arrival, dtype=np.float64)
    weight = boosted_weight(arrival, alpha=alpha, walks=walks).astype(np.float32)
    # Treated: reachable, and never at a weight the noise score reads as a walk row's.
    treated = (arrival > 0.0) & (weight != np.float32(1.0)) & (weight != np.float32(alpha))

    at_treated = np.isin(rows, drawn[treated])
    natural_weight = np.ones(rows.shape[0], dtype=np.float32)
    natural_weight[at_treated] = np.float32(alpha)
    return Injected(
        rows=np.concatenate([rows, drawn[treated].astype(np.int32)]),
        targets=np.concatenate([targets, np.asarray(regret, dtype=np.float32)[treated]]),
        weight=np.concatenate([natural_weight, weight[treated]]),
        boosted=int(treated.sum()),
        skipped=int(drawn.size - treated.sum()),
        turned_down=int(at_treated.sum()),
    )

def _drawn_rows(drawn: np.ndarray) -> np.ndarray:
    """`drawn` as a 1-D int64 array, strictly increasing: a repeat would give one row two boosted rows."""
    drawn = np.asarray(drawn)
    if drawn.ndim != 1 or (drawn.size and drawn.dtype.kind not in "iu"):
        raise ValueError(f"drawn must be a 1-D integer array of table rows, got {drawn.dtype} {drawn.shape}")
    drawn = drawn.astype(np.int64)
    if drawn.size and np.any(np.diff(drawn) <= 0):
        raise ValueError("drawn rows must be sorted and distinct: a repeated row would be boosted twice")
    return drawn

def _is_number(value: object) -> bool:
    return isinstance(value, (numbers.Real, np.floating, np.integer)) and not isinstance(value, (bool, np.bool_))

def _is_int(value: object) -> bool:
    return isinstance(value, (numbers.Integral, np.integer)) and not isinstance(value, (bool, np.bool_))

# ---------------------------------------------------------------------------
# A run's boost: the refresher and the oracle, built once
# ---------------------------------------------------------------------------

class Boost:
    """The stage's fixed parts for one run: who picks the rows, and what computes them.

    `refresher` is the `targeting.Refresher` the script layer built (the
    grader-backed `GraderRefresher`, or a test's stand-in); `exact` the
    `BoostTargets` the boosted rows and σ come from — the whole game's,
    built on first use, unless a restricted one is given. `train.step` asks
    `due`, calls `refresh` every R iterations before seat 0's walks, and
    `phase` after each seat's walks; the set itself lives on the run
    (`DeepRun.targets`), since it is resume state.
    """

    def __init__(self, refresher: Refresher, *, exact: BoostTargets | None = None) -> None:
        self.refresher = refresher
        self._exact = exact

    @property
    def exact(self) -> BoostTargets:
        """The boost oracle, the whole game's when none was given; ~5 s to build, cached per process."""
        if self._exact is None:
            self._exact = BoostTargets.build()
        return self._exact

    def due(self, targets: TargetSet | None, t: int) -> bool:
        """Whether iteration t needs a refresh: no set yet, or the set's R-iteration window has ended."""
        return targets is None or t >= targets.built_at + targets.spec.refresh

    def refresh(self, *, t: int, table, memories, spec: TargetSpec) -> TargetSet:
        """The refresher's set for iteration t, refused by name if it was built for another t or another spec."""
        targets = self.refresher(t=t, table=table, memories=memories)
        if not isinstance(targets, TargetSet):
            raise ValueError(f"the refresher must return a TargetSet, got {type(targets).__name__}")
        if targets.built_at != t:
            raise ValueError(f"the refresher returned a set with built_at {targets.built_at}, asked for iteration {t}")
        if targets.spec != spec:
            raise ValueError(
                f"the refresher's set follows the spec {targets.spec!r}, not the run's {spec!r}: "
                "the two must be one object"
            )
        return targets

    def phase(
        self,
        targets: TargetSet,
        offered,
        *,
        table,
        seat: int,
        t: int,
        seed: int,
        sampling: Sampling,
        alpha: float,
        walks: int,
    ) -> Injected:
        """One seat's advantage offer with this iteration's boosted rows.

        σ is read from `table` now, and the rows are `targets.rows_for(t,
        seat, seed)`: seat 0 and seat 1 draw different rows at the same t.
        `offered` is the walk's `Collected`; `table` the collector's table
        after this phase's `learner.write`, so σ is what the walks read.
        """
        sigma = self.exact.sigma(table)
        drawn = targets.rows_for(t, seat, seed)
        return inject(
            offered.advantage_rows,
            offered.advantage_targets,
            drawn=drawn,
            exact=self.exact,
            sigma=sigma,
            seat=seat,
            t=t,
            seed=seed,
            sampling=sampling,
            alpha=alpha,
            walks=walks,
        )
