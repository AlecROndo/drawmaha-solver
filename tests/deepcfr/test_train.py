"""The loop: one iteration in the spec's order, the oracle pin, the resume, the driver.

No torch here: every learner in this file is a spy or the oracle, so the
whole file runs in the default suite. The claims, in order: a `DeepConfig`
is the spec's and refuses what a run could not use; one `step` runs both
seats' phases as "write, observe, collect, merge, fit" with the right seat,
t, seed and memory at every join; with the oracle for a net the loop IS
rung 3's LCFR, bit for bit over three iterations once the samples are
rounded through the memory's float32, and to float32 precision against the
untouched `train_lockstep`; a run saved and loaded continues exactly; and
`run_deep` keeps its grid, its nets and its log, stops on a signal or a
deadline with its state saved, and refuses a changed sample by name.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.deepcfr.collect import Collector
from drawmaha_solver.deepcfr.reservoir import TARGET_WIDTH, Memory, Reservoir, reservoir_stream
import drawmaha_solver.deepcfr.train as train
from drawmaha_solver.deepcfr.boost import Boost
from drawmaha_solver.deepcfr.targeting import TargetSet, TargetSpec, row_offsets
from drawmaha_solver.deepcfr.train import (
    ADDED_LATER,
    GRID,
    MARKS,
    NETS,
    PROGRESS,
    RESUME,
    SAMPLE_FIELDS,
    STATE,
    DeepConfig,
    DeepRun,
    Fitted,
    OracleLearner,
    OracleModel,
    config_from_saved,
    current_iterate,
    load_run,
    mark_name,
    net_name,
    new_run,
    require_same_run,
    run_deep,
    save_run,
    saved_config,
    step,
)
from drawmaha_solver.minidrawmaha.rules.enumeration import private_keys, public_decision_points
from drawmaha_solver.minidrawmaha.rules.features import layout
from drawmaha_solver.minidrawmaha.rules.game import Action, InfoSet, chip_state, random_deal
from drawmaha_solver.minidrawmaha.solver.lockstep import (
    SEATS,
    apply_arrays,
    flatten,
    new_lockstep,
    train_lockstep,
    walk,
)
from drawmaha_solver.minidrawmaha.solver.boost_target import EXACT, BoostTargets, Sampling
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.solver.regret_rules import Average, RegretRule
from drawmaha_solver.minidrawmaha.answer.strategy import StrategyInfo
from drawmaha_solver.regret_matching import RegretMatcher

# ---------------------------------------------------------------------------
# Spies
# ---------------------------------------------------------------------------

class Spy:
    """One event list shared by a learner, a collector wrapper and an observer.

    As a learner its `write` touches nothing, so the table stays uniform and
    the walks can be replayed by hand; its models are the fit's seed, so a
    write's models say which fit they came from.
    """

    def __init__(self, inner: Collector) -> None:
        self.inner = inner
        self.events: list[tuple] = []

    # -- the learner --
    def fit(self, rows, t, targets, weight, *, seed) -> Fitted:
        self.events.append(("fit", seed, (rows.copy(), t.copy(), targets.copy(), weight.copy())))
        return Fitted(("model", seed), None)

    def write(self, table, models) -> None:
        self.events.append(("write", tuple(models)))

    def save(self, model, path: Path) -> None:
        Path(path).write_text(json.dumps(model))

    def load(self, path: Path):
        name, seed = json.loads(Path(path).read_text())
        return (name, tuple(seed))

    # -- the collector --
    @property
    def table(self):
        return self.inner.table

    def collect(self, *, seat, t, walks):
        self.events.append(("collect", seat, t, walks))
        return self.inner.collect(seat=seat, t=t, walks=walks)

    # -- the observer --
    def observe(self, table, t, seat) -> None:
        self.events.append(("observe", t, seat))

    def of(self, kind: str) -> list[tuple]:
        return [event[1:] for event in self.events if event[0] == kind]

@pytest.fixture(autouse=True)
def importable_tests(monkeypatch):
    # Spawned workers unpickle a test-defined job by module name, which
    # resolves only with the repo root on the path they inherit.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))

@pytest.fixture(scope="module")
def collector():
    with Collector(seed=3, workers=0) as run:
        yield run

SMALL = DeepConfig(seed=3, walks=4, workers=0, capacity=100_000)

def same_memory(first: Reservoir, second: Reservoir) -> bool:
    return (
        first.offered == second.offered
        and first.capacity == second.capacity
        and all(
            a.dtype == b.dtype and np.array_equal(a, b)
            for a, b in zip(first.contents(), second.contents(), strict=True)
        )
    )

# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------

def test_config_defaults_are_the_specs():
    config = DeepConfig()
    assert (config.seed, config.walks, config.capacity) == (0, 10_000, 50_000_000)
    assert config.grid == GRID == (10, 20, 50, 100, 200, 500, 1_000, 2_000)
    assert config.save_every_s == 1_800.0
    assert (config.width, config.depth, config.steps, config.batch) == (64, 3, 4_000, 10_000)
    assert (config.lr, config.clip, config.threads) == (1e-3, 1.0, 8)
    # The one departure from the 2026-10-07 spec: targets divided by √pot
    # before the refit, chosen 2026-10-27 from the pot arms in the run log.
    assert config.pot_power == 0.5
    # No aim, so no boost; what a boosted row would sample is the walk's own sampling.
    assert config.target is None and config.boost == Sampling()

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", -1),
        ("seed", True),
        ("walks", 0),
        ("workers", -1),
        ("capacity", 0),
        ("width", 0),
        ("depth", 0),
        ("steps", 0),
        ("batch", 0),
        ("threads", 0),
        ("lr", 0.0),
        ("clip", float("nan")),
        ("grid", (10, 10)),
        ("grid", (20, 10)),
        ("grid", (0, 10)),
        ("save_every_s", -1.0),
        ("device", "cuda"),
        ("pot_power", -0.1),
        ("pot_power", 1.5),
        ("pot_power", float("nan")),
        ("target", "line"),
        ("target", {"granularity": "line", "alpha": 0.0}),
        ("boost", EXACT),
        ("boost", Sampling(opponent_actions=None, board_cards=14, opponent_draws=12, hero_draws=12)),
        ("boost", "walk"),
    ],
)
def test_config_refuses_bad_values_by_name(field, value):
    with pytest.raises(ValueError, match=field):
        DeepConfig(**{field: value})

LINE = TargetSpec(granularity="line", alpha=0.0)

def test_the_boost_fields_are_sample_fields_added_later():
    # A run saved before the fields existed had no aim; its sampling is the
    # default so a resume under today's defaults is not refused.
    assert ADDED_LATER["target"] is None and ADDED_LATER["boost"] == Sampling()
    assert {"target", "boost"} <= set(SAMPLE_FIELDS)
    with pytest.raises(ValueError, match="target"):
        require_same_run(DeepConfig(), DeepConfig(target=LINE))
    with pytest.raises(ValueError, match="boost"):
        require_same_run(DeepConfig(), DeepConfig(boost=Sampling(playouts=2)))

def test_a_config_with_a_target_and_a_sampling_round_trips_through_json():
    config = DeepConfig(
        target=TargetSpec(granularity="line_class", alpha=0.0, budget=7, leak="current"),
        boost=Sampling(opponent_range=3, playouts=2),
    )
    assert config_from_saved(json.loads(json.dumps(asdict(config)))) == config
    assert config_from_saved(json.loads(json.dumps(asdict(DeepConfig())))) == DeepConfig()

def test_a_saved_aim_missing_or_carrying_an_unknown_field_is_refused_by_name():
    carried = json.loads(json.dumps(asdict(DeepConfig(target=LINE))))
    del carried["target"]["leak"]
    with pytest.raises(ValueError, match="leak"):
        config_from_saved(carried)
    carried = json.loads(json.dumps(asdict(DeepConfig(target=LINE))))
    carried["boost"]["coinage"] = 3
    with pytest.raises(ValueError, match="coinage"):
        config_from_saved(carried)

def test_require_same_run_names_every_changed_sample_field():
    asked = DeepConfig(seed=1, walks=5, width=32, workers=2, grid=(1,), save_every_s=0.0)
    with pytest.raises(ValueError, match=r"seed.*walks.*width") as caught:
        require_same_run(DeepConfig(), asked)
    for free in ("workers", "grid", "save_every_s"):
        assert free not in str(caught.value)
    assert free not in SAMPLE_FIELDS
    require_same_run(DeepConfig(), DeepConfig(workers=2, grid=(1,), save_every_s=0.0))

def test_a_resume_may_move_the_refit_to_the_gpu_but_not_change_its_threads():
    assert "device" not in SAMPLE_FIELDS
    require_same_run(DeepConfig(), DeepConfig(device="mps"))
    with pytest.raises(ValueError, match="threads") as caught:
        require_same_run(DeepConfig(), DeepConfig(device="mps", threads=4))
    assert "device" not in str(caught.value)

def test_new_run_is_empty():
    run = new_run(SMALL)
    assert isinstance(run, DeepRun)
    assert run.t == 0 and run.nets == [None, None] and run.config is SMALL
    for memory in (*run.advantage, run.policy):
        assert memory.capacity == SMALL.capacity and memory.offered == 0

# ---------------------------------------------------------------------------
# One iteration
# ---------------------------------------------------------------------------

def test_one_iteration_runs_both_phases_in_the_specs_order(collector):
    spy = Spy(collector)
    run = new_run(SMALL)
    stats = step(run, spy, spy, observers=(spy.observe,))
    assert [event[0] for event in spy.events] == ["write", "observe", "collect", "fit"] * 2
    assert spy.of("collect") == [(0, 1, 4), (1, 1, 4)]
    assert spy.of("observe") == [(1, 1), (1, 0)]
    assert [seed for seed, _ in spy.of("fit")] == [(3, 1, 0), (3, 1, 1)]
    assert [models for (models,) in spy.of("write")] == [
        (None, None),
        (("model", (3, 1, 0)), None),
    ]
    assert run.t == 1
    assert run.nets == [("model", (3, 1, 0)), ("model", (3, 1, 1))]
    assert stats.t == 1 and stats.seconds > 0
    for stage in (stats.materialise, stats.collect, stats.merge, stats.fit):
        assert len(stage) == 2 and all(s >= 0 for s in stage)
    assert stats.loss == (None, None)
    assert stats.rows == (run.advantage[0].offered, run.advantage[1].offered)
    # No aim: the boost's stages took no time and added nothing.
    assert stats.refresh == 0.0 and stats.boost == (0.0, 0.0)
    assert stats.boosted == (0, 0) and stats.skipped == (0, 0)

def test_the_second_iteration_writes_both_nets_and_the_fresh_one_before_seat_one(collector):
    spy = Spy(collector)
    run = new_run(SMALL)
    step(run, spy, spy)
    step(run, spy, spy)
    m = lambda t, p: ("model", (3, t, p))  # noqa: E731
    assert [models for (models,) in spy.of("write")] == [
        (None, None),
        (m(1, 0), None),
        (m(1, 0), m(1, 1)),
        (m(2, 0), m(1, 1)),
    ]
    assert spy.of("collect") == [(0, 1, 4), (1, 1, 4), (0, 2, 4), (1, 2, 4)]
    assert run.t == 2 and run.nets == [m(2, 0), m(2, 1)]

def test_fit_receives_the_memory_as_it_stands_after_the_merge(collector):
    spy = Spy(collector)
    run = new_run(SMALL)
    step(run, spy, spy)
    step(run, spy, spy)
    fits = spy.of("fit")
    for seat in SEATS:
        _, (rows, t, targets, weight) = [fit for fit in fits if fit[0][2] == seat][-1]
        for given, resident in zip((rows, t, targets, weight), run.advantage[seat].contents(), strict=True):
            assert given.dtype == resident.dtype
            np.testing.assert_array_equal(given, resident)
        assert rows.shape[0] == run.advantage[seat].size > 0

def test_each_merge_uses_its_own_stream_and_the_policy_memory_takes_both_seats(collector):
    config = replace(SMALL, walks=8, capacity=100)
    spy = Spy(collector)
    run = new_run(config)
    for _ in range(2):
        step(run, spy, spy)
    # The spy's write leaves the table uniform, so the same walks replay here.
    advantage = (Reservoir(100), Reservoir(100))
    policy = Reservoir(100)
    policy_rows = 0
    for t in (1, 2):
        for seat in SEATS:
            rows = collector.collect(seat=seat, t=t, walks=8)
            advantage[seat].merge(
                rows.advantage_rows,
                rows.advantage_targets,
                t=t,
                rng=reservoir_stream(seed=3, t=t, seat=seat, memory=Memory.ADVANTAGE),
            )
            policy.merge(
                rows.policy_rows,
                rows.policy_targets,
                t=t,
                rng=reservoir_stream(seed=3, t=t, seat=seat, memory=Memory.POLICY),
            )
            policy_rows += rows.policy_rows.shape[0]
    for seat in SEATS:
        assert run.advantage[seat].offered > 100  # over capacity: the draws mattered
        assert same_memory(run.advantage[seat], advantage[seat])
    assert same_memory(run.policy, policy)
    assert run.policy.offered == policy_rows > 100

# ---------------------------------------------------------------------------
# Gate 1: with a perfect net, the loop is LCFR
# ---------------------------------------------------------------------------

def lcfr_rounded(seed: int, walks: int, iterations: int):
    """LCFR from rung 3's own walk, flatten and apply_arrays, each sample rounded through float32.

    The one difference from `train_lockstep` is the cast `collect.rows_of`
    applies to every sample on its way into the memory. Yields the table
    after each iteration.
    """
    table = PackedTable.whole_game()
    for t in range(1, iterations + 1):
        for seat in SEATS:
            flats = [flatten(walk(table, random_deal, seed, t, seat, k, ()), 0) for k in range(walks)]
            flats = [
                replace(f, regret_values=f.regret_values.astype(np.float32).astype(np.float64))
                for f in flats
            ]
            apply_arrays(table, flats, RegretRule.LCFR, t)
        yield table

def lcfr_exact(seed: int, walks: int, iterations: int):
    """The untouched reference: `train_lockstep` with K workers under the LCFR rule."""
    solve = new_lockstep(seed, workers=walks, rule="lcfr")
    for _ in range(iterations):
        yield train_lockstep(solve, 1).table

def oracle_loop(seed: int, walks: int, workers: int, iterations: int):
    """The loop with the oracle and an unbounded memory; yields (run, table) after each iteration.

    The table is written once more after each step: the last write inside an
    iteration precedes seat 1's refit, and what LCFR's table holds after
    iteration t is what the loop's table holds as iteration t + 1 opens.
    """
    config = DeepConfig(seed=seed, walks=walks, workers=workers, capacity=walks * iterations * 100)
    run = new_run(config)
    learner = OracleLearner()
    with Collector(seed=seed, workers=workers) as collector:
        for _ in range(iterations):
            step(run, collector, learner)
            learner.write(collector.table, run.nets)
            yield run, collector.table

@pytest.mark.parametrize("workers", [0, 2])
def test_the_oracle_loop_is_lcfr_bit_for_bit(workers):
    iterations = 0
    for (run, deep), reference in zip(oracle_loop(5, 40, workers, 3), lcfr_rounded(5, 40, 3)):
        np.testing.assert_array_equal(deep.cumulative_regret, reference.cumulative_regret)
        assert deep.cumulative_regret.any()
        iterations += 1
    assert iterations == 3
    for memory in run.advantage:
        assert 0 < memory.offered <= memory.capacity  # nothing was ever evicted

def test_the_oracle_loop_equals_train_lockstep_to_float32_precision():
    for (_, deep), reference in zip(oracle_loop(5, 40, 0, 3), lcfr_exact(5, 40, 3)):
        exact = reference.cumulative_regret
        np.testing.assert_allclose(deep.cumulative_regret, exact, rtol=1e-5, atol=1e-6)
        # A row the walks never reached is exactly 0.0 in both.
        np.testing.assert_array_equal(deep.cumulative_regret == 0.0, exact == 0.0)
        # The float32 cast on the way into the memory is why there are two pins.
        assert not np.array_equal(deep.cumulative_regret, exact)

def test_the_uniform_column_leaves_the_walks_regrets_untouched():
    table = PackedTable.whole_game(extra_averages=1)
    table.cumulative_regret[:] = np.random.default_rng(1).standard_normal(table.cumulative_regret.size)
    for k in range(20):
        bare = walk(table, random_deal, 5, 2, k % 2, k, ())
        asked = walk(table, random_deal, 5, 2, k % 2, k, (Average.UNIFORM,))
        assert [(key, slot, r.tobytes()) for key, slot, r in bare.regret] == [
            (key, slot, r.tobytes()) for key, slot, r in asked.regret
        ]

def test_the_oracle_writes_into_the_table_it_is_handed_and_never_rebinds_it():
    table = PackedTable.whole_game()
    before = table.cumulative_regret
    before[:] = 7.0
    OracleLearner().write(table, (None, None))
    assert table.cumulative_regret is before and not before.any()

def test_oracle_models_round_trip(tmp_path, collector):
    run = new_run(SMALL)
    learner = OracleLearner()
    step(run, collector, learner)
    learner.save(run.nets[0], tmp_path / "p0-t00001.pt")
    back = learner.load(tmp_path / "p0-t00001.pt")
    for name in ("rows", "t", "targets", "weight"):
        a, b = getattr(run.nets[0], name), getattr(back, name)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    assert run.nets[0].rows.shape[0] == run.advantage[0].size
    assert run.nets[0].weight.dtype == np.float32 and (run.nets[0].weight == 1.0).all()

def test_an_oracle_model_saved_without_a_weight_loads_with_ones(tmp_path):
    old = tmp_path / "p0-t00003.pt"
    with old.open("wb") as file:
        np.savez(file, rows=np.arange(4, dtype=np.int32), t=np.full(4, 3, np.int32), targets=np.ones((4, TARGET_WIDTH), np.float32))
    back = OracleLearner().load(old)
    assert back.weight.dtype == np.float32 and back.weight.tolist() == [1.0] * 4

def oracle_table(*models: OracleModel) -> np.ndarray:
    table = PackedTable.whole_game()
    OracleLearner().write(table, models)
    return table.cumulative_regret

def test_the_oracle_weighs_a_row_by_its_weight():
    # One real row with the sample r = (0.5, −0.25, 0, 0) at t = 3. A weight
    # of 2 is the row offered twice; a weight of 0 is the row never offered.
    # The equality is to the bit because these are the only samples at the
    # column: 0 + r + r is 2r exactly. Beside other samples the two would
    # agree only up to rounding (see the `OracleLearner` docstring).
    row = np.array([first_row_at(((),))] * 2, dtype=np.int32)
    t = np.array([3, 3], dtype=np.int32)
    r = np.array([[0.5, -0.25, 0.0, 0.0]] * 2, dtype=np.float32)
    w = lambda *values: np.array(values, dtype=np.float32)  # noqa: E731
    twice = oracle_table(OracleModel(row, t, r, w(1.0, 1.0)))
    weighted = oracle_table(OracleModel(row[:1], t[:1], r[:1], w(2.0)))
    np.testing.assert_array_equal(twice, weighted)
    assert twice.any()
    once = oracle_table(OracleModel(row[:1], t[:1], r[:1], w(1.0)))
    with_an_inert_row = oracle_table(OracleModel(row, t, r, w(1.0, 0.0)))
    np.testing.assert_array_equal(once, with_an_inert_row)
    assert not np.array_equal(once, twice)

def test_the_oracle_keeps_each_weight_on_its_own_row_across_widths():
    # Two rows of different widths in one model — the opening spot, 2 wide
    # (check or pot, no bet to fold to), and the spot after pot, pot, 3 wide
    # — each with its own sample and weight. `write` flattens the samples
    # row by row through `legal` and repeats each weight by its row's width;
    # a weight landing on a neighbour's columns would make the table differ
    # from the rows offered weight-many times. Exact as above: 0 + r + r + r
    # and 3r are each one rounding of the same real number.
    rows = np.array([first_row_at(((),)), first_row_at(((Action.POT, Action.POT),))], dtype=np.int32)
    widths = layout().width[layout().locate_rows(rows.astype(np.int64))[0]]
    assert widths.tolist() == [2, 3]
    t = np.array([3, 3], dtype=np.int32)
    r = np.array([[0.5, -0.25, 0.0, 0.0], [-0.75, 0.375, 0.125, 0.0]], dtype=np.float32)
    weighted = oracle_table(OracleModel(rows, t, r, np.array([3.0, 2.0], np.float32)))
    repeated = oracle_table(
        OracleModel(np.repeat(rows, [3, 2]), np.repeat(t, [3, 2]), np.repeat(r, [3, 2], axis=0), np.ones(5, np.float32))
    )
    np.testing.assert_array_equal(weighted, repeated)
    swapped = oracle_table(OracleModel(rows, t, r, np.array([2.0, 3.0], np.float32)))
    assert not np.array_equal(weighted, swapped)

# ---------------------------------------------------------------------------
# The resume state
# ---------------------------------------------------------------------------

def same_targets(first: TargetSet | None, second: TargetSet | None) -> bool:
    if (first is None) != (second is None):
        return False
    if first is None:
        return True
    return (
        first.built_at == second.built_at
        and first.spec == second.spec
        and all(np.array_equal(first.rows[s], second.rows[s]) and np.array_equal(first.weight[s], second.weight[s]) for s in SEATS)
    )

def same_run(first: DeepRun, second: DeepRun) -> bool:
    """t, config, the three memories, the oracle models and the target set, all bytewise."""
    if first.t != second.t or first.config != second.config or not same_targets(first.targets, second.targets):
        return False
    if not all(same_memory(a, b) for a, b in zip((*first.advantage, first.policy), (*second.advantage, second.policy), strict=True)):
        return False
    for a, b in zip(first.nets, second.nets, strict=True):
        if (a is None) != (b is None):
            return False
        if a is not None and not all(
            getattr(a, name).dtype == getattr(b, name).dtype
            and np.array_equal(getattr(a, name), getattr(b, name))
            for name in ("rows", "t", "targets")
        ):
            return False
    return True

def written_by(run: DeepRun) -> np.ndarray:
    """The regret slots the oracle writes for the run's nets, on a fresh whole-game table."""
    table = PackedTable.whole_game()
    OracleLearner().write(table, run.nets)
    return table.cumulative_regret

def straight(config: DeepConfig, iterations: int) -> DeepRun:
    run = new_run(config)
    learner = OracleLearner()
    with Collector(seed=config.seed, workers=config.workers) as collector:
        for _ in range(iterations):
            step(run, collector, learner)
    return run

def test_save_and_load_round_trip(tmp_path, collector):
    run = new_run(SMALL)
    learner = OracleLearner()
    step(run, collector, learner)
    step(run, collector, learner)
    save_run(run, tmp_path, learner=learner)
    back = load_run(tmp_path, learner=learner)
    assert same_run(run, back)
    assert not list(tmp_path.glob("*.tmp"))
    # No stateful observer, so no observer file: the PR 4 layout exactly.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "advantage-p0-t00002.npz",
        "advantage-p1-t00002.npz",
        "p0-t00002.pt",
        "p1-t00002.pt",
        "policy-t00002.npz",
        STATE,
    ]
    assert json.loads((tmp_path / STATE).read_text())["t"] == 2

def test_resume_equals_uninterrupted(tmp_path):
    config = DeepConfig(seed=5, walks=8, workers=0, capacity=200)
    learner = OracleLearner()
    resumed = new_run(config)
    with Collector(seed=5, workers=0) as collector:
        step(resumed, collector, learner)
        step(resumed, collector, learner)
    save_run(resumed, tmp_path, learner=learner)
    resumed = load_run(tmp_path, learner=learner)
    with Collector(seed=5, workers=0) as collector:
        step(resumed, collector, learner)
    uninterrupted = straight(config, 3)
    assert same_run(resumed, uninterrupted)
    np.testing.assert_array_equal(written_by(resumed), written_by(uninterrupted))
    for memory in resumed.advantage:
        assert memory.offered > memory.capacity  # the resumed draws evicted, as the straight ones did

def test_a_save_interrupted_before_state_json_keeps_the_previous_state(tmp_path, collector, monkeypatch):
    run = new_run(SMALL)
    learner = OracleLearner()
    step(run, collector, learner)
    save_run(run, tmp_path, learner=learner)
    first = load_run(tmp_path, learner=learner)
    step(run, collector, learner)

    def crash(directory, state):
        raise OSError("disk gone")

    monkeypatch.setattr(train, "_write_state", crash)
    with pytest.raises(OSError, match="disk gone"):
        save_run(run, tmp_path, learner=learner)
    assert same_run(load_run(tmp_path, learner=learner), first)
    monkeypatch.undo()
    save_run(run, tmp_path, learner=learner)
    assert load_run(tmp_path, learner=learner).t == 2
    assert not any("t00001" in p.name for p in tmp_path.iterdir())

def test_load_refuses_a_missing_state_or_a_missing_file(tmp_path, collector):
    with pytest.raises(ValueError, match="state.json"):
        load_run(tmp_path, learner=OracleLearner())
    run = new_run(SMALL)
    learner = OracleLearner()
    step(run, collector, learner)
    save_run(run, tmp_path, learner=learner)
    (tmp_path / "policy-t00001.npz").unlink()
    with pytest.raises(ValueError, match="policy-t00001.npz"):
        load_run(tmp_path, learner=learner)

# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

DRIVEN = DeepConfig(seed=5, walks=5, workers=0, capacity=100_000, grid=(4, 8))

def drive(out: Path, target: int, config: DeepConfig = DRIVEN, **kw) -> DeepRun:
    return run_deep(out, config, target=target, learner=OracleLearner(), log=lambda _: None, **kw)

def lines(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / PROGRESS).read_text().splitlines()]

def loaded(out: Path, *parts: str) -> DeepRun:
    return load_run(out.joinpath(*parts), learner=OracleLearner())

def test_a_run_keeps_its_grid_its_nets_and_its_log_and_ends_at_its_target(tmp_path):
    commits = []
    run = drive(tmp_path, 10, commit=lambda: commits.append(1))
    assert run.t == 10
    for t in (4, 8):
        kept = loaded(tmp_path, MARKS, mark_name(t))
        assert kept.t == t and same_run(kept, straight(DRIVEN, t))
    assert sorted(p.name for p in (tmp_path / NETS).iterdir()) == sorted(
        net_name(seat, t) for t in range(1, 11) for seat in SEATS
    )
    assert same_run(loaded(tmp_path, RESUME), run)
    log = lines(tmp_path)
    assert [line["t"] for line in log] == list(range(1, 11))
    assert log[-1]["stopped"] == "done" and "stopped" not in log[0]
    for line in log:
        assert line["target"] == 10
        assert all(len(line["seconds"][stage]) == 2 for stage in ("materialise", "collect", "merge", "fit"))
        assert line["seconds"]["iteration"] > 0
        assert line["rows"]["advantage"][0] > 0 and line["rows"]["resident"]["policy"] > 0
        assert line["loss"] == [None, None]
        assert line["iterations_per_s"] > 0 and line["eta_h"] >= 0
    assert set(log[3]["wrote"]) >= {f"{NETS}/{net_name(0, 4)}", f"{MARKS}/{mark_name(4)}", RESUME}
    assert log[0]["wrote"] == [f"{NETS}/{net_name(0, 1)}", f"{NETS}/{net_name(1, 1)}"]
    assert RESUME in log[-1]["wrote"]
    assert len(commits) == 10
    assert not list(tmp_path.rglob("*.tmp"))

def test_a_run_stopped_and_restarted_equals_one_straight_run(tmp_path):
    assert drive(tmp_path, 6).t == 6
    run = drive(tmp_path, 10)
    assert same_run(run, straight(DRIVEN, 10))
    np.testing.assert_array_equal(written_by(run), written_by(straight(DRIVEN, 10)))
    before = lines(tmp_path)
    assert drive(tmp_path, 10).t == 10  # already there: nothing to do, nothing written
    assert lines(tmp_path) == before

def test_sigterm_finishes_the_iteration_saves_and_resumes_exactly(tmp_path):
    raised = []

    def log(line: str) -> None:
        # The first progress record, i.e. after the first iteration's save.
        if line.startswith("{") and not raised:
            raised.append(1)
            signal.raise_signal(signal.SIGTERM)

    run = run_deep(tmp_path, DRIVEN, target=10, learner=OracleLearner(), log=log)
    # The signal landed during iteration 1's save; the run must not pay for
    # iteration 2 before noticing it.
    assert run.t == 1
    assert loaded(tmp_path, RESUME).t == 1
    log_lines = lines(tmp_path)
    assert log_lines[-1]["stopped"] == "SIGTERM" and log_lines[-1]["t"] == 1
    assert RESUME in log_lines[-1]["wrote"]
    assert same_run(drive(tmp_path, 10), straight(DRIVEN, 10))

def test_a_deadline_stops_the_run_with_its_state_saved(tmp_path):
    # Read before each iteration as well as after: a deadline already past
    # starts nothing, and the resume state still says where the run stands.
    run = drive(tmp_path, 10, deadline_s=0.0)
    assert run.t == 0
    assert loaded(tmp_path, RESUME).t == 0
    assert lines(tmp_path)[-1] | {"stopped": "deadline", "t": 0} == lines(tmp_path)[-1]
    drive(tmp_path, 3)
    run = drive(tmp_path, 10, deadline_s=0.0)
    assert run.t == 3 and loaded(tmp_path, RESUME).t == 3
    assert lines(tmp_path)[-1]["stopped"] == "deadline" and lines(tmp_path)[-1]["wrote"] == []

def test_a_resume_refuses_a_changed_config_by_name(tmp_path):
    drive(tmp_path, 2)
    with pytest.raises(ValueError, match="walks"):
        drive(tmp_path, 3, replace(DRIVEN, walks=6))
    assert drive(tmp_path, 3, replace(DRIVEN, workers=0, save_every_s=0.0, grid=(3,))).t == 3
    assert loaded(tmp_path, MARKS, mark_name(3)).t == 3

def test_the_resume_file_is_rewritten_on_the_clock_and_at_the_end_only_once(tmp_path):
    drive(tmp_path, 3, replace(DRIVEN, grid=(), save_every_s=1e9))
    assert [RESUME in line["wrote"] for line in lines(tmp_path)] == [False, False, True]
    drive(tmp_path, 5, replace(DRIVEN, grid=(), save_every_s=0.0))
    assert [RESUME in line["wrote"] for line in lines(tmp_path)[3:]] == [True, True]

# ---------------------------------------------------------------------------
# The boost inside the loop
# ---------------------------------------------------------------------------

def round_two_rows(seat: int, count: int = 3) -> np.ndarray:
    """The first `count` rows of `seat`'s first round-2 point: real rows of the right seat, cheap to boost."""
    points = public_decision_points()
    offsets = row_offsets()
    p = next(i for i, point in enumerate(points) if point.player == seat and point.board_cards == 2)
    return np.arange(offsets[p], offsets[p] + count)

class FixedRefresher:
    """An `infoset` set of fixed rows built at every call; records each call's t and what the memories had been offered."""

    def __init__(self, spec: TargetSpec, rows: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        self.spec = spec
        self.rows = rows if rows is not None else (round_two_rows(0), round_two_rows(1))
        self.calls: list[tuple[int, tuple[int, int]]] = []

    def __call__(self, *, t, table, memories):
        self.calls.append((t, tuple(memory.offered for memory in memories)))
        weight = tuple(np.ones(rows.size) for rows in self.rows)
        return TargetSet(spec=self.spec, built_at=t, rows=self.rows, weight=weight, groups=self.rows)

class FakeExact:
    """σ is the table's first regret slot as it stands when read; every drawn row arrives at 0.3 with a zero regret."""

    def __init__(self) -> None:
        self.sigmas: list[float] = []

    def sigma(self, table):
        self.sigmas.append(float(table.cumulative_regret[0]))
        return "σ"

    def targets(self, sigma, *, seat, rows, sampling, seed, t):
        return np.zeros((len(rows), TARGET_WIDTH)), np.full(len(rows), 0.3)

BOOST_SPEC = TargetSpec(granularity="infoset", alpha=0.0, budget=3, refresh=2)
BOOSTED = replace(DRIVEN, target=BOOST_SPEC)

def a_boost(spec: TargetSpec = BOOST_SPEC) -> tuple[Boost, FixedRefresher]:
    refresher = FixedRefresher(spec)
    return Boost(refresher, exact=FakeExact()), refresher

def straight_boosted(config: DeepConfig, iterations: int) -> DeepRun:
    run = new_run(config)
    with Collector(seed=config.seed, workers=config.workers) as collector:
        for _ in range(iterations):
            step(run, collector, OracleLearner(), boost=a_boost(config.target)[0])
    return run

def test_the_refresh_runs_at_the_first_iteration_and_every_r_after(tmp_path):
    boost, refresher = a_boost()
    run = drive(tmp_path, 5, BOOSTED, boost=boost)
    assert [t for t, _ in refresher.calls] == [1, 3, 5]
    assert run.targets.built_at == 5

def test_boosted_rows_land_after_the_natural_rows_and_the_policy_memory_is_untouched(collector):
    exact = BoostTargets.build()
    collector.table.cumulative_regret[:] = 0.0  # uniform σ, so every round-2 row the deck allows is reachable
    sigma = exact.sigma(collector.table)
    drawn = tuple(
        rows[exact.arrivals(sigma, seat=seat, rows=rows) > 0][:3] for seat, rows in ((0, round_two_rows(0, 8)), (1, round_two_rows(1, 8)))
    )
    assert all(rows.size == 3 for rows in drawn)
    spec = replace(BOOST_SPEC, refresh=25)
    plain, boosted = Spy(collector), Spy(collector)
    run_plain = new_run(SMALL)
    step(run_plain, plain, plain)
    run_boosted = new_run(replace(SMALL, target=spec))
    stats = step(run_boosted, boosted, boosted, boost=Boost(FixedRefresher(spec, drawn), exact=exact))
    for seat in SEATS:
        rows, t, targets, weight = run_boosted.advantage[seat].contents()
        natural_rows, natural_t, natural_targets, natural_weight = run_plain.advantage[seat].contents()
        a = natural_rows.size
        np.testing.assert_array_equal(rows[:a], natural_rows)
        np.testing.assert_array_equal(t, np.ones(a + 3, dtype=np.int32))
        np.testing.assert_array_equal(targets[:a], natural_targets)
        np.testing.assert_array_equal(natural_weight, np.ones(a, dtype=np.float32))
        np.testing.assert_array_equal(weight[:a], np.where(np.isin(natural_rows, drawn[seat]), 0, 1).astype(np.float32))
        np.testing.assert_array_equal(rows[a:], drawn[seat])
        expected = (SMALL.walks * exact.arrivals(sigma, seat=seat, rows=drawn[seat])).astype(np.float32)
        np.testing.assert_array_equal(weight[a:], expected)
        assert (weight[a:] > 0).all() and not np.isin(weight[a:], [np.float32(1.0)]).any()
    assert same_memory(run_plain.policy, run_boosted.policy)
    assert stats.boosted == (3, 3) and stats.skipped == (0, 0)
    assert all(s > 0 for s in stats.boost) and stats.refresh > 0

class Stamping(Spy):
    """A Spy whose `write` stamps the table's first slot with how many writes there have been."""

    def __init__(self, inner: Collector) -> None:
        super().__init__(inner)
        self.writes = 0

    def write(self, table, models) -> None:
        self.writes += 1
        table.cumulative_regret[0] = self.writes
        super().write(table, models)

def test_the_boost_reads_sigma_after_this_phases_write(collector):
    spy = Stamping(collector)
    boost, _ = a_boost(replace(BOOST_SPEC, refresh=25))
    run = new_run(replace(SMALL, target=replace(BOOST_SPEC, refresh=25)))
    try:
        step(run, spy, spy, boost=boost)
        step(run, spy, spy, boost=boost)
    finally:
        collector.table.cumulative_regret[0] = 0.0
    # Four phases, four writes, and the σ of each phase read after its own write — never the previous phase's.
    assert boost.exact.sigmas == [1.0, 2.0, 3.0, 4.0]

def test_stats_and_the_progress_line_carry_the_boost(tmp_path):
    boost, _ = a_boost()
    drive(tmp_path, 2, BOOSTED, boost=boost)
    log = lines(tmp_path)
    # The fake refresher and oracle are instant, so the rounded seconds read 0.0; the keys and
    # the counts are what this pins (the real costs are read unrounded in the σ test above and
    # measured in the whole-game smoke).
    assert log[0]["seconds"]["refresh"] >= 0.0 and log[1]["seconds"]["refresh"] == 0.0
    for line in log:
        assert len(line["seconds"]["boost"]) == 2 and all(s >= 0 for s in line["seconds"]["boost"])
        assert line["rows"]["boosted"] == [3, 3] and line["rows"]["skipped"] == [0, 0]
    drive(tmp_path / "plain", 1)
    line = lines(tmp_path / "plain")[0]
    assert line["seconds"]["refresh"] == 0.0 and line["seconds"]["boost"] == [0.0, 0.0]
    assert line["rows"]["boosted"] == [0, 0] and line["rows"]["skipped"] == [0, 0]

def test_the_target_set_rides_the_resume_state_and_is_not_regraded(tmp_path):
    spec = replace(BOOST_SPEC, refresh=4)
    config = replace(DRIVEN, target=spec)
    boost, refresher = a_boost(spec)
    drive(tmp_path, 3, config, boost=boost)
    assert [t for t, _ in refresher.calls] == [1]
    state = json.loads((tmp_path / RESUME / STATE).read_text())
    assert state["files"]["targets"] == "targets-t00003.npz"
    assert (tmp_path / RESUME / "targets-t00003.npz").exists()
    assert loaded(tmp_path, RESUME).targets.built_at == 1
    boost, refresher = a_boost(spec)
    run = drive(tmp_path, 6, config, boost=boost)
    assert [t for t, _ in refresher.calls] == [5]  # the saved set served iteration 4
    assert (tmp_path / MARKS / mark_name(4) / "targets-t00004.npz").exists()
    assert same_run(run, straight_boosted(config, 6))
    assert same_run(loaded(tmp_path, MARKS, mark_name(4)), straight_boosted(config, 4))

def test_a_resume_whose_saved_set_has_expired_refreshes_before_its_first_walks(tmp_path):
    boost, _ = a_boost()
    drive(tmp_path, 2, BOOSTED, boost=boost)  # the set built at 1 served iterations 1 and 2
    offered_at_save = tuple(memory.offered for memory in loaded(tmp_path, RESUME).advantage)
    boost, refresher = a_boost()
    drive(tmp_path, 3, BOOSTED, boost=boost)
    assert refresher.calls == [(3, offered_at_save)]

def test_a_run_stopped_and_restarted_with_the_boost_equals_one_straight_run(tmp_path):
    assert drive(tmp_path, 6, BOOSTED, boost=a_boost()[0]).t == 6
    run = drive(tmp_path, 10, BOOSTED, boost=a_boost()[0])
    assert same_run(run, straight_boosted(BOOSTED, 10))
    np.testing.assert_array_equal(written_by(run), written_by(straight_boosted(BOOSTED, 10)))

def test_run_deep_refuses_a_target_without_a_boost_and_a_boost_without_a_target(tmp_path):
    with pytest.raises(ValueError, match="boost"):
        drive(tmp_path, 1, BOOSTED)
    with pytest.raises(ValueError, match="target"):
        drive(tmp_path, 1, DRIVEN, boost=a_boost()[0])
    assert not (tmp_path / NETS).exists()

def test_an_old_state_without_a_target_file_loads_with_none(tmp_path):
    drive(tmp_path, 1)
    state = json.loads((tmp_path / RESUME / STATE).read_text())
    assert state["files"]["targets"] is None
    assert loaded(tmp_path, RESUME).targets is None

@pytest.mark.skipif(
    not os.environ.get("MINIDRAWMAHA_FULL_DEEP"),
    reason="a real refresh (two grader walks, ~2 min) and two boosted iterations on the whole game; set MINIDRAWMAHA_FULL_DEEP=1",
)
def test_two_boosted_iterations_on_the_whole_game_record_their_cost(tmp_path):
    # The line arm at the settled B = 500 with the real grader-backed refresher
    # (leak="current", so no exact average is needed) and the real boost oracle.
    # Prints the refresh and boost seconds so the run log and `train.py`'s
    # timings paragraph can quote them.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "targeting_refresh", Path(__file__).parents[2] / "scripts" / "deepcfr" / "targeting_refresh.py"
    )
    refresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refresh)
    aim = TargetSpec(granularity="line", alpha=0.0, budget=500, refresh=25, leak="current")
    config = DeepConfig(seed=1, walks=200, workers=0, capacity=1_000_000, target=aim, grid=())
    boost = Boost(refresh.GraderRefresher(aim, walks=config.walks, seed=config.seed))
    run = run_deep(tmp_path, config, target=2, learner=OracleLearner(), log=lambda _: None, boost=boost)
    assert run.t == 2 and run.targets.built_at == 1
    log = lines(tmp_path)
    assert len(log) == 2
    assert log[0]["seconds"]["refresh"] > 0 and log[1]["seconds"]["refresh"] == 0.0
    for line in log:
        assert all(s > 0 for s in line["seconds"]["boost"])
        # B rows a seat, less the rare row whose weight would read as a walk row's.
        assert sum(line["rows"]["boosted"]) >= 2 * aim.budget - 10
        assert sum(line["rows"]["skipped"]) <= 10
    print(
        f"\nwhole-game smoke, line arm, B = {aim.budget}, K = {config.walks}: "
        f"refresh {log[0]['seconds']['refresh']} s; boost per seat "
        f"{log[0]['seconds']['boost']} s at t = 1, {log[1]['seconds']['boost']} s at t = 2; "
        f"boosted {log[0]['rows']['boosted']} / {log[1]['rows']['boosted']}, skipped {log[0]['rows']['skipped']} / {log[1]['rows']['skipped']}"
    )

# ---------------------------------------------------------------------------
# The current iterate
# ---------------------------------------------------------------------------

def some_keys() -> list[InfoSet]:
    """Two keys on every public point: both seats, every width, without walking the enumeration."""
    keys = []
    for point in public_decision_points():
        shape = private_keys(board_cards=point.board_cards, discards=point.discards)
        for hole, thrown, board in (shape[0], shape[len(shape) // 2]):
            keys.append(
                InfoSet(
                    player=point.player,
                    hole=hole,
                    discarded=thrown,
                    board=board,
                    draws=point.draws,
                    betting=point.betting,
                )
            )
    return keys

def test_current_iterate_is_regret_matching_row_by_row():
    table = PackedTable.whole_game()
    keys = some_keys()
    rng = np.random.default_rng(2)
    for index, key in enumerate(keys):
        _, start, width = table.slot_of(key)
        kind = index % 4
        if kind == 0:  # a signed mix with something positive: regret matching
            table.cumulative_regret[start : start + width] = rng.standard_normal(width) + 0.3
        elif kind == 1:  # what materialise writes for an all-negative net: a one-hot
            table.cumulative_regret[start + rng.integers(width)] = 1.0
        elif kind == 2:  # all negative, raw: uniform, as the ledger reads it
            table.cumulative_regret[start : start + width] = -rng.random(width) - 0.1
        # kind 3: all zero, uniform
    frozen = current_iterate(table, iteration=7, seed=5)
    assert frozen.info == StrategyInfo(rule="deep", column="current", iteration=7, seed=5, workers=0)
    for key in keys:
        np.testing.assert_array_equal(frozen[key], table[key].strategy().astype(np.float32))
    assert len(frozen) == len(table)

# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def test_cli_parses_the_flags_into_a_config():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_deep_cfr", Path(__file__).parents[2] / "scripts" / "deepcfr" / "run_deep_cfr.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = cli.parse(["--out", "runs/x", "--walks", "7", "--width", "32", "--target", "3"])
    assert cli.config_from_args(args) == DeepConfig(walks=7, width=32)
    assert args.out == Path("runs/x") and args.target == 3
    assert cli.config_from_args(cli.parse(["--out", "y"])) == DeepConfig()
    assert cli.parse(["--out", "y"]).target == 2_000
    assert cli.config_from_args(cli.parse(["--out", "y", "--lr", "5e-4"])).lr == 5e-4
    # A rate DeepConfig cannot use is refused by name, not trained on.
    with pytest.raises(ValueError, match="lr"):
        cli.config_from_args(cli.parse(["--out", "y", "--lr", "0"]))
    assert cli.config_from_args(cli.parse(["--out", "y", "--save-every-s", "600"])).save_every_s == 600.0
    assert cli.config_from_args(cli.parse(["--out", "y", "--pot-power", "0"])).pot_power == 0.0
    assert cli.config_from_args(cli.parse(["--out", "y"])).pot_power == DeepConfig().pot_power == 0.5
    with pytest.raises(ValueError, match="pot_power"):
        cli.config_from_args(cli.parse(["--out", "y", "--pot-power", "2"]))

def load_cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_deep_cfr", Path(__file__).parents[2] / "scripts" / "deepcfr" / "run_deep_cfr.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli

def test_cli_parses_the_boost_flags_into_the_aim_and_the_sampling():
    cli = load_cli()
    args = cli.parse(
        ["--out", "y", "--boost", "line", "--boost-budget", "7", "--boost-alpha", "0", "--boost-leak", "current",
         "--boost-opponent-range", "3", "--boost-playouts", "2"]
    )
    config = cli.config_from_args(args)
    assert config.target == TargetSpec(granularity="line", alpha=0.0, budget=7, leak="current")
    assert config.boost == Sampling(opponent_range=3, playouts=2)
    plain = cli.config_from_args(cli.parse(["--out", "y"]))
    assert plain.target is None and plain.boost == Sampling()
    assert cli.config_from_args(cli.parse(["--out", "y", "--boost-opponent-range", "all"])).boost.opponent_range is None
    assert cli.config_from_args(cli.parse(["--out", "y", "--boost", "none"])).target is None
    full = cli.config_from_args(
        cli.parse(["--out", "y", "--boost", "random", "--boost-refresh", "9", "--boost-pool", "5", "--boost-noise-cap", "3.5",
                   "--boost-round-two-only", "--boost-board-cards", "2", "--boost-opponent-draws", "3", "--boost-hero-draws", "4",
                   "--boost-opponent-actions", "all"])
    )
    assert full.target == TargetSpec(granularity="random", alpha=0.0, refresh=9, pool=5, noise_cap=3.5, round_two_only=True)
    assert full.boost == Sampling(board_cards=2, opponent_draws=3, hero_draws=4, opponent_actions=None)
    # The default α is the settled 0; the settled B and R are TargetSpec's own defaults.
    assert cli.config_from_args(cli.parse(["--out", "y", "--boost", "line"])).target == TargetSpec(granularity="line", alpha=0.0)

def test_cli_refuses_an_exact_boost_at_the_command_line():
    cli = load_cli()
    with pytest.raises(SystemExit):
        cli.parse(["--out", "y", "--boost", "line", "--boost-board-cards", "all", "--boost-opponent-draws", "all",
                   "--boost-hero-draws", "all", "--boost-opponent-actions", "all"])
    with pytest.raises(SystemExit):
        cli.parse(["--out", "y", "--boost-playouts", "0"])
    with pytest.raises(SystemExit):
        cli.parse(["--out", "y", "--boost", "hand"])

@pytest.mark.parametrize(
    "flags",
    [["--boost-budget", "100"], ["--boost-refresh", "9"], ["--boost-alpha", "0.5"], ["--boost-leak", "current"],
     ["--boost-pool", "3"], ["--boost-noise-cap", "5"], ["--boost-round-two-only"]],
)
def test_cli_refuses_an_aim_flag_without_an_arm(flags):
    # `--boost-budget 100` with `--boost` left at none would be today's run with nothing in the log.
    cli = load_cli()
    with pytest.raises(SystemExit):
        cli.parse(["--out", "y", *flags])
    cli.parse(["--out", "y", "--boost", "line", *flags])  # with an arm the same flags are fine
    # The sampling flags describe `DeepConfig.boost`, a field every run carries, so they stand alone.
    cli.parse(["--out", "y", "--boost-playouts", "2"])

def test_main_hands_the_loop_a_boost_only_when_an_arm_is_named(monkeypatch):
    cli = load_cli()
    handed = []
    built = []

    def fake_run_deep(out, config, *, target, observers, boost=None, **kw):
        handed.append((out, config, target, observers, boost))

    class FakeRefresher:
        def __init__(self, spec, *, walks, average=None, seed=0, exact=None):
            built.append((spec, walks, average, seed))

    monkeypatch.setattr(cli, "run_deep", fake_run_deep)
    monkeypatch.setattr(cli, "GraderRefresher", FakeRefresher)
    monkeypatch.setattr(sys, "argv", ["run_deep_cfr.py", "--out", "runs/plain", "--target", "3"])
    cli.main()
    _, config, target, observers, boost = handed[-1]
    assert target == 3 and config.target is None and boost is None and built == []
    assert len(observers) == 1 and observers[0].name == "exact"
    monkeypatch.setattr(sys, "argv", ["run_deep_cfr.py", "--out", "runs/aimed", "--boost", "line_class", "--seed", "4", "--walks", "20"])
    cli.main()
    _, config, _, observers, boost = handed[-1]
    assert isinstance(boost, Boost) and isinstance(boost.refresher, FakeRefresher)
    assert built == [(config.target, 20, observers[0], 4)]
    assert config.target.granularity == "line_class"

def test_cli_passes_the_device_through():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_deep_cfr", Path(__file__).parents[2] / "scripts" / "deepcfr" / "run_deep_cfr.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.config_from_args(cli.parse(["--out", "y", "--device", "cpu"])).device == "cpu"
    cli.mps_available = lambda: True
    assert cli.config_from_args(cli.parse(["--out", "y", "--device", "mps"])).device == "mps"
    # Refused at the command line, before a run is loaded or a walk is spent.
    cli.mps_available = lambda: False
    with pytest.raises(SystemExit):
        cli.parse(["--out", "y", "--device", "mps"])
    with pytest.raises(SystemExit):
        cli.parse(["--out", "y", "--device", "cuda"])

def test_a_resume_that_moves_the_refit_says_so(tmp_path):
    # The oracle learner fits nothing, so an mps config runs on any machine.
    drive(tmp_path, 1)
    moved = replace(DRIVEN, device="mps")
    said = []
    run_deep(tmp_path, moved, target=2, learner=OracleLearner(), log=said.append)
    assert "deep: the refit moves from cpu to mps at iteration 1" in said
    said.clear()
    run_deep(tmp_path, moved, target=3, learner=OracleLearner(), log=said.append)
    assert not any("moves" in line for line in said)

def test_a_state_written_before_the_device_existed_loads_on_the_cpu(tmp_path):
    drive(tmp_path, 1)
    state_path = tmp_path / RESUME / STATE
    state = json.loads(state_path.read_text())
    del state["config"]["device"]
    state_path.write_text(json.dumps(state))
    assert saved_config(tmp_path / RESUME).device == "cpu"
    assert drive(tmp_path, 2).t == 2

def test_a_state_written_before_pot_scaling_existed_was_a_raw_run_and_resumes_as_one(tmp_path):
    # Every run saved before `pot_power` existed fitted raw chips. Its state
    # must read as 0, never as today's default: a resume under the default is
    # refused by name, and one that asks for the run's own weighting goes on.
    drive(tmp_path, 1)
    state_path = tmp_path / RESUME / STATE
    state = json.loads(state_path.read_text())
    del state["config"]["pot_power"]
    state_path.write_text(json.dumps(state))
    assert saved_config(tmp_path / RESUME).pot_power == 0.0
    with pytest.raises(ValueError, match="pot_power: 0.0 → 0.5"):
        drive(tmp_path, 2)
    assert drive(tmp_path, 2, replace(DRIVEN, pot_power=0.0)).t == 2

def test_a_state_written_before_the_boost_existed_resumes_without_one(tmp_path):
    drive(tmp_path, 1)
    state_path = tmp_path / RESUME / STATE
    state = json.loads(state_path.read_text())
    del state["config"]["target"]
    del state["config"]["boost"]
    state_path.write_text(json.dumps(state))
    assert saved_config(tmp_path / RESUME).target is None
    assert saved_config(tmp_path / RESUME).boost == Sampling()
    assert drive(tmp_path, 2).t == 2

def test_a_saved_config_with_a_field_this_code_does_not_know_is_refused_by_name():
    # The mirror case of `ADDED_LATER`: a record from newer code. It is a
    # `ValueError` naming the field, like every other refusal, not the bare
    # `TypeError` `DeepConfig(**saved)` would raise.
    carried = {**asdict(DeepConfig()), "grid": list(GRID), "coinage": 3}
    with pytest.raises(ValueError, match="coinage"):
        train.config_from_saved(carried)

def test_a_changed_config_is_refused_before_the_memories_are_read(tmp_path):
    drive(tmp_path, 2)
    # A refusal must cost a read of state.json, not of the 3.6 GB behind it:
    # with the memory files gone, the config is still what is refused.
    for memory in (tmp_path / RESUME).glob("*.npz"):
        memory.unlink()
    with pytest.raises(ValueError, match="walks"):
        drive(tmp_path, 3, replace(DRIVEN, walks=6))

def dispositions() -> tuple[int, object, object]:
    """This process's id and what it does on SIGINT and SIGTERM; slow, so a job per worker."""
    import time as clock

    clock.sleep(0.3)
    return os.getpid(), signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)

def test_every_worker_ignores_the_signals_the_parent_catches():
    # A terminal's Ctrl-C reaches the whole process group. The parent turns
    # it into a stop request; a worker that died on it would break the pool
    # and the run would end without its save. The pool starts a worker only
    # when none is idle, so the claim is that the warm-up phase started ALL
    # of them under the ignore: every distinct worker must report SIG_IGN.
    before = signal.getsignal(signal.SIGINT)
    with Collector(seed=1, workers=3) as collector:
        train._spawn_ignoring_signals(collector, workers=3)
        assert signal.getsignal(signal.SIGINT) is before
        jobs = [collector._pool.submit(dispositions) for _ in range(9)]
        seen = {pid: (on_int, on_term) for pid, on_int, on_term in (job.result() for job in jobs)}
    assert len(seen) == 3
    assert all(handlers == (signal.SIG_IGN, signal.SIG_IGN) for handlers in seen.values())

# ---------------------------------------------------------------------------
# The regret rules' per-sample form
# ---------------------------------------------------------------------------

def test_dcfr_targets_carry_the_discounts_owed_since_their_iteration():
    from drawmaha_solver.deepcfr.train import rule_targets

    t = np.array([1, 3, 5])
    targets = np.tile(np.array([[2.0, -2.0, 0.0, 0.0]], dtype=np.float32), (3, 1))
    out = rule_targets(targets, t, now=5, rule="dcfr")
    owed = [np.prod([k**1.5 / (k**1.5 + 1) for k in range(s, 5)]) for s in (1, 3, 5)]
    np.testing.assert_allclose(out[:, 0], 2.0 * np.array(owed), rtol=1e-6)
    np.testing.assert_allclose(out[:, 1], -2.0 * 0.5 ** np.array([4, 2, 0]), rtol=1e-6)

def test_cfr_plus_targets_keep_positives_and_only_this_iterations_negatives():
    from drawmaha_solver.deepcfr.train import rule_targets

    t = np.array([2, 7])
    targets = np.array([[1.0, -1.0, 0.5, -0.5]] * 2, dtype=np.float32)
    out = rule_targets(targets, t, now=7, rule="cfr+")
    np.testing.assert_array_equal(out, [[1.0, 0.0, 0.5, 0.0], [1.0, -1.0, 0.5, -0.5]])

# ---------------------------------------------------------------------------
# Pot-normalised targets
# ---------------------------------------------------------------------------

# Three spots and their pots, re-derived from the rules rather than read from
# `chip_state`: the ante pot is 2; a pot bet makes it 4, and the pot-limit
# raise over it is 2 to call plus 6 more, so the first bettor faces a raise
# at 12; re-raising is 6 to call plus min(18, the 17 behind) = 23, then the
# call of 17: 2 + 2 + 8 + 23 + 17 = 52, the biggest pot the game reaches.
SPOTS = {
    ((),): 2,
    ((Action.POT, Action.POT),): 12,
    ((Action.POT, Action.POT, Action.POT, Action.CHECK_CALL),): 52,
}

def first_row_at(betting) -> int:
    # Several points can share a closed betting line (each seat's throw, say);
    # the pot is the line's, so any of them serves. The first is taken.
    built = layout()
    index = next(i for i, point in enumerate(built.points) if point.betting == betting)
    return int(built.row_offset[index])

def first_row_point(betting) -> int:
    return next(i for i, point in enumerate(layout().points) if point.betting == betting)

def test_the_game_reaches_seven_pots_and_spot_pots_reads_them_from_the_betting():
    # The seven the `spot_pots` docstring lists, re-derived here from the
    # pot-limit ladder rather than copied. The ante puts 2 in the middle. A
    # pot bet adds 2: the caller decides facing 4, and the call makes 6. The
    # raise over that bet is 2 to call plus the 6 then in the pot, 8, so the
    # opener decides facing 12, and calling the 6 owed makes 18. The third
    # bet is 6 to call plus min(18, the 17 behind), 23, so the raiser decides
    # facing 35, all in, and the call of 17 makes 52, every chip in.
    from drawmaha_solver.deepcfr.train import spot_pots

    pots = spot_pots()
    assert pots.shape == (len(layout().points),) and not pots.flags.writeable
    # The independent checks: the set the ladder above derives, and two
    # spots' pots read off it by hand.
    assert sorted(set(pots.tolist())) == [2, 4, 6, 12, 18, 35, 52]
    assert pots[first_row_point(((Action.POT, Action.POT),))] == 12
    assert pots[first_row_point(((Action.POT, Action.POT, Action.POT, Action.CHECK_CALL),))] == 52
    # And the array is in the layout's point order, each entry its own line's.
    for index, point in enumerate(layout().points):
        assert pots[index] == chip_state(point.betting).pot

def test_pot_scaling_at_power_zero_is_the_identity():
    from drawmaha_solver.deepcfr.train import pot_scaled

    targets = np.array([[3.0, -1.0, 0.0, 0.0], [0.5, 0.25, -2.0, 0.0]], dtype=np.float32)
    rows = np.array([first_row_at(((),)), first_row_at(((Action.POT, Action.POT),))])
    assert pot_scaled(rows, targets, power=0.0) is targets

@pytest.mark.parametrize("power", [0.5, 1.0])
def test_each_regret_sample_is_divided_by_its_own_spots_pot(power):
    from drawmaha_solver.deepcfr.train import pot_scaled

    rows = np.array([first_row_at(betting) + 1 for betting in SPOTS])
    pots = np.array(list(SPOTS.values()), dtype=np.float64)
    targets = np.array([[6.0, -3.0, 1.5, 0.0]] * len(rows), dtype=np.float32)
    scaled = pot_scaled(rows, targets, power=power)
    assert scaled.dtype == np.float32 and scaled.shape == targets.shape
    np.testing.assert_allclose(scaled, targets / (pots**power)[:, None], rtol=1e-6)
    # The divisor is the spot's, so the padding column stays zero and every
    # column of one row is divided by the same number.
    assert (scaled[:, 3] == 0).all()
    np.testing.assert_allclose(scaled[:, 0] / scaled[:, 1], -2.0, rtol=1e-6)

def test_pot_scaling_leaves_the_strategy_regret_matching_plays_unchanged():
    from drawmaha_solver.deepcfr.train import pot_scaled

    rng = np.random.default_rng(3)
    built = layout()
    rows = rng.integers(0, built.row_offset[-1], size=500)
    targets = rng.normal(scale=10.0, size=(500, 4)).astype(np.float32)
    scaled = pot_scaled(rows, targets, power=0.5)
    # The ledger's own `strategy()` normalises the positive part; a row with
    # none is the argmax one-hot (`materialise`, torch-bound, so its rule is
    # run on scaled regrets in `test_train_torch.py`). Both read only ratios.
    for before, after in zip(targets, scaled, strict=True):
        assert (before > 0).tolist() == (after > 0).tolist()
        if (before > 0).any():
            raw, pot = RegretMatcher(4), RegretMatcher(4)
            raw.cumulative_regret = before.astype(np.float64)
            pot.cumulative_regret = after.astype(np.float64)
            np.testing.assert_allclose(pot.strategy(), raw.strategy(), rtol=1e-5)
        else:
            assert np.argmax(after) == np.argmax(before)

def test_the_pot_power_is_part_of_the_run():
    assert "pot_power" in SAMPLE_FIELDS
    with pytest.raises(ValueError, match="pot_power"):
        require_same_run(DeepConfig(), DeepConfig(pot_power=0.0))

def test_the_rule_is_part_of_the_run():
    with pytest.raises(ValueError, match="rule"):
        DeepConfig(rule="vanilla")
    with pytest.raises(ValueError, match="rule"):
        require_same_run(DeepConfig(), DeepConfig(rule="dcfr"))
