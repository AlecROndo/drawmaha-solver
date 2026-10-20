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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.deepcfr.collect import Collector
from drawmaha_solver.deepcfr.reservoir import Memory, Reservoir, reservoir_stream
from drawmaha_solver.deepcfr.train import (
    GRID,
    SAMPLE_FIELDS,
    DeepConfig,
    DeepRun,
    Fitted,
    STATE,
    OracleLearner,
    load_run,
    new_run,
    require_same_run,
    save_run,
    step,
)
import drawmaha_solver.deepcfr.train as train
from drawmaha_solver.minidrawmaha.game import random_deal
from drawmaha_solver.minidrawmaha.lockstep import (
    SEATS,
    apply_arrays,
    flatten,
    new_lockstep,
    train_lockstep,
    walk,
)
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.regret_rules import Average, RegretRule

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
    def fit(self, rows, t, targets, *, seed) -> Fitted:
        self.events.append(("fit", seed, (rows.copy(), t.copy(), targets.copy())))
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
    ],
)
def test_config_refuses_bad_values_by_name(field, value):
    with pytest.raises(ValueError, match=field):
        DeepConfig(**{field: value})

def test_require_same_run_names_every_changed_sample_field():
    asked = DeepConfig(seed=1, walks=5, width=32, workers=2, grid=(1,), save_every_s=0.0)
    with pytest.raises(ValueError, match=r"seed.*walks.*width") as caught:
        require_same_run(DeepConfig(), asked)
    for free in ("workers", "grid", "save_every_s"):
        assert free not in str(caught.value)
    assert free not in SAMPLE_FIELDS
    require_same_run(DeepConfig(), DeepConfig(workers=2, grid=(1,), save_every_s=0.0))

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
        _, (rows, t, targets) = [fit for fit in fits if fit[0][2] == seat][-1]
        for given, resident in zip((rows, t, targets), run.advantage[seat].contents(), strict=True):
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
    for name in ("rows", "t", "targets"):
        a, b = getattr(run.nets[0], name), getattr(back, name)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    assert run.nets[0].rows.shape[0] == run.advantage[0].size

# ---------------------------------------------------------------------------
# The resume state
# ---------------------------------------------------------------------------

def same_run(first: DeepRun, second: DeepRun) -> bool:
    """t, config, the three memories and the oracle models, all bytewise."""
    if first.t != second.t or first.config != second.config:
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
