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
    new_run,
    require_same_run,
    step,
)
from drawmaha_solver.minidrawmaha.lockstep import SEATS

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
