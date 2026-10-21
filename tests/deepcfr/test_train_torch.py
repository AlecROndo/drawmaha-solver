"""The real learner: a resumed run refits the same bits, and gate 5 behind the slow flag.

`NetLearner` is the loop's one torch-bound piece. The claims: a run saved
and loaded refits to the same weights as the uninterrupted run (the memory
round-trips, the fit is a function of its seed, and the thread count is
fixed); the kept net files load with `load_net` and reproduce materialise's
write; and, behind `MINIDRAWMAHA_FULL_DEEP=1`, a few real iterations leave
the exact average less exploitable than the current iterate, and both less
than uniform play (4.842), the average replays from the kept nets bit for
bit, and Π grades below uniform.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drawmaha_solver.deepcfr.average import ExactAverage  # noqa: E402
from drawmaha_solver.deepcfr.collect import Collector  # noqa: E402
from drawmaha_solver.deepcfr.materialise import materialise  # noqa: E402
from drawmaha_solver.deepcfr.net import load_net  # noqa: E402
from drawmaha_solver.deepcfr.policy import fit_policy, policy_strategy  # noqa: E402
from drawmaha_solver.deepcfr.train import (  # noqa: E402
    NETS,
    RESUME,
    DeepConfig,
    NetLearner,
    current_iterate,
    load_run,
    net_name,
    new_run,
    run_deep,
    save_run,
    step,
)
from drawmaha_solver.minidrawmaha.exploitability import exploitability  # noqa: E402
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable  # noqa: E402
from drawmaha_solver.minidrawmaha.answer.strategy import uniform_strategy  # noqa: E402

FULL_DEEP = os.environ.get("MINIDRAWMAHA_FULL_DEEP") == "1"

TINY = DeepConfig(seed=9, walks=20, workers=0, capacity=100_000, steps=5, batch=64, threads=2)

def same_weights(first, second) -> bool:
    a, b = first.state_dict(), second.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

def written_by(nets, learner: NetLearner) -> np.ndarray:
    table = PackedTable.whole_game()
    learner.write(table, nets)
    return table.cumulative_regret

def test_net_learner_resume_equals_uninterrupted_bytewise(tmp_path):
    learner = NetLearner(TINY)
    straight = new_run(TINY)
    with Collector(seed=9, workers=0) as collector:
        for _ in range(3):
            step(straight, collector, learner)
    resumed = new_run(TINY)
    with Collector(seed=9, workers=0) as collector:
        step(resumed, collector, learner)
        step(resumed, collector, learner)
    save_run(resumed, tmp_path, learner=learner)
    resumed = load_run(tmp_path, learner=learner)
    with Collector(seed=9, workers=0) as collector:
        step(resumed, collector, learner)
    assert resumed.t == straight.t == 3
    for seat in (0, 1):
        assert same_weights(resumed.nets[seat], straight.nets[seat])
        for a, b in zip(resumed.advantage[seat].contents(), straight.advantage[seat].contents(), strict=True):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(written_by(resumed.nets, learner), written_by(straight.nets, learner))

def test_saved_nets_load_back_and_reproduce_the_write(tmp_path):
    learner = NetLearner(TINY)
    run = run_deep(tmp_path, TINY, target=2, learner=learner, log=lambda _: None)
    loaded = [load_net(tmp_path / NETS / net_name(seat, 2)) for seat in (0, 1)]
    for seat in (0, 1):
        assert same_weights(loaded[seat], run.nets[seat])
    table = PackedTable.whole_game()
    materialise(table, loaded)
    np.testing.assert_array_equal(table.cumulative_regret, written_by(run.nets, learner))
    assert table.cumulative_regret.any()
    assert same_weights(load_run(tmp_path / RESUME, learner=learner).nets[1], run.nets[1])

def test_deep_config_carries_every_fit_config_field():
    # The fit fields are copied flat into DeepConfig (fit.py imports torch at
    # its top); a field added to FitConfig later must not drift past them.
    from dataclasses import fields

    from drawmaha_solver.deepcfr.fit import FitConfig

    fit_fields = {f.name: f.default for f in fields(FitConfig)}
    deep_fields = {f.name: f.default for f in fields(DeepConfig)}
    assert all(deep_fields.get(name) == default for name, default in fit_fields.items())
    assert NetLearner(DeepConfig())._fit() == FitConfig()

def test_fit_reports_a_finite_loss_and_a_run_logs_it(tmp_path):
    learner = NetLearner(TINY)
    run = new_run(TINY)
    with Collector(seed=9, workers=0) as collector:
        rows = collector.collect(seat=0, t=1, walks=5)
    fitted = learner.fit(rows.advantage_rows, np.ones(len(rows.advantage_rows), np.int32), rows.advantage_targets, seed=(9, 1, 0))
    assert isinstance(fitted.loss, float) and math.isfinite(fitted.loss)
    again = learner.fit(rows.advantage_rows, np.ones(len(rows.advantage_rows), np.int32), rows.advantage_targets, seed=(9, 1, 0))
    assert again.loss == fitted.loss  # the held batch is a function of the fit's seed
    # ... but not the fit's own first minibatch, which the net has just trained on.
    from drawmaha_solver.deepcfr.fit import _fold_seed, batch_loss

    first = torch.randint(len(rows.advantage_rows), (TINY.batch,), generator=torch.Generator().manual_seed(_fold_seed((9, 1, 0)))).numpy()
    ones = np.ones(len(rows.advantage_rows), np.int32)
    with torch.inference_mode():
        trained_on = float(batch_loss(fitted.model, rows.advantage_rows[first], ones[first], rows.advantage_targets[first]).item())
    assert fitted.loss != trained_on
    lines = []
    run_deep(tmp_path, TINY, target=1, learner=learner, log=lines.append)
    import json

    assert all(math.isfinite(x) for x in json.loads(lines[-1])["loss"])
    assert run.t == 0

@pytest.mark.skipif(
    not FULL_DEEP,
    reason="20 real iterations at K = 1,000 and three exact grades, ~8 min; set MINIDRAWMAHA_FULL_DEEP=1",
)
def test_gate_five_the_exact_average_beats_the_current_iterate_and_uniform(tmp_path):
    config = DeepConfig(seed=0, walks=1_000, workers=8, capacity=10_000_000, steps=1_000, threads=8)
    learner = NetLearner(config)
    exact = ExactAverage()
    started = time.monotonic()
    # Twenty iterations, not PR 4's five: the average carries iteration 1's
    # uniform play at weight 1/Σt and seat 1's share lags a refit, so at t = 5
    # it grades 1.522 against the current iterate's 1.391 and at t = 10
    # 1.089 against 0.865; the current iterate then swings back up (0.927 at
    # 20, 0.998 at 40) while the average keeps falling (0.869, 0.737), so
    # from t = 20 the spec's ordering holds (measured 2026-10-21, seed 0).
    run = run_deep(tmp_path, config, target=20, learner=learner, observers=(exact,), log=print)
    trained_s = time.monotonic() - started
    assert exact.last == [20, 20]
    table = PackedTable.whole_game()
    learner.write(table, run.nets)
    grades, seconds = {}, {}
    strategies = {
        "current": current_iterate(table, iteration=run.t, seed=config.seed),
        "exact": exact.frozen(iteration=run.t, seed=config.seed),
    }
    started = time.monotonic()
    pi = fit_policy(*run.policy.contents(), seed=config.seed, iteration=run.t, config=learner._fit())
    strategies["pi"] = policy_strategy(pi, iteration=run.t, seed=config.seed)
    seconds["pi fit + freeze"] = time.monotonic() - started
    for name, strategy in strategies.items():
        started = time.monotonic()
        grades[name] = exploitability(strategy)
        seconds[name] = time.monotonic() - started
    uniform = exploitability(uniform_strategy())
    print(
        f"gate 5 at t={run.t}: current {grades['current']:.3f}, exact average {grades['exact']:.3f}, "
        f"pi {grades['pi']:.3f}, uniform {uniform:.3f}; trained in {trained_s:.0f} s; "
        + ", ".join(f"{k} {v:.0f} s" for k, v in seconds.items())
    )
    assert uniform == pytest.approx(4.842, abs=0.001)
    assert grades["exact"] < grades["current"] < uniform
    # Π's bar is the floor; its number is reported, not pinned.
    assert grades["pi"] < uniform
    # Single Deep CFR both ways: the sums replay from the kept nets, bit for bit.
    replay = ExactAverage()
    kept = lambda seat, t: None if t == 0 else load_net(tmp_path / NETS / net_name(seat, t))  # noqa: E731
    for t in range(1, run.t + 1):
        materialise(table, (kept(0, t - 1), kept(1, t - 1)))
        replay(table, t, 1)
        materialise(table, (kept(0, t), kept(1, t - 1)))
        replay(table, t, 0)
    np.testing.assert_array_equal(replay.sums, exact.sums)
    saved = ExactAverage()
    saved.load(tmp_path / RESUME / f"exact-t{run.t:05d}.npz")
    np.testing.assert_array_equal(saved.sums, exact.sums)
