"""The real learner: a resumed run refits the same bits, and gate 5 behind the slow flag.

`NetLearner` is the loop's one torch-bound piece. The claims: a run saved
and loaded refits to the same weights as the uninterrupted run (the memory
round-trips, the fit is a function of its seed, and the thread count is
fixed); the kept net files load with `load_net` and reproduce materialise's
write; and, behind `MINIDRAWMAHA_FULL_DEEP=1`, a few real iterations leave
the current iterate less exploitable than uniform play (4.842).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drawmaha_solver.deepcfr.collect import Collector  # noqa: E402
from drawmaha_solver.deepcfr.materialise import materialise  # noqa: E402
from drawmaha_solver.deepcfr.net import load_net  # noqa: E402
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
    reason="a few real iterations at K = 1,000 and one exact grade, ~10 min; set MINIDRAWMAHA_FULL_DEEP=1",
)
def test_gate_five_the_current_iterate_beats_uniform(tmp_path):
    config = DeepConfig(seed=0, walks=1_000, workers=8, capacity=10_000_000, steps=1_000, threads=8)
    learner = NetLearner(config)
    started = time.monotonic()
    run = run_deep(tmp_path, config, target=5, learner=learner, log=print)
    trained_s = time.monotonic() - started
    table = PackedTable.whole_game()
    learner.write(table, run.nets)
    started = time.monotonic()
    grade = exploitability(current_iterate(table, iteration=run.t, seed=config.seed))
    graded_s = time.monotonic() - started
    uniform = exploitability(uniform_strategy())
    print(f"gate 5: current iterate at t={run.t} grades {grade:.3f} against uniform {uniform:.3f}; "
          f"trained in {trained_s:.0f} s, graded in {graded_s:.0f} s")
    assert uniform == pytest.approx(4.842, abs=0.001)
    assert grade < uniform
