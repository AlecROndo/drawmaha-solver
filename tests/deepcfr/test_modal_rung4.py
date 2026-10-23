"""Rung 4's Modal launcher without Modal: the launch, the claim, the grader's bookkeeping, the sweep's verdict.

`scripts/deepcfr/modal_rung4.py` imports `modal` at the top, so the module is
loaded with a stand-in whose decorators hand the functions back unchanged,
as `tests/minidrawmaha/test_modal_rung3.py` does for rung 3 (the stand-in is
restated here: test modules never import one another). The claim helpers are
rung 3's, lifted; their tests are kept to the cases a lift could break. What
is new is tested in full: `train` and `sweep` build a `DeepConfig` and spawn
on the deployed app, refusing a bad flag before any spawn; `grade` fans one
container per mark and writes `grades.json` keyed `deep/<t>/<column>`;
`grade_mark_dir` grades a mark's three objects — the exact average, Π and
the current iterate — with the grader stubbed (the real one is a minute a
call) and torch needed only for Π; `_wall_clock_s` reads the training clock
off `progress.jsonl`; `sweep_winner` applies the pre-registered rule; and
`_speed_report` turns the smoke's lines into seconds, hours and dollars.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from drawmaha_solver.deepcfr.train import MARKS, PROGRESS, DeepConfig, mark_name

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deepcfr" / "modal_rung4.py"
START = 1_000_000.0

class FakeApp:
    """`modal.App` whose decorators return the function itself."""

    def __init__(self, name: str) -> None:
        self.name = name

    def function(self, **options):
        return lambda function: function

    def local_entrypoint(self):
        return lambda function: function

@pytest.fixture
def launcher(monkeypatch):
    fake_modal = mock.MagicMock()
    fake_modal.App = FakeApp
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    spec = importlib.util.spec_from_file_location("modal_rung4_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = {"now": START}

    def sleep(seconds):
        clock["now"] += seconds

    now = lambda: clock["now"]  # noqa: E731
    monkeypatch.setattr(
        module, "time", mock.Mock(time=now, sleep=sleep, monotonic=now, perf_counter=now)
    )
    monkeypatch.setenv("MODAL_TASK_ID", "me")
    module.clock = clock
    module.modal = fake_modal
    return module

def held_by(out: Path, task: str, heartbeat: float) -> None:
    (out / "owner.json").write_text(json.dumps({"task": task, "heartbeat": heartbeat}))

def progress(out: Path, *lines: dict) -> None:
    (out / PROGRESS).write_text("".join(json.dumps(line) + "\n" for line in lines))

def owner(out: Path) -> str:
    return json.loads((out / "owner.json").read_text())["task"]

def line(t: int, iteration_s: float, **extra) -> dict:
    """One progress line as `train._Saves` writes it, with only what the launcher reads."""
    return {"t": t, "target": 2000, "seconds": {"iteration": iteration_s,
            "materialise": [0.5, 0.5], "collect": [1.0, 1.0], "merge": [0.0, 0.0],
            "fit": [10.0, 10.0]}, **extra}

# ---------------------------------------------------------------------------
# The claim, lifted from rung 3
# ---------------------------------------------------------------------------

def test_a_fresh_run_is_claimed_and_a_preempted_one_taken_over(launcher, tmp_path):
    launcher._claim(tmp_path)
    assert owner(tmp_path) == "me"
    held_by(tmp_path, "dead", START)
    progress(tmp_path, {"t": 20, "stopped": "SIGTERM"})
    launcher._claim(tmp_path)
    assert owner(tmp_path) == "me"
    assert json.loads((tmp_path / PROGRESS).read_text().splitlines()[-1])["claimed_by"] == "me"

def test_a_live_trainer_refuses_a_newcomer_after_the_stale_wait(launcher, tmp_path, monkeypatch):
    held_by(tmp_path, "other", START)
    progress(tmp_path, {"t": 20})

    def live_owner_beats(seconds):
        launcher.clock["now"] += seconds
        held_by(tmp_path, "other", launcher.clock["now"])

    launcher.time.sleep = live_owner_beats
    with pytest.raises(RuntimeError, match="being trained by other"):
        launcher._claim(tmp_path)
    assert launcher.clock["now"] - START == launcher.STALE_S

def test_only_the_deadline_spawns_a_continuation(launcher):
    deadline = launcher.DEADLINE_S
    assert launcher._continues(10, 20, deadline)
    assert not launcher._continues(10, 20, deadline - 1)
    assert not launcher._continues(20, 20, deadline)

def test_claiming_outside_a_modal_container_is_refused(launcher, tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_TASK_ID")
    with pytest.raises(KeyError, match="MODAL_TASK_ID"):
        launcher._claim(tmp_path)

# ---------------------------------------------------------------------------
# Launching: train and sweep
# ---------------------------------------------------------------------------

def test_train_spawns_the_run_on_the_deployed_app_with_the_specs_config(launcher):
    launcher.train()
    launcher.modal.Function.from_name.assert_called_once_with("drawmaha-rung4", "train_run")
    spawn = launcher.modal.Function.from_name.return_value.spawn
    (name, target, config), _ = spawn.call_args
    assert (name, target) == (launcher.RUN, 2_000)
    # The spec's run: K = 10,000, C = 50M, 3 × 64, 4,000 × 10,000, on W = 32.
    assert launcher._config_from(config) == DeepConfig(workers=launcher.WORKERS)

def test_train_refuses_a_bad_flag_before_spawning(launcher):
    with pytest.raises(ValueError, match="width"):
        launcher.train(width=0)
    launcher.modal.Function.from_name.return_value.spawn.assert_not_called()

def test_a_config_round_trips_through_the_spawn_as_plain_data(launcher):
    config = DeepConfig(walks=3, workers=2, grid=(1, 2))
    carried = launcher._config_to(config)
    assert json.loads(json.dumps(carried)) == carried
    assert launcher._config_from(json.loads(json.dumps(carried))) == config

def test_sweep_spawns_one_trainer_per_size_under_its_own_directory(launcher):
    launcher.sweep()
    spawn = launcher.modal.Function.from_name.return_value.spawn
    launched = [(args[0], args[1], launcher._config_from(args[2])) for args, _ in spawn.call_args_list]
    assert [(name, target) for name, target, _ in launched] == [
        ("sweep/64x3", 100), ("sweep/64x4", 100), ("sweep/128x3", 100),
    ]
    assert [(c.width, c.depth) for _, _, c in launched] == [(64, 3), (64, 4), (128, 3)]
    assert all(c.walks == 10_000 and c.workers == launcher.WORKERS for _, _, c in launched)

def test_the_sweep_winner_is_the_default_unless_a_size_is_clearly_better(launcher):
    def sizes(exact: dict[str, float]) -> dict:
        return {size: {"exact": {"exploitability": value}} for size, value in exact.items()}

    base = {"64x3": 1.0, "64x4": 0.95, "128x3": 0.99}
    assert launcher.sweep_winner(sizes(base)) == "64x3"
    clear = {**base, "128x3": 1.0 - launcher.CLEAR_MARGIN - 0.01}
    assert launcher.sweep_winner(sizes(clear)) == "128x3"
    at_the_line = {**base, "128x3": 1.0 - launcher.CLEAR_MARGIN}
    assert launcher.sweep_winner(sizes(at_the_line)) == "64x3"

def test_a_sweep_grade_refuses_a_run_short_of_its_target_before_spawning(launcher, tmp_path, capsys):
    fakes = {
        "read_status": mock.Mock(remote=lambda: {"sweep/64x3": {"t": 100}, "sweep/64x4": {"t": 40}}),
        "grade_mark": mock.Mock(),
    }
    launcher.modal.Function.from_name.side_effect = lambda app, name: fakes[name]
    with pytest.raises(RuntimeError, match="sweep/64x4.*40"):
        launcher.sweep(grade=True, out=str(tmp_path / "sweep.json"))
    fakes["grade_mark"].spawn.assert_not_called()

def test_a_sweep_grade_fans_out_the_three_sizes_and_names_the_winner(launcher, tmp_path, capsys):
    exact = {"sweep/64x3": 1.0, "sweep/64x4": 0.97, "sweep/128x3": 0.5}

    def spawn(name, t):
        assert t == 100
        graded = {column: {"exploitability": exact[name] + k * 0.1, "hands_per_seat": 1_000_000}
                  for k, column in enumerate(("exact", "pi", "current"))}
        return mock.Mock(get=lambda: graded)

    fakes = {
        "read_status": mock.Mock(remote=lambda: {name: {"t": 100} for name in exact}),
        "grade_mark": mock.Mock(spawn=spawn),
    }
    launcher.modal.Function.from_name.side_effect = lambda app, name: fakes[name]
    out = tmp_path / "sweep.json"
    launcher.sweep(grade=True, out=str(out))
    written = json.loads(out.read_text())
    assert written["winner"] == "128x3"
    assert written["sizes"]["64x4"]["pi"]["exploitability"] == pytest.approx(1.07)
    assert written["iterations"] == 100
    # The sheet's hands are the graded run's, not the grading command's flags.
    assert written["hands_per_seat"] == 1_000_000
    assert "128x3" in capsys.readouterr().out

def test_a_sweep_grade_refuses_sizes_that_disagree_on_hands(launcher, tmp_path):
    hands = {"sweep/64x3": 1_000_000, "sweep/64x4": 1_000_000, "sweep/128x3": 500_000}

    def spawn(name, t):
        return mock.Mock(get=lambda: {c: {"exploitability": 1.0, "hands_per_seat": hands[name]} for c in ("exact", "pi", "current")})

    fakes = {
        "read_status": mock.Mock(remote=lambda: {name: {"t": 100} for name in hands}),
        "grade_mark": mock.Mock(spawn=spawn),
    }
    launcher.modal.Function.from_name.side_effect = lambda app, name: fakes[name]
    out = tmp_path / "sweep.json"
    with pytest.raises(ValueError, match="hands per seat"):
        launcher.sweep(grade=True, out=str(out))
    assert not out.exists()

# ---------------------------------------------------------------------------
# Grading: the fan-out and grades.json
# ---------------------------------------------------------------------------

def test_grade_fans_one_container_per_reached_mark_and_writes_the_keys(launcher, tmp_path):
    def spawn(name, t):
        assert name == launcher.RUN
        return mock.Mock(get=lambda: {
            column: {"exploitability": t / 100 + k, "hands_per_seat": t * 10_000}
            for k, column in enumerate(("exact", "pi", "current"))
        })

    fakes = {
        "read_marks": mock.Mock(remote=lambda name: [10, 20]),
        "grade_mark": mock.Mock(spawn=spawn),
    }
    launcher.modal.Function.from_name.side_effect = lambda app, name: fakes[name]
    out = tmp_path / "grades.json"
    out.write_text(json.dumps({"deep/00010/exact": {"exploitability": -1.0}, "other": {"x": 1}}))
    launcher.grade(grades=str(out))
    written = json.loads(out.read_text())
    assert written["deep/00010/exact"]["exploitability"] == pytest.approx(0.1), "a mark asked for is regraded"
    assert written["deep/00020/current"]["exploitability"] == pytest.approx(2.2)
    assert written["deep/00020/current"]["hands_per_seat"] == 200_000
    assert written["other"] == {"x": 1}, "keys the grader did not write are kept"
    assert list(written) == sorted(written)

def test_grade_takes_an_explicit_list_of_marks(launcher, tmp_path):
    spawned = []

    def spawn(name, t):
        spawned.append(t)
        return mock.Mock(get=lambda: {"exact": {}, "pi": {}, "current": {}})

    fakes = {"read_marks": mock.Mock(remote=lambda name: [10, 20, 50]), "grade_mark": mock.Mock(spawn=spawn)}
    launcher.modal.Function.from_name.side_effect = lambda app, name: fakes[name]
    launcher.grade(marks="50,10", grades=str(tmp_path / "grades.json"))
    assert spawned == [10, 50]

def test_grade_refuses_a_mark_the_run_has_not_kept(launcher, tmp_path):
    fakes = {"read_marks": mock.Mock(remote=lambda name: [10]), "grade_mark": mock.Mock()}
    launcher.modal.Function.from_name.side_effect = lambda app, name: fakes[name]
    with pytest.raises(ValueError, match="20"):
        launcher.grade(marks="10,20", grades=str(tmp_path / "grades.json"))
    fakes["grade_mark"].spawn.assert_not_called()

def test_grade_local_grades_a_mac_runs_marks_here_and_touches_no_container(launcher, tmp_path, monkeypatch):
    run = tmp_path / "run"
    for t in (10, 20):
        (run / MARKS / mark_name(t)).mkdir(parents=True)
        (run / MARKS / mark_name(t) / "state.json").write_text("{}")
    graded = []

    def fake_grade_mark_dir(run_dir, t):
        graded.append((Path(run_dir), t))
        return {column: {"exploitability": float(t)} for column in ("exact", "pi", "current")}

    monkeypatch.setattr(launcher, "grade_mark_dir", fake_grade_mark_dir)
    out = tmp_path / "grades.json"
    launcher.grade(local=str(run), grades=str(out))
    assert graded == [(run, 10), (run, 20)]
    launcher.modal.Function.from_name.assert_not_called()
    assert json.loads(out.read_text())["deep/00020/pi"] == {"exploitability": 20.0}
    with pytest.raises(ValueError, match="50"):
        launcher.grade(local=str(run), marks="50", grades=str(out))

def test_the_training_clock_is_the_sum_of_the_last_line_per_iteration(launcher, tmp_path):
    progress(
        tmp_path,
        line(1, 10.0), line(2, 10.0), line(3, 10.0),
        # Crashed after 3, resumed from 2: the re-run of 3 is the one that counts.
        line(3, 12.0), line(4, 10.0),
        {"t": 4, "target": 2000, "stopped": "deadline", "wrote": []},
    )
    assert launcher._wall_clock_s(tmp_path / PROGRESS, 3) == pytest.approx(32.0)
    assert launcher._wall_clock_s(tmp_path / PROGRESS, 4) == pytest.approx(42.0)
    with pytest.raises(ValueError, match="5"):
        launcher._wall_clock_s(tmp_path / PROGRESS, 5)

def test_read_marks_lists_the_kept_marks_in_order(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    assert launcher.read_marks("deep") == []
    for t in (50, 10, 20):
        (tmp_path / "deep" / MARKS / mark_name(t)).mkdir(parents=True)
        (tmp_path / "deep" / MARKS / mark_name(t) / "state.json").write_text("{}")
    (tmp_path / "deep" / MARKS / "t00099.tmp").mkdir()  # a save that never reached its state
    assert launcher.read_marks("deep") == [10, 20, 50]

def test_read_status_reports_every_started_run_by_its_last_iteration_line(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    (tmp_path / "deep").mkdir()
    progress(tmp_path / "deep", line(1, 1.0), {"claimed_by": "x"}, line(2, 1.0, stopped="deadline"))
    (tmp_path / "sweep" / "64x4").mkdir(parents=True)
    progress(tmp_path / "sweep" / "64x4", line(7, 1.0))
    status = launcher.read_status()
    assert status["deep"]["t"] == 2 and status["deep"]["stopped"] == "deadline"
    assert status["sweep/64x4"]["t"] == 7
    assert "sweep/64x3" not in status

# ---------------------------------------------------------------------------
# Grading one mark: the three objects
# ---------------------------------------------------------------------------

def test_grade_mark_dir_grades_the_exact_average_pi_and_the_current_iterate(launcher, tmp_path):
    pytest.importorskip("torch")
    from drawmaha_solver.deepcfr.average import ExactAverage
    from drawmaha_solver.deepcfr.train import NetLearner, run_deep
    from drawmaha_solver.minidrawmaha.answer.strategy import load_strategy

    tiny = DeepConfig(seed=9, walks=20, workers=0, capacity=100_000, steps=5, batch=64, threads=2, grid=(2,))
    lines = []
    run_deep(tmp_path, tiny, target=2, learner=NetLearner(tiny), observers=(ExactAverage(),), log=lines.append)
    graded = []

    def fake_grade(strategy):
        graded.append(strategy)
        return {"br0": 0.0, "br1": 0.0, "exploitability": float(len(graded)), "value_p0": 0.0}

    result = launcher.grade_mark_dir(tmp_path, 2, grade=fake_grade)
    assert list(result) == ["exact", "pi", "current"]
    assert [s.info.column for s in graded] == ["exact", "pi", "current"]
    assert all(s.info.rule == "deep" and s.info.iteration == 2 and s.info.seed == 9 for s in graded)
    for column, entry in result.items():
        assert entry["hands_per_seat"] == 2 * 20
        assert entry["config"] == asdict(tiny) | {"grid": [2]}
        assert entry["wall_clock_s"] == pytest.approx(
            sum(json.loads(l)["seconds"]["iteration"] for l in lines if l.startswith("{") and "seconds" in l)
        )
        assert entry["grade_s"] >= 0.0
        frozen = load_strategy(tmp_path / launcher.FROZEN / f"{mark_name(2)}-{column}.npz")
        np.testing.assert_array_equal(frozen.probabilities, graded[list(result).index(column)].probabilities)
    assert result["pi"]["pi_fit_s"] >= 0.0
    assert result["exact"]["exploitability"] == 1.0 and result["current"]["exploitability"] == 3.0

def test_grade_mark_dir_refuses_a_mark_that_was_not_kept(launcher, tmp_path):
    with pytest.raises(ValueError, match="state.json"):
        launcher.grade_mark_dir(tmp_path, 7, grade=lambda s: pytest.fail("graded"))

# ---------------------------------------------------------------------------
# The smoke's report
# ---------------------------------------------------------------------------

def test_the_speed_report_projects_the_full_run_from_the_last_iteration(launcher):
    # Iteration 1 fits a tiny memory and is cheap; the LAST iteration is the one
    # that looks like the run. Iteration 3 was re-run after the resume and its
    # earlier line must not count twice.
    cheap = line(1, 12.0)
    cheap["seconds"]["fit"] = [5.0, 5.0]
    lines = [cheap, line(2, 22.0), line(3, 22.0), line(3, 22.0), line(4, 22.0)]
    graded = {"exact": {"grade_s": 60.0}, "pi": {"grade_s": 60.0, "pi_fit_s": 30.0}, "current": {"grade_s": 60.0}}
    report = launcher._speed_report(lines, graded, walks=1_000, workers=32, cpu_count=33)
    assert report["iterations"] == 4
    assert report["smoke_mean_seconds_per_iteration"] == pytest.approx((12.0 + 22.0 * 3) / 4)
    assert report["seconds"]["fit"] == pytest.approx(20.0), "the last iteration's stages"
    assert report["seconds"]["collect"] == pytest.approx(2.0)
    # Only the walks scale with K: 2 s of collecting at K = 1,000 is 20 s at 10,000.
    assert report["projected_seconds_per_iteration"] == pytest.approx(22.0 - 2.0 + 20.0)
    hours = 40.0 * 2_000 / 3600
    assert report["projected_hours_for_full_run"] == pytest.approx(hours, abs=0.01)
    expected = hours * 3600 * (33 * launcher.CPU_USD_PER_CORE_S + 16 * launcher.MEMORY_USD_PER_GIB_S)
    assert report["projected_usd_for_full_run"] == pytest.approx(expected, abs=0.01)
    assert report["grade_seconds"] == {"exact": 60.0, "pi": 60.0, "current": 60.0, "pi_fit": 30.0}
    assert report["cpu_count"] == 33
