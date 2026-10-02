"""The Modal launcher's one-trainer-per-rule claim and its launch, without Modal.

`scripts/modal_rung3.py` imports `modal` at the top, so the module is loaded
with a stand-in for it whose decorators hand the functions back unchanged, so
`main` and the claim helpers (`_claim`, `_stopped`, `_settle`, `_continues`)
are called as plain functions. The helpers touch nothing of Modal's but
`volume.reload()`; `main` touches only the deployed function's `spawn`. The
clock and the sleep are replaced by a fake that advances when slept, so the
15-minute wait for a dead trainer's heartbeat runs instantly. `train_rule`
itself only wires these helpers to Modal and `run_to`, and is left to the real
runs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "modal_rung3.py"
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
    spec = importlib.util.spec_from_file_location("modal_rung3_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = {"now": START}

    def sleep(seconds):
        clock["now"] += seconds

    monkeypatch.setattr(module, "time", mock.Mock(time=lambda: clock["now"], sleep=sleep))
    monkeypatch.setenv("MODAL_TASK_ID", "me")
    module.clock = clock
    module.modal = fake_modal
    return module

def held_by(out: Path, task: str, heartbeat: float) -> None:
    (out / "owner.json").write_text(json.dumps({"task": task, "heartbeat": heartbeat}))

def progress(out: Path, *lines: dict) -> None:
    (out / "progress.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))

def last_line(out: Path) -> dict:
    return json.loads((out / "progress.jsonl").read_text().splitlines()[-1])

def owner(out: Path) -> str:
    return json.loads((out / "owner.json").read_text())["task"]

def test_stopped_reads_only_the_last_progress_line(launcher, tmp_path):
    assert not launcher._stopped(tmp_path)
    progress(tmp_path, {"iteration": 10, "stopped": "SIGTERM"}, {"iteration": 20})
    assert not launcher._stopped(tmp_path)
    progress(tmp_path, {"iteration": 10}, {"iteration": 20, "stopped": "SIGTERM"})
    assert launcher._stopped(tmp_path)

def test_a_fresh_rule_is_claimed_without_a_takeover_line(launcher, tmp_path):
    launcher._claim(tmp_path)
    assert owner(tmp_path) == "me"
    assert not (tmp_path / "progress.jsonl").exists()

def test_a_preempted_run_is_taken_over_and_marked(launcher, tmp_path):
    # Its final commit refreshed the heartbeat, but its last line says it stopped.
    held_by(tmp_path, "dead", START)
    progress(tmp_path, {"iteration": 20, "stopped": "SIGTERM"})
    launcher._claim(tmp_path)
    assert owner(tmp_path) == "me"
    assert last_line(tmp_path)["claimed_by"] == "me"
    assert launcher.clock["now"] == START  # no waiting

def test_a_second_newcomer_after_a_takeover_is_refused(launcher, tmp_path, monkeypatch):
    held_by(tmp_path, "dead", START)
    progress(tmp_path, {"iteration": 20, "stopped": "SIGTERM"})
    launcher._claim(tmp_path)
    monkeypatch.setenv("MODAL_TASK_ID", "second")

    def live_owner_beats(seconds):
        # The trainer that took over keeps beating while the newcomer waits.
        launcher.clock["now"] += seconds
        held_by(tmp_path, "me", launcher.clock["now"])

    launcher.time.sleep = live_owner_beats
    with pytest.raises(RuntimeError, match="being trained by me"):
        launcher._claim(tmp_path)
    assert launcher.clock["now"] - START == launcher.STALE_S
    assert owner(tmp_path) == "me"

def test_a_dead_trainers_heartbeat_is_waited_out_then_taken(launcher, tmp_path):
    # Killed outright, before it could write its stopped line.
    held_by(tmp_path, "dead", START - 60)
    progress(tmp_path, {"iteration": 20})
    launcher._claim(tmp_path)
    assert owner(tmp_path) == "me"
    assert START < launcher.clock["now"] <= START + launcher.STALE_S
    assert "claimed_by" not in last_line(tmp_path)

def test_a_stale_heartbeat_is_taken_at_once(launcher, tmp_path):
    held_by(tmp_path, "dead", START - launcher.STALE_S)
    progress(tmp_path, {"iteration": 20})
    launcher._claim(tmp_path)
    assert owner(tmp_path) == "me"
    assert launcher.clock["now"] == START

def test_the_settle_keeps_the_container_that_won_the_claim(launcher, tmp_path):
    held_by(tmp_path, "me", START)
    launcher._settle(tmp_path)
    assert launcher.clock["now"] - START == launcher.CLAIM_SETTLE_S

def test_the_settle_refuses_a_container_whose_claim_was_overwritten(launcher, tmp_path):
    # Both passed `_claim`; the other's commit landed last.
    held_by(tmp_path, "other", START)
    with pytest.raises(RuntimeError, match="claimed by other at the same time"):
        launcher._settle(tmp_path)

def test_only_the_deadline_spawns_a_continuation(launcher):
    deadline = launcher.DEADLINE_S
    assert launcher._continues(10, 20, deadline)
    assert not launcher._continues(10, 20, deadline - 1)  # preempted: Modal restarts it
    assert not launcher._continues(20, 20, deadline)  # done

def test_claiming_outside_a_modal_container_is_refused(launcher, tmp_path, monkeypatch):
    # Every container would otherwise claim as the same name and wave the others through.
    monkeypatch.delenv("MODAL_TASK_ID")
    with pytest.raises(KeyError, match="MODAL_TASK_ID"):
        launcher._claim(tmp_path)

def test_main_spawns_each_listed_rule_on_the_deployed_app(launcher):
    launcher.main(target=2_000_000, rules="vanilla,cfr+", seed=3)
    launcher.modal.Function.from_name.assert_called_once_with("drawmaha-rung3", "train_rule")
    spawn = launcher.modal.Function.from_name.return_value.spawn
    assert spawn.call_args_list == [
        mock.call("vanilla", 2_000_000, 3),
        mock.call("cfr+", 2_000_000, 3),
    ]

def test_main_refuses_an_unknown_rule_before_spawning_any(launcher):
    with pytest.raises(ValueError, match="dcfrr"):
        launcher.main(rules="vanilla,dcfrr")
    launcher.modal.Function.from_name.return_value.spawn.assert_not_called()
