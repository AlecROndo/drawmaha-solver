"""Lockstep MCCFR in one process: the reference `parallel.py` must equal bit for bit.

* **One worker is the one-hand walk.** On Leduc, where no walk reaches a row
  twice, a lockstep iteration with W = 1 banks exactly what `mccfr.traverse`
  writes as it goes, given the same random stream — every rule, both columns.
* **The packed path is the dict path.** On mini-drawmaha with W = 3, the
  vectorised apply over a packed table equals the per-ledger apply over a dict
  of the same keys, every rule.
* **Checkpoints** round-trip and resume exactly, and refuse another run.
* **Leduc gate:** every rule with W = 4 reaches the LP value's neighbourhood.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.leduc.exploitability import exploitability, expected_value
from drawmaha_solver.leduc.game import DEALS, LeducState
from drawmaha_solver.leduc.infoset_table import new_infoset_table as new_leduc_table
from drawmaha_solver.minidrawmaha.lockstep import (
    load_lockstep,
    new_lockstep,
    read_lockstep,
    save_lockstep,
    stream,
    train_lockstep,
)
from drawmaha_solver.minidrawmaha.mccfr import traverse
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.regret_rules import (
    RULES,
    Average,
    RegretRule,
    column_average,
    extra_weights,
)
from drawmaha_solver.regret_matching import RegretMatcher

# The Leduc helpers `test_mccfr.py` uses, restated: test modules here do not
# import one another (importlib mode, no packages).
REFEREE = json.loads((Path(__file__).parents[1] / "leduc" / "referee.json").read_text())
LP_VALUE = REFEREE["lp_value_to_p0"]
BOTH_COLUMNS = (Average.UNIFORM, Average.QUADRATIC)

def leduc_deal(rng: np.random.Generator) -> LeducState:
    return LeducState(cards=DEALS[rng.integers(len(DEALS))])

def new_leduc_table_with(extra_averages: int) -> dict:
    return {
        key: RegretMatcher(ledger.n_actions, extra_averages=extra_averages)
        for key, ledger in new_leduc_table().items()
    }

def ledgers_equal(first, second) -> bool:
    return list(first) == list(second) and all(
        getattr(first[key], slot).tobytes() == getattr(second[key], slot).tobytes()
        for key in first
        for slot in ("cumulative_regret", "strategy_sum", "extra_sums", "stamp")
    )

SLOTS = ("cumulative_regret", "strategy_sum", "extra_sums", "stamp")

class GrowingTable(dict):
    """A dict that grows a ledger on first touch — how a test meets a run's keys."""

    def __init__(self, extra_averages: int) -> None:
        super().__init__()
        self.extra_averages = extra_averages

    def __missing__(self, key) -> RegretMatcher:
        ledger = self[key] = RegretMatcher(
            len(key.legal_actions()), extra_averages=self.extra_averages
        )
        return ledger

def keys_met(seed: int, workers: int, iterations: int, rule: RegretRule) -> list:
    grown = GrowingTable(len(BOTH_COLUMNS))
    solve = new_lockstep(seed, workers=workers, table=grown, rule=rule, averages=BOTH_COLUMNS)
    train_lockstep(solve, iterations)
    return list(grown)

def packed_equal(first: PackedTable, second: PackedTable) -> bool:
    return all(np.array_equal(getattr(first, s), getattr(second, s)) for s in SLOTS)

# ---------------------------------------------------------------------------
# One worker is the one-hand walk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", list(RegretRule))
def test_one_worker_is_the_one_hand_walk_on_leduc(rule):
    iterations, seed = 300, 4
    walked = new_leduc_table_with(len(BOTH_COLUMNS))
    for t in range(1, iterations + 1):
        for seat in (0, 1):
            rng = stream(seed, t, seat, 0)
            traverse(
                leduc_deal(rng),
                walked,
                seat,
                rng,
                t,
                bank=RULES[rule],
                column_weights=extra_weights(BOTH_COLUMNS, t),
            )
    lockstep = new_lockstep(
        seed,
        workers=1,
        table=new_leduc_table_with(len(BOTH_COLUMNS)),
        deal=leduc_deal,
        rule=rule,
        averages=BOTH_COLUMNS,
    )
    train_lockstep(lockstep, iterations)
    assert ledgers_equal(lockstep.table, walked)

def test_a_lockstep_walk_writes_nothing_until_the_phase_is_applied():
    from drawmaha_solver.minidrawmaha.lockstep import walk

    table = new_leduc_table_with(0)
    before = {key: ledger.cumulative_regret.copy() for key, ledger in table.items()}
    recorded = walk(table, leduc_deal, 1, 1, 0, 0, ())
    assert recorded.regret and recorded.strategy
    assert all(not ledger.strategy_sum.any() for ledger in table.values())
    assert all(np.array_equal(table[key].cumulative_regret, before[key]) for key in table)

# ---------------------------------------------------------------------------
# The packed path is the dict path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", list(RegretRule))
def test_the_packed_apply_equals_the_per_ledger_apply(rule):
    seed, workers, iterations = 3, 3, 12
    keys = keys_met(seed, workers, iterations, rule)
    run = {"workers": workers, "rule": rule, "averages": BOTH_COLUMNS}
    objects = {
        key: RegretMatcher(len(key.legal_actions()), extra_averages=2) for key in keys
    }
    train_lockstep(new_lockstep(seed, table=objects, **run), iterations)
    packed = PackedTable.listed(keys, extra_averages=2)
    train_lockstep(new_lockstep(seed, table=packed, **run), iterations)
    assert packed.strategy_sum.any()
    for key, ledger in objects.items():
        window = packed[key]
        for slot in SLOTS:
            assert np.array_equal(getattr(window, slot), getattr(ledger, slot)), (key, slot)

def test_more_workers_bank_more_hands_per_iteration():
    # W hands per seat per iteration: at the same t, three workers have
    # banked three walks' worth of opponent mass at the shared first spot.
    one = new_lockstep(0, workers=1, table=GrowingTable(0))
    three = new_lockstep(0, workers=3, table=GrowingTable(0))
    train_lockstep(one, 5)
    train_lockstep(three, 5)
    mass = lambda table: sum(ledger.strategy_sum.sum() for ledger in table.values())
    assert mass(three.table) > 2 * mass(one.table)

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dcfr_keys():
    return keys_met(7, 2, 20, RegretRule.DCFR)

def dcfr_run(keys, **kw):
    return new_lockstep(
        7,
        workers=2,
        table=PackedTable.listed(keys, extra_averages=2),
        rule="dcfr",
        averages=BOTH_COLUMNS,
        **kw,
    )

def test_a_checkpoint_resumes_to_equal_an_uninterrupted_run(tmp_path, dcfr_keys):
    path = tmp_path / "run.npz"
    save_lockstep(train_lockstep(dcfr_run(dcfr_keys), 10), path)
    assert not (tmp_path / "run.npz.tmp").exists()
    assert read_lockstep(path) == {
        "iteration": 10,
        "seed": 7,
        "workers": 2,
        "rule": "dcfr",
        "averages": ["uniform", "quadratic"],
    }
    resumed = load_lockstep(
        path,
        table=PackedTable.listed(dcfr_keys, extra_averages=2),
        workers=2,
        rule="dcfr",
        averages=BOTH_COLUMNS,
    )
    assert resumed.iteration == 10
    train_lockstep(resumed, 10)
    straight = train_lockstep(dcfr_run(dcfr_keys), 20)
    assert packed_equal(resumed.table, straight.table)
    assert straight.table.stamp.any()

@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"workers": 3}, "ran 2 workers"),
        ({"rule": "lcfr"}, "asks for lcfr"),
        ({"averages": ("uniform",)}, "asks for dcfr"),
    ],
)
def test_a_checkpoint_refuses_another_run(tmp_path, dcfr_keys, change, message):
    path = tmp_path / "run.npz"
    save_lockstep(train_lockstep(dcfr_run(dcfr_keys), 3), path)
    asked = {"workers": 2, "rule": "dcfr", "averages": BOTH_COLUMNS} | change
    table = PackedTable.listed(dcfr_keys, extra_averages=len(asked["averages"]))
    with pytest.raises(ValueError, match=message):
        load_lockstep(path, table=table, **asked)
    assert not table.cumulative_regret.any()

def test_a_serial_checkpoint_is_not_a_lockstep_one(tmp_path):
    from drawmaha_solver.minidrawmaha.mccfr import new_solve, save_solve, train

    serial = train(new_solve(1, table=GrowingTable(2), averages=BOTH_COLUMNS), 1)
    keys = list(serial.table)
    save_solve(serial, tmp_path / "serial.npz")
    with pytest.raises(ValueError, match="not a lockstep checkpoint"):
        load_lockstep(
            tmp_path / "serial.npz",
            table=PackedTable.listed(keys, extra_averages=2),
            workers=1,
            rule="vanilla",
            averages=BOTH_COLUMNS,
        )

# ---------------------------------------------------------------------------
# Leduc gate
# ---------------------------------------------------------------------------

def leduc_grade(solve, column: Average) -> tuple[float, float]:
    strategies = {
        key: column_average(ledger, column, solve.averages)
        for key, ledger in solve.table.items()
    }
    return exploitability(strategies), expected_value(strategies)[0]

def leduc_lockstep(rule: RegretRule, seed: int, workers: int, iterations: int):
    solve = new_lockstep(
        seed,
        workers=workers,
        table=new_leduc_table_with(len(BOTH_COLUMNS)),
        deal=leduc_deal,
        rule=rule,
        averages=BOTH_COLUMNS,
    )
    return train_lockstep(solve, iterations)

def leduc_gate_grades(rule: RegretRule) -> dict[Average, tuple[float, float]]:
    """Seed 1, W = 4, 12,500 iterations (50,000 hands a seat): every column's grade.

    Module-level so a process pool can run the four rules side by side.
    """
    solve = leduc_lockstep(rule, 1, 4, 12_500)
    return {column: leduc_grade(solve, column) for column in Average}

# Calibrated 2026-09-30, W = 4, seeds 1-3, at 12,500 iterations — the same
# 50,000 hands a seat as the one-hand gate in test_mccfr.py, and the same
# picture (vanilla-linear 0.18-0.24 here, 0.15-0.26 there; LCFR 0.059-0.072
# against 0.053-0.070). Bounds sit ~1.4x above the worst seed.
LOCKSTEP_GATES = [
    (RegretRule.VANILLA, Average.LINEAR, 0.34),
    (RegretRule.VANILLA, Average.UNIFORM, 0.11),
    (RegretRule.CFR_PLUS, Average.LINEAR, 0.13),
    (RegretRule.LCFR, Average.LINEAR, 0.10),
    (RegretRule.DCFR, Average.LINEAR, 0.13),
    (RegretRule.DCFR, Average.QUADRATIC, 0.13),
]

@pytest.fixture(scope="module")
def lockstep_gate_grades():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    rules = list(RegretRule)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).parents[2]))
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(rules), mp_context=context) as pool:
            return dict(zip(rules, pool.map(leduc_gate_grades, rules), strict=True))

@pytest.mark.parametrize(
    ("rule", "column", "bound"),
    LOCKSTEP_GATES,
    ids=[f"{r.value}-{c.value}" for r, c, _ in LOCKSTEP_GATES],
)
def test_every_rule_converges_on_leduc_with_four_workers(lockstep_gate_grades, rule, column, bound):
    exploitability_, value = lockstep_gate_grades[rule][column]
    assert exploitability_ < bound
    assert abs(value - LP_VALUE) < 0.025

def test_workers_must_be_a_real_int_not_a_bool():
    with pytest.raises(ValueError, match="positive int"):
        new_lockstep(0, workers=True, table={})

def test_asking_the_recorder_whether_it_holds_a_row_records_no_visit():
    from drawmaha_solver.minidrawmaha.lockstep import _Recorder

    key = ("row",)
    recorder = _Recorder({key: RegretMatcher(2)}, extra_rows=0)
    assert key in recorder and ("other",) not in recorder
    assert recorder.get(("other",)) is None
    assert recorder.recorded().strategy == []
    recorder.get(key)
    assert len(recorder.recorded().strategy) == 1
