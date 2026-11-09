"""Friction toward a reference (RTCFR+-style) in the sampled walk.

* **Off is today.** `friction=None` and friction at μ = 0 bank exactly today's regrets.
* **A reference equal to play cancels.** At t = 1 every σ is uniform; a uniform
  reference has the same reach, so friction at any μ changes nothing.
* **One spot, by hand.** The adjustment is μ·(own reach·σ − reference reach·reference).
* **The reset** copies every row's current strategy, the same numbers
  `RegretMatcher.strategy` gives.
* **Refusals.** A friction run is neither checkpointed nor parallelised.
* **Leduc gate (slow).** Friction + CFR+ makes the current strategy converge where
  plain CFR+'s current strategy does not.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from drawmaha_solver.leduc.exploitability import exploitability
from drawmaha_solver.leduc.game import DEALS, LeducState
from drawmaha_solver.leduc.infoset_table import new_infoset_table as new_leduc_table
from drawmaha_solver.minidrawmaha.rules.enumeration import all_infosets
from drawmaha_solver.minidrawmaha.solver.friction import Friction, friction_adjustment
from drawmaha_solver.minidrawmaha.solver.lockstep import (
    new_lockstep,
    save_lockstep,
    train_lockstep,
)
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.solver.parallel import ParallelLockstep
from drawmaha_solver.regret_matching import RegretMatcher

def leduc_deal(rng: np.random.Generator) -> LeducState:
    return LeducState(cards=DEALS[rng.integers(len(DEALS))])

def fresh_leduc() -> dict:
    return {key: RegretMatcher(ledger.n_actions) for key, ledger in new_leduc_table().items()}

def regrets(table: dict) -> bytes:
    return b"".join(table[key].cumulative_regret.tobytes() for key in table)

def leduc_run(rule: str, workers: int, *, mu: float | None, iterations: int) -> dict:
    table = fresh_leduc()
    solve = new_lockstep(7, workers=workers, table=table, deal=leduc_deal, rule=rule)
    if mu is not None:
        solve.friction = Friction.new(table, mu)
    train_lockstep(solve, iterations)
    return table

@pytest.mark.parametrize("rule", ["lcfr", "cfr+"])
def test_friction_at_mu_zero_banks_todays_regrets(rule):
    plain = leduc_run(rule, 4, mu=None, iterations=30)
    zero = leduc_run(rule, 4, mu=0.0, iterations=30)
    assert regrets(plain) == regrets(zero)

def test_a_uniform_reference_cancels_at_the_first_iteration():
    plain = leduc_run("lcfr", 4, mu=None, iterations=1)
    pulled = leduc_run("lcfr", 4, mu=5.0, iterations=1)
    assert regrets(plain) == regrets(pulled)

def test_friction_moves_regrets_once_play_leaves_the_reference():
    plain = leduc_run("lcfr", 4, mu=None, iterations=20)
    pulled = leduc_run("lcfr", 4, mu=5.0, iterations=20)
    assert regrets(plain) != regrets(pulled)

def test_strong_friction_toward_a_fixed_uniform_reference_keeps_play_nearer_uniform():
    # Pins the sign and the reach plumbing end to end: with the reference held at
    # uniform (no resets) a large μ must leave the current strategy closer to
    # uniform than plain play does. A flipped sign or a dropped reach would push
    # play away from the reference, or not at all.
    def distance_to_uniform(table: dict) -> float:
        return sum(np.abs(ledger.strategy() - 1.0 / ledger.n_actions).sum() for ledger in table.values())

    plain = leduc_run("lcfr", 8, mu=None, iterations=300)
    pulled = leduc_run("lcfr", 8, mu=50.0, iterations=300)
    assert distance_to_uniform(pulled) < 0.8 * distance_to_uniform(plain)

@pytest.mark.parametrize("mu", [-1.0, float("nan"), float("inf")])
def test_a_negative_or_non_finite_mu_is_refused(mu):
    with pytest.raises(ValueError, match="mu"):
        Friction.new(fresh_leduc(), mu)

def test_the_adjustment_at_one_spot_by_hand():
    sigma = np.array([0.5, 0.25, 0.25])
    reference = np.array([0.2, 0.2, 0.6])
    # own reach 0.4, reference reach 0.5, μ = 2:
    # x = 0.4σ = (0.2, 0.1, 0.1); r = 0.5·ref = (0.1, 0.1, 0.3); μ(x − r) = (0.2, 0, −0.4)
    got = friction_adjustment(sigma, reference, own_reach=0.4, reference_reach=0.5, mu=2.0)
    np.testing.assert_allclose(got, [0.2, 0.0, -0.4], rtol=0, atol=1e-15)

def test_the_reset_copies_every_rows_current_strategy():
    keys = list(all_infosets())[:5000]
    table = PackedTable.listed(keys)
    rng = np.random.default_rng(3)
    table.cumulative_regret[:] = rng.normal(size=table.cumulative_regret.size)
    # some rows all non-positive, so the uniform fallback is exercised
    table.cumulative_regret[: 40] = -1.0
    friction = Friction.new(table, 1.0)
    # Equal to the last bit on most rows; `np.add.reduceat` sums a 3- or 4-wide row
    # in a different order than `ndarray.sum`, which moves ~2% of rows by one ulp.
    # A reference only sets where the pull points, so one ulp is immaterial.
    # every 37th row, plus every row touching the all-negative columns (the uniform branch)
    for key in sorted(set(keys[::37]) | set(keys[:20]), key=keys.index):
        np.testing.assert_allclose(friction.of(key), table[key].strategy(), rtol=0, atol=1e-15)

def test_a_dict_reset_tracks_play():
    table = fresh_leduc()
    friction = Friction.new(table, 1.0)
    key = next(iter(table))
    table[key].cumulative_regret[:] = np.arange(table[key].n_actions, dtype=float)
    friction.reset()
    assert friction.of(key).tobytes() == table[key].strategy().tobytes()

def test_a_friction_run_is_not_checkpointed(tmp_path):
    table = fresh_leduc()
    solve = new_lockstep(1, workers=2, table=table, deal=leduc_deal, rule="lcfr")
    solve.friction = Friction.new(table, 1.0)
    with pytest.raises(ValueError, match="friction"):
        save_lockstep(solve, tmp_path / "solve.npz")

def test_a_friction_run_is_not_parallelised():
    table = PackedTable.listed(list(all_infosets())[:100])
    solve = new_lockstep(1, workers=2, table=table, rule="lcfr")
    solve.friction = Friction.new(table, 1.0)
    with pytest.raises(ValueError, match="friction"):
        ParallelLockstep(solve)

@pytest.mark.skipif(
    not os.environ.get("MINIDRAWMAHA_FULL_CALIBRATION"),
    reason="Leduc friction gate: ~minutes; set MINIDRAWMAHA_FULL_CALIBRATION=1",
)
def test_leduc_friction_makes_cfr_plus_current_strategy_converge():
    def current_exploitability(table: dict) -> float:
        return exploitability({key: ledger.strategy() for key, ledger in table.items()})

    plain = fresh_leduc()
    solve = new_lockstep(0, workers=32, table=plain, deal=leduc_deal, rule="cfr+")
    train_lockstep(solve, 3000)

    # μ = 0.05 with resets every 100 iterations: Leduc payoffs are a few chips, so
    # this tax is small against them, so the bias toward each reference stays small
    # while the pull still damps CFR+'s overshoot. Seed 0;
    # the strict < is the claim itself: friction's current strategy beats plain's.
    pulled = fresh_leduc()
    solve = new_lockstep(0, workers=32, table=pulled, deal=leduc_deal, rule="cfr+")
    solve.friction = Friction.new(pulled, 0.05)
    for _ in range(30):
        train_lockstep(solve, 100)
        solve.friction.reset()
    assert current_exploitability(pulled) < current_exploitability(plain)
