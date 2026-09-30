"""The four regret rules and the averaging columns, one ledger at a time.

The walk is tested in `test_mccfr.py`; this file tests only what a rule does to
the ledger it is handed. Each rule is pinned to its formula on a hand-worked
row, and DCFR's lazy discount — the one piece with real arithmetic in it — is
pinned twice over: against an eager loop that discounts every iteration, which
is what the lazy form must equal, and against brute-force sums for the running
log-discount it reads, out to fifty million iterations, which is how far a run
is planned to go.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.regret_rules import (
    RULES,
    Average,
    RegretRule,
    bank_cfr_plus,
    bank_discounted,
    bank_linear,
    bank_vanilla,
    column_average,
    cumulative_log_discount,
    extra_weights,
    settled_regret,
    validate_averages,
)
from drawmaha_solver.regret_matching import RegretMatcher

def ledger_with(regret, *, stamp=0, extra_averages=0) -> RegretMatcher:
    ledger = RegretMatcher(len(regret), extra_averages=extra_averages)
    ledger.cumulative_regret[:] = regret
    ledger.stamp[0] = stamp
    return ledger

def positive_factor(t: int) -> float:
    """DCFR's per-iteration shrink of a positive regret, straight from the paper."""
    return t**1.5 / (t**1.5 + 1)

def eager_discounted(visits: dict[int, np.ndarray], width: int) -> dict[int, list[float]]:
    """DCFR as the paper writes it: every iteration adds (zero if unvisited), then discounts.

    Returns the row as it stands right after each visit. Scalar Python rather
    than NumPy, because tens of thousands of three-entry array operations cost
    seconds and this loop is the reference, not the thing under test.
    """
    regret, after = [0.0] * width, {}
    for t in range(1, max(visits) + 1):
        added = visits.get(t)
        shrink = positive_factor(t)
        for a in range(width):
            value = regret[a] + (added[a] if added is not None else 0.0)
            regret[a] = value * (shrink if value > 0.0 else 0.5)
        if added is not None:
            after[t] = list(regret)
    return after

# ---------------------------------------------------------------------------
# The rules by name
# ---------------------------------------------------------------------------

def test_every_rule_has_a_bank():
    assert set(RULES) == set(RegretRule)
    assert [rule.value for rule in RegretRule] == ["vanilla", "cfr+", "lcfr", "dcfr"]

def test_a_rule_name_the_module_does_not_know_is_refused():
    with pytest.raises(ValueError, match="'rm-plus'"):
        RegretRule("rm-plus")

# ---------------------------------------------------------------------------
# Vanilla, LCFR, CFR+
# ---------------------------------------------------------------------------

def test_vanilla_adds_the_regret_at_weight_one():
    ledger = ledger_with([1.0, -2.0, 0.5])
    bank_vanilla(ledger, np.array([0.25, 0.5, -1.0]), 9)
    assert ledger.cumulative_regret.tolist() == [1.25, -1.5, -0.5]

def test_lcfr_adds_the_regret_weighted_by_the_iteration():
    ledger = ledger_with([1.0, -2.0, 0.5])
    bank_linear(ledger, np.array([0.25, 0.5, -1.0]), 4)
    assert ledger.cumulative_regret.tolist() == [2.0, 0.0, -3.5]

def test_cfr_plus_clips_the_sum_at_zero_after_adding():
    # Clipping the SUM, not the increment: a negative increment still eats into
    # a positive balance before anything is floored.
    ledger = ledger_with([1.0, -2.0, 0.5])
    bank_cfr_plus(ledger, np.array([-0.25, 0.5, -1.0]), 9)
    assert ledger.cumulative_regret.tolist() == [0.75, 0.0, 0.0]

@pytest.mark.parametrize("bank", [bank_vanilla, bank_linear, bank_cfr_plus])
def test_only_dcfr_touches_the_stamp(bank):
    ledger = ledger_with([1.0, -1.0])
    bank(ledger, np.array([0.5, -0.5]), 12)
    assert ledger.stamp[0] == 0

@pytest.mark.parametrize("bank", list(RULES.values()))
def test_every_rule_writes_in_place(bank):
    # The ledger may be a view into a packed table's buffer; a rule that
    # rebinds the attribute would bank into a copy the table never sees.
    buffer = np.array([1.0, -1.0])
    ledger = RegretMatcher.over(
        cumulative_regret=buffer,
        strategy_sum=np.zeros(2),
        extra_sums=np.zeros((0, 2)),
        stamp=np.zeros(1, dtype=np.int64),
    )
    bank(ledger, np.array([0.5, -0.5]), 3)
    assert ledger.cumulative_regret is buffer
    assert not np.array_equal(buffer, [1.0, -1.0])

# ---------------------------------------------------------------------------
# DCFR: one visit, then the lazy discount against the eager one
# ---------------------------------------------------------------------------

def test_dcfr_discounts_by_sign_after_adding():
    # At t = 4, t^1.5 = 8: positives keep 8/9, negatives keep 1/2. The sign is
    # read AFTER the add — the first entry is positive only because of it.
    ledger = ledger_with([-0.5, 0.0, 0.0], stamp=3)
    bank_discounted(ledger, np.array([1.4, 0.9, -0.6]), 4)
    np.testing.assert_allclose(
        ledger.cumulative_regret, [0.9 * 8 / 9, 0.9 * 8 / 9, -0.3], rtol=1e-15
    )
    assert ledger.stamp[0] == 4

def test_dcfr_at_the_first_iteration_halves_both_signs():
    # t = 1: 1^1.5 / (1^1.5 + 1) = 1/2, the same as the negative factor.
    ledger = ledger_with([0.0, 0.0])
    bank_discounted(ledger, np.array([2.0, -2.0]), 1)
    assert ledger.cumulative_regret.tolist() == [1.0, -1.0]
    assert ledger.stamp[0] == 1

def test_dcfr_owes_nothing_for_a_gap_of_zero():
    # Visited at t - 1 and again at t: no iteration was missed, so only day t's
    # own factor applies.
    ledger = ledger_with([3.0, -3.0], stamp=6)
    bank_discounted(ledger, np.zeros(2), 7)
    np.testing.assert_allclose(
        ledger.cumulative_regret, [3.0 * positive_factor(7), -1.5], rtol=1e-15
    )

@pytest.mark.parametrize("seed", range(3))
def test_dcfr_lazy_equals_dcfr_eager(seed):
    # Sparse visits with random regrets, including a gap long enough to push
    # the negatives through a thousand halvings (to exactly zero) and one that
    # crosses the exact table's end into the tail formula.
    rng = np.random.default_rng(seed)
    visit_at = [1, 3, 10, 11, 500, 2_000, 66_000, 70_000]
    visits = {t: rng.normal(size=3) for t in visit_at}
    eager = eager_discounted(visits, 3)
    ledger = ledger_with([0.0, 0.0, 0.0])
    for t in visit_at:
        bank_discounted(ledger, visits[t], t)
        np.testing.assert_allclose(ledger.cumulative_regret, eager[t], rtol=1e-12, atol=1e-300)

def test_a_pending_discount_never_changes_the_strategy():
    # Why reads need no catch-up: every positive entry of a row is owed the
    # same factor, and the strategy normalises the positives.
    ledger = ledger_with([3.0, -1.0, 1.0, 0.0], stamp=10)
    settled = ledger_with(settled_regret(ledger, 5_000))
    np.testing.assert_allclose(settled.strategy(), ledger.strategy(), rtol=1e-15)

def test_settled_regret_brings_a_row_up_to_date_without_touching_it():
    ledger = ledger_with([3.0, -1.0], stamp=4)
    settled = settled_regret(ledger, 9)
    expected = [3.0 * math.prod(positive_factor(s) for s in range(5, 10)), -(0.5**5)]
    np.testing.assert_allclose(settled, expected, rtol=1e-14)
    assert ledger.cumulative_regret.tolist() == [3.0, -1.0]
    assert settled_regret(ledger, 4).tolist() == [3.0, -1.0]

def test_a_row_cannot_be_settled_or_banked_into_its_past():
    # A stamp ahead of the iteration means the row came from another run, or
    # was banked twice in one iteration — neither can be discounted backwards.
    ledger = ledger_with([1.0, -1.0], stamp=9)
    with pytest.raises(ValueError, match="stamped at iteration 9"):
        settled_regret(ledger, 8)
    with pytest.raises(ValueError, match="stamped at iteration 9"):
        bank_discounted(ledger, np.zeros(2), 9)

# ---------------------------------------------------------------------------
# DCFR: the running log-discount, exact table and tail formula
# ---------------------------------------------------------------------------

@functools.cache
def brute_force_log_discount(n: int) -> float:
    """−Σ_{k ≤ n} log(1 + k^−1.5), summed accurately in million-term chunks.

    Each chunk is a NumPy pairwise sum of same-signed terms and the chunk
    totals are added with `math.fsum`, so the reference is good to ~1e-15
    at fifty million terms. A literal product of fifty million factors would
    not be: its own rounding, one per factor, would reach ~5e-9.
    """
    totals = [
        float(np.log1p(np.arange(start, min(start + 1_000_000, n + 1), dtype=np.float64) ** -1.5).sum())
        for start in range(1, n + 1, 1_000_000)
    ]
    return -math.fsum(totals)

TABLE_END = 2**16

@pytest.mark.parametrize(
    "n",
    [0, 1, 2, 100, TABLE_END - 1, TABLE_END, TABLE_END + 1, TABLE_END + 2, 10**6, 10**7, 5 * 10**7],
)
def test_the_running_log_discount_matches_a_brute_force_sum(n):
    # Measured within 4.4e-16 (one unit in the last place of L ≈ −2.2) at every
    # n here. 1e-14 keeps a 20x margin and still fails a tail formula missing
    # its a^−2.5 term, which is off by 1.1e-13 at a million.
    assert cumulative_log_discount(n) == pytest.approx(
        brute_force_log_discount(n), rel=0, abs=1e-14
    )

@pytest.mark.parametrize(
    ("since", "until"),
    [
        (0, 1),
        (0, TABLE_END + 5),
        (TABLE_END - 3, TABLE_END + 3),
        (TABLE_END, TABLE_END + 1),
        (10, 5 * 10**7),
        (10**7, 5 * 10**7),
    ],
)
def test_a_missed_stretch_is_discounted_by_the_product_it_stands_for(since, until):
    # What a visit actually multiplies by: the positive factors for iterations
    # since+1 … until, against the accurately summed product. A relative error
    # in the factor is the absolute error in the log difference, so this holds
    # to the same 1e-14.
    owed = math.exp(cumulative_log_discount(until) - cumulative_log_discount(since))
    exact = math.exp(brute_force_log_discount(until) - brute_force_log_discount(since))
    assert owed == pytest.approx(exact, rel=1e-14)

def test_the_log_discount_refuses_a_negative_iteration():
    with pytest.raises(ValueError, match="-1"):
        cumulative_log_discount(-1)

# ---------------------------------------------------------------------------
# The averaging columns
# ---------------------------------------------------------------------------

def test_the_averaging_weights_are_one_t_and_t_squared():
    weights = extra_weights((Average.UNIFORM, Average.QUADRATIC), 7)
    assert weights.shape == (2, 1)
    assert weights[:, 0].tolist() == [1.0, 49.0]

def test_no_extra_columns_means_no_weights():
    assert extra_weights((), 7) is None

def test_averages_are_validated_by_name():
    assert validate_averages(("uniform", "quadratic")) == (Average.UNIFORM, Average.QUADRATIC)
    with pytest.raises(ValueError, match="'cubic'"):
        validate_averages(("cubic",))
    with pytest.raises(ValueError, match="twice"):
        validate_averages(("uniform", "uniform"))
    # The primary strategy sum IS the linear average; a linear extra row would
    # bank the same numbers twice.
    with pytest.raises(ValueError, match="primary"):
        validate_averages(("linear",))

def test_a_column_average_reads_its_own_row():
    ledger = RegretMatcher(2, extra_averages=2)
    ledger.strategy_sum[:] = [1.0, 3.0]
    ledger.extra_sums[:] = [[1.0, 1.0], [0.0, 2.0]]
    averages = (Average.UNIFORM, Average.QUADRATIC)
    assert column_average(ledger, Average.LINEAR, averages).tolist() == [0.25, 0.75]
    assert column_average(ledger, Average.UNIFORM, averages).tolist() == [0.5, 0.5]
    assert column_average(ledger, Average.QUADRATIC, averages).tolist() == [0.0, 1.0]

def test_an_empty_column_averages_to_uniform():
    ledger = RegretMatcher(4, extra_averages=1)
    assert column_average(ledger, Average.UNIFORM, (Average.UNIFORM,)).tolist() == [0.25] * 4

def test_a_column_the_run_did_not_bank_is_refused():
    ledger = RegretMatcher(2, extra_averages=1)
    with pytest.raises(ValueError, match="quadratic"):
        column_average(ledger, Average.QUADRATIC, (Average.UNIFORM,))
