"""Π: fit on the policy memory under its own seed, frozen as a strategy, and it is the average.

Torch only. The claims: the policy fit's seed is distinct from both
advantage fits' and the fit is deterministic in it; a regret memory handed
by mistake is refused; the freeze is a distribution at every key and equals
the net read key by key through `encode`; an all-negative net plays uniform
everywhere; and on a memory with three snapshots of σ at one key the fit
lands on their t-weighted mean — the whole reason Π is a regression.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drawmaha_solver.deepcfr.collect import Collector  # noqa: E402
from drawmaha_solver.deepcfr.fit import FitConfig, fit_from_scratch  # noqa: E402
from drawmaha_solver.deepcfr.net import predict  # noqa: E402
from drawmaha_solver.deepcfr.policy import (  # noqa: E402
    POLICY_TAG,
    fit_policy,
    policy_seed,
    policy_strategy,
)
from drawmaha_solver.deepcfr.reservoir import TARGET_WIDTH  # noqa: E402
from drawmaha_solver.minidrawmaha.rules.enumeration import (  # noqa: E402
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.rules.features import encode  # noqa: E402
from drawmaha_solver.minidrawmaha.rules.game import InfoSet  # noqa: E402
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable  # noqa: E402
from drawmaha_solver.minidrawmaha.answer.strategy import StrategyInfo  # noqa: E402

TINY = FitConfig(width=16, depth=2, steps=40, batch=64, threads=2)

def some_keys() -> list[InfoSet]:
    """Two keys on every public point: both seats, every width (restated from test_train)."""
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

def policy_memory(seed: int, walks: int):
    """One phase's policy rows over a random-regret table, all at t = 1."""
    with Collector(seed=seed, workers=0) as collector:
        regret = collector.table.cumulative_regret
        regret[:] = np.random.default_rng(seed).standard_normal(regret.size)
        rows = collector.collect(seat=0, t=1, walks=walks)
    return rows.policy_rows, np.ones(len(rows.policy_rows), np.int32), rows.policy_targets

def same_weights(first, second) -> bool:
    a, b = first.state_dict(), second.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

def expected_row(net, key: InfoSet) -> np.ndarray:
    out = predict(net, encode(key)[None])[0][[int(a) for a in key.legal_actions()]]
    clipped = np.maximum(out.astype(np.float64), 0.0)
    if clipped.sum() <= 0.0:
        return np.full(len(clipped), 1.0 / len(clipped), np.float32)
    return (clipped / clipped.sum()).astype(np.float32)

def test_policy_seed_is_distinct_from_the_advantage_fits_and_the_fit_is_deterministic():
    assert POLICY_TAG == 0x5049  # "PI"
    assert policy_seed(9, 1) == (9, 1, POLICY_TAG)
    assert policy_seed(9, 1) not in {(9, 1, 0), (9, 1, 1)}
    rows, t, targets = policy_memory(9, 10)
    pi = fit_policy(rows, t, targets, seed=9, iteration=1, config=TINY)
    again = fit_policy(rows, t, targets, seed=9, iteration=1, config=TINY)
    assert same_weights(pi, again)
    for seat in (0, 1):
        advantage = fit_from_scratch(rows, t, targets, seed=(9, 1, seat), config=TINY)
        assert not same_weights(pi, advantage)
    assert not same_weights(pi, fit_policy(rows, t, targets, seed=9, iteration=2, config=TINY))

def test_fit_policy_refuses_a_regret_memory():
    rows, t, targets = policy_memory(9, 5)
    regrets = targets.copy()
    regrets[0, 0] = 3.0
    with pytest.raises(ValueError, match="distribution"):
        fit_policy(rows, t, regrets, seed=9, iteration=1, config=TINY)
    assert targets.shape[1] == TARGET_WIDTH

def test_policy_strategy_is_a_distribution_and_matches_the_net_key_by_key():
    rows, t, targets = policy_memory(9, 10)
    net = fit_policy(rows, t, targets, seed=9, iteration=1, config=TINY)
    frozen = policy_strategy(net, iteration=1, seed=9)
    assert frozen.info == StrategyInfo(rule="deep", column="pi", iteration=1, seed=9, workers=0)
    assert len(frozen) == len(PackedTable.whole_game())
    for key in some_keys():
        np.testing.assert_allclose(frozen[key], expected_row(net, key), atol=1e-6)
    # Every head output pushed far below zero: a distribution still, uniform everywhere.
    with torch.no_grad():
        head = net.layers[-1]
        head.weight.zero_()
        head.bias.fill_(-10.0)
    uniform = policy_strategy(net, iteration=1, seed=9)
    widths = uniform.widths
    np.testing.assert_array_equal(uniform.probabilities, np.repeat(1.0 / widths, widths).astype(np.float32))

def test_the_fit_lands_on_the_t_weighted_mean_at_a_repeated_key():
    # Three snapshots of σ at one 3-wide key, weights 1, 2, 3: the regression's
    # minimiser is their t-weighted mean, (0.47, 0.53, 0), with no row holding it.
    key = next(k for k in some_keys() if len(k.legal_actions()) == 3)
    row = PackedTable.whole_game().row_of(key)
    snapshots = [(1, (1.0, 0.0, 0.0)), (2, (0.0, 1.0, 0.0)), (3, (0.6, 0.4, 0.0))]
    rows = np.full(150, row, np.int32)
    t = np.repeat([t for t, _ in snapshots], 50).astype(np.int32)
    targets = np.zeros((150, TARGET_WIDTH), np.float32)
    targets[:, :3] = np.repeat([s for _, s in snapshots], 50, axis=0)
    net = fit_policy(rows, t, targets, seed=1, iteration=3, config=FitConfig(width=16, depth=2, steps=400, batch=64, threads=2))
    frozen = policy_strategy(net, iteration=3, seed=1)
    expected = np.array([(1 * 1.0 + 3 * 0.6) / 6, (2 * 1.0 + 3 * 0.4) / 6, 0.0])
    # 400 Adam steps of a 16-wide net on three distinct rows: within 0.02 of the mean.
    np.testing.assert_allclose(frozen[key], expected, atol=0.02)
