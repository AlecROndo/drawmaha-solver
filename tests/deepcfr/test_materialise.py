"""Materialise: two seats' nets into the table's regret slots, eq. (3) by the write.

What is pinned here, and why:

- `regret_slots_from_outputs` is the spec's three clauses: some output
  positive passes through unchanged (the ledger clips), all non-positive
  and not all equal writes a one-hot at the argmax (shared by tied maxima,
  so the strategy does not depend on `Action` order), all equal writes
  zeros (uniform). `RegretMatcher.strategy()` on the written slots is then
  Deep CFR's eq. (3), which is correctness gate 3.
- `materialise` writes every row's slots where the table's own `slot_of`
  puts them, from the net of the seat whose decision it is, gathered
  through the point's legal actions — checked key by key against the
  reference encoder on a sample that touches all 141 points.
- An untrained seat (`None`) writes zeros, uniform, without touching the
  other seat's rows; nothing but `cumulative_regret` is written.
- A table that is not the whole game — the wrong size, or the whole
  game's rows in another order — or the wrong number of nets, is refused
  by name.

The whole-game tests run the real 6.2M-row materialise; they are kept to
the four that need it. Every test skips without torch.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drawmaha_solver.deepcfr.materialise import (  # noqa: E402
    materialise,
    regret_slots_from_outputs,
)
from drawmaha_solver.deepcfr.net import Net, predict  # noqa: E402
from drawmaha_solver.minidrawmaha.rules.enumeration import all_infosets  # noqa: E402
from drawmaha_solver.minidrawmaha.rules.features import encode  # noqa: E402
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable  # noqa: E402
from drawmaha_solver.regret_matching import RegretMatcher  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: the whole table, a sample of keys over every point, seeded nets
# ---------------------------------------------------------------------------

def point_of(key):
    return (len(key.board), key.draws, key.betting)

@pytest.fixture(scope="module")
def sampled_keys():
    # 4,000 keys spread over the whole enumeration, plus the first key of
    # every public point, so the small points the stride skips are covered.
    seen = set()
    keys = []
    for index, key in enumerate(all_infosets()):
        if index % 1555 == 0 or point_of(key) not in seen:
            seen.add(point_of(key))
            keys.append(key)
    return keys

@pytest.fixture(scope="module")
def whole_table():
    return PackedTable.whole_game()

@pytest.fixture(autouse=True)
def _clean_slots(request):
    # Each whole-table test starts from the all-zero table it would be
    # handed at iteration 1, whatever the previous test wrote.
    if "whole_table" in request.fixturenames:
        request.getfixturevalue("whole_table").cumulative_regret[:] = 0.0

def seeded_net(seed: int) -> Net:
    torch.manual_seed(seed)
    return Net()

def ledger_over(slots: np.ndarray) -> RegretMatcher:
    return RegretMatcher.over(
        cumulative_regret=slots.astype(np.float64),
        strategy_sum=np.zeros(len(slots)),
        extra_sums=np.zeros((0, len(slots))),
        stamp=np.zeros(1, dtype=np.int64),
    )

# ---------------------------------------------------------------------------
# The three clauses
# ---------------------------------------------------------------------------

def test_some_positive_passes_through():
    out = np.array([[0.5, -2.0, 0.1]])
    got = regret_slots_from_outputs(out)
    np.testing.assert_array_equal(got, out)
    assert got.dtype == np.float64

def test_all_nonpositive_writes_the_argmax_one_hot():
    np.testing.assert_array_equal(
        regret_slots_from_outputs(np.array([[-3.0, -0.5, -1.0, -2.0]])), [[0.0, 1.0, 0.0, 0.0]]
    )

def test_zero_is_not_positive():
    # A maximum of exactly 0 with negatives beside it is clause two, not one:
    # nothing is positive, so the argmax gets the one-hot.
    np.testing.assert_array_equal(
        regret_slots_from_outputs(np.array([[-1.0, 0.0, -2.0]])), [[0.0, 1.0, 0.0]]
    )

def test_tied_maxima_share_the_one_hot():
    np.testing.assert_array_equal(
        regret_slots_from_outputs(np.array([[-1.0, -0.5, -0.5]])), [[0.0, 1.0, 1.0]]
    )

def test_all_equal_writes_zeros():
    np.testing.assert_array_equal(
        regret_slots_from_outputs(np.full((2, 3), -0.7)), np.zeros((2, 3))
    )

def test_rows_are_classified_one_by_one():
    out = np.array([[0.5, -2.0, 0.1], [-3.0, -0.5, -1.0], [-0.7, -0.7, -0.7]])
    np.testing.assert_array_equal(
        regret_slots_from_outputs(out),
        [[0.5, -2.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    )

def test_outputs_are_not_written_through():
    out = np.array([[-3.0, -0.5]])
    regret_slots_from_outputs(out)
    np.testing.assert_array_equal(out, [[-3.0, -0.5]])

def test_ledger_strategy_is_eq_three():
    # Eq. (3): max(out, 0) / sum when some output is positive, else uniform
    # over the argmax set. The ledger's own `strategy()` on the written
    # slots must produce it, row by row, over random signed outputs.
    rng = np.random.default_rng(0)
    for width in (2, 3, 4):
        outputs = rng.uniform(-1.0, 1.0, size=(200, width))
        outputs[:50] -= 2.0  # all non-positive rows, so clause two is exercised
        outputs[50] = outputs[50].max()  # one all-equal row
        slots = regret_slots_from_outputs(outputs)
        for out, written in zip(outputs, slots, strict=True):
            positive = np.maximum(out, 0.0)
            if positive.sum() > 0.0:
                expect = positive / positive.sum()
            else:
                top = out == out.max()
                expect = top / top.sum()
            np.testing.assert_allclose(ledger_over(written).strategy(), expect)

def test_non_finite_outputs_are_refused():
    # A NaN output would classify as "nothing positive, nothing equal" and
    # write zeros: a diverged net playing uniform with nothing in the log.
    with pytest.raises(ValueError, match="finite"):
        regret_slots_from_outputs(np.array([[np.nan, -1.0, -2.0]]))

def test_materialise_refuses_a_net_with_non_finite_weights(whole_table):
    net = seeded_net(0)
    with torch.no_grad():
        net.layers[0].weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        materialise(whole_table, (net, None))

def test_refuses_anything_but_a_batch_of_rows():
    with pytest.raises(ValueError, match="2-D"):
        regret_slots_from_outputs(np.array([1.0, -1.0]))

# ---------------------------------------------------------------------------
# The whole table
# ---------------------------------------------------------------------------

def test_untrained_seats_write_uniform(whole_table):
    whole_table.cumulative_regret[:] = 5.0
    materialise(whole_table, (None, None))
    assert not whole_table.cumulative_regret.any()

def test_materialise_equals_the_per_key_path(whole_table, sampled_keys):
    nets = (seeded_net(0), seeded_net(1))
    materialise(whole_table, nets)
    for key in sampled_keys:
        _, start, width = whole_table.slot_of(key)
        gather = [int(action) for action in key.legal_actions()]
        outputs = predict(nets[key.player], encode(key)[None])[:, gather]
        expect = regret_slots_from_outputs(outputs)[0]
        np.testing.assert_allclose(
            whole_table.cumulative_regret[start : start + width], expect, rtol=1e-5, atol=1e-6
        )

def test_each_point_is_scored_by_its_own_seats_net(whole_table, sampled_keys):
    materialise(whole_table, (seeded_net(0), None))
    seat_0_rows = 0
    for key in sampled_keys:
        _, start, width = whole_table.slot_of(key)
        written = whole_table.cumulative_regret[start : start + width]
        if key.player == 1:
            assert not written.any()
        else:
            seat_0_rows += written.any()
    # Seat 0's rows are scored: a random net all but never writes a zero row.
    assert seat_0_rows > 0.9 * sum(key.player == 0 for key in sampled_keys)

def test_materialise_touches_only_cumulative_regret(whole_table):
    whole_table.strategy_sum[:] = 2.0
    materialise(whole_table, (seeded_net(0), seeded_net(1)))
    assert (whole_table.strategy_sum == 2.0).all()
    assert not whole_table.stamp.any()
    whole_table.strategy_sum[:] = 0.0

# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_materialise_refuses_a_listed_table(sampled_keys):
    with pytest.raises(ValueError, match="whole game"):
        materialise(PackedTable.listed(sampled_keys), (None, None))

def test_materialise_refuses_the_whole_games_rows_in_another_order(whole_table):
    # The same 6.2M rows and the same columns, but not in the layout's
    # order: the size check alone would write every point's outputs into
    # the wrong slots. A listed table over every key would cost gigabytes
    # to build, so the stand-in is the whole table with its widths reversed.
    class Reordered:
        cumulative_regret = whole_table.cumulative_regret

        def __len__(self):
            return len(whole_table)

        def widths(self):
            return whole_table.widths()[::-1]

    with pytest.raises(ValueError, match="order"):
        materialise(Reordered(), (None, None))

def test_materialise_refuses_wrong_number_of_nets(whole_table):
    with pytest.raises(ValueError, match="two"):
        materialise(whole_table, (None,))
