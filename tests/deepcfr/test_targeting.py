"""Targeting: which table rows the boost stage aims at, and how they are drawn.

Five claims. The groups are the four granularities the plan names, read off
the rules alone (141 betting lines, 3,870 line × hand-class groups). The noise
score is one walk's relative error at a group, from the walk's own rows only.
The aimed arms take the groups of highest leak × noise per reached infoset
until the pool is `pool × B` reached rows. A target set's draws are
reproducible in (seed, t, seat, stream), never repeat a row, and are
successive sampling ∝ the pool's weights. And no module of the trainer's
package can import the grader.
"""

from __future__ import annotations

import ast
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.deepcfr import targeting
from drawmaha_solver.deepcfr.reservoir import TARGET_WIDTH, Reservoir
from drawmaha_solver.deepcfr.targeting import TargetSet, TargetSpec, choose, groups, noise, target_stream
from drawmaha_solver.minidrawmaha.rules.enumeration import private_keys, public_decision_points

POINTS = public_decision_points()
SIZES = [len(private_keys(board_cards=p.board_cards, discards=p.discards)) for p in POINTS]
OFFSET = np.concatenate([[0], np.cumsum(SIZES)])
N_ROWS = int(OFFSET[-1])

def rows_of_point(p: int) -> np.ndarray:
    return np.arange(OFFSET[p], OFFSET[p + 1])

def seat_points(seat: int) -> list[int]:
    return [p for p, point in enumerate(POINTS) if point.player == seat]

# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def test_the_spec_carries_the_plans_proposals_and_no_alpha_default():
    spec = TargetSpec(granularity="line", alpha=0.0)
    assert (spec.budget, spec.refresh, spec.noise_cap, spec.pool) == (2_000, 25, 20.0, 20)
    assert spec.round_two_only is False  # the plan's rule: every row eligible
    with pytest.raises(TypeError):
        TargetSpec(granularity="line")  # α is PR 4's to settle: every caller states it

@pytest.mark.parametrize(
    "fields, match",
    [
        ({"granularity": "none"}, "granularity"),
        ({"granularity": "hand"}, "granularity"),
        ({"budget": 0}, "budget"),
        ({"budget": 2.5}, "budget"),
        ({"budget": True}, "budget"),
        ({"refresh": 0}, "refresh"),
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"alpha": True}, "alpha"),
        ({"noise_cap": 0.0}, "noise_cap"),
        ({"noise_cap": float("inf")}, "noise_cap"),
        ({"pool": 0}, "pool"),
        ({"stream": -1}, "stream"),
        ({"round_two_only": 1}, "round_two_only"),
    ],
)
def test_the_spec_refuses_a_bad_field_by_name(fields, match):
    with pytest.raises(ValueError, match=match):
        TargetSpec(**({"granularity": "line", "alpha": 0.0} | fields))

# ---------------------------------------------------------------------------
# The groups
# ---------------------------------------------------------------------------

def test_the_layout_is_the_tables():
    from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable

    assert targeting.n_rows() == N_ROWS == 6_220_050
    # The packed table lays its rows out the same way: point by point, keys in order.
    assert len(PackedTable.whole_game()) == N_ROWS

@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize("granularity", ["random", "infoset", "line", "line_class"])
def test_a_seats_groups_cover_its_rows_and_no_others(granularity, seat):
    group = groups(granularity, seat)
    assert group.shape == (N_ROWS,) and group.dtype == np.int64
    assert not group.flags.writeable
    mine = np.zeros(N_ROWS, dtype=bool)
    for p in seat_points(seat):
        mine[rows_of_point(p)] = True
    assert (group[mine] >= 0).all() and (group[~mine] == -1).all()

def test_an_infoset_is_its_row_and_a_line_is_its_point():
    for seat in (0, 1):
        assert np.array_equal(groups("infoset", seat)[groups("infoset", seat) >= 0], np.flatnonzero(groups("infoset", seat) >= 0))
        line = groups("line", seat)
        for p in seat_points(seat):
            assert (line[rows_of_point(p)] == p).all()
        assert set(np.unique(groups("random", seat))) == {-1, 0}
    lines = {int(g) for seat in (0, 1) for g in np.unique(groups("line", seat)) if g >= 0}
    assert len(lines) == len(POINTS) == 141

def test_line_class_is_point_by_inner_by_outer_with_3870_groups():
    from drawmaha_solver.minidrawmaha.rules.hands import inner_score, outer_score

    found = {int(g) for seat in (0, 1) for g in np.unique(groups("line_class", seat)) if g >= 0}
    assert len(found) == 3_870
    # Decode a sample of rows by hand from the key itself.
    rng = np.random.default_rng(0)
    for p in rng.choice(len(POINTS), size=12, replace=False):
        point = POINTS[p]
        keys = private_keys(board_cards=point.board_cards, discards=point.discards)
        group = groups("line_class", point.player)
        for position in rng.choice(len(keys), size=5, replace=False):
            hole, _, board = keys[position]
            inner = int(inner_score(hole).category)
            outer = int(outer_score(hole, board).category) if point.board_cards == 2 else 0
            assert group[OFFSET[p] + position] == targeting.line_class_id(int(p), inner, outer)

def test_line_class_ids_decode_and_never_collide():
    seen = {}
    for p in range(len(POINTS)):
        for inner in range(1, 7):
            for outer in range(0, 8):
                gid = targeting.line_class_id(p, inner, outer)
                assert gid not in seen
                seen[gid] = (p, inner, outer)
                assert targeting.decode_line_class(gid) == (p, inner, outer)

# ---------------------------------------------------------------------------
# The noise score
# ---------------------------------------------------------------------------

A, B_, C = seat_points(0)[:3]  # three of seat 0's lines

def memory_of(entries, *, capacity=1_000) -> Reservoir:
    """A memory holding `entries`, (row, t, target row, weight) each, offered in order at their t."""
    memory = Reservoir(capacity)
    for t in sorted({t for _, t, _, _ in entries}):
        batch = [e for e in entries if e[1] == t]
        target = np.zeros((len(batch), TARGET_WIDTH), dtype=np.float32)
        for i, (_, _, values, _) in enumerate(batch):
            target[i, : len(values)] = values
        memory.merge(
            np.array([row for row, _, _, _ in batch]),
            target,
            t=t,
            rng=np.random.default_rng(t),
            weight=np.array([w for _, _, _, w in batch], dtype=np.float32),
        )
    return memory

def test_noise_is_one_walks_relative_error_at_the_largest_mean_column():
    a, b = rows_of_point(A)[:2], rows_of_point(B_)[:1]
    entries = [
        # line A: column 0 has mean 2 and E[r²] = (1 + 9 + 4 + 36) / 4 = 12.5; column 1 has mean 0.5
        (a[0], 1, (1.0, 0.5), 1.0),
        (a[1], 1, (3.0, 0.5), 1.0),
        (a[0], 2, (-2.0, 0.5), 1.0),
        (a[1], 2, (6.0, 0.5), 1.0),
        # line B: column 1 has the largest |mean|, −4; E[r²] there = (9 + 25) / 2 = 17
        (b[0], 1, (1.0, -3.0), 1.0),
        (b[0], 2, (1.0, -5.0), 1.0),
    ]
    memory = memory_of(entries)
    walks = 5  # K: walks per iteration; the memory's latest t is 2, so 10 walks fed it
    out = noise(memory, groups("line", 0), walks=walks, alpha=0.0, cap=1e9)
    # p = (n_g / size) · offered / (K · t_last): 4/6 · 6/10 = 0.4 for A, 0.2 for B;
    # noise = √(E[r²]/p − μ²)/|μ|, one walk's total read as one visit or none.
    assert out[A] == pytest.approx(np.sqrt(12.5 / 0.4 - 2.0**2) / 2.0, rel=1e-12)
    assert out[B_] == pytest.approx(np.sqrt(17.0 / 0.2 - 4.0**2) / 4.0, rel=1e-12)
    assert out.shape == (max(seat_points(0)) + 1,)  # one entry per id up to the seat's largest

def test_p_reads_the_memorys_share_once_the_memory_is_full():
    # Ten offers into four slots: the memory keeps 4 of 10, so n_g / size · offered
    # = 4/4 · 10 and p = 10 / (K · t_last) = 10 / 20 = 0.5. Every row is the same
    # sample, so σ = 0 and noise = √(μ²/p − μ²)/|μ| = √(1/p − 1) = 1. Reading
    # n_g alone as the visits would give p = 4/20 and noise 2.
    a = rows_of_point(A)[:1]
    memory = memory_of([(a[0], 1, (3.0,), 1.0)] * 10, capacity=4)
    assert (memory.size, memory.offered) == (4, 10)
    assert noise(memory, groups("line", 0), walks=20, alpha=0.0, cap=1e9)[A] == pytest.approx(1.0, rel=1e-12)

def test_a_line_every_walk_reaches_reads_as_the_scatter_of_one_visit():
    # p = 1: every walk banks one sample, so the per-walk relative error is σ/|μ|.
    a = rows_of_point(A)[:1]
    memory = memory_of([(a[0], 1, (1.0,), 1.0), (a[0], 1, (3.0,), 1.0)])  # μ 2, σ 1
    assert noise(memory, groups("line", 0), walks=2, alpha=0.0, cap=1e9)[A] == pytest.approx(0.5, rel=1e-12)

def test_only_the_walks_own_rows_count():
    a = rows_of_point(A)[:1]
    walk = [(a[0], 1, (1.0,), 1.0), (a[0], 1, (3.0,), 1.0), (a[0], 2, (-2.0,), 1.0), (a[0], 2, (6.0,), 1.0)]
    alone = noise(memory_of(walk), groups("line", 0), walks=5, alpha=0.5, cap=1e9)[A]
    # A walk row turned down to α is still a sample; an exact row (any other weight) is a mean, not a sample.
    turned_down = [(row, t, v, 0.5 if i == 3 else w) for i, (row, t, v, w) in enumerate(walk)]
    exact = [(a[0], 2, (100.0,), 0.37)]
    mixed = noise(memory_of(turned_down + exact), groups("line", 0), walks=5, alpha=0.5, cap=1e9)[A]
    # The exact row still takes a slot and an offer: p reads n_g / size · offered, 4/5 · 5/10 = 0.4 as before.
    assert mixed == pytest.approx(alone, rel=1e-12)

def test_too_few_rows_or_a_zero_mean_or_an_empty_memory_reads_as_the_cap():
    a, b, c = rows_of_point(A)[:1], rows_of_point(B_)[:1], rows_of_point(C)[:1]
    entries = [
        (a[0], 1, (5.0,), 1.0),  # one row: no spread to read
        (b[0], 1, (2.0, -2.0), 1.0),
        (b[0], 1, (-2.0, 2.0), 1.0),  # every column's mean is zero
        (c[0], 1, (1.0,), 1.0),
        (c[0], 1, (1.0,), 1.0),  # no scatter, frequent: well under the cap
    ]
    out = noise(memory_of(entries), groups("line", 0), walks=4, alpha=0.0, cap=20.0)
    assert out[A] == 20.0 and out[B_] == 20.0
    # p = 2/5 · 5/4 = 0.5; √(1/0.5 − 1)/1 = 1
    assert out[C] == pytest.approx(1.0, rel=1e-12)
    untouched = [p for p in seat_points(0) if p not in (A, B_, C)]
    assert (out[untouched] == 20.0).all()
    assert (noise(Reservoir(10), groups("line", 0), walks=1, alpha=0.0, cap=20.0) == 20.0).all()

def test_the_score_is_capped():
    a = rows_of_point(A)[:1]
    rare = [(a[0], 1, (1.0,), 1.0), (a[0], 1, (-0.9,), 1.0)]  # mean 0.05 against spread ~1: far over 20
    assert noise(memory_of(rare), groups("line", 0), walks=1_000, alpha=0.0, cap=20.0)[A] == 20.0

def test_a_memory_row_of_the_other_seat_is_refused():
    other = rows_of_point(seat_points(1)[0])[:1]
    with pytest.raises(ValueError, match="seat"):
        noise(memory_of([(other[0], 1, (1.0,), 1.0)]), groups("line", 0), walks=1, alpha=0.0, cap=20.0)

def test_noise_refuses_a_bad_walk_count():
    with pytest.raises(ValueError, match="walks"):
        noise(Reservoir(10), groups("line", 0), walks=0, alpha=0.0, cap=20.0)

# ---------------------------------------------------------------------------
# Scores and selection
# ---------------------------------------------------------------------------

def flat_noise(granularity: str, seat: int, value: float = 20.0) -> np.ndarray:
    return np.full(int(groups(granularity, seat).max()) + 1, value)

def everywhere(value: float) -> np.ndarray:
    return np.full(N_ROWS, value)

def test_infoset_takes_the_b_best_reached_rows_by_leak_times_noise():
    seat_rows = rows_of_point(A)
    leak, frequency = everywhere(0.0), everywhere(1.0)
    leak[seat_rows[[5, 1, 7, 3]]] = [4.0, 3.0, 3.0, 9.0]
    frequency[seat_rows[3]] = 0.0  # the leakiest row is never reached: not eligible
    spec = TargetSpec(granularity="infoset", alpha=0.0, budget=3)
    pool, chosen = choose(spec, seat=0, leak=leak, frequency=frequency, noise_of_group=flat_noise("infoset", 0))
    # 4.0 first, then the 3.0 tie goes to the lower row
    assert chosen.tolist() == [seat_rows[5], seat_rows[1], seat_rows[7]]
    assert pool.tolist() == sorted(chosen.tolist())

def test_noise_breaks_an_equal_leak():
    seat_rows = rows_of_point(A)
    leak = everywhere(0.0)
    leak[seat_rows[[0, 1]]] = 1.0
    noisy = flat_noise("infoset", 0, 2.0)
    noisy[seat_rows[1]] = 3.0
    spec = TargetSpec(granularity="infoset", alpha=0.0, budget=1)
    _, chosen = choose(spec, seat=0, leak=leak, frequency=everywhere(1.0), noise_of_group=noisy)
    assert chosen.tolist() == [seat_rows[1]]

def test_line_takes_groups_by_leak_per_reached_row_until_the_pool_is_full():
    a, b, c = rows_of_point(A), rows_of_point(B_), rows_of_point(C)
    leak, frequency = everywhere(0.0), everywhere(0.0)
    frequency[a[:13]] = 1.0  # A: 13 reached rows, leak 13 → 1 per reached row
    leak[a[:13]] = 1.0
    leak[a[20]] = 0.0  # an unreached row of A, not in the pool
    frequency[b[:500]] = 1.0  # B: 500 reached rows, leak 100 → 0.2 per row
    leak[b[:500]] = 0.2
    frequency[c[:5]] = 1.0  # C: 5 reached rows, leak 50 → 10 per row
    leak[c[:5]] = 10.0
    spec = TargetSpec(granularity="line", alpha=0.0, budget=1, pool=12)
    pool, chosen = choose(spec, seat=0, leak=leak, frequency=frequency, noise_of_group=flat_noise("line", 0))
    # C brings 5 reached rows, A 13 more: 18 ≥ 12, so A is the group that crosses and is in.
    assert chosen.tolist() == [C, A]
    assert pool.tolist() == sorted(c[:5].tolist() + a[:13].tolist())

def test_line_takes_every_eligible_group_when_the_seat_is_smaller_than_the_pool():
    a = rows_of_point(A)
    frequency = everywhere(0.0)
    frequency[a[:3]] = 1.0
    leak = everywhere(0.0)
    leak[a[0]] = 1.0
    spec = TargetSpec(granularity="line", alpha=0.0, budget=100)
    pool, chosen = choose(spec, seat=0, leak=leak, frequency=frequency, noise_of_group=flat_noise("line", 0))
    assert chosen.tolist() == [A] and pool.tolist() == a[:3].tolist()

def test_line_class_pools_the_reached_rows_of_its_classes():
    group = groups("line_class", 0)
    a = rows_of_point(A)
    target = group[a[0]]
    members = a[group[a] == target]
    leak, frequency = everywhere(0.0), everywhere(1.0)
    leak[members] = 1.0
    frequency[members[-1]] = 0.0
    spec = TargetSpec(granularity="line_class", alpha=0.0, budget=1, pool=1)
    pool, chosen = choose(spec, seat=0, leak=leak, frequency=frequency, noise_of_group=flat_noise("line_class", 0))
    assert chosen.tolist() == [target]
    assert pool.tolist() == members[:-1].tolist()

ROUND_TWO = np.zeros(N_ROWS, dtype=bool)
for _p, _point in enumerate(POINTS):
    if _point.board_cards == 2:
        ROUND_TWO[OFFSET[_p] : OFFSET[_p + 1]] = True

def test_round_two_only_leaves_the_one_card_rows_out_of_every_arm():
    # Round 1 and the draw are the leakiest lines and the dearest exact rows
    # (a round-1 row's exact target costs ~100× a round-2 row's), so the
    # opt-in keeps the budget where the sampling is sparsest and the rows cheap.
    one_card = next(p for p in seat_points(0) if POINTS[p].board_cards == 1)
    two_card = next(p for p in seat_points(0) if POINTS[p].board_cards == 2)
    leak, frequency = everywhere(0.0), everywhere(1.0)
    leak[rows_of_point(one_card)[:10]] = 100.0
    leak[rows_of_point(two_card)[:10]] = 1.0
    for granularity in ("infoset", "line", "line_class"):
        spec = TargetSpec(granularity=granularity, alpha=0.0, budget=5, pool=1, round_two_only=True)
        pool, _ = choose(
            spec, seat=0, leak=leak, frequency=frequency, noise_of_group=flat_noise(granularity, 0)
        )
        assert pool.size and ROUND_TWO[pool].all(), granularity
        assert set(pool.tolist()) & set(rows_of_point(two_card)[:10].tolist()), granularity
        plan = TargetSpec(granularity=granularity, alpha=0.0, budget=5, pool=1)
        pool, _ = choose(plan, seat=0, leak=leak, frequency=frequency, noise_of_group=flat_noise(granularity, 0))
        assert not ROUND_TWO[pool].all(), granularity  # off by default: the plan's rule picks round 1 first
    spec = TargetSpec(granularity="random", alpha=0.0, round_two_only=True)
    pool, _ = choose(spec, seat=1, leak=None, frequency=None, noise_of_group=None)
    assert np.array_equal(pool, np.flatnonzero((groups("random", 1) >= 0) & ROUND_TWO))

def test_random_pools_the_whole_seat_and_reads_no_leak():
    spec = TargetSpec(granularity="random", alpha=0.0)
    pool, chosen = choose(spec, seat=1, leak=None, frequency=None, noise_of_group=None)
    assert np.array_equal(pool, np.flatnonzero(groups("random", 1) >= 0)) and chosen.tolist() == [0]

@pytest.mark.parametrize(
    "leak, frequency, noise_value, match",
    [
        (None, everywhere(1.0), 20.0, "leak"),
        (everywhere(0.0), None, 20.0, "frequency"),
        (np.zeros(10), everywhere(1.0), 20.0, "leak"),
        (everywhere(-1.0), everywhere(1.0), 20.0, "leak"),
        (everywhere(np.nan), everywhere(1.0), 20.0, "leak"),
        (everywhere(0.0), everywhere(np.inf), 20.0, "frequency"),
        (everywhere(0.0), everywhere(1.0), np.nan, "noise"),
    ],
)
def test_choose_refuses_bad_inputs_by_name(leak, frequency, noise_value, match):
    spec = TargetSpec(granularity="line", alpha=0.0)
    with pytest.raises(ValueError, match=match):
        choose(spec, seat=0, leak=leak, frequency=frequency, noise_of_group=flat_noise("line", 0, noise_value))

def test_a_gain_an_ulp_under_zero_reads_as_zero_and_a_real_negative_is_refused():
    # The grader's best − Σ mix · value lands a few 1e-12 under zero where the mix plays the best action.
    seat_rows = rows_of_point(A)
    leak = everywhere(0.0)
    leak[seat_rows[0]] = 1.0
    leak[seat_rows[1]] = -6.8e-12
    spec = TargetSpec(granularity="infoset", alpha=0.0, budget=2)
    _, chosen = choose(spec, seat=0, leak=leak, frequency=everywhere(1.0), noise_of_group=flat_noise("infoset", 0))
    assert chosen[0] == seat_rows[0]
    leak[seat_rows[1]] = -1e-6
    with pytest.raises(ValueError, match="leak"):
        choose(spec, seat=0, leak=leak, frequency=everywhere(1.0), noise_of_group=flat_noise("infoset", 0))

def test_choose_refuses_a_noise_array_of_another_granularity():
    spec = TargetSpec(granularity="line", alpha=0.0)
    with pytest.raises(ValueError, match="noise"):
        choose(spec, seat=0, leak=everywhere(0.0), frequency=everywhere(1.0), noise_of_group=flat_noise("infoset", 0))

# ---------------------------------------------------------------------------
# The frozen set and its draws
# ---------------------------------------------------------------------------

S1 = seat_points(1)[0]  # one of seat 1's lines
WIDE = max(seat_points(0), key=lambda p: SIZES[p])  # seat 0's widest line, 100,400 rows

def a_set(
    weight0, *, budget, granularity="line", built_at=26, refresh=25, stream=0x7A52, weight1=(1.0, 2.0), alpha=0.0,
    **spec_fields,
) -> TargetSet:
    """A set whose seat-0 pool is the first rows of seat 0's widest line, with the given weights; seat 1 a two-row pool."""
    spec = TargetSpec(
        granularity=granularity, alpha=alpha, budget=budget, refresh=refresh, stream=stream, **spec_fields
    )
    rows0 = rows_of_point(WIDE)[: len(weight0)]
    rows1 = rows_of_point(S1)[: len(weight1)]
    return TargetSet(
        spec=spec,
        built_at=built_at,
        rows=(rows0, rows1),
        weight=(np.asarray(weight0, dtype=np.float64), np.asarray(weight1, dtype=np.float64)),
        groups=(np.array([WIDE]), np.array([S1])),
    )

def test_a_draw_is_reproducible_and_moves_with_every_word_of_its_stream():
    rng = np.random.default_rng(1)
    targets = a_set(rng.random(1_000), budget=10)
    first = targets.rows_for(30, 0, 7)
    assert np.array_equal(first, targets.rows_for(30, 0, 7))
    assert first.dtype == np.int64 and np.all(np.diff(first) > 0) and first.size == 10
    assert not np.array_equal(first, targets.rows_for(31, 0, 7))
    assert not np.array_equal(first, targets.rows_for(30, 0, 8))
    other = a_set(targets.weight[0], budget=10, stream=0x7A53)
    assert not np.array_equal(first, other.rows_for(30, 0, 7))
    assert set(targets.rows_for(30, 1, 7)) <= set(targets.rows[1].tolist())

def test_the_stream_is_its_own():
    from drawmaha_solver.deepcfr.reservoir import Memory, reservoir_stream
    from drawmaha_solver.minidrawmaha.solver import lockstep

    words = dict(seed=3, t=40, seat=1)
    ours = target_stream(**words, stream=0).random(4)
    assert not np.array_equal(ours, reservoir_stream(**words, memory=Memory.ADVANTAGE).random(4))
    assert not np.array_equal(ours, lockstep.stream(3, 40, 1, 0).random(4))
    assert targeting._STREAM_TAG == int.from_bytes(b"TARG", "big")

def test_a_draw_never_repeats_a_row():
    rng = np.random.default_rng(2)
    budget = 20
    targets = a_set(rng.random(budget + 3) ** 4, budget=budget)  # skewed weights, a pool barely over B
    for seed in range(200):
        drawn = targets.rows_for(27, 0, seed)
        assert drawn.size == budget == np.unique(drawn).size

def test_a_short_pool_gives_every_positive_row_and_never_a_weightless_one():
    targets = a_set([0.0, 3.0, 0.0, 1.0, 0.0], budget=4)
    pool = targets.rows[0]
    assert targets.rows_for(26, 0, 0).tolist() == [pool[1], pool[3]]
    big = a_set([0.0] * 50 + [1.0] * 50, budget=10)
    for seed in range(50):
        assert set(big.rows_for(26, 0, seed)) <= set(big.rows[0][50:].tolist())

def test_the_draw_is_successive_sampling_proportional_to_the_weight():
    weight = np.array([1.0, 2.0, 3.0, 6.0])
    total = weight.sum()
    # Exact inclusion probability of each row when B = 2 rows are drawn one at a
    # time ∝ weight, without replacement: every ordered pair, enumerated.
    exact = np.zeros(4)
    for first, second in permutations(range(4), 2):
        p = weight[first] / total * weight[second] / (total - weight[first])
        exact[[first, second]] += p
    assert exact.sum() == pytest.approx(2.0)
    targets = a_set(weight, budget=2)
    pool = targets.rows[0].tolist()
    trials = 20_000
    seen = np.zeros(4)
    for seed in range(trials):
        for row in targets.rows_for(26, 0, seed).tolist():
            seen[pool.index(row)] += 1
    se = np.sqrt(exact * (1 - exact) / trials)
    assert np.all(np.abs(seen / trials - exact) < 4 * se), (seen / trials, exact)

def test_an_infoset_set_serves_its_rows_at_every_iteration():
    targets = a_set([1.0, 1.0, 1.0], budget=3, granularity="infoset")
    for t in range(26, 51):
        assert targets.rows_for(t, 0, t).tolist() == targets.rows[0].tolist()

@pytest.mark.parametrize("t", [25, 51, 100])
def test_a_set_refuses_an_iteration_outside_its_window(t):
    with pytest.raises(ValueError, match="iteration"):
        a_set([1.0, 2.0], budget=1).rows_for(t, 0, 0)

def test_a_set_refuses_a_seat_by_name():
    with pytest.raises(ValueError, match="seat"):
        a_set([1.0, 2.0], budget=1).rows_for(26, 2, 0)

@pytest.mark.parametrize(
    "change, match",
    [
        (dict(built_at=0), "built_at"),
        (dict(rows=(rows_of_point(WIDE)[[3, 1]], rows_of_point(S1)[:2])), "sorted"),
        (dict(rows=(rows_of_point(WIDE)[[1, 1]], rows_of_point(S1)[:2])), "sorted"),
        (dict(rows=(rows_of_point(S1)[:2], rows_of_point(S1)[:2])), "seat"),
        (dict(rows=(np.array([-1, 0]), rows_of_point(S1)[:2])), "outside"),
        (dict(weight=(np.array([1.0]), np.array([1.0, 2.0]))), "weight"),
        (dict(weight=(np.array([1.0, -1.0]), np.array([1.0, 2.0]))), "weight"),
        (dict(weight=(np.array([1.0, np.nan]), np.array([1.0, 2.0]))), "weight"),
    ],
)
def test_a_set_refuses_a_malformed_field_by_name(change, match):
    good = a_set([1.0, 2.0], budget=1)
    fields = dict(spec=good.spec, built_at=good.built_at, rows=good.rows, weight=good.weight, groups=good.groups)
    with pytest.raises(ValueError, match=match):
        TargetSet(**(fields | change))

def test_an_infoset_set_holds_at_most_b_rows():
    with pytest.raises(ValueError, match="budget"):
        a_set([1.0, 1.0, 1.0], budget=2, granularity="infoset")

def test_a_set_is_read_only():
    targets = a_set([1.0, 2.0], budget=1)
    with pytest.raises(ValueError):
        targets.rows[0][0] = 5

def test_a_saved_set_loads_as_the_same_set(tmp_path):
    rng = np.random.default_rng(3)
    targets = a_set(rng.random(300), budget=7, built_at=51, stream=99, alpha=0.25, round_two_only=True, noise_cap=7.5)
    path = tmp_path / "t00051.npz"
    targets.save(path)
    loaded = TargetSet.load(path)
    assert loaded.spec == targets.spec and loaded.built_at == 51
    for seat in (0, 1):
        assert np.array_equal(loaded.rows[seat], targets.rows[seat])
        assert np.array_equal(loaded.weight[seat], targets.weight[seat])
        assert np.array_equal(loaded.groups[seat], targets.groups[seat])
        assert np.array_equal(loaded.rows_for(60, seat, 4), targets.rows_for(60, seat, 4))
    assert not (tmp_path / "t00051.npz.tmp").exists()

def test_load_refuses_a_file_that_is_not_a_target_set(tmp_path):
    path = tmp_path / "memory.npz"
    Reservoir(4).save(path)
    with pytest.raises(ValueError, match="memory.npz"):
        TargetSet.load(path)

# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------

class FakeArrival:
    """A stand-in for `BoostTargets.arrivals`: a weight per row from its number, and a log of what was asked."""

    def __init__(self) -> None:
        self.asked: list[tuple[int, np.ndarray]] = []

    def __call__(self, seat: int, rows: np.ndarray) -> np.ndarray:
        self.asked.append((seat, np.array(rows)))
        return (rows % 7) * 0.25

def leaky(points: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """Leak and frequency over the whole game: every row reached, leak on the named lines' first ten rows."""
    leak, frequency = everywhere(0.0), everywhere(1.0)
    for p, value in points.items():
        leak[rows_of_point(p)[:10]] = value
    return leak, frequency

def test_target_set_weighs_the_pool_by_arrival_and_asks_for_nothing_else():
    leak, frequency = leaky({A: 1.0, S1: 2.0})
    arrival = FakeArrival()
    spec = TargetSpec(granularity="line", alpha=0.0, budget=1, pool=1)
    targets = targeting.target_set(
        spec, t=26, leak=leak, frequency=frequency, memories=(Reservoir(4), Reservoir(4)), walks=10, arrival=arrival
    )
    assert targets.built_at == 26 and targets.spec == spec
    for seat, line in ((0, A), (1, S1)):
        assert targets.groups[seat].tolist() == [line]
        assert np.array_equal(targets.rows[seat], rows_of_point(line))
        assert np.array_equal(targets.weight[seat], (rows_of_point(line) % 7) * 0.25)
    assert [seat for seat, _ in arrival.asked] == [0, 1]
    assert all(np.array_equal(rows, targets.rows[seat]) for seat, rows in arrival.asked)

def test_target_set_reads_the_memory_for_noise():
    # Two lines with equal leak per row; the walk's rows make A quiet and B_ noisy. A has the lower id,
    # so it wins a tie: only the noise can put B_ first.
    leak, frequency = leaky({A: 1.0, B_: 1.0})
    for p in (A, B_):
        frequency[rows_of_point(p)[10:]] = 0.0
    a, b = rows_of_point(A)[0], rows_of_point(B_)[0]
    memory = memory_of([(a, 1, (1.0,), 1.0), (a, 1, (1.0,), 1.0), (b, 1, (1.0,), 1.0), (b, 1, (-0.8,), 1.0)])
    spec = TargetSpec(granularity="line", alpha=0.0, budget=1, pool=1)
    targets = targeting.target_set(
        spec, t=1, leak=leak, frequency=frequency, memories=(memory, Reservoir(4)), walks=4, arrival=FakeArrival()
    )
    assert targets.groups[0].tolist() == [B_]  # A: √(1/0.5 − 1) = 1; B_: √(0.82/0.5 − 0.01)/0.1 ≈ 12.8

def test_an_infoset_set_never_asks_for_arrival():
    leak, frequency = leaky({A: 1.0, S1: 1.0})

    def refuse(seat, rows):
        raise AssertionError("an infoset set draws nothing, so it needs no arrival")

    spec = TargetSpec(granularity="infoset", alpha=0.0, budget=3)
    targets = targeting.target_set(
        spec, t=1, leak=leak, frequency=frequency, memories=(Reservoir(4), Reservoir(4)), walks=10, arrival=refuse
    )
    assert targets.rows[0].tolist() == rows_of_point(A)[:3].tolist()
    assert targets.rows_for(1, 0, 0).tolist() == rows_of_point(A)[:3].tolist()

def test_a_random_set_needs_no_meter():
    arrival = FakeArrival()
    spec = TargetSpec(granularity="random", alpha=0.0)
    targets = targeting.target_set(
        spec, t=1, leak=None, frequency=None, memories=(Reservoir(4), Reservoir(4)), walks=10, arrival=arrival
    )
    for seat in (0, 1):
        assert np.array_equal(targets.rows[seat], np.flatnonzero(groups("random", seat) >= 0))
    assert targets.rows_for(1, 0, 0).size == 2_000

def test_target_set_refuses_an_arrival_of_the_wrong_shape_or_sign():
    leak, frequency = leaky({A: 1.0, S1: 1.0})
    spec = TargetSpec(granularity="line", alpha=0.0, budget=1, pool=1)
    for bad in (lambda seat, rows: np.ones(rows.size + 1), lambda seat, rows: -np.ones(rows.size)):
        with pytest.raises(ValueError, match="arrival"):
            targeting.target_set(
                spec, t=1, leak=leak, frequency=frequency, memories=(Reservoir(4), Reservoir(4)), walks=1, arrival=bad
            )

# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "drawmaha_solver" / "deepcfr"

def imported_names(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.add(module)
            names |= {f"{module}.{alias.name}" for alias in node.names}
    return names

def test_no_module_of_the_trainer_imports_the_grader():
    # The trainer feeds what the grader certifies; a bug they shared would
    # cancel and pass. Only scripts/deepcfr/targeting_refresh.py runs the
    # meter on the trainer's side. Every import statement in each file is
    # read, those inside functions included, and `from ... import exploitability` too.
    modules = sorted(PACKAGE.glob("*.py"))
    assert PACKAGE / "targeting.py" in modules and len(modules) >= 9
    for path in modules:
        offending = {name for name in imported_names(path) if "exploitability" in name}
        assert not offending, (path.name, offending)
