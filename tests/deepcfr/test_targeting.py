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

import numpy as np
import pytest

from drawmaha_solver.deepcfr import targeting
from drawmaha_solver.deepcfr.reservoir import TARGET_WIDTH, Reservoir
from drawmaha_solver.deepcfr.targeting import TargetSpec, choose, groups, noise
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
        ({"noise_cap": 0.0}, "noise_cap"),
        ({"noise_cap": float("inf")}, "noise_cap"),
        ({"pool": 0}, "pool"),
        ({"stream": -1}, "stream"),
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
    # p = (n_g / size) · offered / (K · t_last): 4/6 · 6/10 = 0.4 for A, 0.2 for B
    assert out[A] == pytest.approx(np.sqrt(12.5 / 0.4) / 2.0, rel=1e-12)
    assert out[B_] == pytest.approx(np.sqrt(17.0 / 0.2) / 4.0, rel=1e-12)
    assert out.shape == (max(seat_points(0)) + 1,)  # one entry per id up to the seat's largest

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
    out = noise(memory_of(entries), groups("line", 0), walks=1, alpha=0.0, cap=20.0)
    assert out[A] == 20.0 and out[B_] == 20.0
    assert out[C] == pytest.approx(np.sqrt(1.0 / (2 / 5 * 5 / 1)), rel=1e-12)
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

def test_choose_refuses_a_noise_array_of_another_granularity():
    spec = TargetSpec(granularity="line", alpha=0.0)
    with pytest.raises(ValueError, match="noise"):
        choose(spec, seat=0, leak=everywhere(0.0), frequency=everywhere(1.0), noise_of_group=flat_noise("infoset", 0))
