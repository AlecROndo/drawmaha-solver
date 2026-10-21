"""The reservoir: Algorithm R over (row, t, target) rows, pinned on what the loop relies on.

Four claims. Below capacity every offer is kept, in order. Above it the memory
never grows, counts every offer, and every offer ever made is resident with
probability C/n (a chi-square over thousands of trials, correctness gate 2).
A memory saved and loaded merges the next phase exactly as the uninterrupted
memory does, because the stream is a function of (seed, t, seat, memory) and
the count of offers travels with the checkpoint. And the file format refuses
anything that is not a reservoir.
"""

from __future__ import annotations

import numpy as np
import pytest

from drawmaha_solver.deepcfr.reservoir import (
    TARGET_WIDTH,
    Memory,
    Reservoir,
    reservoir_stream,
)
from drawmaha_solver.minidrawmaha.solver import lockstep

def targets(n: int) -> np.ndarray:
    """n target rows, column 0 numbering them so a kept row can be told from another."""
    out = np.zeros((n, TARGET_WIDTH), dtype=np.float32)
    out[:, 0] = np.arange(n)
    return out

def offers(t: int) -> np.ndarray:
    """Forty distinct row numbers that only phase t offers."""
    return np.arange(40) + 1_000 * t

class FixedDraws:
    """A stand-in for a Generator whose `integers` hands back the given values."""

    def __init__(self, values) -> None:
        self._values = np.asarray(values, dtype=np.int64)

    def integers(self, low, high, size=None):
        return self._values

def filled_memory() -> Reservoir:
    memory = Reservoir(capacity=8)
    memory.merge(np.arange(5), targets(5), t=1, rng=np.random.default_rng(0))
    return memory

# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def test_below_capacity_everything_is_kept_in_order():
    memory = Reservoir(capacity=10)
    memory.merge(np.arange(6), targets(6), t=1, rng=np.random.default_rng(0))
    rows, t, kept = memory.contents()
    assert memory.offered == 6 and memory.size == 6
    np.testing.assert_array_equal(rows, np.arange(6))
    assert (t == 1).all()
    np.testing.assert_array_equal(kept, targets(6))

def test_a_second_merge_below_capacity_appends_after_the_first():
    memory = Reservoir(capacity=10)
    memory.merge(np.arange(4), targets(4), t=1, rng=np.random.default_rng(0))
    memory.merge(np.arange(3) + 100, targets(3), t=2, rng=np.random.default_rng(0))
    rows, t, _ = memory.contents()
    assert rows.tolist() == [0, 1, 2, 3, 100, 101, 102]
    assert t.tolist() == [1, 1, 1, 1, 2, 2, 2]

def test_never_exceeds_capacity_and_counts_every_offer():
    memory = Reservoir(capacity=10)
    for t in range(1, 6):
        memory.merge(np.arange(7) + 100 * t, targets(7), t=t, rng=np.random.default_rng(t))
    assert memory.offered == 35 and memory.size == 10
    rows, t_col, kept = memory.contents()
    assert set(rows.tolist()) <= {r + 100 * t for t in range(1, 6) for r in range(7)}
    # Each row kept its own t and its own target: the three columns move together.
    np.testing.assert_array_equal(t_col, rows // 100)
    np.testing.assert_array_equal(kept[:, 0], rows % 100)

def test_a_merge_that_straddles_capacity_fills_then_samples():
    memory = Reservoir(capacity=5)
    memory.merge(np.arange(3), targets(3), t=1, rng=np.random.default_rng(0))
    memory.merge(np.arange(4) + 10, targets(4), t=2, rng=FixedDraws([0, 7]))
    # Offers 10 and 11 fill slots 3 and 4; offer 12 (global index 5) draws 0 and
    # takes slot 0; offer 13 (global index 6) draws 7 >= C and is dropped.
    assert memory.contents()[0].tolist() == [12, 1, 2, 10, 11]
    assert memory.offered == 7 and memory.size == 5

def test_every_offer_is_resident_with_probability_c_over_n():
    # 16,000 trials, C = 10, 100 offers in four merges of 25: each offer is
    # resident with probability p = 10/100, so 1,600 times in expectation with
    # variance 1,600 · (1 − p). Over the 100 counts, Σ (count − E)² / (E (1 − p))
    # is asymptotically (N/(N−1)) · chi-square with N − 1 = 99 degrees of freedom
    # (the counts sum to a constant); 148.23 is the 0.999 point of chi-square(99).
    # Measured: the correct bound scores about 105 here; an exclusive draw bound
    # (the off-by-one this gate exists to catch) scores about 287.
    trials, capacity, per_merge, merges = 16_000, 10, 25, 4
    counts = np.zeros(per_merge * merges)
    rng = np.random.default_rng(2026)
    for _ in range(trials):
        memory = Reservoir(capacity=capacity)
        for phase in range(merges):
            rows = np.arange(per_merge) + per_merge * phase
            memory.merge(rows, targets(per_merge), t=phase + 1, rng=rng)
        resident = memory.contents()[0]
        assert resident.size == capacity and np.unique(resident).size == capacity
        counts[resident] += 1
    p = capacity / counts.size
    expected = trials * p
    chi_square = float(((counts - expected) ** 2 / (expected * (1 - p))).sum())
    assert chi_square < 148.23, chi_square

def test_the_later_offer_wins_a_shared_slot():
    memory = Reservoir(capacity=2)
    memory.merge(np.array([0, 1]), targets(2), t=1, rng=np.random.default_rng(0))
    memory.merge(np.array([7, 8]), targets(2), t=2, rng=FixedDraws([1, 1]))
    rows, t, kept = memory.contents()
    assert rows.tolist() == [0, 8]
    assert t.tolist() == [1, 2]
    assert kept[1, 0] == 1.0  # offer 8 was the second target of its merge

def test_an_empty_merge_changes_nothing():
    memory = filled_memory()
    before = [a.copy() for a in memory.contents()]
    memory.merge(np.zeros(0, dtype=np.int64), targets(0), t=2, rng=np.random.default_rng(0))
    assert memory.offered == 5
    for a, b in zip(before, memory.contents(), strict=True):
        np.testing.assert_array_equal(a, b)

def test_merge_draws_once_per_late_offer_and_nothing_below_capacity():
    class Counting:
        asked = []

        def integers(self, low, high, size=None):
            Counting.asked.append(np.asarray(high).copy())
            return np.zeros(np.asarray(high).shape, dtype=np.int64)

    memory = Reservoir(capacity=3)
    memory.merge(np.arange(2), targets(2), t=1, rng=Counting())
    memory.merge(np.arange(3) + 10, targets(3), t=2, rng=Counting())
    # Nothing was late in the first merge, so the stream was not touched; the
    # second had global indexes 2, 3, 4 of which 3 and 4 are late, each drawing
    # from [0, g] inclusive.
    assert [a.tolist() for a in Counting.asked] == [[4, 5]]

# ---------------------------------------------------------------------------
# The stream and the resume
# ---------------------------------------------------------------------------

def test_a_resumed_reservoir_merges_identically(tmp_path):
    def stream(t):
        return reservoir_stream(seed=3, t=t, seat=0, memory=Memory.ADVANTAGE)

    straight = Reservoir(capacity=50)
    for t in (1, 2):
        straight.merge(offers(t), targets(40), t=t, rng=stream(t))
    straight.save(tmp_path / "m.npz")
    resumed = Reservoir.load(tmp_path / "m.npz")
    assert resumed.capacity == 50 and resumed.offered == 80 and resumed.size == 50
    for memory in (straight, resumed):
        memory.merge(offers(3), targets(40), t=3, rng=stream(3))
    for a, b in zip(straight.contents(), resumed.contents(), strict=True):
        np.testing.assert_array_equal(a, b)
    assert straight.offered == resumed.offered

def test_streams_differ_by_memory_and_never_equal_a_walk_stream():
    a = reservoir_stream(seed=1, t=2, seat=0, memory=Memory.ADVANTAGE).integers(0, 2**62, 4)
    b = reservoir_stream(seed=1, t=2, seat=0, memory=Memory.POLICY).integers(0, 2**62, 4)
    c = reservoir_stream(seed=1, t=3, seat=0, memory=Memory.ADVANTAGE).integers(0, 2**62, 4)
    w = lockstep.stream(1, 2, 0, 0).integers(0, 2**62, 4)
    assert not np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, w)

def test_the_stream_is_a_function_of_its_four_words():
    first = reservoir_stream(seed=1, t=2, seat=1, memory=Memory.POLICY).integers(0, 2**62, 4)
    again = reservoir_stream(seed=1, t=2, seat=1, memory=Memory.POLICY).integers(0, 2**62, 4)
    np.testing.assert_array_equal(first, again)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_merge_refuses_bad_rows():
    rng0 = np.random.default_rng(0)
    memory = Reservoir(capacity=4)
    with pytest.raises(ValueError, match="same length"):
        memory.merge(np.arange(3), targets(2), t=1, rng=rng0)
    with pytest.raises(ValueError, match="4"):
        memory.merge(np.arange(2), np.zeros((2, 3), dtype=np.float32), t=1, rng=rng0)
    with pytest.raises(ValueError, match="integer"):
        memory.merge(np.arange(2, dtype=np.float64), targets(2), t=1, rng=rng0)
    with pytest.raises(ValueError, match="negative"):
        memory.merge(np.array([0, -1]), targets(2), t=1, rng=rng0)
    with pytest.raises(ValueError, match="int32"):
        memory.merge(np.array([0, 2**31]), targets(2), t=1, rng=rng0)
    with pytest.raises(ValueError, match="t"):
        memory.merge(np.arange(2), targets(2), t=0, rng=rng0)
    # t is stamped into an int32 array beside the rows.
    with pytest.raises(ValueError, match="t must fit int32"):
        memory.merge(np.arange(2), targets(2), t=2**31, rng=rng0)
    with pytest.raises(ValueError, match="capacity"):
        Reservoir(capacity=0)
    with pytest.raises(ValueError, match="capacity"):
        Reservoir(capacity=True)

def test_contents_are_read_only_and_feed_the_fit():
    rows, t, tg = filled_memory().contents()
    assert rows.dtype == np.int32 and t.dtype == np.int32 and tg.dtype == np.float32
    assert rows.shape == (5,) and t.shape == (5,) and tg.shape == (5, TARGET_WIDTH)
    with pytest.raises(ValueError):
        rows[0] = 1
    with pytest.raises(ValueError):
        tg[0, 0] = 1.0

# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

def test_save_is_atomic_and_load_refuses_a_foreign_file(tmp_path):
    memory = filled_memory()
    memory.save(tmp_path / "m.npz")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.npz"]
    loaded = Reservoir.load(tmp_path / "m.npz")
    assert (loaded.capacity, loaded.offered, loaded.size) == (8, 5, 5)
    for a, b in zip(memory.contents(), loaded.contents(), strict=True):
        np.testing.assert_array_equal(a, b)
    foreign = tmp_path / "lockstep.npz"
    np.savez(foreign, regret=np.zeros(3), workers=np.int64(4))
    with pytest.raises(ValueError, match="lockstep.npz"):
        Reservoir.load(foreign)

def test_load_refuses_a_file_whose_counts_disagree(tmp_path):
    # offered = 5 and capacity = 8 mean five resident rows, not three.
    broken = tmp_path / "broken.npz"
    np.savez(
        broken,
        rows=np.zeros(3, np.int32),
        t=np.ones(3, np.int32),
        targets=np.zeros((3, TARGET_WIDTH), np.float32),
        offered=np.int64(5),
        capacity=np.int64(8),
    )
    with pytest.raises(ValueError, match="broken.npz"):
        Reservoir.load(broken)

def test_load_refuses_a_capacity_that_holds_nothing_naming_the_file(tmp_path):
    broken = tmp_path / "zero.npz"
    np.savez(
        broken,
        rows=np.zeros(0, np.int32),
        t=np.zeros(0, np.int32),
        targets=np.zeros((0, TARGET_WIDTH), np.float32),
        offered=np.int64(0),
        capacity=np.int64(0),
    )
    with pytest.raises(ValueError, match="zero.npz.*capacity"):
        Reservoir.load(broken)

def test_load_refuses_arrays_of_the_wrong_type(tmp_path):
    # Float rows would be truncated into the int32 slots without a word.
    broken = tmp_path / "floats.npz"
    np.savez(
        broken,
        rows=np.array([1.7, 2.2, 3.9]),
        t=np.ones(3, np.int32),
        targets=np.zeros((3, TARGET_WIDTH), np.float32),
        offered=np.int64(3),
        capacity=np.int64(8),
    )
    with pytest.raises(ValueError, match="rows must be int32"):
        Reservoir.load(broken)

def test_a_loaded_memory_can_keep_filling_to_its_capacity(tmp_path):
    memory = filled_memory()
    memory.save(tmp_path / "m.npz")
    loaded = Reservoir.load(tmp_path / "m.npz")
    loaded.merge(np.arange(3) + 50, targets(3), t=2, rng=np.random.default_rng(0))
    assert loaded.contents()[0].tolist() == [0, 1, 2, 3, 4, 50, 51, 52]

def test_target_width_matches_the_encoder():
    from drawmaha_solver.minidrawmaha.rules.features import MAX_WIDTH

    assert TARGET_WIDTH == MAX_WIDTH

def test_memory_ids_are_the_two_memories():
    assert int(Memory.ADVANTAGE) == 0 and int(Memory.POLICY) == 1
