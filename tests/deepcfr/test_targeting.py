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
from drawmaha_solver.deepcfr.targeting import TargetSpec, groups
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
