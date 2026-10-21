"""The exact average: own reach against a brute-force walk, the running sum, its file, its resume.

No torch here. The claims, in order: the predecessor table names, for every
public point, an earlier OWN decision point with one position and column per
key — three for a round-2 point after a throw; `own_reach` equals a brute
force that walks `MiniState` from a representative deal along the key's own
line, on a random non-uniform table, at keys of every predecessor case
(gate 4); `ExactAverage` adds `t · reach · σ` for one seat at a time and
refuses a t out of order; its freeze is the normalised sums; its file round
trips; the sums an oracle run accumulates equal a replay from the models it
kept; and a run stopped and restarted keeps the same sums bit for bit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.deepcfr.average import (
    ExactAverage,
    current_strategy,
    own_reach,
    predecessors,
)
from drawmaha_solver.deepcfr.collect import Collector
from drawmaha_solver.deepcfr.train import (
    MARKS,
    RESUME,
    STATE,
    DeepConfig,
    OracleLearner,
    load_run,
    mark_name,
    new_run,
    run_deep,
    step,
)
from drawmaha_solver.minidrawmaha.rules.cards import DECK, SUIT_RELABELLINGS, relabel
from drawmaha_solver.minidrawmaha.rules.enumeration import (
    PublicPoint,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.rules.features import layout
from drawmaha_solver.minidrawmaha.rules.game import (
    DRAW_ACTIONS,
    Action,
    InfoSet,
    MiniState,
    NodeKind,
    throw_count,
)
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.answer.strategy import StrategyInfo, uniform_strategy

# ---------------------------------------------------------------------------
# Helpers (restated, as the suite's rule is: test modules never import one another)
# ---------------------------------------------------------------------------

def key_on(point: PublicPoint, private) -> InfoSet:
    hole, thrown, board = private
    return InfoSet(
        player=point.player,
        hole=hole,
        discarded=thrown,
        board=board,
        draws=point.draws,
        betting=point.betting,
    )

def shape_of(point: PublicPoint):
    return private_keys(board_cards=point.board_cards, discards=point.discards)

def some_keys(seat: int | None = None) -> list[InfoSet]:
    """Two keys on every public point (of `seat`, if given): the first and the middle private key."""
    keys = []
    for point in public_decision_points():
        if seat is not None and point.player != seat:
            continue
        shape = shape_of(point)
        keys += [key_on(point, shape[0]), key_on(point, shape[len(shape) // 2])]
    return keys

def random_table(seed: int) -> PackedTable:
    """The whole game with N(0, 1) regrets: rows with no positive slot play uniform."""
    table = PackedTable.whole_game()
    table.cumulative_regret[:] = np.random.default_rng(seed).standard_normal(
        table.cumulative_regret.size
    )
    return table

def is_first_action(point: PublicPoint) -> bool:
    return point.board_cards == 1 and not point.is_draw_decision and len(point.betting[0]) == point.player

def is_throw_opening(point: PublicPoint) -> bool:
    """A round-2 point where the actor threw a card and has not yet acted in round 2."""
    return point.board_cards == 2 and point.discards == 1 and len(point.betting[1]) == point.player

def brute_force_own_reach(table: PackedTable, key: InfoSet) -> float:
    """Σ over the seat's physical histories in `key` of Π own σ, walked through `MiniState`, per suit frame.

    The key's picture is put in every one of the six physical suit frames,
    and each frame's histories are walked; the mean over frames is the sum
    over physical histories up to the per-row constant the average cancels.
    One frame is not enough: at a post-draw key whose pre-draw picture has a
    suit symmetry, the physical histories split between the thrown card's
    column and its twin's, and a single frame sees only one of them.

    Within a frame, the opponent's hole, throw and replacement are FIXED to
    one choice disjoint from every card the seat has seen, so the sum runs
    over the seat's own histories only. A key with a discard has three
    pre-draw holes, `hole − d + discarded` for each d in the hole, and the
    walk deals d back as the replacement. Every own action is taken through
    `MiniState.apply`, so no column convention is assumed; the throw is the
    action whose discard pile comes out as the frame's thrown card.
    """
    return sum(
        _own_reach_in_frame(
            table,
            key,
            hole=relabel(frame, key.hole),
            thrown=relabel(frame, key.discarded),
            board=relabel(frame, key.board),
        )
        for frame in SUIT_RELABELLINGS
    ) / len(SUIT_RELABELLINGS)

def _own_reach_in_frame(table: PackedTable, key: InfoSet, *, hole, thrown, board) -> float:
    seat = key.player
    seen = set(hole) | set(thrown) | set(board)
    spare = [card for card in DECK if card not in seen]
    opponent = tuple(sorted(spare[:3]))
    if thrown:
        starts = [(tuple(sorted((set(hole) - {d}) | set(thrown))), d) for d in hole]
    else:
        starts = [(hole, None)]

    def walk(state: MiniState, drawn, probability: float) -> float:
        if state.kind is NodeKind.TERMINAL:
            return 0.0
        if state.kind is NodeKind.CHANCE:
            short = [p for p, cards in enumerate(state.holes) if len(cards) < 3]
            if short:
                if short[0] == seat:
                    cards = (drawn,)
                else:
                    cards = next(
                        dealt
                        for dealt, _ in state.chance_outcomes()
                        if dealt[0] != drawn and dealt[0] not in seen
                    )
            else:
                cards = (board[len(state.board)],)
            return walk(state.apply_chance(cards), drawn, probability)
        if state.current_player == seat and state.infoset() == key:
            return probability
        player = state.current_player
        if state.is_draw_decision():
            count = key.draws[player].count
            if count == 0:
                action = Action.THROW_NONE
            elif player == seat:
                action = next(
                    a
                    for a in DRAW_ACTIONS
                    if throw_count(a) == 1 and state.apply(a).discards[seat] == thrown
                )
            else:
                action = DRAW_ACTIONS[1]
        else:
            line = state.betting[-1]
            action = key.betting[len(state.betting) - 1][len(line)]
        if player == seat:
            column = state.legal_actions().index(action)
            probability *= table[state.infoset()].strategy()[column]
        return walk(state.apply(action), drawn, probability)

    total = 0.0
    for pre, drawn in starts:
        holes = (pre, opponent) if seat == 0 else (opponent, pre)
        total += walk(MiniState(holes=holes), drawn, 1.0)
    return total

# ---------------------------------------------------------------------------
# The predecessor table
# ---------------------------------------------------------------------------

def test_predecessors_cover_the_six_cases():
    preds = predecessors()
    points = public_decision_points()
    assert len(preds) == len(points) == 141
    for index, (point, pred) in enumerate(zip(points, preds, strict=True)):
        if pred is None:
            assert is_first_action(point)
            continue
        assert pred.point < index  # parents before children, in walk order
        before = points[pred.point]
        assert before.player == point.player  # an OWN decision
        n = len(shape_of(point))
        assert pred.positions.shape == pred.columns.shape == pred.weights.shape
        assert pred.positions.shape[0] == n
        assert pred.columns.min() >= 0 and pred.columns.max() < before.width
        assert pred.positions.min() >= 0 and pred.positions.max() < len(shape_of(before))
        if is_throw_opening(point):
            # Three pre-draw holes, each sharing 1 over the thrown card's twins.
            assert pred.positions.shape[1] >= 3
            np.testing.assert_allclose(pred.weights.sum(axis=1), 3.0)
            assert set(np.unique(pred.weights).tolist()) <= {0.0, 1 / 3, 0.5, 1.0}
        else:
            assert pred.positions.shape[1] == 1 and (pred.weights == 1.0).all()
        if point.board_cards == 2 and len(point.betting[1]) == point.player:
            assert before.is_draw_decision and before.draws == point.draws[: point.player]
        elif point.is_draw_decision:
            assert not before.is_draw_decision and before.board_cards == 1
        else:
            assert before.betting[-1] == point.betting[-1][:-2]
            assert (pred.columns == before.actions.index(point.betting[-1][-2])).all()
    # P0 opens once; P1's first action faces a check or a bet, two points.
    assert sum(pred is None for pred in preds) == 3

def test_a_throw_keys_three_predecessors_are_its_three_pre_draw_holes():
    preds = predecessors()
    points = public_decision_points()
    index = next(i for i, point in enumerate(points) if is_throw_opening(point))
    pred = preds[index]
    distinct = np.array(
        [len({at for at, w in zip(row, weights) if w > 0}) for row, weights in zip(pred.positions.tolist(), pred.weights.tolist())]
    )
    assert set(distinct.tolist()) == {2, 3}
    assert (distinct == 3).mean() > 0.9  # the spec's sample: 96 of 101 have three distinct
    # A tied thrown card is charged to two columns at a half each.
    halves = (pred.weights == 0.5).sum(axis=1)
    assert set(halves.tolist()) >= {0, 2} and 0 < (halves > 0).mean() < 0.5
    # Stand-pat openings have one predecessor at the stand-pat column.
    pat = next(
        preds[i]
        for i, point in enumerate(points)
        if point.board_cards == 2 and point.discards == 0 and len(point.betting[1]) == point.player
    )
    assert (pat.columns == DRAW_ACTIONS.index(Action.THROW_NONE)).all()
    # A throw column is never the stand-pat column.
    assert (pred.columns[pred.weights > 0] >= 1).all()

# ---------------------------------------------------------------------------
# Own reach (gate 4)
# ---------------------------------------------------------------------------

def test_current_strategy_is_regret_matching_row_by_row():
    table = random_table(1)
    sigma = current_strategy(table)
    assert sigma.shape == table.cumulative_regret.shape and sigma.dtype == np.float64
    for key in some_keys():
        _, start, width = table.slot_of(key)
        # `reduceat` adds a row left to right; `ndarray.sum` may pair, one ulp apart.
        np.testing.assert_allclose(sigma[start : start + width], table[key].strategy(), rtol=1e-15, atol=0)

def throw_keys_with(distinct: int, seat: int) -> list[InfoSet]:
    """One round-2 throw-opening key of `seat` whose three predecessors are `distinct` draw keys."""
    preds = predecessors()
    points = public_decision_points()
    for index, point in enumerate(points):
        if point.player != seat or not is_throw_opening(point):
            continue
        pred = preds[index]
        rows = [
            len({at for at, w in zip(row, weights) if w > 0})
            for row, weights in zip(pred.positions.tolist(), pred.weights.tolist())
        ]
        position = rows.index(distinct)
        return [key_on(point, shape_of(point)[position])]
    raise AssertionError("no throw-opening point")

def tied_throw_keys(seat: int, count: int) -> list[InfoSet]:
    """Round-2 throw-opening keys of `seat` where a pre-draw picture's suit symmetry moves the thrown card.

    There the physical histories split between the thrown card's column and
    its twin's; one frame sees only one of them, and a map that reads a
    single column is wrong by the other's share.
    """
    point = next(p for p in public_decision_points() if p.player == seat and is_throw_opening(p))
    keys = []
    for hole, thrown, board in shape_of(point):
        for drawn in hole:
            pre = tuple(sorted((set(hole) - {drawn}) | set(thrown)))
            stabiliser = [
                frame
                for frame in SUIT_RELABELLINGS
                if set(relabel(frame, pre)) == set(pre) and relabel(frame, board[:1]) == board[:1]
            ]
            if len({relabel(frame, thrown) for frame in stabiliser}) > 1:
                keys.append(key_on(point, (hole, thrown, board)))
                break
        if len(keys) == count:
            return keys
    raise AssertionError("no tied throw key")

@pytest.mark.parametrize("seat", [0, 1])
def test_own_reach_equals_the_brute_force_walk(seat):
    table = random_table(2)
    reach = own_reach(table, seat=seat)
    assert reach.shape == (len(table),) and reach.dtype == np.float64
    keys = (
        some_keys(seat)
        + throw_keys_with(3, seat)
        + throw_keys_with(2, seat)
        + tied_throw_keys(seat, 12)
    )
    assert len(keys) > 100
    for key in keys:
        expected = brute_force_own_reach(table, key)
        assert reach[table.row_of(key)] == pytest.approx(expected, rel=1e-12, abs=0.0), str(key)
    built = layout()
    for p, point in enumerate(built.points):
        rows = slice(built.row_offset[p], built.row_offset[p + 1])
        if point.player != seat:
            assert not reach[rows].any()
        elif is_first_action(point):
            assert (reach[rows] == 1.0).all()
        else:
            # A pure predecessor gives 1.0; three predecessors can sum past it,
            # and every later row behind a throw inherits that.
            assert reach[rows].min() >= 0.0 and np.isfinite(reach[rows]).all()
            assert reach[rows].max() <= (3.0 if point.discards == 1 else 1.0)

def test_own_reach_refuses_a_listed_table_and_a_bad_seat():
    listed = PackedTable.listed(some_keys()[:5])
    with pytest.raises(ValueError, match="whole game"):
        own_reach(listed, seat=0)
    with pytest.raises(ValueError, match="seat"):
        own_reach(random_table(0), seat=2)

# ---------------------------------------------------------------------------
# The running sum
# ---------------------------------------------------------------------------

def columns_of(seat: int) -> np.ndarray:
    """Every column of `seat`'s rows, as a boolean mask over the flat arrays."""
    built = layout()
    mask = np.zeros(built.column_offset[-1], dtype=bool)
    for p, point in enumerate(built.points):
        if point.player == seat:
            mask[built.column_offset[p] : built.column_offset[p + 1]] = True
    return mask

def test_a_new_average_is_empty_and_plays_uniform():
    acc = ExactAverage()
    assert acc.name == "exact" and acc.last == [0, 0]
    assert acc.sums.dtype == np.float64 and acc.sums.shape == (layout().column_offset[-1],)
    assert not acc.sums.any()
    frozen = acc.frozen(iteration=0, seed=0)
    assert frozen.info == StrategyInfo(rule="deep", column="exact", iteration=0, seed=0, workers=0)
    np.testing.assert_array_equal(frozen.probabilities, uniform_strategy().probabilities)

def test_accumulate_adds_t_times_reach_times_sigma_for_one_seat_only():
    widths = PackedTable.whole_game().widths()
    first, second = random_table(3), random_table(4)
    acc = ExactAverage()
    acc(first, 1, 1)
    expected = 1 * np.repeat(own_reach(first, seat=1), widths) * current_strategy(first)
    np.testing.assert_array_equal(acc.sums, expected)
    assert not acc.sums[columns_of(0)].any() and acc.sums[columns_of(1)].any()
    assert acc.last == [0, 1]
    acc(first, 1, 0)
    assert acc.last == [1, 1] and acc.sums[columns_of(0)].any()
    acc.accumulate(second, 2, seat=1)
    expected += 1 * np.repeat(own_reach(first, seat=0), widths) * current_strategy(first)
    expected += 2 * np.repeat(own_reach(second, seat=1), widths) * current_strategy(second)
    np.testing.assert_array_equal(acc.sums, expected)
    assert acc.last == [1, 2]

def test_the_frozen_average_is_the_normalised_sums_uniform_where_unreached():
    table = random_table(5)
    acc = ExactAverage()
    acc(table, 1, 1)
    frozen = acc.frozen(iteration=1, seed=5)
    checked = 0
    for key in some_keys(1):
        _, start, width = table.slot_of(key)
        row = acc.sums[start : start + width]
        if row.sum() > 0:
            np.testing.assert_allclose(frozen[key], (row / row.sum()).astype(np.float32), rtol=1e-6)
            checked += 1
    assert checked > 0  # the guarded loop above must not pass vacuously
    for key in some_keys(0):  # seat 0 was never accumulated: uniform everywhere
        np.testing.assert_array_equal(frozen[key], np.full(len(key.legal_actions()), 1 / len(key.legal_actions()), np.float32))

def test_accumulate_refuses_a_t_out_of_order_and_a_listed_table():
    acc = ExactAverage()
    table = random_table(6)
    acc(table, 1, 0)
    for bad in (1, 3, 0):
        with pytest.raises(ValueError, match=r"seat 0.*expected t = 2.*got"):
            acc(table, bad, 0)
    acc(table, 1, 1)  # the seats are independent
    with pytest.raises(ValueError, match="whole game"):
        acc(PackedTable.listed(some_keys()[:5]), 2, 0)
    with pytest.raises(ValueError, match="seat"):
        acc(table, 2, 2)
    assert acc.last == [1, 1]

def test_the_file_round_trips_atomically(tmp_path):
    acc = ExactAverage()
    acc(random_table(7), 1, 1)
    path = tmp_path / "exact-t00001.npz"
    acc.save(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["exact-t00001.npz"]
    back = ExactAverage()
    back.load(path)
    np.testing.assert_array_equal(back.sums, acc.sums)
    assert back.last == acc.last == [0, 1]
    with np.load(path) as saved:
        keys = dict(saved)
    for field, broken in (
        ("last", None),
        ("sums", keys["sums"][:-1]),
        ("sums", keys["sums"].astype(np.float32)),
        ("last", np.array([-1, 0])),
        ("last", np.array([1, 2, 3])),
    ):
        bad = dict(keys)
        if broken is None:
            del bad[field]
        else:
            bad[field] = broken
        with (tmp_path / "bad.npz").open("wb") as file:
            np.savez(file, **bad)
        with pytest.raises(ValueError, match=f"bad.npz.*{field}"):
            ExactAverage().load(tmp_path / "bad.npz")

def test_the_sums_accumulated_in_an_oracle_run_equal_a_replay_from_its_models():
    config = DeepConfig(seed=5, walks=8, workers=0, capacity=100_000)
    learner = OracleLearner()
    acc = ExactAverage()
    run = new_run(config)
    kept = {0: (None, None)}
    with Collector(seed=5, workers=0) as collector:
        for _ in range(3):
            step(run, collector, learner, observers=(acc,))
            kept[run.t] = tuple(run.nets)
    assert acc.last == [3, 3] and acc.sums.any()
    replay = ExactAverage()
    table = PackedTable.whole_game()
    for t in (1, 2, 3):
        learner.write(table, kept[t - 1])
        replay(table, t, 1)
        learner.write(table, (kept[t][0], kept[t - 1][1]))
        replay(table, t, 0)
    np.testing.assert_array_equal(replay.sums, acc.sums)
    # The freeze is a distribution on every key (the constructor checks each
    # row); the whole-game grade of it is gate 5, behind the slow flag.
    assert len(acc.frozen(iteration=3, seed=5)) == len(table)

# ---------------------------------------------------------------------------
# The resume: the sums join the resume set and every mark
# ---------------------------------------------------------------------------

DRIVEN = DeepConfig(seed=5, walks=5, workers=0, capacity=100_000, grid=(4, 8))

def drive(out: Path, target: int, observers=()) -> None:
    run_deep(out, DRIVEN, target=target, learner=OracleLearner(), observers=observers, log=lambda _: None)

def test_a_run_stopped_and_restarted_keeps_the_same_sums_bitwise(tmp_path):
    first = ExactAverage()
    drive(tmp_path / "run", 6, observers=(first,))
    assert first.last == [6, 6]
    second = ExactAverage()
    drive(tmp_path / "run", 10, observers=(second,))
    straight = ExactAverage()
    drive(tmp_path / "straight", 10, observers=(straight,))
    assert second.last == straight.last == [10, 10] and straight.sums.any()
    np.testing.assert_array_equal(second.sums, straight.sums)
    resume = tmp_path / "run" / RESUME
    assert (resume / "exact-t00010.npz").exists() and not (resume / "exact-t00006.npz").exists()
    assert json.loads((resume / STATE).read_text())["files"]["observers"] == {"exact": "exact-t00010.npz"}
    assert not list((tmp_path / "run").rglob("*.tmp"))
    # A mark carries the sums at its own t, equal to a straight run stopped there.
    at_mark = ExactAverage()
    at_mark.load(tmp_path / "run" / MARKS / mark_name(8) / "exact-t00008.npz")
    to_eight = ExactAverage()
    drive(tmp_path / "eight", 8, observers=(to_eight,))
    assert at_mark.last == [8, 8]
    np.testing.assert_array_equal(at_mark.sums, to_eight.sums)

def test_a_resume_refuses_a_state_without_the_observers_file(tmp_path):
    drive(tmp_path, 2)  # saved with no observer
    with pytest.raises(ValueError, match="exact"):
        drive(tmp_path, 3, observers=(ExactAverage(),))
    assert load_run(tmp_path / RESUME, learner=OracleLearner()).t == 2  # nothing was overwritten

def test_a_resume_refuses_to_drop_a_saved_observer(tmp_path):
    drive(tmp_path, 2, observers=(ExactAverage(),))
    with pytest.raises(ValueError, match="exact"):
        drive(tmp_path, 3)
    # A plain load — what PR 6's grader does with a mark — still works without it.
    assert load_run(tmp_path / RESUME, learner=OracleLearner()).t == 2
    assert (tmp_path / RESUME / "exact-t00002.npz").exists()

def test_two_stateful_observers_with_one_name_are_refused(tmp_path):
    with pytest.raises(ValueError, match="exact"):
        drive(tmp_path, 1, observers=(ExactAverage(), ExactAverage()))

def test_the_cli_passes_the_exact_average(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "run_deep_cfr", Path(__file__).parents[2] / "scripts" / "deepcfr" / "run_deep_cfr.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    monkeypatch.setattr(cli, "run_deep", lambda *args, **kw: calls.append((args, kw)))
    monkeypatch.setattr("sys.argv", ["run_deep_cfr.py", "--out", "x"])
    cli.main()
    (_, kw), = calls
    assert [type(o) for o in kw["observers"]] == [ExactAverage]
