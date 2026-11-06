"""The boost target: extra samples of a row's regret, each unbiased for the exact sum the tests first pin down.

The module under test is rung 4's boost target. Exactly (`EXACT`), for a
table row it sums, over every physical entry of the key and every opponent
hand, the regret the walk would bank there, weighted by how often the walk
arrives; sampled, it estimates that sum from play-outs that keep k of n of
each source below the row. The claims, in
order: it imports the rules and never the grader; `sigma` is the ledger's
own σ row by row; the back-solved opponent reach equals the grader's carried
vector on every board of every public point, in the full game and in a
restricted-deal one (test a); at a seat's first decision the arrival IS the
grader's frequency, and a seat's first decisions sum to one; on a small
restricted game the regret and the arrival equal a world-by-world
expectation walked through `MiniState` (test b, exact form); on a wider
restricted game they match the mean and the visit count of thousands of
`Collector` walks, row by row within four standard errors and pooled (tests
b and c); rows of the wrong seat are refused and an unreachable row is zero;
and the cost per round-2 row is recorded (test d, a benchmark). Then the
sampled forms (e): counts that are not positive ints are refused by name; a
count at or above its fan is enumeration, bit for bit; each chance source
kept one branch at a time and averaged over its branches is the exact sum
to 1e-10, in the full game and a restricted one, which pins every n/k scale
with no noise; each source alone, and all of them together, is unbiased
against the exact sum at a row where it is live, and actually moves the
target there; the range draw is unbiased where a key's two physical entries
face wide and different ranges, before and after a throw; a row's sample depends on
(seed, t, seat, row) only; play-outs shrink the spread as 1/M; an
unreachable row is zero under any sampling; and sampling a draw or round-1
row costs a fraction of its exact sum.

Helpers are restated rather than imported from other test modules, as the
suite's rule has it.
"""

from __future__ import annotations

import ast
import os
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.deepcfr.collect import Collector
from drawmaha_solver.minidrawmaha import exploitability as grader
from drawmaha_solver.minidrawmaha.rules.cards import DECK, SUIT_RELABELLINGS, parse_cards, relabel
from drawmaha_solver.minidrawmaha.rules.enumeration import (
    PublicPoint,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.rules.game import (
    Action,
    InfoSet,
    MiniState,
    NodeKind,
    canonical_picture,
    random_deal,
)
from drawmaha_solver.minidrawmaha.solver.boost_target import (
    HOLES,
    N_STATES,
    SLOTS,
    EXACT,
    BoostTargets,
    Sampling,
    board_row,
    state_index,
)
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable

REPO = Path(__file__).parents[2]
MODULE = REPO / "src" / "drawmaha_solver" / "minidrawmaha" / "solver" / "boost_target.py"
FULL_TARGET = os.environ.get("MINIDRAWMAHA_FULL_TARGET") == "1"

# ---------------------------------------------------------------------------
# Helpers (restated)
# ---------------------------------------------------------------------------

def key_on(point: PublicPoint, private) -> InfoSet:
    hole, thrown, board = private
    return InfoSet(
        player=point.player, hole=hole, discarded=thrown, board=board,
        draws=point.draws, betting=point.betting,
    )

def shape_of(point: PublicPoint):
    return private_keys(board_cards=point.board_cards, discards=point.discards)

def random_table(seed: int) -> PackedTable:
    """The whole game with N(0, 1) regrets: every row plays a different, mostly sparse σ."""
    table = PackedTable.whole_game()
    table.cumulative_regret[:] = np.random.default_rng(seed).standard_normal(
        table.cumulative_regret.size
    )
    return table

def positive_table(seed: int) -> PackedTable:
    """The whole game with regrets in (0.05, 1.05): every action has positive probability, so every line is reachable."""
    table = PackedTable.whole_game()
    table.cumulative_regret[:] = 0.05 + np.random.default_rng(seed).random(table.cumulative_regret.size)
    return table

def point_index() -> dict[tuple, int]:
    """Public signature → position among the 141 points, the table's order."""
    return {
        (point.board_cards, point.draws, point.betting): index
        for index, point in enumerate(public_decision_points())
    }

def grader_node_to_point(game: grader.VectorGame) -> dict[int, int]:
    """The grader's own decision-node indices → table point indices, by public signature."""
    by_signature = point_index()
    return {
        node.index: by_signature[(node.level, node.public[0], node.public[1])]
        for node in game.decisions
    }

def grader_sigma(game: grader.VectorGame, sigma) -> dict[int, np.ndarray]:
    """`sigma` re-keyed by the grader's node index, as `_snapshot` would hand it to the walk."""
    to_point = grader_node_to_point(game)
    return {node.index: sigma[to_point[node.index]] for node in game.decisions}

def grader_reach(game: grader.VectorGame, sigma, *, carried: int) -> dict[int, np.ndarray]:
    """The grader's forward pass, stopping at nothing: seat `carried`'s reach at every node, all boards.

    `_walk`'s chance and opponent branches verbatim (the opponent here is
    `carried`); the other seat's nodes pass the reach straight down, since its
    actions never enter a counterfactual reach.
    """
    tables = grader_sigma(game, sigma)
    out: dict[int, np.ndarray] = {}

    def forward(node, reach):
        out[node.index] = reach
        if node.kind is NodeKind.TERMINAL:
            return
        if node.kind is NodeKind.CHANCE:
            child = node.child
            if node.deals_board:
                level = game.levels[child.level]
                forward(child, node.prob * reach[level.parent] * level.ok)
            else:
                forward(child, node.prob * reach * game.levels[node.level].ok)
            return
        if node.player != carried:
            for edge in node.edges:
                forward(edge.child, reach)
            return
        keys = game.keys[(node.level, node.discards[node.player])]
        valid = keys.index >= 0
        safe = np.where(valid, keys.index, 0)
        table = tables[node.index]
        for edge in node.edges:
            pushed = np.zeros_like(reach)
            for column, draw in zip(edge.actions, edge.draws, strict=True):
                played = reach * np.where(valid, table[safe, column], 0.0)
                if draw is None:
                    pushed += played
                else:
                    grader._push(pushed, played, draw)
            forward(edge.child, pushed)

    forward(game.root, game.weight[carried][None, :])
    return out

def grader_board_row(level: int, b: int) -> int:
    """My board index → the grader's batch row for the same cards (a guard against a convention drift)."""
    if level == 1:
        return grader.board_row((DECK[b],))
    first = b // (len(DECK) - 1)
    rest = b % (len(DECK) - 1)
    second = rest if rest < first else rest + 1
    return grader.board_row((DECK[first], DECK[second]))

class RowProfile(Mapping):
    """A profile over the whole game read from flat σ, float64, for the grader's functions."""

    def __init__(self, table: PackedTable, flat: np.ndarray):
        self._table, self._flat = table, flat

    def __getitem__(self, key):
        _, start, width = self._table.slot_of(key)
        return self._flat[start : start + width]

    def __iter__(self):
        return iter(self._table)

    def __len__(self):
        return len(self._table)

# -- a restricted deal the walks can sample ------------------------------------

_ALLOWED: list[tuple[tuple, tuple]] = []

def restricted_deal(rng: np.random.Generator) -> MiniState:
    """Uniform over `_ALLOWED` (seat-0 hole, seat-1 hole) pairs; module-level so the collector may pickle it."""
    return MiniState(holes=_ALLOWED[int(rng.integers(len(_ALLOWED)))])

def allowed_pairs(seat0, seat1) -> list[tuple[tuple, tuple]]:
    return [
        (tuple(sorted(a)), tuple(sorted(b)))
        for a in seat0
        for b in seat1
        if not set(a) & set(b)
    ]

# -- the world-by-world expectation ------------------------------------------

def world_by_world(table: PackedTable, key: InfoSet, pairs) -> tuple[np.ndarray, float]:
    """E[sampled target | the walk reaches `key`] and P(reach), by enumerating every world.

    Walks `MiniState` from every allowed deal (each at 1/len(pairs)), following
    every chance outcome at its probability and every action of the other
    seat at its σ, and every action of the key's seat unweighted — the walk
    enumerates its own actions. At the key, the exact value of each action
    (`ev`: chance averaged, both seats at σ) stands in for the walk's sampled
    continuation, whose expectation it is. Nothing here shares code with the
    module under test: it reads σ through the ledger and the rules through
    the state machine.
    """
    seat = key.player
    width = len(key.legal_actions())
    numerator = np.zeros(width)
    denominator = 0.0

    def ev(state: MiniState) -> float:
        if state.kind is NodeKind.TERMINAL:
            return state.returns()[seat]
        if state.kind is NodeKind.CHANCE:
            return sum(q * ev(state.apply_chance(cards)) for cards, q in state.chance_outcomes())
        sigma = table[state.infoset()].strategy()
        return sum(
            s * ev(state.apply(action))
            for action, s in zip(state.legal_actions(), sigma, strict=True)
            if s > 0.0
        )

    def consistent(state: MiniState) -> bool:
        """Could this history still reach `key`? Its public record must be a prefix of the key's, and the seat's own cards must fit."""
        if len(state.board) > len(key.board) or len(state.draws) > len(key.draws):
            return False
        if state.draws != key.draws[: len(state.draws)]:
            return False
        for round_index, line in enumerate(state.betting):
            if round_index >= len(key.betting) or key.betting[round_index][: len(line)] != line:
                return False
            if round_index < len(state.betting) - 1 and line != key.betting[round_index]:
                return False
        own = set(state.holes[seat]) | set(state.discards[seat])
        # The seat's cards so far, in the key's suit frame, must be a subset of the key's picture.
        return any(
            own <= set(relabel(frame, key.hole)) | set(relabel(frame, key.discarded))
            and tuple(relabel(frame, key.board))[: len(state.board)] == state.board
            for frame in SUIT_RELABELLINGS
        )

    def arrive(state: MiniState, p: float) -> None:
        nonlocal denominator
        if state.kind is NodeKind.TERMINAL or not consistent(state):
            return
        if state.kind is NodeKind.CHANCE:
            for cards, q in state.chance_outcomes():
                arrive(state.apply_chance(cards), p * q)
            return
        actions = state.legal_actions()
        if state.current_player == seat:
            if state.infoset() == key:
                utilities = np.array([ev(state.apply(action)) for action in actions])
                sigma = table[key].strategy()
                numerator[:] += p * (utilities - sigma @ utilities)
                denominator += p
                return  # perfect recall: the key is not below itself
            for action in actions:
                arrive(state.apply(action), p)
            return
        sigma = table[state.infoset()].strategy()
        for action, s in zip(actions, sigma, strict=True):
            if s > 0.0:
                arrive(state.apply(action), p * s)

    for holes in pairs:
        arrive(MiniState(holes=holes), 1.0 / len(pairs))
    return (numerator / denominator if denominator > 0 else numerator), denominator

def rows_at(table: PackedTable, keys) -> np.ndarray:
    return np.asarray([table.row_of(key) for key in keys], dtype=np.int64)

def assert_pooled_fair(z: np.ndarray) -> None:
    """The z of many rows, pooled: mean within 4/√n of 0, mean of z² within 4√(2/n) of 1.

    The mean of z is the sharp bias test: at n rows its standard deviation
    is 1/√n, so a bias of a tenth of a standard error in every row reaches
    the band at n ≈ 1,600 and is caught reliably by about 3,000 rows. The
    widths take the z's as independent and normal, which they are not
    quite: rows on one path share walks (a positive correlation, so the
    regret band is a little narrower than four true standard deviations),
    visit counts over rows come from one multinomial (negative, so that
    band is conservative), and a z built from a plug-in standard error over
    40-odd samples has heavier tails than normal. The seeds are fixed, so
    the runs are deterministic. The arrival bands were checked at the
    gated test's seed 12 for both seats and at seeds 13, 14, 15, 16, 17,
    18 and 19 for seat 1 (2026-11-05, mean z within 0.01 of zero on every
    fair selection); they are a calibrated regression gate, not a
    distributional guarantee. At small n the z² band is wide: 1 ± 1.79 at
    n = 10; its low side bites once the half-width drops under 1, from
    n = 33 (1 ± 0.98), and is 1 ± 0.84 at n = 45.
    """
    n = z.size
    assert abs(np.mean(z)) < 4 / np.sqrt(n), (n, np.mean(z))
    assert abs(np.mean(z**2) - 1.0) < 4 * np.sqrt(2 / n), (n, np.mean(z**2))

# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------

def test_the_module_imports_the_rules_and_never_the_grader_or_the_net():
    # The boost feeds the trainer; a bug shared with the grader would cancel
    # and certify itself. The check is static, on the import block, so a
    # review of the module starts where the root instructions say it should.
    tree = ast.parse(MODULE.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("exploitability" in name for name in imported), imported
    assert not any("deepcfr" in name for name in imported), imported
    assert not any(name.startswith("torch") for name in imported), imported
    assert all(
        name.startswith(("drawmaha_solver.minidrawmaha.rules", "drawmaha_solver.minidrawmaha.solver.packed_table"))
        for name in imported
        if name.startswith("drawmaha_solver")
    ), imported

# ---------------------------------------------------------------------------
# The description
# ---------------------------------------------------------------------------

def test_build_is_one_object_per_deal_set_over_a_shared_description():
    full = BoostTargets.build()
    assert BoostTargets.build() is full  # the default is cached: the description takes seconds to build
    assert len(full.points) == 141
    # The reach memo's bound, as the module states it: every node with one
    # board card out on the path from a decision to the root, and every
    # round-2 opening decision (one per round-1 line and pair of draw counts).
    d = full.description
    memoised = {}
    for node in d.decision_of_point:
        while node is not d.root:
            above, _ = d.parent[node.index]
            if node.level <= 1 or (above.kind is NodeKind.CHANCE and above.deals_board):
                memoised[node.index] = node
            node = above
    level_one = [node for node in memoised.values() if node.level <= 1]
    opening = [node for node in memoised.values() if node.level == 2]
    assert len(level_one) == 66 and sum(node.kind is NodeKind.DECISION for node in level_one) == 29
    assert len(opening) == 20 and all(node.kind is NodeKind.DECISION for node in opening)
    assert len(level_one) * 15 + len(opening) * 210 == 5_190
    # Every physical state of the frame is one of (hole, nothing thrown | a card not in the hole).
    assert N_STATES == len(HOLES) * SLOTS == 455 * 16
    assert state_index(parse_cards("2c 2d 2h"), ()) == 0  # the deck is ordered by rank, then suit
    assert state_index(parse_cards("2c 2d 2h"), parse_cards("3c")) == 1 + DECK.index(parse_cards("3c")[0])
    assert board_row(parse_cards("2c")) == 0 and board_row(parse_cards("2d 2c")) == 14
    # Every key of every shape has its physical entries, and the entry counts
    # are the frame's: boards × states the board allows.
    counts = {(1, 0): 15 * 364, (2, 0): 210 * 286, (2, 1): 210 * 286 * 10}
    for shape, expected in counts.items():
        n_keys = len(private_keys(board_cards=shape[0], discards=shape[1]))
        total = 0
        sizes = set()
        for position in range(n_keys):
            boards, states = full.entries_of(shape, position)
            assert boards.shape == states.shape and boards.size >= 1
            total += boards.size
            sizes.add(boards.size)
        assert total == expected
        # A key is a suit orbit of one picture: 6 relabellings over its stabiliser.
        assert sizes <= {1, 2, 3, 6}

def test_sigma_is_the_ledgers_strategy_row_by_row():
    full = BoostTargets.build()
    table = random_table(1)
    sigma = full.sigma(table)
    assert len(sigma) == 141
    for p, point in enumerate(public_decision_points()):
        shape = shape_of(point)
        assert sigma[p].shape == (len(shape), point.width) and sigma[p].dtype == np.float64
        for position in (0, len(shape) // 3, len(shape) - 1):
            key = key_on(point, shape[position])
            np.testing.assert_allclose(sigma[p][position], table[key].strategy(), rtol=1e-15, atol=0)
    with pytest.raises(ValueError, match="whole game"):
        full.sigma(PackedTable.listed([key_on(public_decision_points()[0], shape_of(public_decision_points()[0])[0])]))

# ---------------------------------------------------------------------------
# (a) the back-solved reach is the grader's carried reach
# ---------------------------------------------------------------------------

RESTRICTED = (
    [parse_cards("2c 3c 5d"), parse_cards("2d 3d 5c"), parse_cards("4h 5h 6h")],
    [parse_cards("2d 4h 6h"), parse_cards("3h 4c 4d"), parse_cards("2c 2d 6c"), parse_cards("3c 5h 6d")],
)

@pytest.mark.parametrize("holes", [None, RESTRICTED], ids=["full", "restricted"])
@pytest.mark.parametrize("carried", [0, 1])
def test_reach_equals_the_graders_carried_reach_on_every_board(holes, carried):
    exact = BoostTargets.build(holes=holes)
    game = grader.minidrawmaha_game(holes=holes)
    table = random_table(2)
    sigma = exact.sigma(table)
    reference = grader_reach(game, sigma, carried=carried)
    to_point = grader_node_to_point(game)
    checked = 0
    for node in game.decisions:
        p = to_point[node.index]
        for b in range(reference[node.index].shape[0]):
            mine = exact.reach(sigma, carried=carried, point=p, board=b)
            theirs = reference[node.index][grader_board_row(node.level, b)]
            np.testing.assert_allclose(mine, theirs, rtol=1e-10, atol=1e-300)
            checked += bool(theirs.any())
    assert checked > 1000  # most (point, board) pairs carry reach under a random table

# ---------------------------------------------------------------------------
# arrival at a first decision is the grader's frequency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seat", [0, 1])
def test_arrival_at_a_seats_first_decision_is_the_graders_frequency_and_sums_to_one(seat):
    # Before a seat has acted, own reach is the deal alone, so the grader's
    # `frequency` (own × compat) at its opening points is exactly the walk's
    # arrival; and every walk reaches one of them.
    exact = BoostTargets.build()
    game = grader.minidrawmaha_game()
    table = random_table(3)
    sigma = exact.sigma(table)
    reach = grader_reach(game, sigma, carried=1 - seat)
    to_point = grader_node_to_point(game)
    opening = [
        node for node in game.decisions
        if node.player == seat and node.level == 1 and len(node.public[1][0]) == seat
    ]
    assert len(opening) == 1 + seat
    total = 0.0
    for node in opening:
        p = to_point[node.index]
        keys = game.keys[(1, 0)]
        valid = keys.index >= 0
        _, compat = game.showdown.sums(1, reach[node.index], discards=0, shares=False)
        frequency = game.weight[seat][None, :] * compat
        per_key = np.zeros(keys.n_keys)
        np.add.at(per_key, keys.index[valid], frequency[valid])
        rows = np.arange(len(per_key)) + table.row_of(key_on(public_decision_points()[p], shape_of(public_decision_points()[p])[0]))
        arrival = exact.arrivals(sigma, seat=seat, rows=rows)
        assert arrival.shape == (len(rows),)
        np.testing.assert_allclose(arrival, per_key, rtol=1e-10, atol=1e-300)
        total += arrival.sum()
        # The full computation agrees with the arrival-only path (a round-1
        # row's value walks the whole game below it, so a handful of rows).
        some = rows[:: max(1, len(rows) // 4)]
        regret, with_values = exact.targets(sigma, seat=seat, rows=some, sampling=EXACT)
        assert regret.shape == (len(some), 4) and np.isfinite(regret).all()
        np.testing.assert_array_equal(with_values, arrival[:: max(1, len(rows) // 4)])
    assert total == pytest.approx(1.0, abs=1e-12)

# ---------------------------------------------------------------------------
# (b) exact form: a world-by-world expectation on a small restricted game
# ---------------------------------------------------------------------------

SMALL = (
    # Two suit relabellings of one hole for seat 0: one key, two physical entries.
    [parse_cards("2c 3c 5d"), parse_cards("2d 3d 5c")],
    # An opponent set that is NOT closed under that relabelling, so the two
    # entries face different ranges and the sum over entries is load-bearing.
    [parse_cards("2d 4h 6h"), parse_cards("3h 4c 4d"), parse_cards("2c 2h 6c")],
)

def small_game_keys(seat: int) -> list[InfoSet]:
    """One round-2 stood-pat row, one post-draw row, and (seat 1) one draw row, all reachable from SMALL."""
    points = public_decision_points()
    picks = []
    hero = SMALL[seat][0]
    if seat == 1:
        # Seat 1's draw after seat 0 stood pat: the value below it expands over
        # 37 hero continuations and 7 boards, which the exact walk can afford.
        draw = next(p for p in points if p.player == 1 and p.is_draw_decision and p.draws[0].count == 0)
        picks.append((draw, (hero, (), parse_cards("4c"))))
    pat = next(p for p in points if p.player == seat and p.board_cards == 2 and p.discards == 0 and len(p.betting[1]) == seat)
    picks.append((pat, (hero, (), parse_cards("4c 6d"))))
    threw = next(p for p in points if p.player == seat and p.board_cards == 2 and p.discards == 1 and len(p.betting[1]) == seat)
    kept = hero[:2] + (parse_cards("6d")[0],)
    picks.append((threw, (kept, hero[2:], parse_cards("4c 3h"))))
    return [key_on(point, canonical_key(private)) for point, private in picks]

def canonical_key(private):
    hole, thrown, board = private
    return canonical_picture(tuple(sorted(hole)), tuple(sorted(thrown)), board)

@pytest.mark.parametrize("seat", [0, 1])
def test_regret_and_arrival_equal_a_world_by_world_expectation(seat):
    pairs = allowed_pairs(*SMALL)
    exact = BoostTargets.build(holes=SMALL)
    table = positive_table(4)
    sigma = exact.sigma(table)
    keys = small_game_keys(seat)
    rows = rows_at(table, keys)
    regret, arrival = exact.targets(sigma, seat=seat, rows=rows, sampling=EXACT)
    for key, row_regret, row_arrival in zip(keys, regret, arrival, strict=True):
        expected, reach = world_by_world(table, key, pairs)
        assert reach > 0, str(key)
        width = len(key.legal_actions())
        assert row_arrival == pytest.approx(reach, rel=1e-9, abs=0), str(key)
        np.testing.assert_allclose(row_regret[:width], expected, rtol=1e-9, atol=1e-12, err_msg=str(key))
        assert not row_regret[width:].any()
    # The seat-0 keys stand for two physical entries whose weights differ,
    # so a representative entry would have failed above.
    if seat == 0:
        for key in keys:
            own = [
                world_by_world(table, key, [pair for pair in pairs if pair[0] == hole])[1]
                for hole in SMALL[0]
            ]
            assert own[0] != pytest.approx(own[1], rel=1e-3)

# ---------------------------------------------------------------------------
# (b) and (c) sampled: thousands of walks against the exact numbers
# ---------------------------------------------------------------------------

WIDE = ([parse_cards("2c 3c 5d")], list(HOLES))  # one hero hole for seat 0, any opponent

def test_regret_and_arrival_match_the_sampled_walks_row_by_row_and_pooled():
    # Seat 0 holds one hole, so its rows are reached often enough to test:
    # 6,000 walks reach the opening row every time and the draw rows a few
    # hundred times each. Per row the sample mean is within 4 s.e. of the
    # exact regret and the visit count within 4 s.e. of the exact arrival;
    # pooled, the mean of z² over rows sits near 1 — a bias of half a
    # standard error in every row would read 1.25 and a convention bug
    # (swapped reach, a dropped entry, the wrong chance factor) far above.
    walks = 6_000
    _ALLOWED[:] = allowed_pairs(*WIDE)
    exact = BoostTargets.build(holes=WIDE)
    table = random_table(5)
    sigma = exact.sigma(table)
    with Collector(seed=11, workers=0, deal=restricted_deal) as collector:
        collector.table.cumulative_regret[:] = table.cumulative_regret
        offered = collector.collect(seat=0, t=1, walks=walks)
    rows, counts = np.unique(offered.advantage_rows, return_counts=True)
    enough = counts >= 40
    rows, counts = rows[enough], counts[enough]
    regret, arrival = exact.targets(sigma, seat=0, rows=rows, sampling=EXACT)
    # Rows are kept by their EXPECTED count, 80 or more, so the observed
    # cutoff of 40 sits 4.5 standard errors below every kept row's mean and
    # the selection cannot bias z: a row chosen by its observed count alone
    # is there because it fluctuated high.
    fair = walks * arrival >= 80
    rows, counts, regret, arrival = rows[fair], counts[fair], regret[fair], arrival[fair]
    assert rows.size >= 8
    widths = table.widths()[rows]
    z_regret, z_visits = [], []
    for row, count, width, exact_regret, exact_arrival in zip(rows, counts, widths, regret, arrival, strict=True):
        samples = offered.advantage_targets[offered.advantage_rows == row][:, :width].astype(np.float64)
        mean = samples.mean(axis=0)
        se = samples.std(axis=0, ddof=1) / np.sqrt(count)
        live = se > 0
        z = (mean - exact_regret[:width])[live] / se[live]
        assert (np.abs(z) < 4).all(), (row, mean, exact_regret[:width], se)
        z_regret += z.tolist()
        # Visits: binomial over walks (a row reached twice in one walk is rarer than one in a thousand).
        se_visits = np.sqrt(walks * exact_arrival * (1 - exact_arrival))
        zv = (count - walks * exact_arrival) / se_visits
        assert abs(zv) < 4, (row, count, walks * exact_arrival)
        z_visits.append(zv)
    for z in (np.asarray(z_regret), np.asarray(z_visits)):
        assert z.size >= 8
        assert_pooled_fair(z)

@pytest.mark.skipif(
    not FULL_TARGET, reason="200,000 whole-game walks on 8 workers, ~2 min; set MINIDRAWMAHA_FULL_TARGET=1"
)
@pytest.mark.parametrize("seat", [0, 1])
def test_arrival_is_the_visit_rate_of_two_hundred_thousand_whole_game_walks(seat):
    walks = 200_000
    exact = BoostTargets.build()
    table = random_table(6)
    sigma = exact.sigma(table)
    with Collector(seed=12, workers=8, deal=random_deal) as collector:
        collector.table.cumulative_regret[:] = table.cumulative_regret
        offered = collector.collect(seat=seat, t=1, walks=walks)
    rows, counts = np.unique(offered.advantage_rows, return_counts=True)
    enough = counts >= 40
    rows, counts = rows[enough], counts[enough]
    arrival = exact.arrivals(sigma, seat=seat, rows=rows)
    # Kept by EXPECTED count, 80 or more: the observed cutoff of 40 is then
    # 4.5 standard errors below every kept row's mean, so the selection
    # cannot bias z. (Selecting on the observed count alone put seat 1's
    # draw rows, whose means sit near 80, at a mean z of +1.8 on two seeds:
    # the rows that cleared the cutoff were the ones that fluctuated high.
    # Seat 1's commonest rows, its two opening points, expect about 110.)
    fair = walks * arrival >= 80
    rows, counts, arrival = rows[fair], counts[fair], arrival[fair]
    assert rows.size >= 50
    z = (counts - walks * arrival) / np.sqrt(walks * arrival * (1 - arrival))
    # The sharp tests are the pooled ones (`assert_pooled_fair`). The
    # per-row cap is loose on purpose: of some 3,000 rows the largest |z|
    # passes 5 in a run out of a thousand, and the deal alone can do it — at
    # seed 12 one seat-1 picture (3c 3d 4h with 2c on the board, six
    # relabellings) was dealt 292 times against an expected 220, 4.9
    # standard errors, in a count that involves no σ and nothing of this
    # module, and 221, 219, 222 times on three other seeds.
    assert (np.abs(z) < 6).all(), (np.abs(z).max(), rows[np.argmax(np.abs(z))])
    assert_pooled_fair(z)

# ---------------------------------------------------------------------------
# Refusals and the unreachable row
# ---------------------------------------------------------------------------

def test_rows_of_the_other_seat_and_outside_the_table_are_refused():
    exact = BoostTargets.build()
    table = random_table(7)
    sigma = exact.sigma(table)
    points = public_decision_points()
    seat1 = next(i for i, point in enumerate(points) if point.player == 1)
    row_of_seat_1 = table.row_of(key_on(points[seat1], shape_of(points[seat1])[0]))
    with pytest.raises(ValueError, match="seat"):
        exact.targets(sigma, seat=0, rows=np.array([row_of_seat_1]), sampling=EXACT)
    with pytest.raises(ValueError, match="seat"):
        exact.targets(sigma, seat=2, rows=np.array([0]), sampling=EXACT)
    for bad in (-1, len(table)):
        with pytest.raises(IndexError):
            exact.targets(sigma, seat=0, rows=np.array([bad]), sampling=EXACT)

def test_an_unreachable_row_has_zero_arrival_and_zero_regret():
    # Seat 1 never bets or raises: every seat-0 round-1 row facing a bet is
    # unreachable, and the exact target says so with zeros rather than NaN.
    exact = BoostTargets.build()
    table = random_table(8)
    points = public_decision_points()
    for p, point in enumerate(points):
        if point.player == 1 and not point.is_draw_decision and Action.POT in point.actions:
            column = point.actions.index(Action.POT)
            for position in range(len(shape_of(point))):
                _, start, width = table.slot_of(key_on(point, shape_of(point)[position]))
                table.cumulative_regret[start + column] = -1.0
                if not (table.cumulative_regret[start : start + width] > 0).any():
                    table.cumulative_regret[start + point.actions.index(Action.CHECK_CALL)] = 1.0
    sigma = exact.sigma(table)
    facing = next(
        i for i, point in enumerate(points)
        if point.player == 0 and point.board_cards == 1 and not point.is_draw_decision
        and point.betting[0][-1:] == (Action.POT,)
    )
    rows = np.arange(len(shape_of(points[facing]))) + table.row_of(key_on(points[facing], shape_of(points[facing])[0]))
    regret, arrival = exact.targets(sigma, seat=0, rows=rows, sampling=EXACT)
    assert not arrival.any() and not regret.any()
    assert np.isfinite(regret).all()

# ---------------------------------------------------------------------------
# (e) sampling below the row: unbiased against the exact sum, seeded per row
# ---------------------------------------------------------------------------

def only(**sampled) -> Sampling:
    """A Sampling that enumerates every source but the ones named."""
    return Sampling(**{
        "opponent_range": None, "board_cards": None, "opponent_draws": None,
        "hero_draws": None, "opponent_actions": None, **sampled,
    })

def row_on(table: PackedTable, point: int, position: int = 7) -> np.ndarray:
    points = public_decision_points()
    return np.array([table.row_of(key_on(points[point], shape_of(points[point])[position]))])

def sampled(exact, sigma, seat, rows, sampling, seeds) -> np.ndarray:
    """One sampled target per seed, `(len(seeds), 4)`, t equal to the seed."""
    return np.stack([
        exact.targets(sigma, seat=seat, rows=rows, sampling=sampling, seed=0, t=s)[0][0] for s in seeds
    ])

@pytest.mark.parametrize(
    "bad",
    [dict(board_cards=0), dict(hero_draws=-1), dict(opponent_actions=True), dict(opponent_range=1.5), dict(playouts=0)],
    ids=lambda d: next(iter(d)),
)
def test_sampling_refuses_counts_that_are_not_positive_ints(bad):
    field = next(iter(bad))
    with pytest.raises(ValueError, match=f"Sampling.{field}"):
        Sampling(**bad)

def test_sampling_counts_take_any_integer_type():
    assert Sampling(board_cards=np.int64(2), playouts=np.int32(3)).board_cards == 2

def test_exact_with_more_than_one_playout_is_refused():
    with pytest.raises(ValueError, match="playouts must be 1"):
        only(playouts=2)

def test_a_sampled_call_names_its_seed_and_t():
    exact = BoostTargets.build()
    table = positive_table(3)
    sigma = exact.sigma(table)
    rows = row_on(table, 4)
    with pytest.raises(ValueError, match="seed"):
        exact.targets(sigma, seat=0, rows=rows, sampling=Sampling(), t=0)
    with pytest.raises(ValueError, match="t,|non-negative int t"):
        exact.targets(sigma, seat=0, rows=rows, sampling=Sampling(), seed=0)
    with pytest.raises(TypeError, match="Sampling"):
        exact.targets(sigma, seat=0, rows=rows, sampling=None)

@pytest.mark.parametrize("point", [2, 0], ids=["draw row", "round-1 row"])
def test_a_count_at_or_above_its_fan_is_enumeration_bit_for_bit(point):
    exact = BoostTargets.build()
    table = positive_table(3)
    sigma = exact.sigma(table)
    rows = row_on(table, point)
    full = only(board_cards=14, opponent_draws=12, hero_draws=99)
    # The fans `is_exact` reads are the description's own.
    assert exact.description.levels[2].children_of.shape[1] == 14
    assert all(
        draw.targets.shape[-1] == 12
        for node_edges in [n.edges for n in exact.description.decision_of_point if n.edges]
        for edge in node_edges for draw in edge.draws if draw is not None
    )
    assert full.is_exact
    with pytest.raises(ValueError, match="playouts must be 1"):
        only(board_cards=14, playouts=2)
    want = exact.targets(sigma, seat=0, rows=rows, sampling=EXACT)
    got = exact.targets(sigma, seat=0, rows=rows, sampling=full, seed=0, t=0)
    assert np.array_equal(want[0], got[0]) and np.array_equal(want[1], got[1])

# Each source alone at a row where it is live, then all of them at once. The
# draw-row cases walk the whole draw and round 2 below them, about 0.2 s a
# call when nothing prunes, so they take fewer seeds. At these seeds the
# smallest relative bias the 4.5-s.e. gate resolves (4.5 · s.e. / |mean|,
# measured 2026-11-06 at these rows) is about 3% for the opponent's actions,
# 8% for the range and 2–8% for the opponent's replacements; for the hero's
# replacements, the board card and the mixed forms it is 13% to over 100%,
# because those sources move the target by more than its size. Their scale
# factors are pinned exactly instead, by the branch oracle below; these
# cases check that the forms compose and that each one actually samples.
UNBIASED = [
    ("opponent_range", only(opponent_range=1), 4, 0, 400),
    ("opponent_range after a throw", only(opponent_range=1), 29, 0, 400),
    ("opponent_actions", only(opponent_actions=1), 4, 0, 400),
    ("board_cards", only(board_cards=1), 2, 0, 200),
    ("hero_draws, seat 0", only(hero_draws=1), 2, 0, 60),
    ("hero_draws, seat 1", only(hero_draws=1), 3, 1, 60),
    ("opponent_draws", only(opponent_draws=1), 2, 0, 60),
    ("the walk's sampling, whole range", Sampling(), 2, 0, 200),
    ("everything at one", Sampling(opponent_range=1), 2, 0, 200),
    ("two of each, two play-outs", Sampling(playouts=2, opponent_range=2, board_cards=2, opponent_draws=2,
                                            hero_draws=2, opponent_actions=2), 2, 0, 100),
]

@pytest.mark.parametrize(("sampling", "point", "seat", "n"), [c[1:] for c in UNBIASED], ids=[c[0] for c in UNBIASED])
def test_each_sampled_source_is_unbiased_and_actually_samples(sampling, point, seat, n):
    exact = BoostTargets.build()
    table = positive_table(3)
    sigma = exact.sigma(table)
    rows = row_on(table, point)
    want, arrival = exact.targets(sigma, seat=seat, rows=rows, sampling=EXACT)
    assert arrival[0] > 0.0
    draws = sampled(exact, sigma, seat, rows, sampling, range(n))
    spread = draws.std(axis=0, ddof=1)
    live = spread > 1e-9  # above float rounding, which a z would blow up
    # Not vacuous: the source really varied the target at this row.
    assert live.any(), "the sampled source never moved the target here"
    z = (draws.mean(axis=0)[live] - want[0][live]) / (spread[live] / np.sqrt(n))
    assert (np.abs(z) < 4.5).all(), z
    # A column the sample never moves is the exact one, bit for bit or near.
    np.testing.assert_allclose(draws[:, ~live], np.broadcast_to(want[0][~live], draws[:, ~live].shape), atol=1e-9)
    _, sampled_arrival = exact.targets(sigma, seat=seat, rows=rows, sampling=sampling, seed=0, t=0)
    assert sampled_arrival[0] == arrival[0] == exact.arrivals(sigma, seat=seat, rows=rows)[0]

def small_draw_row(table: PackedTable, seat: int) -> np.ndarray:
    """A draw row reachable from SMALL: seat 0's (both throws below it, two physical entries) or seat 1's."""
    if seat == 1:
        return rows_at(table, small_game_keys(1)[:1])
    point = public_decision_points()[2]
    return rows_at(table, [key_on(point, canonical_key((SMALL[0][0], (), parse_cards("4c"))))])

# Each chance source crosses a path below a draw row once, and the target is
# linear in it, so keeping branch j at every position, scaled n, and
# averaging over j = 0 … n − 1 is the exact sum with no noise at all: a
# scale factor off by one branch in twelve fails at 1e-10 instead of hiding
# inside a standard error.
ORACLE = [
    ("board_cards", 14, None, 0), ("hero_draws", 12, None, 0), ("opponent_draws", 12, None, 0),
    ("hero_draws", 12, None, 1),
    ("board_cards", 14, SMALL, 0), ("hero_draws", 12, SMALL, 0), ("opponent_draws", 12, SMALL, 0),
    ("board_cards", 14, SMALL, 1),
]

@pytest.mark.parametrize(("source", "fan", "holes", "seat"), ORACLE,
                         ids=[f"{c[0]}, {'small' if c[2] else 'full'}, seat {c[3]}" for c in ORACLE])
def test_a_chance_source_kept_one_branch_at_a_time_averages_to_the_exact_sum(source, fan, holes, seat, monkeypatch):
    from drawmaha_solver.minidrawmaha.solver import boost_target
    exact = BoostTargets.build(holes=holes)
    table = positive_table(3)
    sigma = exact.sigma(table)
    if holes is None:
        rows = row_on(table, 2 if seat == 0 else 3)
    else:
        rows = small_draw_row(table, seat)
    want, arrival = exact.targets(sigma, seat=seat, rows=rows, sampling=EXACT)
    assert arrival[0] > 0.0
    branch = [0]
    monkeypatch.setattr(
        boost_target._Draws, "subset",
        lambda self, k, n, shape: None if k is None or k >= n else np.full((*shape, 1), branch[0]),
    )
    total = np.zeros_like(want)
    for j in range(fan):
        branch[0] = j
        total += exact.targets(sigma, seat=seat, rows=rows, sampling=only(**{source: 1}), seed=0, t=0)[0]
    assert np.abs(want).max() > 0.0
    np.testing.assert_allclose(total / fan, want, rtol=1e-10, atol=1e-13)

# SMALL's seat-0 hole and its clubs ↔ diamonds relabelling against every
# hole holding more clubs than diamonds: a set the relabelling maps onto its
# complement, so the key's two physical entries face wide and different
# ranges (SMALL's three opponent holes leave one live hand at these rows,
# and a range draw there has nothing to vary).
LOPSIDED = (
    SMALL[0],
    [hole for hole in HOLES if sum(c.suit == 0 for c in hole) > sum(c.suit == 1 for c in hole)],  # suit 0 = c, 1 = d
)

def test_the_range_draw_is_unbiased_where_two_entries_face_different_ranges():
    # At a stood-pat row and at one after a throw (where the thrown card
    # enters the clash mask): a draw that mixed one entry's total or mask
    # into the other's would shift the mean.
    exact = BoostTargets.build(holes=LOPSIDED)
    table = positive_table(4)
    sigma = exact.sigma(table)
    rows = rows_at(table, small_game_keys(0))
    want, arrival = exact.targets(sigma, seat=0, rows=rows, sampling=EXACT)
    assert (arrival > 0.0).all()
    n = 600
    draws = np.stack([
        exact.targets(sigma, seat=0, rows=rows, sampling=only(opponent_range=1), seed=0, t=s)[0] for s in range(n)
    ])  # (n, rows, 4)
    spread = draws.std(axis=0, ddof=1)
    live = spread > 1e-9
    assert live[0].any() and live[1].any(), spread
    z = (draws.mean(axis=0)[live] - want[live]) / (spread[live] / np.sqrt(n))
    assert (np.abs(z) < 4.5).all(), z

def test_a_rows_sample_depends_on_seed_t_seat_and_row_only():
    exact = BoostTargets.build()
    table = positive_table(3)
    sigma = exact.sigma(table)
    rows = np.concatenate([row_on(table, 4, 7), row_on(table, 2, 11), row_on(table, 4, 30)])
    at = lambda r, t=5: exact.targets(sigma, seat=0, rows=r, sampling=Sampling(opponent_range=1), seed=1, t=t)[0]
    together = at(rows)
    assert np.array_equal(at(rows[::-1]), together[::-1])
    assert np.array_equal(np.concatenate([at(rows[:1]), at(rows[1:])]), together)
    assert np.array_equal(at(rows[[0, 0, 1, 2]]), together[[0, 0, 1, 2]])
    assert not np.array_equal(at(rows, t=6), together)

def test_more_playouts_shrink_the_spread_as_one_over_m():
    exact = BoostTargets.build()
    table = positive_table(3)
    sigma = exact.sigma(table)
    rows = row_on(table, 4)
    one = sampled(exact, sigma, 0, rows, only(opponent_actions=1), range(200)).var(axis=0, ddof=1)
    eight = sampled(exact, sigma, 0, rows, only(opponent_actions=1, playouts=8), range(200)).var(axis=0, ddof=1)
    live = one > 0.0
    ratio = eight[live] / one[live]
    assert ((1 / 16 < ratio) & (ratio < 1 / 4)).all(), ratio

def test_an_unreachable_row_is_zero_under_any_sampling():
    # The same construction as the exact test above: seat 1 never bets, so a
    # seat-0 round-1 row facing a bet has no weight to draw a range from.
    exact = BoostTargets.build()
    table = random_table(8)
    points = public_decision_points()
    for p, point in enumerate(points):
        if point.player == 1 and not point.is_draw_decision and Action.POT in point.actions:
            column = point.actions.index(Action.POT)
            for position in range(len(shape_of(point))):
                _, start, width = table.slot_of(key_on(point, shape_of(point)[position]))
                table.cumulative_regret[start + column] = -1.0
                if not (table.cumulative_regret[start : start + width] > 0).any():
                    table.cumulative_regret[start + point.actions.index(Action.CHECK_CALL)] = 1.0
    sigma = exact.sigma(table)
    facing = next(
        i for i, point in enumerate(points)
        if point.player == 0 and point.board_cards == 1 and not point.is_draw_decision
        and point.betting[0][-1:] == (Action.POT,)
    )
    rows = np.arange(4) + table.row_of(key_on(points[facing], shape_of(points[facing])[0]))
    regret, arrival = exact.targets(sigma, seat=0, rows=rows, sampling=Sampling(opponent_range=1), seed=0, t=0)
    assert not arrival.any() and not regret.any()
    assert np.isfinite(regret).all()

# ---------------------------------------------------------------------------
# (d) cost per row: a benchmark, recorded, with a loose ceiling
# ---------------------------------------------------------------------------

def test_a_round_two_row_costs_a_few_milliseconds():
    exact = BoostTargets.build()
    table = random_table(9)
    sigma = exact.sigma(table)
    rng = np.random.default_rng(0)
    points = public_decision_points()
    offsets = np.cumsum([0] + [len(shape_of(point)) for point in points])
    round_two = [p for p, point in enumerate(points) if point.player == 0 and point.board_cards == 2]
    rows = np.concatenate([
        offsets[p] + rng.choice(len(shape_of(points[p])), size=10, replace=False) for p in round_two
    ])
    exact.targets(sigma, seat=0, rows=rows[:5], sampling=EXACT)  # warm the description
    started = time.perf_counter()
    exact.targets(sigma, seat=0, rows=rows, sampling=EXACT)
    per_row = (time.perf_counter() - started) / rows.size
    print(f"\nexact target: {per_row * 1e3:.2f} ms per round-2 row over {rows.size} rows")
    # The plan's figure is 1 ms a row and the measured one 0.8 ms. The
    # ceiling is fifty times that: wide enough that a laptop busy with a
    # training run (0.95 ms measured beside a 9-worker LCFR run) cannot
    # cross it, narrow enough to catch a regression to the grader's
    # whole-board update, which is about a second a row.
    assert per_row < 0.050

def test_sampling_below_a_draw_or_round_one_row_costs_a_fraction_of_the_exact_sum():
    # The board card's fourteen branches and the hero's thirty-six
    # replacements are what an exact draw or round-1 row pays for; the walk's
    # sampling keeps one of each. Printed for the record; the assertion is a
    # ratio, which a busy laptop slows on both sides alike.
    exact = BoostTargets.build()
    table = positive_table(3)
    sigma = exact.sigma(table)
    lines = []
    for point, name in ((4, "round 2"), (2, "draw"), (0, "round 1")):
        rows = np.concatenate([row_on(table, point, position) for position in (3, 7, 11)])
        exact.targets(sigma, seat=0, rows=rows[:1], sampling=EXACT)  # warm
        cost = {}
        for label, sampling in (("exact", EXACT), ("walk's sampling", Sampling())):
            started = time.perf_counter()
            exact.targets(sigma, seat=0, rows=rows, sampling=sampling, seed=0, t=0)
            cost[label] = (time.perf_counter() - started) / rows.size
        lines.append(f"{name:>8}: exact {cost['exact'] * 1e3:7.1f} ms, walk's sampling {cost["walk's sampling"] * 1e3:6.1f} ms a row")
        if point != 4:
            assert cost["walk's sampling"] < cost["exact"] / 2, (name, cost)
    print("\nboost target, a positive table:\n" + "\n".join(lines))
