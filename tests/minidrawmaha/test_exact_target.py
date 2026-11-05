"""The exact target: what one sampled regret row is worth on average, computed instead of sampled.

The module under test is rung 4's boost oracle: for a table row it sums, over
every physical entry of the key and every opponent hand, the regret the walk
would bank there, weighted by how often the walk arrives. The claims, in
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
and the cost per round-2 row is recorded (test d, a benchmark).

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
from drawmaha_solver.minidrawmaha.solver.exact_target import (
    HOLES,
    N_STATES,
    SLOTS,
    ExactTargets,
    board_row,
    state_index,
)
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable

REPO = Path(__file__).parents[2]
MODULE = REPO / "src" / "drawmaha_solver" / "minidrawmaha" / "solver" / "exact_target.py"
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
    the runs are deterministic, and the thresholds were checked at those
    seeds and on several others; they are a calibrated regression gate,
    not a distributional guarantee.
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
    full = ExactTargets.build()
    assert ExactTargets.build() is full  # the default is cached: the description takes seconds to build
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
    full = ExactTargets.build()
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
    exact = ExactTargets.build(holes=holes)
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
    exact = ExactTargets.build()
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
        regret, with_values = exact.targets(sigma, seat=seat, rows=some)
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
    exact = ExactTargets.build(holes=SMALL)
    table = positive_table(4)
    sigma = exact.sigma(table)
    keys = small_game_keys(seat)
    rows = rows_at(table, keys)
    regret, arrival = exact.targets(sigma, seat=seat, rows=rows)
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
    exact = ExactTargets.build(holes=WIDE)
    table = random_table(5)
    sigma = exact.sigma(table)
    with Collector(seed=11, workers=0, deal=restricted_deal) as collector:
        collector.table.cumulative_regret[:] = table.cumulative_regret
        offered = collector.collect(seat=0, t=1, walks=walks)
    rows, counts = np.unique(offered.advantage_rows, return_counts=True)
    enough = counts >= 40
    rows, counts = rows[enough], counts[enough]
    regret, arrival = exact.targets(sigma, seat=0, rows=rows)
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
        assert_pooled_fair(z)  # (−1.3, 1.3) and (0.4, 1.6) at n = 10

@pytest.mark.skipif(
    not FULL_TARGET, reason="200,000 whole-game walks on 8 workers, ~2 min; set MINIDRAWMAHA_FULL_TARGET=1"
)
@pytest.mark.parametrize("seat", [0, 1])
def test_arrival_is_the_visit_rate_of_two_hundred_thousand_whole_game_walks(seat):
    walks = 200_000
    exact = ExactTargets.build()
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
    exact = ExactTargets.build()
    table = random_table(7)
    sigma = exact.sigma(table)
    points = public_decision_points()
    seat1 = next(i for i, point in enumerate(points) if point.player == 1)
    row_of_seat_1 = table.row_of(key_on(points[seat1], shape_of(points[seat1])[0]))
    with pytest.raises(ValueError, match="seat"):
        exact.targets(sigma, seat=0, rows=np.array([row_of_seat_1]))
    with pytest.raises(ValueError, match="seat"):
        exact.targets(sigma, seat=2, rows=np.array([0]))
    for bad in (-1, len(table)):
        with pytest.raises(IndexError):
            exact.targets(sigma, seat=0, rows=np.array([bad]))

def test_an_unreachable_row_has_zero_arrival_and_zero_regret():
    # Seat 1 never bets or raises: every seat-0 round-1 row facing a bet is
    # unreachable, and the exact target says so with zeros rather than NaN.
    exact = ExactTargets.build()
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
    regret, arrival = exact.targets(sigma, seat=0, rows=rows)
    assert not arrival.any() and not regret.any()
    assert np.isfinite(regret).all()

# ---------------------------------------------------------------------------
# (d) cost per row: a benchmark, recorded, with a loose ceiling
# ---------------------------------------------------------------------------

def test_a_round_two_row_costs_a_few_milliseconds():
    exact = ExactTargets.build()
    table = random_table(9)
    sigma = exact.sigma(table)
    rng = np.random.default_rng(0)
    points = public_decision_points()
    offsets = np.cumsum([0] + [len(shape_of(point)) for point in points])
    round_two = [p for p, point in enumerate(points) if point.player == 0 and point.board_cards == 2]
    rows = np.concatenate([
        offsets[p] + rng.choice(len(shape_of(points[p])), size=10, replace=False) for p in round_two
    ])
    exact.targets(sigma, seat=0, rows=rows[:5])  # warm the description
    started = time.perf_counter()
    exact.targets(sigma, seat=0, rows=rows)
    per_row = (time.perf_counter() - started) / rows.size
    print(f"\nexact target: {per_row * 1e3:.2f} ms per round-2 row over {rows.size} rows")
    # The plan's figure is 1 ms a row and the measured one 0.8 ms. The
    # ceiling is fifty times that: wide enough that a laptop busy with a
    # training run (0.95 ms measured beside a 9-worker LCFR run) cannot
    # cross it, narrow enough to catch a regression to the grader's
    # whole-board update, which is about a second a row.
    assert per_row < 0.050
