"""The sampled learner, pinned on three games: one scripted, one refereed, one real.

Mini-drawmaha has no referee — no engine plays it and no LP has solved it — so
the learner cannot be graded on the game it exists for. It is graded instead on
the two games that CAN say what is right, and the same `traverse()` is pointed
at all three, because a Leduc-specific copy would prove the copy and not the
file:

* **A scripted ten-node tree** pins the four branches exactly. Its states log
  every node they are asked about, so "the traverser walks every action, the
  opponent follows one, the deck follows one" is a count of log entries rather
  than an inference from ledger sums. It is also where the one real trap —
  reach multiplied into the regret weight — is pinned on a spot whose wrong
  weight can be computed by hand.
* **Leduc** grades the learner against the LP referee in `tests/leduc/`. The
  game value converges fast and cleanly under sampling; exploitability falls
  slowly and noisily, so it is held to a trend and a loose bound, never to
  rung 2's vanilla curve. Every bound was calibrated against this exact
  `traverse()`, and the measured numbers sit beside it — including the
  finding that short runs cannot tell a double-weighted walk from a correct
  one, which is why the weights are pinned directly.
* **Mini-drawmaha** is smoke-tested for the things only it has: a four-wide
  draw ledger and a root that is a chance node. On a lazy table by default,
  and for one iteration on the real 1.77 GB table behind
  `MINIDRAWMAHA_FULL_TABLE=1`.

The million-iteration Leduc run — the only convergence test long enough to
separate a double-weighted walk from a correct one — sits behind
`MINIDRAWMAHA_FULL_CALIBRATION=1`, so the default suite stays short.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.leduc.exploitability import exploitability, expected_value
from drawmaha_solver.leduc.game import DEALS, LeducState
from drawmaha_solver.leduc.infoset_table import (
    average_strategy as leduc_average_strategy,
)
from drawmaha_solver.leduc.infoset_table import new_infoset_table as new_leduc_table
from drawmaha_solver.minidrawmaha.game import DRAW_ACTIONS, InfoSet, random_deal
from drawmaha_solver.minidrawmaha.mccfr import (
    Solve,
    load_solve,
    new_solve,
    run_iteration,
    save_solve,
    train,
    traverse,
)
from drawmaha_solver.regret_matching import RegretMatcher

REFEREE = json.loads(
    (Path(__file__).parents[1] / "leduc" / "referee.json").read_text()
)
LP_VALUE = REFEREE["lp_value_to_p0"]

FULL_TABLE = os.environ.get("MINIDRAWMAHA_FULL_TABLE") == "1"
FULL_CALIBRATION = os.environ.get("MINIDRAWMAHA_FULL_CALIBRATION") == "1"

def leduc_deal(rng: np.random.Generator) -> LeducState:
    """Leduc's root sampler: one of the 30 ordered deals, uniformly."""
    return LeducState(cards=DEALS[rng.integers(len(DEALS))])

def leduc_solve(seed: int, *, table=None) -> Solve:
    return new_solve(
        seed, table=new_leduc_table() if table is None else table, deal=leduc_deal
    )

class SpyLedger(RegretMatcher):
    """A RegretMatcher that records every bank it receives."""

    def __init__(self, n_actions: int):
        super().__init__(n_actions)
        self.banked: list[tuple[np.ndarray, float, float]] = []

    def update(self, utilities, *, regret_weight=1.0, strategy_weight=1.0):
        self.banked.append((np.array(utilities), regret_weight, strategy_weight))
        super().update(
            utilities, regret_weight=regret_weight, strategy_weight=strategy_weight
        )

def ledgers_equal(first, second) -> bool:
    return list(first) == list(second) and all(
        np.array_equal(first[key].cumulative_regret, second[key].cumulative_regret)
        and np.array_equal(first[key].strategy_sum, second[key].strategy_sum)
        for key in first
    )

# ---------------------------------------------------------------------------
# A scripted tree: every branch of the walk, counted
# ---------------------------------------------------------------------------

# The deck picks a side; on the left P0 chooses between asking P1 (who has
# three answers) and ending the hand; on the right P0 has two ways to end it.
# Payoffs are chips to P0 and every terminal pays a different amount, so a
# returned value names the leaves it was built from.
_TREE = {
    "root": ("chance", (("L", 0.25), ("R", 0.75))),
    "L": ("decision", 0, "I-left", ("ask", "stop")),
    "L.ask": ("decision", 1, "J", ("c", "d", "e")),
    "L.ask.c": ("terminal", 1.0),
    "L.ask.d": ("terminal", 2.0),
    "L.ask.e": ("terminal", 4.0),
    "L.stop": ("terminal", 8.0),
    "R": ("decision", 0, "I-right", ("x", "y")),
    "R.x": ("terminal", 16.0),
    "R.y": ("terminal", -32.0),
}

@dataclass(frozen=True)
class ScriptedState:
    """One node of `_TREE`, logging every node the walk asks about."""

    node: str
    log: list[str] = field(default_factory=list, compare=False)

    def __post_init__(self) -> None:
        self.log.append(self.node)

    @property
    def _spec(self):
        return _TREE[self.node]

    def is_terminal(self) -> bool:
        return self._spec[0] == "terminal"

    def is_chance_node(self) -> bool:
        return self._spec[0] == "chance"

    def chance_outcomes(self):
        return self._spec[1]

    def apply_chance(self, outcome: str) -> ScriptedState:
        return ScriptedState(outcome, self.log)

    @property
    def current_player(self) -> int:
        return self._spec[1]

    def legal_actions(self):
        return self._spec[3]

    def apply(self, action: str) -> ScriptedState:
        return ScriptedState(f"{self.node}.{action}", self.log)

    def infoset(self) -> str:
        return self._spec[2]

    def returns(self) -> tuple[float, float]:
        return (self._spec[1], -self._spec[1])

def scripted_table() -> dict[str, SpyLedger]:
    return {"I-left": SpyLedger(2), "J": SpyLedger(3), "I-right": SpyLedger(2)}

def test_a_terminal_returns_the_traversers_chips():
    rng = np.random.default_rng(0)
    for traverser, chips in ((0, 8.0), (1, -8.0)):
        leaf = ScriptedState("L.stop")
        assert traverse(leaf, scripted_table(), traverser, rng, 1) == chips

def test_a_chance_node_follows_exactly_one_outcome():
    for seed in range(20):
        root = ScriptedState("root")
        traverse(root, scripted_table(), 0, np.random.default_rng(seed), 1)
        assert ("L" in root.log) != ("R" in root.log)

def test_the_deck_is_sampled_by_its_probabilities():
    # 0.25 left and 0.75 right: a uniform-index sampler would split them evenly
    # and pass every other test in this file.
    rng = np.random.default_rng(0)
    lefts = 0
    for _ in range(4_000):
        root = ScriptedState("root")
        traverse(root, scripted_table(), 0, rng, 1)
        lefts += "L" in root.log
    assert lefts / 4_000 == pytest.approx(0.25, abs=0.02)

def test_the_traverser_walks_every_action_and_the_opponent_one():
    for seed in range(20):
        start = ScriptedState("L")
        traverse(start, scripted_table(), 0, np.random.default_rng(seed), 1)
        assert {"L.ask", "L.stop"} <= set(start.log)
        answers = [node for node in start.log if node.startswith("L.ask.")]
        assert len(answers) == 1

def test_seated_the_other_way_the_same_spots_swap_roles():
    # With P1 traversing, P0's spot is the opponent's: one of its two actions
    # is followed, and only when that action is "ask" does P1 walk all three.
    for seed in range(20):
        start = ScriptedState("L")
        traverse(start, scripted_table(), 1, np.random.default_rng(seed), 1)
        followed = [node for node in start.log if node in ("L.ask", "L.stop")]
        assert len(followed) == 1
        if followed == ["L.ask"]:
            assert {"L.ask.c", "L.ask.d", "L.ask.e"} <= set(start.log)

def test_the_traversers_bank_carries_regret_weight_one_and_no_strategy():
    table = scripted_table()
    traverse(ScriptedState("L"), table, 0, np.random.default_rng(3), 7)
    ((utilities, regret_weight, strategy_weight),) = table["I-left"].banked
    assert (regret_weight, strategy_weight) == (1.0, 0.0)
    # "stop" pays 8; "ask" pays whichever answer P1 was sampled into.
    assert utilities[1] == 8.0
    assert utilities[0] in (1.0, 2.0, 4.0)

def test_regret_weight_is_one_however_rarely_the_spot_is_reached():
    # The gotcha, on a tree small enough to name the wrong answer: with P1
    # traversing, J sits behind the deck's 0.25 and P0's uniform 0.5, so a walk
    # threading reach down would bank it at 0.125. Sampling already charged
    # that probability — J is simply banked on one traversal in eight.
    banks = []
    for seed in range(64):
        table = scripted_table()
        traverse(ScriptedState("root"), table, 1, np.random.default_rng(seed), 1)
        banks += table["J"].banked
    assert banks
    assert all((regret_weight, strategy_weight) == (1.0, 0.0) for _, regret_weight, strategy_weight in banks)

def test_the_opponents_bank_carries_the_iteration_and_no_regret():
    table = scripted_table()
    traverse(ScriptedState("L"), table, 0, np.random.default_rng(3), 7)
    ((utilities, regret_weight, strategy_weight),) = table["J"].banked
    assert (regret_weight, strategy_weight) == (0.0, 7.0)
    assert not utilities.any()
    assert table["J"].cumulative_regret.tolist() == [0.0, 0.0, 0.0]

def test_the_value_is_the_traversers_mix_over_what_each_action_earned():
    # Fresh ledgers play uniformly, so the traverser's value at "L" is the plain
    # average of "stop" (8) and the one answer the opponent was sampled into.
    for seed in range(20):
        start = ScriptedState("L")
        value = traverse(start, scripted_table(), 0, np.random.default_rng(seed), 1)
        (answer,) = (node for node in start.log if node.startswith("L.ask."))
        assert value == pytest.approx((_TREE[answer][1] + 8.0) / 2)

def test_an_action_the_opponent_never_plays_is_never_followed():
    # Regret matching zeroes an action outright, and a sampler that fell back to
    # the last index on a rounding tail would still reach it occasionally.
    table = scripted_table()
    table["J"].cumulative_regret[:] = [1.0, 1.0, 0.0]
    rng = np.random.default_rng(0)
    for _ in range(2_000):
        start = ScriptedState("L")
        traverse(start, table, 0, rng, 1)
        assert "L.ask.e" not in start.log

# ---------------------------------------------------------------------------
# Training on Leduc: the weights, the seed, the chunks
# ---------------------------------------------------------------------------

def test_every_leduc_bank_has_one_of_the_two_weight_shapes():
    # Over whole iterations, every update anywhere is either a traverser's
    # (1, 0) or an opponent's (0, t) with zero utilities — the gotcha pinned at
    # every ledger, not just one spot. Iteration t runs at weight t.
    table = {key: SpyLedger(ledger.n_actions) for key, ledger in new_leduc_table().items()}
    solve = leduc_solve(5, table=table)
    for t in (1, 2, 3):
        before = {key: len(ledger.banked) for key, ledger in table.items()}
        run_iteration(solve)
        new = [
            bank
            for key, ledger in table.items()
            for bank in ledger.banked[before[key]:]
        ]
        assert new
        for utilities, regret_weight, strategy_weight in new:
            assert (regret_weight, strategy_weight) in ((1.0, 0.0), (0.0, float(t)))
            if regret_weight == 0.0:
                assert not utilities.any()
    assert solve.iteration == 3

def test_training_is_reproducible_at_a_seed():
    first, second = train(leduc_solve(3), 200), train(leduc_solve(3), 200)
    assert ledgers_equal(first.table, second.table)
    other = train(leduc_solve(4), 200)
    assert not ledgers_equal(first.table, other.table)

def test_chunked_training_equals_straight_training():
    chunked = train(train(leduc_solve(11), 10), 10)
    straight = train(leduc_solve(11), 20)
    assert chunked.iteration == straight.iteration == 20
    assert ledgers_equal(chunked.table, straight.table)

def test_zero_iterations_are_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        train(leduc_solve(0), 0)

def test_every_leduc_ledger_is_reached():
    # Reached means banked at all. A nonzero STRATEGY sum is the wrong test:
    # that is banked only when the owner is sampled into the spot, and a spot
    # behind the owner's own zero-probability action is never sampled — its
    # own-reach weight is genuinely 0 (8 of the 288 at this seed and length).
    table = {key: SpyLedger(ledger.n_actions) for key, ledger in new_leduc_table().items()}
    train(leduc_solve(2, table=table), 2_000)
    assert all(ledger.banked for ledger in table.values())

# ---------------------------------------------------------------------------
# Convergence on Leduc, against the LP referee
# ---------------------------------------------------------------------------

def grade(solve: Solve) -> tuple[float, float]:
    """(exploitability, P0's game value) of a solve's average strategy."""
    strategies = leduc_average_strategy(solve.table)
    return exploitability(strategies), expected_value(strategies)[0]

# Calibrated against this traverse() on 2026-09-29, seeds 1-3 (exploitability /
# |value - LP|): 10k 0.18-0.40 / 0.0004-0.012; 50k 0.15-0.26 / 0.0004-0.0077;
# 200k 0.10-0.13 / 0.0001-0.0034; 1M 0.046-0.055 / 0.0003-0.0014. Seed 1 is the
# slowest of the three at 50k, so the default gate runs on the worst case.
#
# What these gates do NOT do at 50k is catch a double-weighted walk. A walk
# that threads the opponent's reach into the regret weight measured 0.24-0.34
# at 50k with a value gap of 0.0002-0.0054 — inside every bound here. The spy
# tests above are what catch it in the default run; only the million-iteration
# run below separates the two by convergence (the mutant stalls at 0.14-0.20).
# Threading the CHANCE reach in is not detectable at all on either game, and
# not wrong in effect: every chance probability at one public point is the
# same number, so it rescales a ledger's regrets uniformly and regret matching
# reads only their ratios.

@pytest.fixture(scope="module")
def seed_one_trajectory():
    solve = leduc_solve(1)
    early = grade(train(solve, 10_000))
    late = grade(train(solve, 40_000))
    return early, late

def test_leduc_game_value_approaches_the_lp_value(seed_one_trajectory):
    # Measured 0.0004 at seed 1; 0.0077 is the widest of three seeds.
    _, (_, value) = seed_one_trajectory
    assert abs(value - LP_VALUE) < 0.02

def test_leduc_exploitability_falls_under_sampling(seed_one_trajectory):
    # Measured 0.404 -> 0.256. The bound is a smoke test, not a target: rung 2's
    # vanilla walk reaches 0.011 in 2,000 full-tree iterations, and a sampled
    # walk is never judged against that curve.
    (early, _), (late, _) = seed_one_trajectory
    assert late < early
    assert late < 0.35

@pytest.mark.skipif(
    not FULL_CALIBRATION,
    reason="a million Leduc iterations is ~5.5 min; set MINIDRAWMAHA_FULL_CALIBRATION=1",
)
def test_a_million_iterations_separate_the_walk_from_a_double_weighted_one():
    # Correct walk at 1M: 0.046-0.055 over seeds 1-3. Opponent-reach mutant:
    # 0.144-0.200. The bound sits between them with room on both sides.
    solve = leduc_solve(1)
    start, _ = grade(train(solve, 2_000))
    final, value = grade(train(solve, 998_000))
    assert final < 0.10
    assert final < start / 5
    assert abs(value - LP_VALUE) < 0.005

# ---------------------------------------------------------------------------
# Mini-drawmaha: the draw, and a root that is a chance node
# ---------------------------------------------------------------------------

class LazyTable(dict):
    """A ledger table that allocates on first touch — a test double only.

    The real table refuses unknown keys on purpose; this one exists so the
    walk can run on mini-drawmaha without paying 1.77 GB for keys it will
    mostly never reach.
    """

    def __missing__(self, infoset: InfoSet) -> RegretMatcher:
        ledger = self[infoset] = SpyLedger(len(infoset.legal_actions()))
        return ledger

def test_draw_spots_bank_four_wide_utilities():
    table = LazyTable()
    solve = train(new_solve(0, table=table, deal=random_deal), 20)
    draws = [key for key in table if key.is_draw_decision()]
    assert draws
    assert all(table[key].n_actions == len(DRAW_ACTIONS) == 4 for key in draws)
    assert {key.player for key in table} == {0, 1}
    for ledger in table.values():
        for utilities, _, _ in ledger.banked:
            assert np.all(np.isfinite(utilities))
    assert solve.iteration == 20

def test_mini_drawmahas_root_is_dealt_by_the_deck_first():
    # `random_deal` hands back a state with no board card: the walk's first act
    # is a chance sample, so the first key it reaches already holds one card.
    table = LazyTable()
    train(new_solve(0, table=table, deal=random_deal), 1)
    assert all(len(key.board) >= 1 for key in table)

@pytest.mark.skipif(
    not FULL_TABLE,
    reason="the whole 3.1M-ledger table is 1.77 GB; set MINIDRAWMAHA_FULL_TABLE=1",
)
def test_one_iteration_on_the_real_table():
    solve = train(new_solve(0), 1)
    changed = sum(
        1
        for ledger in solve.table.values()
        if ledger.strategy_sum.any() or ledger.cumulative_regret.any()
    )
    assert 0 < changed <= 200

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def test_save_load_round_trip_is_exact(tmp_path):
    solve = train(leduc_solve(9), 100)
    path = tmp_path / "solve.npz"
    save_solve(solve, path)
    loaded = load_solve(path, table=new_leduc_table(), deal=leduc_deal)
    assert ledgers_equal(solve.table, loaded.table)
    assert (loaded.iteration, loaded.seed) == (100, 9)
    assert loaded.rng.bit_generator.state == solve.rng.bit_generator.state

def test_a_resumed_solve_matches_an_uninterrupted_one(tmp_path):
    path = tmp_path / "solve.npz"
    save_solve(train(leduc_solve(9), 100), path)
    resumed = train(load_solve(path, table=new_leduc_table(), deal=leduc_deal), 100)
    straight = train(leduc_solve(9), 200)
    assert resumed.iteration == 200
    assert ledgers_equal(resumed.table, straight.table)

def test_load_refuses_a_table_of_another_shape(tmp_path):
    path = tmp_path / "solve.npz"
    save_solve(train(leduc_solve(9), 10), path)
    table = new_leduc_table()
    last = list(table)[-1]
    table[last] = RegretMatcher(table[last].n_actions + 1)
    with pytest.raises(ValueError, match="width"):
        load_solve(path, table=table, deal=leduc_deal)

def test_load_refuses_a_table_of_another_size(tmp_path):
    path = tmp_path / "solve.npz"
    save_solve(train(leduc_solve(9), 10), path)
    table = new_leduc_table()
    del table[list(table)[-1]]
    with pytest.raises(ValueError, match="ledgers"):
        load_solve(path, table=table, deal=leduc_deal)
