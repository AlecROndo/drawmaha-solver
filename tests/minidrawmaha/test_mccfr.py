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
  and for one iteration on the real 3.4 GB table behind
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
from drawmaha_solver.minidrawmaha.game import (
    DRAW_ACTIONS,
    Action,
    InfoSet,
    random_deal,
)
from drawmaha_solver.minidrawmaha import regret_rules
from drawmaha_solver.minidrawmaha.mccfr import (
    Solve,
    _pick,
    load_solve,
    new_solve,
    run_iteration,
    save_solve,
    train,
    traverse,
)
from drawmaha_solver.minidrawmaha.regret_rules import (
    Average,
    RegretRule,
    bank_vanilla,
    extra_weights,
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

BOTH_COLUMNS = (Average.UNIFORM, Average.QUADRATIC)

def new_leduc_table_with(extra_averages: int) -> dict:
    """Leduc's 288 ledgers, each carrying `extra_averages` averaging rows."""
    return {
        key: RegretMatcher(ledger.n_actions, extra_averages=extra_averages)
        for key, ledger in new_leduc_table().items()
    }

def leduc_solve(seed: int, *, table=None, rule=RegretRule.VANILLA, averages=()) -> Solve:
    return new_solve(
        seed,
        table=new_leduc_table_with(len(averages)) if table is None else table,
        deal=leduc_deal,
        rule=rule,
        averages=averages,
    )

@dataclass
class SpyBank:
    """A regret rule that records every bank it is handed, then banks it the vanilla way."""

    banked: list[tuple[RegretMatcher, np.ndarray, int]] = field(default_factory=list)

    def __call__(self, ledger: RegretMatcher, regret: np.ndarray, t: int) -> None:
        self.banked.append((ledger, regret.copy(), t))
        bank_vanilla(ledger, regret, t)

@pytest.fixture
def spy(monkeypatch) -> SpyBank:
    """Stand a SpyBank in for vanilla, so whole iterations bank through it."""
    bank = SpyBank()
    monkeypatch.setitem(regret_rules.RULES, RegretRule.VANILLA, bank)
    return bank

def ledgers_equal(first, second) -> bool:
    """Every slot of every ledger identical, bit for bit, in the same key order."""
    return list(first) == list(second) and all(
        first[key].cumulative_regret.tobytes() == second[key].cumulative_regret.tobytes()
        and first[key].strategy_sum.tobytes() == second[key].strategy_sum.tobytes()
        and first[key].extra_sums.tobytes() == second[key].extra_sums.tobytes()
        and first[key].stamp.tobytes() == second[key].stamp.tobytes()
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

def scripted_table(extra_averages: int = 0) -> dict[str, RegretMatcher]:
    return {
        key: RegretMatcher(width, extra_averages=extra_averages)
        for key, width in (("I-left", 2), ("J", 3), ("I-right", 2))
    }

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

def test_the_traverser_hands_the_rule_each_actions_regret_and_banks_no_strategy():
    table, spy = scripted_table(), SpyBank()
    value = traverse(ScriptedState("L"), table, 0, np.random.default_rng(3), 7, bank=spy)
    ((ledger, regret, t),) = spy.banked
    assert (ledger, t) == (table["I-left"], 7)
    # The regret is what each action earned minus the mix's value: "stop" pays
    # 8; "ask" pays whichever answer P1 was sampled into.
    utilities = regret + value
    assert utilities[1] == 8.0
    assert utilities[0] in (1.0, 2.0, 4.0)
    assert not table["I-left"].strategy_sum.any()

def test_regret_weight_is_one_however_rarely_the_spot_is_reached():
    # The gotcha, on a tree small enough to name the wrong answer: with P1
    # traversing, J sits behind the deck's 0.25 and P0's uniform 0.5, so a walk
    # threading reach down would bank it at 0.125. Sampling already charged
    # that probability — J is simply banked on one traversal in eight, and each
    # time at full weight: P1's answers pay −1, −2, −4 against a uniform mix.
    answers = np.array([-1.0, -2.0, -4.0])
    full_weight = answers - np.full(3, 1 / 3) @ answers
    banks = 0
    for seed in range(64):
        table, spy = scripted_table(), SpyBank()
        traverse(ScriptedState("root"), table, 1, np.random.default_rng(seed), 1, bank=spy)
        for ledger, regret, _ in spy.banked:
            if ledger is table["J"]:
                banks += 1
                assert regret.tobytes() == full_weight.tobytes()
                assert table["J"].cumulative_regret.tobytes() == full_weight.tobytes()
    assert banks

def test_the_opponents_bank_carries_the_iteration_and_no_regret():
    table, spy = scripted_table(), SpyBank()
    traverse(ScriptedState("L"), table, 0, np.random.default_rng(3), 7, bank=spy)
    assert all(ledger is not table["J"] for ledger, _, _ in spy.banked)
    assert table["J"].strategy_sum.tobytes() == (7 * np.full(3, 1 / 3)).tobytes()
    assert not table["J"].cumulative_regret.any()

def test_the_extra_columns_bank_the_same_mix_at_their_own_weights():
    table = scripted_table(extra_averages=2)
    weights = extra_weights(BOTH_COLUMNS, 7)
    traverse(ScriptedState("L"), table, 0, np.random.default_rng(3), 7, column_weights=weights)
    mix = np.full(3, 1 / 3)
    np.testing.assert_array_equal(table["J"].extra_sums, [mix, 49 * mix])
    # The traverser's own spot was enumerated, not played: no column banks it.
    assert not table["I-left"].extra_sums.any()

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

def test_every_leduc_traversal_banks_regret_or_strategy_and_never_both():
    # Over whole traversals, every ledger touched is either the traverser's
    # (regret handed to the rule, strategy sum untouched) or the opponent's
    # (t·σ into the strategy sum, regret untouched) — the gotcha pinned at
    # every ledger, not just one spot. Per traversal, not per iteration: a
    # ledger is banked both ways in one iteration, once from each seat.
    # Iteration t runs at weight t.
    table, rng = new_leduc_table(), np.random.default_rng(5)
    for t in (1, 2, 3):
        for traverser in (0, 1):
            before = {
                key: (ledger.cumulative_regret.copy(), ledger.strategy_sum.copy(), ledger.strategy())
                for key, ledger in table.items()
            }
            spy = SpyBank()
            traverse(leduc_deal(rng), table, traverser, rng, t, bank=spy)
            regret_banked = {id(ledger) for ledger, _, _ in spy.banked}
            assert regret_banked and all(bank_t == t for _, _, bank_t in spy.banked)
            opponents = 0
            for key, ledger in table.items():
                regret_before, sum_before, sigma = before[key]
                if id(ledger) in regret_banked:
                    assert ledger.strategy_sum.tobytes() == sum_before.tobytes()
                elif not np.array_equal(ledger.strategy_sum, sum_before):
                    opponents += 1
                    np.testing.assert_allclose(ledger.strategy_sum - sum_before, t * sigma)
                    assert ledger.cumulative_regret.tobytes() == regret_before.tobytes()
            assert opponents

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

def test_every_leduc_ledger_is_reached(spy):
    # Reached means banked at all. A nonzero STRATEGY sum alone is the wrong
    # test: that is banked only when the owner is sampled into the spot, and a
    # spot behind the owner's own zero-probability action is never sampled —
    # its own-reach weight is genuinely 0 (8 of the 288 at this seed and
    # length). Those are reached as the traverser, and handed to the rule.
    solve = train(leduc_solve(2), 2_000)
    regret_banked = {id(ledger) for ledger, _, _ in spy.banked}
    assert all(
        id(ledger) in regret_banked or ledger.strategy_sum.any()
        for ledger in solve.table.values()
    )

# ---------------------------------------------------------------------------
# The rules on the walk: today's learner bit for bit, and the other three
# ---------------------------------------------------------------------------

def reference_traverse(state, table, traverser: int, rng, t: int) -> float:
    """The walk as it stood before regret rules: every bank through `RegretMatcher.update`.

    Kept verbatim as the regression oracle. The walk now banks straight into
    the ledger slots through a rule; under vanilla with no extra columns that
    must be the same floating-point arithmetic `update` did at weights (1, 0)
    and (0, t), so the two produce identical bytes, not merely close numbers.
    """
    if state.is_terminal():
        return state.returns()[traverser]
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        picked = _pick(rng, [probability for _, probability in outcomes])
        return reference_traverse(state.apply_chance(outcomes[picked][0]), table, traverser, rng, t)
    ledger = table[state.infoset()]
    actions = state.legal_actions()
    sigma = ledger.strategy()
    if state.current_player == traverser:
        utilities = np.array(
            [reference_traverse(state.apply(action), table, traverser, rng, t) for action in actions]
        )
        ledger.update(utilities, regret_weight=1.0, strategy_weight=0.0)
        return float(sigma @ utilities)
    ledger.update(np.zeros(len(actions)), regret_weight=0.0, strategy_weight=float(t))
    return reference_traverse(state.apply(actions[_pick(rng, sigma)]), table, traverser, rng, t)

def reference_train(solve: Solve, iterations: int) -> Solve:
    for _ in range(iterations):
        t = solve.iteration + 1
        for traverser in (0, 1):
            reference_traverse(solve.deal(solve.rng), solve.table, traverser, solve.rng, t)
        solve.iteration = t
    return solve

@pytest.mark.parametrize(
    ("make", "iterations"),
    [
        (lambda: leduc_solve(3), 2_000),
        (lambda: new_solve(0, table=LazyTable(), deal=random_deal), 50),
    ],
    ids=["leduc", "minidrawmaha"],
)
def test_vanilla_without_columns_is_the_learner_before_rules_bit_for_bit(make, iterations):
    ruled, reference = train(make(), iterations), reference_train(make(), iterations)
    assert ledgers_equal(ruled.table, reference.table)
    assert ruled.rng.bit_generator.state == reference.rng.bit_generator.state

@pytest.mark.parametrize("rule", list(RegretRule))
def test_every_rule_is_reproducible_and_trains_the_same_in_chunks(rule):
    straight = train(leduc_solve(11, rule=rule, averages=BOTH_COLUMNS), 60)
    again = train(leduc_solve(11, rule=rule, averages=BOTH_COLUMNS), 60)
    chunked = train(train(leduc_solve(11, rule=rule, averages=BOTH_COLUMNS), 25), 35)
    assert chunked.iteration == 60
    assert ledgers_equal(straight.table, again.table)
    assert ledgers_equal(straight.table, chunked.table)

def test_the_four_rules_train_four_different_tables():
    tables = [train(leduc_solve(11, rule=rule), 60).table for rule in RegretRule]
    for first in range(4):
        for second in range(first + 1, 4):
            assert not ledgers_equal(tables[first], tables[second])

def test_the_rules_leave_their_marks():
    # CFR+ never holds a negative regret; DCFR stamps every row it banks, with
    # an iteration it has run; the others never stamp.
    for rule in RegretRule:
        table = train(leduc_solve(11, rule=rule), 60).table
        regrets = np.concatenate([ledger.cumulative_regret for ledger in table.values()])
        stamps = np.array([ledger.stamp[0] for ledger in table.values()])
        if rule is RegretRule.CFR_PLUS:
            assert regrets.min() == 0.0
        else:
            assert regrets.min() < 0.0
        if rule is RegretRule.DCFR:
            banked = [ledger.stamp[0] for ledger in table.values() if ledger.cumulative_regret.any()]
            assert banked and all(1 <= stamp <= 60 for stamp in banked)
        else:
            assert not stamps.any()

def test_at_iteration_one_every_column_banks_the_same_numbers():
    # Weights 1, t and t² all equal 1 at t = 1, so one iteration pins that the
    # columns bank the primary's very strategies, in the primary's places.
    solve = train(leduc_solve(4, averages=BOTH_COLUMNS), 1)
    for ledger in solve.table.values():
        for row in ledger.extra_sums:
            assert row.tobytes() == ledger.strategy_sum.tobytes()

def test_a_run_refuses_a_rule_or_column_it_does_not_know():
    with pytest.raises(ValueError, match="'rm-plus'"):
        leduc_solve(0, rule="rm-plus")
    with pytest.raises(ValueError, match="'cubic'"):
        leduc_solve(0, averages=("cubic",))

def test_a_run_refuses_a_table_without_a_row_per_column():
    # A table one row short would crash on the first bank; one with a row too
    # many would broadcast a single column's weights into both rows silently.
    with pytest.raises(ValueError, match="1 extra averaging row"):
        leduc_solve(0, table=new_leduc_table_with(1), averages=BOTH_COLUMNS)
    with pytest.raises(ValueError, match="2 extra averaging rows"):
        leduc_solve(0, table=new_leduc_table_with(2), averages=(Average.UNIFORM,))

def test_rules_and_columns_are_accepted_by_name():
    solve = leduc_solve(0, rule="dcfr", averages=("uniform", "quadratic"))
    assert solve.rule is RegretRule.DCFR
    assert solve.averages == BOTH_COLUMNS

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
# The gates follow ONE deterministic trajectory, so they are not flaky, but
# they are tied to it: anything that changes how the walk consumes the RNG (a
# change to `_pick`, the traversal order, the deal) moves seed 1 onto another
# trajectory and can trip a gate with no real regression. Re-measure the three
# seeds before reading a failure after such a change as a broken walk.
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
    reason="a million Leduc iterations is ~4.5 min; set MINIDRAWMAHA_FULL_CALIBRATION=1",
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
    walk can run on mini-drawmaha without paying 3.4 GB for keys it will
    mostly never reach.
    """

    def __missing__(self, infoset: InfoSet) -> RegretMatcher:
        ledger = self[infoset] = RegretMatcher(len(infoset.legal_actions()))
        return ledger

def test_draw_spots_bank_four_wide_regrets(spy):
    table = LazyTable()
    solve = train(new_solve(0, table=table, deal=random_deal), 20)
    draws = [key for key in table if key.is_draw_decision()]
    assert draws
    assert all(table[key].n_actions == len(DRAW_ACTIONS) == 4 for key in draws)
    assert {key.player for key in table} == {0, 1}
    draw_ledgers = {id(table[key]) for key in draws}
    assert any(id(ledger) in draw_ledgers for ledger, _, _ in spy.banked)
    for ledger, regret, _ in spy.banked:
        assert regret.shape == (ledger.n_actions,)
        assert np.all(np.isfinite(regret))
    assert solve.iteration == 20

def test_mini_drawmahas_root_is_dealt_by_the_deck_first():
    # `random_deal` hands back a state with no board card: the walk's first act
    # is a chance sample, so the first key it reaches already holds one card.
    table = LazyTable()
    train(new_solve(0, table=table, deal=random_deal), 1)
    assert all(len(key.board) >= 1 for key in table)

def test_mini_drawmahas_deck_deals_one_probability_per_public_point():
    # The claim that makes threading the chance reach inert: at one public
    # point every world sees the same chance probabilities, because the deck
    # deals combinations of the stub uniformly and the stub's size is public.
    # Many deals walked down one public line — call every bet, throw one card
    # at every draw — must meet the same probabilities at every chance node.
    rng = np.random.default_rng(0)
    lines = set()
    for _ in range(200):
        state, line = random_deal(rng), []
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                probabilities = {probability for _, probability in outcomes}
                assert len(probabilities) == 1
                line.append((len(outcomes), probabilities.pop()))
                state = state.apply_chance(outcomes[rng.integers(len(outcomes))][0])
            elif state.is_draw_decision():
                state = state.apply(Action.THROW_LOW)
            else:
                state = state.apply(Action.CHECK_CALL)
        lines.add(tuple(line))
    assert len(lines) == 1
    assert len(next(iter(lines))) >= 4

@pytest.mark.skipif(
    not FULL_TABLE,
    reason="the whole 6.2M-ledger table is 3.4 GB; set MINIDRAWMAHA_FULL_TABLE=1",
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

def test_save_refuses_an_empty_table(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_solve(new_solve(0, table={}, deal=leduc_deal), tmp_path / "solve.npz")

def test_save_refuses_a_stream_it_cannot_restore(tmp_path):
    solve = leduc_solve(9)
    solve.rng = np.random.Generator(np.random.Philox(9))
    with pytest.raises(ValueError, match="PCG64"):
        save_solve(solve, tmp_path / "solve.npz")

def test_load_refuses_an_unrestorable_stream_before_touching_the_table(tmp_path):
    path = tmp_path / "solve.npz"
    save_solve(train(leduc_solve(9), 10), path)
    with np.load(path) as saved:
        arrays = dict(saved)
    state = json.loads(str(arrays["rng_state"]))
    state["bit_generator"] = "Philox"
    arrays["rng_state"] = np.str_(json.dumps(state))
    np.savez(path, **arrays)
    table = new_leduc_table()
    with pytest.raises(ValueError, match="PCG64"):
        load_solve(path, table=table, deal=leduc_deal)
    assert not any(
        ledger.cumulative_regret.any() or ledger.strategy_sum.any()
        for ledger in table.values()
    )

def test_load_refuses_a_table_of_another_size(tmp_path):
    path = tmp_path / "solve.npz"
    save_solve(train(leduc_solve(9), 10), path)
    table = new_leduc_table()
    del table[list(table)[-1]]
    with pytest.raises(ValueError, match="ledgers"):
        load_solve(path, table=table, deal=leduc_deal)

def ruled_run(iterations: int, *, rule=RegretRule.DCFR, averages=BOTH_COLUMNS) -> Solve:
    return train(leduc_solve(9, rule=rule, averages=averages), iterations)

def untouched(table) -> bool:
    return not any(
        ledger.cumulative_regret.any()
        or ledger.strategy_sum.any()
        or ledger.extra_sums.any()
        or ledger.stamp.any()
        for ledger in table.values()
    )

def test_a_ruled_run_round_trips_exactly(tmp_path):
    # Every slot: DCFR's stamps and both columns ride along with the sums.
    solve = ruled_run(100)
    path = tmp_path / "solve.npz"
    save_solve(solve, path)
    loaded = load_solve(
        path, table=new_leduc_table_with(2), deal=leduc_deal, rule="dcfr", averages=BOTH_COLUMNS
    )
    assert ledgers_equal(solve.table, loaded.table)
    assert (loaded.rule, loaded.averages, loaded.iteration) == (RegretRule.DCFR, BOTH_COLUMNS, 100)
    assert any(ledger.stamp[0] for ledger in loaded.table.values())
    assert loaded.rng.bit_generator.state == solve.rng.bit_generator.state

@pytest.mark.parametrize("rule", list(RegretRule))
def test_a_resumed_ruled_run_matches_an_uninterrupted_one(tmp_path, rule):
    path = tmp_path / "solve.npz"
    save_solve(ruled_run(50, rule=rule), path)
    resumed = train(
        load_solve(path, table=new_leduc_table_with(2), deal=leduc_deal, rule=rule, averages=BOTH_COLUMNS),
        50,
    )
    assert ledgers_equal(resumed.table, ruled_run(100, rule=rule).table)

def test_load_refuses_another_rule_before_touching_the_table(tmp_path):
    path = tmp_path / "solve.npz"
    save_solve(ruled_run(10), path)
    table = new_leduc_table_with(2)
    with pytest.raises(ValueError, match="'dcfr'.*'lcfr'"):
        load_solve(path, table=table, deal=leduc_deal, rule="lcfr", averages=BOTH_COLUMNS)
    assert untouched(table)

def test_load_refuses_other_columns_before_touching_the_table(tmp_path):
    # Order counts: the same two names in the other order would pour the
    # uniform sums into the quadratic row.
    path = tmp_path / "solve.npz"
    save_solve(ruled_run(10), path)
    table = new_leduc_table_with(2)
    with pytest.raises(ValueError, match="columns"):
        load_solve(path, table=table, deal=leduc_deal, rule="dcfr", averages=("quadratic", "uniform"))
    assert untouched(table)

def test_load_refuses_a_table_without_a_row_per_column(tmp_path):
    path = tmp_path / "solve.npz"
    save_solve(ruled_run(10), path)
    table = new_leduc_table_with(1)
    with pytest.raises(ValueError, match="1 extra averaging row"):
        load_solve(path, table=table, deal=leduc_deal, rule="dcfr", averages=BOTH_COLUMNS)
    assert untouched(table)

def test_a_checkpoint_from_before_rules_loads_as_vanilla(tmp_path):
    # A file saved before this change has no rule, columns, extra sums or
    # stamps; it was a vanilla run with none of them, and loads as one.
    path = tmp_path / "solve.npz"
    solve = train(leduc_solve(9), 100)
    save_solve(solve, path)
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files if key not in ("rule", "averages", "extra_sums", "stamps")}
    np.savez(path, **arrays)
    loaded = load_solve(path, table=new_leduc_table(), deal=leduc_deal)
    assert ledgers_equal(solve.table, loaded.table)
    assert (loaded.rule, loaded.averages) == (RegretRule.VANILLA, ())
    with pytest.raises(ValueError, match="'vanilla'.*'dcfr'"):
        load_solve(path, table=new_leduc_table(), deal=leduc_deal, rule="dcfr")
