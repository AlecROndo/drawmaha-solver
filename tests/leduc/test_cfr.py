import numpy as np
import pytest

from drawmaha_solver.leduc.cfr import run_iteration, train, walk
from drawmaha_solver.leduc.game import (
    DEAL_PROBABILITY,
    DEALS,
    Action,
    DECK,
    InfoSet,
    LeducState,
    Rank,
)
from drawmaha_solver.leduc.infoset_table import average_strategy, new_infoset_table
from drawmaha_solver.regret_matching import RegretMatcher

J, Q, K = Rank.JACK, Rank.QUEEN, Rank.KING
Ja, Jb, Qa, Qb, Ka, Kb = DECK
F, C, R = Action.FOLD, Action.CALL, Action.RAISE

# The referee's ground truth: the sequence-form LP value of Leduc to P0, and
# OpenSpiel CFR's own distance from it along its convergence curve
# (figures/cfr_trace.json). Its updates alternate and ours are simultaneous,
# so the iterates differ; only the destination and the rough pace are shared.
LP_VALUE_P0 = -0.0856064

def expected_value_p0(strategies) -> float:
    """P0's expected chips under `strategies`, computed without `cfr.walk`.

    A test-local evaluator over game.py alone — a plain sigma-weighted
    expectation with no reaches, no banking, no ledgers — so a bug in the
    walk cannot grade itself.
    """

    def value(state) -> float:
        if state.is_terminal():
            return state.returns()[0]
        if state.is_chance_node():
            return sum(
                probability * value(state.apply_chance(card))
                for card, probability in state.chance_outcomes()
            )
        sigma = strategies[state.infoset()]
        return sum(
            sigma[k] * value(state.apply(action))
            for k, action in enumerate(state.legal_actions())
        )

    return DEAL_PROBABILITY * sum(value(LeducState(cards=deal)) for deal in DEALS)

class SpyLedger(RegretMatcher):
    """A RegretMatcher that records the weights every bank arrives with."""

    def __init__(self, n_actions: int):
        super().__init__(n_actions)
        self.banked: list[tuple[float, float]] = []

    def update(self, utilities, *, regret_weight=1.0, strategy_weight=1.0):
        self.banked.append((regret_weight, strategy_weight))
        super().update(
            utilities, regret_weight=regret_weight, strategy_weight=strategy_weight
        )

# ---------------------------------------------------------------------------
# One traversal: values, reach, and the two chance receipts
# ---------------------------------------------------------------------------

def test_a_traversal_returns_zero_sum_values_to_both_seats():
    values = walk(
        LeducState(cards=(Ja, Qa)), new_infoset_table(), (1.0, 1.0), DEAL_PROBABILITY
    )
    assert values[0] == pytest.approx(-values[1], abs=1e-12)

def test_the_walk_leaves_the_state_alone():
    # LeducState is frozen; pin that the walk descends by building children
    # rather than mutating a cursor.
    root = LeducState(cards=(Ja, Qa))
    walk(root, new_infoset_table(), (1.0, 1.0), DEAL_PROBABILITY)
    assert root.betting == ((),)
    assert root.board is None

def test_round_one_banks_the_deal_receipt_and_round_two_the_board_receipt():
    # THE rung-2 delta, pinned at the two spots where every other factor of
    # the weight is known exactly. P0's opening spot is banked before anyone
    # has acted, so its regret weight is the bare deal probability, 1/30. The
    # round-2 opening spot after check-check on the jack board is reached with
    # both players having checked once at uniform sigma (updates land on the
    # unwind, so the walk read those probabilities before any ledger moved):
    # regret weight = pi_c * opponent reach = (1/30 * 1/4) * 1/2 = 1/240, and
    # the strategy weight is P0's own reach, 1/2 — chance deliberately absent.
    table = new_infoset_table()
    open_spot = InfoSet(rank=J, board=None, betting=((),))
    board_spot = InfoSet(rank=J, board=J, betting=((C, C), ()))
    table[open_spot] = SpyLedger(2)
    table[board_spot] = SpyLedger(2)

    walk(LeducState(cards=(Ja, Qa)), table, (1.0, 1.0), DEAL_PROBABILITY)

    assert table[open_spot].banked == [pytest.approx((1 / 30, 1.0))]
    assert table[board_spot].banked == [pytest.approx((1 / 240, 0.5))]

def test_a_chance_node_advances_no_reach():
    # The deck is not a player: crossing the board node must not scale either
    # player's own-reach. The round-2 spot's strategy weight above is 1/2 —
    # P0's single check — not 1/2 * 1/4; this test pins the complement, that
    # the receipt lands in the regret weight ONLY.
    table = new_infoset_table()
    spot = InfoSet(rank=J, board=Q, betting=((C, C), ()))
    table[spot] = SpyLedger(2)
    walk(LeducState(cards=(Ja, Qa)), table, (1.0, 1.0), DEAL_PROBABILITY)
    (regret_weight, strategy_weight) = table[spot].banked[0]
    assert strategy_weight == pytest.approx(0.5)
    assert regret_weight == pytest.approx((1 / 120) * 0.5)

def test_a_traversal_only_touches_the_infosets_of_its_deal():
    # A walk of (Ja, Qa) that banks a king-holding spot is reading a card
    # nobody was dealt. The expected set is enumerated independently of the
    # walk, straight off the game tree.
    expected = set()

    def collect(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for card, _ in state.chance_outcomes():
                collect(state.apply_chance(card))
            return
        expected.add(state.infoset())
        for action in state.legal_actions():
            collect(state.apply(action))

    collect(LeducState(cards=(Ja, Qa)))

    table = new_infoset_table()
    walk(LeducState(cards=(Ja, Qa)), table, (1.0, 1.0), DEAL_PROBABILITY)
    touched = {
        infoset
        for infoset, ledger in table.items()
        if ledger.strategy_sum.any() or ledger.cumulative_regret.any()
    }
    assert touched == expected

def test_a_full_iteration_visits_every_ledger_once_per_hidden_world():
    # 30 deals cover all 288 spots, and each ledger is banked once per node
    # of its infoset — the referee's 10/16/8 histogram, seen from the walk's
    # side: 3,780 update calls per iteration in total. Counted with spies
    # rather than by looking for nonzero sums, because a bank whose weights
    # are both zero (the player's own reach can hit exactly 0 mid-iteration
    # once in-place updates turn a sigma pure) is still a visit.
    table = new_infoset_table()
    for infoset in table:
        table[infoset] = SpyLedger(len(infoset.legal_actions()))
    run_iteration(table)
    visits = [len(ledger.banked) for ledger in table.values()]
    assert min(visits) > 0
    assert sum(visits) == 3780
    histogram = {n: visits.count(n) for n in set(visits)}
    assert histogram == {10: 18, 16: 180, 8: 90}

# ---------------------------------------------------------------------------
# Training mechanics
# ---------------------------------------------------------------------------

def test_training_is_deterministic():
    # Vanilla CFR enumerates the whole tree, so there is no sampling and no
    # seed: two runs must agree to the bit.
    first, second = average_strategy(train(20)), average_strategy(train(20))
    for infoset, probabilities in first.items():
        assert np.array_equal(probabilities, second[infoset])

def test_train_can_continue_an_existing_table():
    table = train(10)
    spot = InfoSet(rank=K, board=None, betting=((),))
    banked = table[spot].strategy_sum.sum()
    train(10, table=table)
    assert table[spot].strategy_sum.sum() > banked

def test_zero_iterations_are_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        train(0)

# ---------------------------------------------------------------------------
# Convergence toward the referee's number
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solved():
    return average_strategy(train(300))

def test_the_game_value_approaches_the_lp_value(solved):
    # The referee's own solver sits 0.007 from the LP value at T=300 on its
    # alternating-update curve; ours updates simultaneously, so hold it to a
    # band of the same order rather than to the iterate.
    assert expected_value_p0(solved) == pytest.approx(LP_VALUE_P0, abs=0.02)

def test_more_iterations_move_the_game_value_closer(solved):
    short = expected_value_p0(average_strategy(train(20)))
    long = expected_value_p0(solved)
    assert abs(long - LP_VALUE_P0) < abs(short - LP_VALUE_P0)

def test_the_solve_finds_leducs_strategic_shape(solved):
    # Three facts pinned by dominance, not folklore. Holding J on a J board
    # is the nuts (the other jack IS the board, so no hand beats it): facing
    # a raise, folding it is dominated and the solve raises back nearly pure.
    # The unpaired jack on a king board facing a raise is beaten by
    # everything that isn't bluffing and folds. And note where the nuts'
    # aggression lives: NOT at the check-check open — the solve slowplays
    # there (checking induces bluffs; leading folds worse hands out) — but
    # one node deeper, as a check-raise. Column order at these 3-wide spots
    # is (fold, call, raise).
    nuts_raised = solved[InfoSet(rank=J, board=J, betting=((C, C), (C, R)))]
    trash_raised = solved[InfoSet(rank=J, board=K, betting=((C, C), (R,)))]
    open_k = solved[InfoSet(rank=K, board=None, betting=((),))]
    assert nuts_raised[0] < 0.02  # never fold the nuts
    assert nuts_raised[2] > 0.9  # the check-raise
    assert trash_raised[0] > 0.9  # fold the beaten jack
    assert open_k[1] > 0.4  # (call, raise) at the open: the king bets
