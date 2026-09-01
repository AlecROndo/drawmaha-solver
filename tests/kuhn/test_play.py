import numpy as np
import pytest

from drawmaha_solver.kuhn.game import Action, Card, InfoSet, KuhnState
from drawmaha_solver.kuhn.infoset_table import new_infoset_table, average_strategy
from drawmaha_solver.kuhn.play import (
    QuitGame,
    Scoreboard,
    _report,
    bot_action,
    deal,
    parse_action,
    prompt_for,
)

J, Q, K = Card.JACK, Card.QUEEN, Card.KING
P, B = Action.PASS, Action.BET
UNIFORM = average_strategy(new_infoset_table())

# ---------------------------------------------------------------------------
# Reading the human's move
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "typed,history,expected",
    [
        ("check", (), P),
        ("c", (), P),
        ("bet", (), B),
        ("b", (), B),
        ("fold", (B,), P),
        ("f", (B,), P),
        ("call", (B,), B),
        ("c", (B,), B),  # 'c' is check with nothing to call and call facing one
    ],
)
def test_parse_action_is_context_sensitive(typed, history, expected):
    assert parse_action(typed, history) == expected

@pytest.mark.parametrize("typed,history", [("fold", ()), ("call", ()), ("check", (B,)), ("bet", (B,))])
def test_words_that_do_not_apply_at_this_node_are_rejected(typed, history):
    # "fold" with nothing to call is not a slip to interpret generously; the
    # player has misread the spot and should be told.
    assert parse_action(typed, history) is None

@pytest.mark.parametrize("typed", ["q", "quit", "exit"])
def test_quitting_raises(typed):
    with pytest.raises(QuitGame):
        parse_action(typed, ())

def test_garbage_is_rejected_without_raising():
    assert parse_action("banana", ()) is None

def test_the_prompt_names_the_two_real_options():
    assert prompt_for(()) == "  [c]heck / [b]et / [q]uit > "
    assert prompt_for((B,)) == "  [f]old / [c]all / [q]uit > "

# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------

def test_the_bot_only_ever_sees_its_own_infoset():
    # Both deals give the bot the king facing a bet; it must act identically,
    # because the human's card is not something it can condition on.
    strategies = dict(UNIFORM)
    strategies[InfoSet(K, (B,))] = np.array([0.0, 1.0])
    rng = np.random.default_rng(0)
    for human_card in (J, Q):
        state = KuhnState(cards=(human_card, K), history=(B,))
        assert bot_action(state, strategies, rng) is Action.BET

def test_the_bot_follows_its_mixed_strategy():
    strategies = dict(UNIFORM)
    strategies[InfoSet(K, (B,))] = np.array([0.25, 0.75])
    rng = np.random.default_rng(0)
    state = KuhnState(cards=(J, K), history=(B,))
    draws = [bot_action(state, strategies, rng) for _ in range(4_000)]
    assert np.mean([a is Action.BET for a in draws]) == pytest.approx(0.75, abs=0.03)

def test_the_bot_is_deterministic_given_a_seed():
    state = KuhnState(cards=(J, K), history=(B,))
    runs = [
        [bot_action(state, UNIFORM, np.random.default_rng(3)) for _ in range(5)]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]

# ---------------------------------------------------------------------------
# Dealing and scoring
# ---------------------------------------------------------------------------

def test_deal_gives_two_distinct_cards_in_seat_order():
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = deal(rng)
        assert state.history == ()
        assert state.cards[0] != state.cards[1]

def test_deal_covers_every_ordering():
    rng = np.random.default_rng(0)
    assert len({deal(rng).cards for _ in range(400)}) == 6

def test_the_scoreboard_tracks_chips_from_the_humans_seat():
    board = Scoreboard()
    # Human is P0 and wins 2; then human is P1 and P0 wins 1, so human is -1.
    board.record(human_seat=0, returns=(2.0, -2.0))
    board.record(human_seat=1, returns=(1.0, -1.0))
    assert board.hands == 2
    assert board.chips == pytest.approx(1.0)

def test_the_scoreboard_reports_a_per_hand_rate():
    board = Scoreboard()
    for _ in range(4):
        board.record(human_seat=0, returns=(1.0, -1.0))
    assert board.per_hand == pytest.approx(1.0)

def test_an_empty_scoreboard_has_no_rate():
    # Dividing by zero hands would print 'nan chips/hand' as if it were a
    # measurement; there is simply nothing to report yet.
    assert Scoreboard().per_hand is None

def test_the_scoreboard_splits_the_result_by_seat():
    # The total is the only number that should approach zero; each seat
    # separately approaches its own game value, so they are banked apart.
    board = Scoreboard()
    board.record(human_seat=0, returns=(-1.0, 1.0))
    board.record(human_seat=1, returns=(-2.0, 2.0))
    board.record(human_seat=0, returns=(1.0, -1.0))
    assert board.seat_chips == [0.0, 2.0]
    assert board.seat_hands == [2, 1]
    assert board.chips == pytest.approx(2.0)

def test_a_seat_that_has_not_played_has_no_rate():
    board = Scoreboard()
    board.record(human_seat=0, returns=(1.0, -1.0))
    assert board.per_hand_in_seat(0) == pytest.approx(1.0)
    assert board.per_hand_in_seat(1) is None

def test_the_report_shows_both_seats_against_their_own_targets(capsys):
    # Reading a P1 result against 0.000 would call an equilibrium player
    # +1/18 ahead a winner, which is the confusion the split exists to stop.
    board = Scoreboard()
    board.record(human_seat=0, returns=(-1.0, 1.0))
    board.record(human_seat=1, returns=(-1.0, 1.0))
    _report(board, UNIFORM)
    out = capsys.readouterr().out
    assert "as P0: -1 over 1 hands (-1.000 per hand, equilibrium -0.056)" in out
    assert "as P1: +1 over 1 hands (+1.000 per hand, equilibrium +0.056)" in out
