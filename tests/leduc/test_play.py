import numpy as np
import pytest

from drawmaha_solver.leduc.game import DEALS, DECK, Action, Card, LeducState, Rank
from drawmaha_solver.leduc.infoset_table import average_strategy, new_infoset_table
from drawmaha_solver.leduc.play import (
    QuitGame,
    Scoreboard,
    _report,
    bot_action,
    deal,
    parse_action,
    prompt_for,
    turn_board,
    verdict,
)

F, C, R = Action.FOLD, Action.CALL, Action.RAISE
J, Q, K = Rank.JACK, Rank.QUEEN, Rank.KING
Ja, Jb, Qa, Qb, Ka, Kb = DECK
UNIFORM = average_strategy(new_infoset_table())

# ---------------------------------------------------------------------------
# Reading the human's move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "typed,line,expected",
    [
        ("check", (), C),
        ("c", (), C),
        ("bet", (), R),
        ("b", (), R),
        ("fold", (R,), F),
        ("f", (R,), F),
        ("call", (R,), C),
        ("c", (R,), C),  # 'c' is check with nothing to call, call facing a raise
        ("raise", (R,), R),
        ("r", (R,), R),
    ],
)
def test_parse_action_is_context_sensitive(typed, line, expected):
    assert parse_action(typed, line) == expected


@pytest.mark.parametrize(
    "typed,line",
    [
        ("fold", ()),  # nothing to fold to
        ("call", ()),  # nothing to call
        ("check", (R,)),  # cannot check facing a raise
        ("bet", (R,)),  # it is a raise here, not a bet
        ("raise", (R, R)),  # the cap is two raises a round
    ],
)
def test_words_that_do_not_apply_at_this_node_are_rejected(typed, line):
    # "fold" with nothing to call is not a slip to interpret generously; the
    # player has misread the spot and should be told.
    assert parse_action(typed, line) is None


@pytest.mark.parametrize("typed", ["q", "quit", "exit"])
def test_quitting_raises(typed):
    with pytest.raises(QuitGame):
        parse_action(typed, ())


def test_garbage_is_rejected_without_raising():
    assert parse_action("banana", ()) is None


def test_the_prompt_names_exactly_the_legal_options():
    # Two wide at the open, three facing a raise — the prompt has to track the
    # spot, unlike rung 1 where every node offered the same two words.
    assert prompt_for(()) == "  [c]heck / [b]et / [q]uit > "
    assert prompt_for((R,)) == "  [f]old / [c]all / [r]aise / [q]uit > "
    assert prompt_for((R, R)) == "  [f]old / [c]all / [q]uit > "


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------


def test_the_bot_only_ever_sees_its_own_infoset():
    # Both deals give the bot a king facing a bet; it must act identically,
    # because the human's card is not something it can condition on.
    rng = np.random.default_rng(0)
    strategies = dict(UNIFORM)
    for human in (Ja, Qa):
        state = LeducState(cards=(human, Ka), betting=((R,),))
        strategies[state.infoset()] = np.array([0.0, 0.0, 1.0])
        assert bot_action(state, strategies, rng) is Action.RAISE


def test_the_bot_follows_its_mixed_strategy():
    state = LeducState(cards=(Ja, Ka), betting=((R,),))
    strategies = dict(UNIFORM)
    strategies[state.infoset()] = np.array([0.5, 0.25, 0.25])
    rng = np.random.default_rng(0)
    draws = [bot_action(state, strategies, rng) for _ in range(4_000)]
    assert np.mean([a is Action.FOLD for a in draws]) == pytest.approx(0.5, abs=0.03)


def test_the_bot_picks_from_the_spots_own_legal_actions():
    # A 2-wide spot must never produce FOLD. Indexing the sampled column
    # straight into `Action` would do exactly that at the open, where
    # column 0 is CALL.
    state = LeducState(cards=(Ja, Ka))
    rng = np.random.default_rng(0)
    drawn = {bot_action(state, UNIFORM, rng) for _ in range(200)}
    assert drawn <= {Action.CALL, Action.RAISE}
    assert Action.FOLD not in drawn


def test_the_bot_is_deterministic_given_a_seed():
    state = LeducState(cards=(Ja, Ka), betting=((R,),))
    runs = [
        [bot_action(state, UNIFORM, np.random.default_rng(3)) for _ in range(5)]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


# ---------------------------------------------------------------------------
# Dealing, and the card that comes out mid-hand
# ---------------------------------------------------------------------------


def test_deal_gives_two_distinct_cards_in_seat_order():
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = deal(rng)
        assert state.betting == ((),)
        assert state.cards[0] != state.cards[1]


def test_deal_covers_every_ordering():
    rng = np.random.default_rng(0)
    assert len({deal(rng).cards for _ in range(2_000)}) == 30


def test_the_board_is_drawn_from_the_four_cards_nobody_holds():
    # Rung 1 had no card to turn. Here the deck acts mid-hand, and it must
    # draw from what is left, or a player could see their own card on the
    # board.
    rng = np.random.default_rng(0)
    state = LeducState(cards=(Ja, Ka), betting=((C, C),))
    assert state.is_chance_node()
    seen = {turn_board(state, rng).board for _ in range(200)}
    assert seen == {Jb, Qa, Qb, Kb}
    assert Ja not in seen and Ka not in seen


def test_turning_the_board_moves_the_hand_into_the_second_round():
    rng = np.random.default_rng(0)
    state = turn_board(LeducState(cards=(Ja, Ka), betting=((C, C),)), rng)
    assert not state.is_chance_node()
    assert state.round_index == 1
    assert isinstance(state.board, Card)


def test_the_board_follows_the_decks_own_probabilities():
    # Uniform over four remaining cards; a bias here would quietly change the
    # game the human is playing against a solver trained on the real one.
    rng = np.random.default_rng(1)
    state = LeducState(cards=(Ja, Ka), betting=((C, C),))
    draws = [turn_board(state, rng).board for _ in range(4_000)]
    for card in (Jb, Qa, Qb, Kb):
        assert draws.count(card) / len(draws) == pytest.approx(0.25, abs=0.03)


# ---------------------------------------------------------------------------
# How a hand is reported
# ---------------------------------------------------------------------------


def test_a_tie_is_reported_as_a_split_pot_not_a_loss():
    # Rung 1 could not tie: three distinct cards, one each. Here there are two
    # of every rank, so two kings meet at showdown and chop. Calling that
    # "bot wins 0" is the reading the player would take away.
    assert verdict(0.0) == "split pot"


@pytest.mark.parametrize(
    "chips,expected", [(4.0, "you win 4"), (-4.0, "bot wins 4"), (1.0, "you win 1")]
)
def test_a_decided_hand_names_the_winner_and_the_size(chips, expected):
    assert verdict(chips) == expected


def test_two_of_a_rank_can_actually_reach_a_showdown_together():
    # The premise of the split-pot case: the deal that makes it possible has
    # to exist, or the branch above is dead code.
    assert (Ka, Kb) in DEALS
    state = LeducState(cards=(Ka, Kb), betting=((C, C),))
    state = state.apply_chance(Qa).apply(C).apply(C)
    assert state.is_terminal()
    assert state.returns() == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_the_scoreboard_tracks_chips_from_the_humans_seat():
    board = Scoreboard()
    # Human is P0 and wins 2; then human is P1 and P0 wins 1, so human is -1.
    board.record(human_seat=0, returns=(2.0, -2.0))
    board.record(human_seat=1, returns=(1.0, -1.0))
    assert board.hands == 2
    assert board.chips == pytest.approx(1.0)


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
    # +0.086 ahead a winner, which is the confusion the split exists to stop.
    board = Scoreboard()
    board.record(human_seat=0, returns=(-1.0, 1.0))
    board.record(human_seat=1, returns=(-1.0, 1.0))
    _report(board, UNIFORM)
    out = capsys.readouterr().out
    assert "as P0: -1 over 1 hands (-1.000 per hand, equilibrium -0.086)" in out
    assert "as P1: +1 over 1 hands (+1.000 per hand, equilibrium +0.086)" in out
