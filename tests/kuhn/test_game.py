import pytest

from drawmaha_solver.kuhn.game import (
    ANTE,
    BET_SIZE,
    DEAL_PROBABILITY,
    DEALS,
    DECISION_HISTORIES,
    DECK,
    N_ACTIONS,
    TERMINAL_HISTORIES,
    Action,
    Card,
    InfoSet,
    KuhnState,
    action_label,
    all_infosets,
)

J, Q, K = Card.JACK, Card.QUEEN, Card.KING
P, B = Action.PASS, Action.BET

# ---------------------------------------------------------------------------
# The behavior table: every terminal history against every deal
# ---------------------------------------------------------------------------

# (history, p0 card, p1 card, p0 chips). Ground truth typed out by hand from
# the rules, not derived — this table is what the implementation must match.
PAYOFFS = [
    # Checked down. Showdown for the antes only: higher card takes 1.
    ((P, P), J, Q, -1.0),
    ((P, P), J, K, -1.0),
    ((P, P), Q, J, 1.0),
    ((P, P), Q, K, -1.0),
    ((P, P), K, J, 1.0),
    ((P, P), K, Q, 1.0),
    # P0 checks, P1 bets, P0 folds. P1 takes P0's ante whatever the cards are.
    ((P, B, P), J, Q, -1.0),
    ((P, B, P), J, K, -1.0),
    ((P, B, P), Q, J, -1.0),
    ((P, B, P), Q, K, -1.0),
    ((P, B, P), K, J, -1.0),
    ((P, B, P), K, Q, -1.0),
    # P0 checks, P1 bets, P0 calls. Showdown for ante plus bet: 2.
    ((P, B, B), J, Q, -2.0),
    ((P, B, B), J, K, -2.0),
    ((P, B, B), Q, J, 2.0),
    ((P, B, B), Q, K, -2.0),
    ((P, B, B), K, J, 2.0),
    ((P, B, B), K, Q, 2.0),
    # P0 bets, P1 folds. P0 takes P1's ante whatever the cards are.
    ((B, P), J, Q, 1.0),
    ((B, P), J, K, 1.0),
    ((B, P), Q, J, 1.0),
    ((B, P), Q, K, 1.0),
    ((B, P), K, J, 1.0),
    ((B, P), K, Q, 1.0),
    # P0 bets, P1 calls. Showdown for ante plus bet: 2.
    ((B, B), J, Q, -2.0),
    ((B, B), J, K, -2.0),
    ((B, B), Q, J, 2.0),
    ((B, B), Q, K, -2.0),
    ((B, B), K, J, 2.0),
    ((B, B), K, Q, 2.0),
]

@pytest.mark.parametrize("history,p0_card,p1_card,p0_chips", PAYOFFS)
def test_every_terminal_payoff(history, p0_card, p1_card, p0_chips):
    state = KuhnState(cards=(p0_card, p1_card), history=history)
    assert state.is_terminal()
    assert state.returns() == (p0_chips, -p0_chips)

def test_payoff_table_covers_every_terminal_history_and_deal():
    assert {(h, (a, b)) for h, a, b, _ in PAYOFFS} == {
        (h, deal) for h in TERMINAL_HISTORIES for deal in DEALS
    }

def test_returns_are_zero_sum():
    for history in TERMINAL_HISTORIES:
        for deal in DEALS:
            assert sum(KuhnState(cards=deal, history=history).returns()) == 0.0

def test_folding_costs_exactly_the_ante():
    # The folder surrenders their ante and nothing more — the bet is returned.
    assert KuhnState(cards=(K, J), history=(B, P)).returns() == (ANTE, -ANTE)

def test_called_bet_doubles_the_showdown_stake():
    checked = KuhnState(cards=(K, J), history=(P, P)).returns()[0]
    called = KuhnState(cards=(K, J), history=(B, B)).returns()[0]
    assert called - checked == BET_SIZE

# ---------------------------------------------------------------------------
# Deck and chance node
# ---------------------------------------------------------------------------

def test_deck_is_three_ranked_cards():
    assert DECK == (Card.JACK, Card.QUEEN, Card.KING)
    assert Card.JACK < Card.QUEEN < Card.KING

def test_deals_are_the_six_distinct_orderings():
    assert len(DEALS) == 6
    assert len(set(DEALS)) == 6
    assert all(a != b for a, b in DEALS)
    assert len(DEALS) * DEAL_PROBABILITY == pytest.approx(1.0)

def test_duplicate_cards_are_rejected():
    # Only three cards exist and each player holds one, so a repeated card is
    # an impossible deal, not a corner case to tolerate.
    with pytest.raises(ValueError, match="distinct"):
        KuhnState(cards=(K, K))

def test_raw_ints_are_not_accepted_as_cards():
    # Card is an IntEnum, so (0, 1) compares and hashes equal to (J, Q) and
    # would otherwise construct silently — then blow up in CARD_SYMBOL the
    # moment the int is out of range. Reject it at the boundary instead.
    with pytest.raises(ValueError, match="Card members"):
        KuhnState(cards=(0, 1))

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def test_game_starts_with_p0_to_act_on_an_empty_history():
    state = KuhnState(cards=(K, J))
    assert state.history == ()
    assert not state.is_terminal()
    assert state.current_player == 0

def test_players_alternate():
    state = KuhnState(cards=(K, J)).apply(Action.PASS)
    assert state.current_player == 1

def test_both_actions_are_always_legal_at_a_decision_node():
    for history in DECISION_HISTORIES:
        state = KuhnState(cards=(K, J), history=history)
        assert state.legal_actions() == (Action.PASS, Action.BET)
    assert N_ACTIONS == 2

def test_apply_leaves_the_original_state_untouched():
    state = KuhnState(cards=(K, J))
    child = state.apply(Action.BET)
    assert state.history == ()
    assert child.history == (Action.BET,)
    assert child.cards == state.cards

def test_every_line_of_play_ends_within_three_actions():
    def walk(state):
        if state.is_terminal():
            assert state.history in TERMINAL_HISTORIES
            assert len(state.history) <= 3
            return
        assert state.history in DECISION_HISTORIES
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(KuhnState(cards=deal))

def test_the_tree_has_four_decision_nodes_and_five_terminals():
    assert DECISION_HISTORIES == ((), (P,), (B,), (P, B))
    assert len(TERMINAL_HISTORIES) == 5

def test_acting_on_a_finished_hand_raises():
    state = KuhnState(cards=(K, J), history=(B, B))
    with pytest.raises(ValueError, match="terminal"):
        state.apply(Action.PASS)
    with pytest.raises(ValueError, match="terminal"):
        state.legal_actions()
    with pytest.raises(ValueError, match="terminal"):
        state.current_player  # noqa: B018 - property access is the call

def test_scoring_an_unfinished_hand_raises():
    with pytest.raises(ValueError, match="not terminal"):
        KuhnState(cards=(K, J), history=(P,)).returns()

def test_unreachable_histories_are_rejected():
    # (PASS, PASS) already ends the hand, so a third action never happens.
    with pytest.raises(ValueError, match="not a reachable"):
        KuhnState(cards=(K, J), history=(P, P, P))

def test_raw_ints_are_not_accepted_as_history_actions():
    # The sharp edge of Action being an IntEnum: (1, 0) is equal to (B, P) for
    # the reachability check but fails the `is Action.PASS` identity test in
    # returns(), which would score the fold as a called showdown — P0 paid 2
    # instead of winning 1. Reject the raw ints rather than answer wrongly.
    with pytest.raises(ValueError, match="Action members"):
        KuhnState(cards=(J, Q), history=(1, 0))

# ---------------------------------------------------------------------------
# Information sets
# ---------------------------------------------------------------------------

def test_there_are_exactly_twelve_infosets():
    # 4 decision nodes x 3 possible own cards. Some sources say 13; that is a
    # miscount — the ladder's rung-1 ledger allocates 12 entries.
    infosets = all_infosets()
    assert len(infosets) == 12
    assert len(set(infosets)) == 12

def test_each_player_owns_six_infosets():
    by_player = [i for i in all_infosets() if i.player == 0], [
        i for i in all_infosets() if i.player == 1
    ]
    assert len(by_player[0]) == 6
    assert len(by_player[1]) == 6
    assert {i.history for i in by_player[0]} == {(), (P, B)}
    assert {i.history for i in by_player[1]} == {(P,), (B,)}

def test_an_infoset_hides_the_opponents_card():
    # The whole point of the game: holding the king, P0 cannot tell the two
    # deals apart, so both must map to one ledger entry.
    against_jack = KuhnState(cards=(K, J)).infoset()
    against_queen = KuhnState(cards=(K, Q)).infoset()
    assert against_jack == against_queen
    assert hash(against_jack) == hash(against_queen)

def test_infoset_belongs_to_the_player_to_act():
    state = KuhnState(cards=(K, J)).apply(Action.PASS)
    assert state.infoset() == InfoSet(card=J, history=(P,))
    assert state.infoset().player == 1

def test_every_reachable_infoset_is_in_the_inventory():
    def walk(state):
        if state.is_terminal():
            return
        assert state.infoset() in all_infosets()
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(KuhnState(cards=deal))

def test_infoset_renders_in_the_literature_notation():
    assert str(InfoSet(card=K, history=())) == "K"
    assert str(InfoSet(card=J, history=(P, B))) == "Jpb"
    assert str(InfoSet(card=Q, history=(B,))) == "Qb"

# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------

# (history the player faces, action, what it is called at that node)
LABELS = [
    ((), P, "check"),
    ((), B, "bet"),
    ((P,), P, "check"),
    ((P,), B, "bet"),
    ((B,), P, "fold"),
    ((B,), B, "call"),
    ((P, B), P, "fold"),
    ((P, B), B, "call"),
]

@pytest.mark.parametrize("history,action,expected", LABELS)
def test_action_labels_are_context_sensitive(history, action, expected):
    assert action_label(action, history) == expected
