import pytest

from drawmaha_solver.leduc.game import (
    ANTE,
    BET_SIZES,
    CARD_SYMBOL,
    DEAL_PROBABILITY,
    DEALS,
    DECK,
    N_ACTIONS,
    N_ROUNDS,
    PAIR_STRENGTH,
    RAISE_CAP,
    RANK_SYMBOL,
    RANKS,
    REACHABLE_ROUND_LINES,
    ROUND_DECISION_LINES,
    Action,
    Card,
    InfoSet,
    LeducState,
    Rank,
    action_label,
    all_infosets,
    hand_strength,
)

J, Q, K = Rank.JACK, Rank.QUEEN, Rank.KING
Ja, Jb, Qa, Qb, Ka, Kb = DECK
F, C, R = Action.FOLD, Action.CALL, Action.RAISE

# ---------------------------------------------------------------------------
# Helpers: drive a state down a named line the way the referee probe did
# ---------------------------------------------------------------------------

def drive(cards, board, *actions):
    """Apply `actions`, dealing `board` whenever the deck's turn comes.

    Pass `board=None` to stop the moment the deck is due, which is how the
    chance-node tests get hold of an undealt state.
    """
    state = LeducState(cards=cards)
    for action in actions:
        if state.is_chance_node():
            if board is None:
                return state
            state = state.apply_chance(board)
        state = state.apply(action)
    if state.is_chance_node() and board is not None:
        state = state.apply_chance(board)
    return state

def census():
    """Walk every node below the private deal, counting the three node kinds."""
    counts = {"board": 0, "decision": 0, "terminal": 0}

    def walk(state):
        if state.is_terminal():
            counts["terminal"] += 1
            return
        if state.is_chance_node():
            counts["board"] += 1
            for card, _ in state.chance_outcomes():
                walk(state.apply_chance(card))
            return
        counts["decision"] += 1
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(LeducState(cards=deal))
    return counts

# ---------------------------------------------------------------------------
# Deck, ranks, and the two chance nodes
# ---------------------------------------------------------------------------

def test_the_deck_is_six_cards_two_of_each_rank():
    assert len(DECK) == 6
    assert [c.rank for c in DECK] == [J, J, Q, Q, K, K]
    assert J < Q < K

def test_a_cards_rank_is_its_index_halved_like_the_referee():
    # OpenSpiel indexes the deck 0..5 and reads rank as card // 2; the rank
    # keying in every downstream table depends on matching that convention.
    for card in DECK:
        assert card.rank == Rank(int(card) // 2)

def test_two_cards_share_a_rank_and_differ_in_suit():
    assert Ja.rank == Jb.rank
    assert Ja.suit != Jb.suit

def test_comparing_two_cards_for_size_raises_instead_of_lying():
    # Card is an IntEnum, so `<` would evaluate happily and answer by deck
    # index: Ja < Jb is True and Jb < Ja is False, though the two hands are
    # equal. Rank ordering is the real comparison and stays legal.
    for compare in (
        lambda: Ja < Jb,
        lambda: Ja <= Jb,
        lambda: Ka > Qa,
        lambda: Ka >= Qa,
    ):
        with pytest.raises(TypeError, match="not ordered"):
            compare()
    assert Ja.rank < Qa.rank

def test_a_card_renders_as_its_rank_and_suit_and_a_rank_as_one_letter():
    assert CARD_SYMBOL[Ja] == "Ja"
    assert CARD_SYMBOL[Kb] == "Kb"
    assert [RANK_SYMBOL[r] for r in RANKS] == ["J", "Q", "K"]

def test_a_board_before_round_two_opens_is_rejected():
    # apply_chance always opens round 2 in the same breath, so this state is
    # unreachable by play — but a hand-built one would silently claim a board
    # that no betting round can act on.
    with pytest.raises(ValueError, match="no board yet"):
        LeducState(cards=(Ja, Qa), board=Ka, betting=((C,),))

def test_there_are_thirty_equally_likely_private_deals():
    assert len(DEALS) == 30
    assert len(set(DEALS)) == 30
    assert all(a != b for a, b in DEALS)
    assert len(DEALS) * DEAL_PROBABILITY == pytest.approx(1.0)

def test_the_board_is_drawn_from_whatever_is_left_in_the_deck():
    # Read the remaining deck from the state; never hard-code four cards.
    state = drive((Ja, Qa), None, C, C)
    assert state.is_chance_node()
    assert [card for card, _ in state.chance_outcomes()] == [Jb, Qb, Ka, Kb]
    assert [prob for _, prob in state.chance_outcomes()] == [0.25] * 4

def test_the_deck_shrinks_by_exactly_the_cards_already_dealt():
    assert LeducState(cards=(Ja, Qa)).remaining_deck() == (Jb, Qb, Ka, Kb)
    assert drive((Ja, Qa), Kb, C, C).remaining_deck() == (Jb, Qb, Ka)

def test_a_board_card_already_in_a_hand_is_rejected():
    state = drive((Ja, Qa), None, C, C)
    with pytest.raises(ValueError, match="already dealt"):
        state.apply_chance(Ja)

def test_duplicate_private_cards_are_rejected():
    with pytest.raises(ValueError, match="distinct"):
        LeducState(cards=(Ka, Ka))

def test_raw_ints_are_not_accepted_as_cards():
    # Card is an IntEnum, so (0, 2) hashes and compares equal to (Ja, Qa) and
    # would construct silently, then read `.rank` off a bare int and crash far
    # from the mistake. Reject it at the boundary, as rung 1 does.
    with pytest.raises(ValueError, match="Card members"):
        LeducState(cards=(0, 2))

# ---------------------------------------------------------------------------
# Betting: the two rounds have one shape
# ---------------------------------------------------------------------------

# Every decision line inside a round, with the legal actions the referee
# offers there. Typed out from the OpenSpiel probe, not derived.
LEGAL_ACTIONS = [
    ((), (C, R)),               # the open: fold is not legal with nothing to call
    ((C,), (C, R)),             # checked to; check behind or bet
    ((C, R), (F, C, R)),        # facing the first raise
    ((C, R, R), (F, C)),        # facing the capped second raise
    ((R,), (F, C, R)),          # facing the first raise
    ((R, R), (F, C)),           # facing the capped second raise
]

@pytest.mark.parametrize("line,expected", LEGAL_ACTIONS)
@pytest.mark.parametrize("round_index", range(N_ROUNDS))
def test_legal_actions_match_the_referee_in_both_rounds(line, expected, round_index):
    # Round 2 repeats round 1's shape exactly: P0 opens, fold needs a raise to
    # face, and the cap bites after two raises.
    prefix = () if round_index == 0 else (C, C)
    state = drive((Ja, Qa), Ka, *prefix, *line)
    assert state.legal_actions() == expected

def test_the_legal_action_order_is_ascending_and_stable():
    # Ledgers are sized and indexed by position within legal_actions(), so this
    # order is load-bearing: a utilities vector built in any other order pairs
    # regrets with the wrong action and numpy will not notice.
    for line, expected in LEGAL_ACTIONS:
        state = drive((Ja, Qa), Ka, *line)
        assert list(state.legal_actions()) == sorted(expected)

def test_folding_is_never_legal_with_nothing_to_call():
    for line in ((), (C,)):
        assert Action.FOLD not in drive((Ja, Qa), Ka, *line).legal_actions()

def test_the_cap_is_two_raises_per_round_counting_both_players():
    assert RAISE_CAP == 2
    capped = drive((Ja, Qa), Ka, R, R)
    assert Action.RAISE not in capped.legal_actions()

def test_the_cap_resets_when_the_second_round_opens():
    reopened = drive((Ja, Qa), Ka, R, R, C)
    assert reopened.legal_actions() == (C, R)

def test_p0_opens_both_rounds():
    assert LeducState(cards=(Ja, Qa)).current_player == 0
    assert drive((Ja, Qa), Ka, C, C).current_player == 0

def test_players_alternate_within_a_round():
    assert drive((Ja, Qa), Ka, C).current_player == 1
    assert drive((Ja, Qa), Ka, C, R).current_player == 0

def test_a_round_one_fold_ends_the_hand_before_any_board_is_dealt():
    folded = drive((Ja, Qa), Ka, R, F)
    assert folded.is_terminal()
    assert not folded.is_chance_node()
    assert folded.board is None

# Every round-1 line that reaches the board, as the referee reports them.
LINES_BEFORE_THE_BOARD = [(C, C), (C, R, C), (C, R, R, C), (R, C), (R, R, C)]

@pytest.mark.parametrize("line", LINES_BEFORE_THE_BOARD)
def test_every_round_one_line_that_does_not_fold_reaches_the_board(line):
    state = drive((Ja, Qa), None, *line)
    assert state.is_chance_node()
    assert state.board is None

def test_the_board_is_dealt_exactly_once():
    dealt = drive((Ja, Qa), Ka, C, C)
    assert dealt.board is Ka
    assert not dealt.is_chance_node()
    with pytest.raises(ValueError, match="not a chance node"):
        dealt.apply_chance(Kb)

def test_a_second_round_call_ends_the_hand():
    assert drive((Ja, Qa), Ka, C, C, C, C).is_terminal()

def test_apply_leaves_the_original_state_untouched():
    state = LeducState(cards=(Ja, Qa))
    child = state.apply(Action.RAISE)
    assert state.betting == ((),)
    assert child.betting == ((R,),)
    assert child.cards == state.cards

def test_acting_on_a_finished_hand_raises():
    done = drive((Ja, Qa), Ka, R, F)
    with pytest.raises(ValueError, match="terminal"):
        done.apply(Action.CALL)
    with pytest.raises(ValueError, match="terminal"):
        done.legal_actions()
    with pytest.raises(ValueError, match="terminal"):
        done.current_player  # noqa: B018 - property access is the call

def test_acting_when_it_is_the_decks_turn_raises():
    waiting = drive((Ja, Qa), None, C, C)
    with pytest.raises(ValueError, match="chance node"):
        waiting.apply(Action.CALL)
    with pytest.raises(ValueError, match="chance node"):
        waiting.current_player  # noqa: B018 - property access is the call

def test_an_illegal_action_is_rejected_rather_than_applied():
    with pytest.raises(ValueError, match="not legal"):
        LeducState(cards=(Ja, Qa)).apply(Action.FOLD)

def test_unreachable_betting_lines_are_rejected():
    # (CALL, CALL) already closes the round, so a third action never happens.
    with pytest.raises(ValueError, match="not a reachable"):
        LeducState(cards=(Ja, Qa), betting=((C, C, C),))

def test_a_second_round_cannot_open_before_the_first_one_closes():
    with pytest.raises(ValueError, match="closed"):
        LeducState(cards=(Ja, Qa), board=Ka, betting=((C,), ()))

def test_a_second_betting_round_without_a_board_is_rejected():
    with pytest.raises(ValueError, match="board"):
        LeducState(cards=(Ja, Qa), betting=((C, C), ()))

def test_raw_ints_are_not_accepted_as_actions():
    # Action is an IntEnum, so (1, 1) passes the reachability check but fails
    # the `is Action.FOLD` identity test in returns(), scoring a fold as a
    # showdown. Reject the raw ints rather than answer wrongly.
    with pytest.raises(ValueError, match="Action members"):
        LeducState(cards=(Ja, Qa), betting=((1, 1),))

# ---------------------------------------------------------------------------
# The census: the whole tree, counted
# ---------------------------------------------------------------------------

def test_the_tree_has_the_referees_census():
    # The referee counts 3,780 decision nodes, 5,520 terminals, and 157 chance
    # nodes. The 157 does not appear here because OpenSpiel deals the private
    # cards one at a time (1 root + 6 second-card nodes) where DEALS deals both
    # at once; 1 + 6 + our 150 board nodes is that 157. The two numbers that
    # depend on the betting rules — and so on this file being right — match
    # exactly.
    assert census() == {"board": 150, "decision": 3780, "terminal": 5520}
    assert 1 + len(DECK) + 150 == 157

def test_a_board_is_dealt_once_per_deal_per_surviving_round_one_line():
    assert len(DEALS) * len(LINES_BEFORE_THE_BOARD) == 150

def test_the_round_shape_is_six_decisions_and_fifteen_reachable_lines():
    assert len(ROUND_DECISION_LINES) == 6
    assert len(REACHABLE_ROUND_LINES) == 15
    assert set(ROUND_DECISION_LINES) <= REACHABLE_ROUND_LINES

def test_the_chance_probabilities_along_any_path_multiply_to_the_referees_two():
    # 1/30 once the private cards are out, 1/120 once the board lands.
    probs = set()

    def walk(state, reach):
        if state.is_terminal():
            probs.add(round(reach, 12))
            return
        if state.is_chance_node():
            for card, prob in state.chance_outcomes():
                walk(state.apply_chance(card), reach * prob)
            return
        for action in state.legal_actions():
            walk(state.apply(action), reach)

    for deal in DEALS:
        walk(LeducState(cards=deal), DEAL_PROBABILITY)

    assert probs == {round(1 / 30, 12), round(1 / 120, 12)}

# ---------------------------------------------------------------------------
# Showdown and payoffs
# ---------------------------------------------------------------------------

def test_a_private_card_that_pairs_the_board_outranks_every_unpaired_card():
    assert hand_strength(Ja, Jb) == PAIR_STRENGTH
    assert hand_strength(Ja, Jb) > hand_strength(Ka, Jb)
    assert hand_strength(Ka, Qa) == K

def test_the_two_bet_sizes_are_two_then_four():
    assert BET_SIZES == (2, 4)
    assert ANTE == 1

# (label, cards, board, actions, chips to P0). Ground truth read off the
# OpenSpiel probe in .context/leduc_rules_probe.py, one line per corner the
# rung-2 design document names.
CORNERS = [
    ("checkdown, K board, Q beats J", (Ja, Qa), Ka, (C, C, C, C), -1.0),
    ("checkdown, J board, P0 pairs", (Ja, Qa), Jb, (C, C, C, C), 1.0),
    ("checkdown, Q board, P1 pairs", (Ja, Qa), Qb, (C, C, C, C), -1.0),
    ("checkdown, K board, J vs J ties", (Ja, Jb), Ka, (C, C, C, C), 0.0),
    ("checkdown, J board, the pair beats the king", (Ka, Ja), Jb, (C, C, C, C), -1.0),
    ("capped round 1, checked round 2", (Ja, Qa), Ka, (R, R, C, C, C), -5.0),
    ("checked round 1, capped round 2", (Ja, Qa), Ka, (C, C, R, R, C), -9.0),
    ("raise-call round 1, capped round 2", (Ja, Qa), Ka, (R, C, R, R, C), -11.0),
    ("raise-call in both rounds", (Ja, Qa), Ka, (R, C, R, C), -7.0),
    ("capped in both rounds", (Ja, Qa), Ka, (R, R, C, R, R, C), -13.0),
    ("P1 folds to the opening raise", (Ja, Qa), Ka, (R, F), 1.0),
    ("P0 folds to the re-raise, having raised once", (Ja, Qa), Ka, (R, R, F), -3.0),
    ("P0 checks, raises are traded, P1 folds", (Ja, Qa), Ka, (C, R, R, F), 3.0),
    ("P0 checks, is raised, and folds", (Ja, Qa), Ka, (C, R, F), -1.0),
]

@pytest.mark.parametrize(
    "cards,board,actions,p0_chips",
    [c[1:] for c in CORNERS],
    ids=[c[0] for c in CORNERS],
)
def test_the_referees_corner_payoffs(cards, board, actions, p0_chips):
    state = drive(cards, board, *actions)
    assert state.is_terminal()
    assert state.returns() == (p0_chips, -p0_chips)

def test_every_terminal_is_zero_sum_and_pays_a_pot_the_rules_can_build():
    # 5,520 terminals is past the point of listing every case by hand, so the
    # discipline becomes: pin the corners above, property-test the rest.
    magnitudes = set()

    def walk(state):
        if state.is_terminal():
            p0, p1 = state.returns()
            assert p0 + p1 == 0.0
            magnitudes.add(abs(p0))
            return
        if state.is_chance_node():
            for card, _ in state.chance_outcomes():
                walk(state.apply_chance(card))
            return
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(LeducState(cards=deal))

    assert magnitudes == {0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0}

def test_swapping_the_private_cards_negates_every_payoff():
    # The rules treat the seats symmetrically, so the only thing a swap can
    # change is who collects. Mirror the betting line too, since P0 opens.
    for line in [(C, C, C, C), (R, C, R, C), (C, R, R, C, R, C)]:
        original = drive((Ja, Qa), Kb, *line).returns()
        swapped = drive((Qa, Ja), Kb, *line).returns()
        assert original == (-swapped[0], -swapped[1])

def test_only_a_showdown_can_pay_zero():
    # A fold always collects the folder's stake, which is at least the ante.
    assert drive((Ja, Jb), Ka, R, F).returns() == (1.0, -1.0)

def test_scoring_an_unfinished_hand_raises():
    with pytest.raises(ValueError, match="not terminal"):
        LeducState(cards=(Ja, Qa)).returns()

# ---------------------------------------------------------------------------
# Information sets: 936 referee strings collapse to 288 rank-keyed ledgers
# ---------------------------------------------------------------------------

def test_the_inventory_holds_two_hundred_eighty_eight_rank_keyed_infosets():
    infosets = all_infosets()
    assert len(infosets) == 288
    assert len(set(infosets)) == 288

def test_eighteen_infosets_live_in_round_one_and_two_hundred_seventy_in_round_two():
    by_round = [i for i in all_infosets() if i.board is None], [
        i for i in all_infosets() if i.board is not None
    ]
    assert len(by_round[0]) == 18  # 3 own ranks x 6 decision lines
    assert len(by_round[1]) == 270  # 3 x 3 boards x 5 surviving lines x 6 lines
    assert {i.board for i in by_round[1]} == set(RANKS)

def test_ninety_six_of_the_ledgers_are_three_wide():
    widths = [len(i.legal_actions()) for i in all_infosets()]
    assert widths.count(3) == 96
    assert widths.count(2) == 192
    assert set(widths) == {2, 3}

def test_swapping_suits_leaves_the_infoset_unchanged():
    # The whole point of keying on rank: the two jacks are the same hand, so
    # they must share one ledger rather than train two half-sized copies.
    assert LeducState(cards=(Ja, Qa)).infoset() == LeducState(cards=(Jb, Qa)).infoset()
    assert LeducState(cards=(Ja, Qa)).infoset() == LeducState(cards=(Ja, Qb)).infoset()
    on_ka = drive((Ja, Qa), Ka, C, C).infoset()
    on_kb = drive((Ja, Qa), Kb, C, C).infoset()
    assert on_ka == on_kb
    assert hash(on_ka) == hash(on_kb)

def test_an_infoset_hides_the_opponents_card():
    assert LeducState(cards=(Ka, Ja)).infoset() == LeducState(cards=(Ka, Qa)).infoset()

def test_an_infoset_remembers_round_one_after_the_board_lands():
    # Perfect recall: the round-2 key carries the round-1 line, so two
    # different routes to the same board are different ledgers.
    checked = drive((Ja, Qa), Ka, C, C).infoset()
    raised = drive((Ja, Qa), Ka, R, C).infoset()
    assert checked != raised

def test_infoset_belongs_to_the_player_to_act():
    assert LeducState(cards=(Ja, Qa)).infoset().player == 0
    assert drive((Ja, Qa), Ka, C).infoset().player == 1

def test_every_reachable_infoset_is_in_the_inventory():
    inventory = set(all_infosets())
    seen = set()

    def walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for card, _ in state.chance_outcomes():
                walk(state.apply_chance(card))
            return
        seen.add(state.infoset())
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(LeducState(cards=deal))

    assert seen == inventory

def test_the_nodes_per_infoset_match_the_referees_histogram():
    # 10 nodes for each round-1 infoset (2 own cards x 5 opponent cards);
    # 16 when the board rank differs from the hand (2 x 2 x 4) and only 8 when
    # it matches, because the board then consumes the hand's twin.
    counts = {}

    def walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for card, _ in state.chance_outcomes():
                walk(state.apply_chance(card))
            return
        counts[state.infoset()] = counts.get(state.infoset(), 0) + 1
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(LeducState(cards=deal))

    histogram = {}
    for n in counts.values():
        histogram[n] = histogram.get(n, 0) + 1
    assert histogram == {10: 18, 16: 180, 8: 90}

def test_an_infoset_at_a_closed_round_is_rejected():
    with pytest.raises(ValueError, match="not a decision"):
        InfoSet(rank=K, board=None, betting=((C, C),))

def test_an_infoset_built_from_raw_values_is_rejected():
    with pytest.raises(ValueError, match="Rank member"):
        InfoSet(rank=0, board=None, betting=((),))
    with pytest.raises(ValueError, match="Action members"):
        InfoSet(rank=K, board=None, betting=((1,),))

def test_infoset_renders_the_hand_the_board_and_both_lines():
    assert str(InfoSet(rank=K, board=None, betting=((),))) == "K:"
    assert str(InfoSet(rank=J, board=None, betting=((C, R),))) == "J:cr"
    assert str(InfoSet(rank=J, board=Q, betting=((C, C), (R,)))) == "J:cc|Q:r"

# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------

# (line the player faces, action, what poker calls it there)
LABELS = [
    ((), C, "check"),
    ((), R, "bet"),
    ((C,), C, "check"),
    ((C,), R, "bet"),
    ((R,), F, "fold"),
    ((R,), C, "call"),
    ((R,), R, "raise"),
    ((C, R), F, "fold"),
    ((C, R), C, "call"),
    ((C, R, R), C, "call"),
]

@pytest.mark.parametrize("line,action,expected", LABELS)
def test_action_labels_are_context_sensitive(line, action, expected):
    assert action_label(action, line) == expected

def test_there_are_three_actions_and_two_rounds():
    assert N_ACTIONS == 3
    assert N_ROUNDS == 2
