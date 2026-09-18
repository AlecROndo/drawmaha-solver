"""The rules: the bomb pot, the betting tree, the draw, and the split pot.

Rung 3 has no referee — no engine implements Drawmaha — so the answer sheet here
is the plan's own enumerated tables: round 1's thirteen betting lines written out
action by action, the four (pot, stacks) states round 2 can open in, the 2 → 8 →
26 commitment ladder truncated to an all-in at 25, and the deck slack at the
bottom of the deepest line. Everything else is a property: returns sum to zero,
a relabelled position plays the same game, a discard never comes back.
"""

from collections import Counter, defaultdict
from itertools import combinations
from random import Random

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.cards import DECK, canonical, parse_cards, relabel
from drawmaha_solver.minidrawmaha.game import (
    ANTE,
    DISCARD_MASK,
    DRAW_ACTIONS,
    STACK,
    THROW_CAP,
    Action,
    DrawSignal,
    InfoSet,
    MiniState,
    NodeKind,
    action_label,
    chip_state,
    draw_order,
    legal_actions_for,
    line_symbol,
    pot_shares,
    random_deal,
    throw_count,
)
from drawmaha_solver.minidrawmaha.hands import HOLE_CARDS, inner_score, outer_score

F, X, P = Action.FOLD, Action.CHECK_CALL, Action.POT

HOLES = (parse_cards("2c 3c 6d"), parse_cards("2d 4h 5h"))
BOARD_ONE, BOARD_TWO = parse_cards("4c"), parse_cards("5c")

def play(*moves, holes=HOLES):
    """Drive a hand: an `Action` goes to the player to act, a card tuple to the deck."""
    state = MiniState(holes=holes)
    for move in moves:
        state = state.apply(move) if isinstance(move, Action) else state.apply_chance(move)
    return state

def betting_subtree(state):
    """Walk one betting round from `state`, returning its lines and decision widths.

    Stops at whatever ends the round — a fold, a matched call, or two empty
    stacks — so what comes back is exactly the round's own shape.
    """
    lines, widths = [], Counter()
    opening = len(state.betting[-1])

    def walk(node):
        if node.is_terminal() or node.is_chance_node() or node.is_draw_decision():
            lines.append(line_symbol(node.betting[-1][opening:]))
            return
        widths[len(node.legal_actions())] += 1
        for action in node.legal_actions():
            walk(node.apply(action))

    walk(state)
    return lines, widths

def playouts(count, *, seed=20260918):
    """Random complete hands: uniform over legal actions and over chance outcomes."""
    rng = Random(seed)
    for _ in range(count):
        deck = list(parse_cards("2c 2d 2h 3c 3d 3h 4c 4d 4h 5c 5d 5h 6c 6d 6h"))
        rng.shuffle(deck)
        state = MiniState(holes=(tuple(sorted(deck[:3])), tuple(sorted(deck[3:6]))))
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                state = state.apply_chance(rng.choice(outcomes)[0])
            else:
                state = state.apply(rng.choice(state.legal_actions()))
        yield state

# ---------------------------------------------------------------------------
# A bomb pot: the deck acts before anybody bets
# ---------------------------------------------------------------------------

def test_the_ante_is_in_before_anyone_decides_anything():
    # The correction this rung's README fix is about: there is no preflop
    # betting. Both players are already in for the ante and the first thing
    # that happens is a board card.
    root = MiniState(holes=HOLES)
    assert (ANTE, STACK) == (1, 26)
    assert root.kind is NodeKind.CHANCE
    assert root.contributions() == (ANTE, ANTE)
    assert chip_state(root.betting).pot == 2 * ANTE

def test_the_first_board_card_comes_from_the_nine_the_deal_left():
    root = MiniState(holes=HOLES)
    outcomes = root.chance_outcomes()
    assert len(outcomes) == 15 - 2 * 3
    assert sum(probability for _, probability in outcomes) == pytest.approx(1.0)
    assert all(len(cards) == 1 for cards, _ in outcomes)

def test_the_streets_run_board_bet_draw_board_bet():
    state = MiniState(holes=HOLES)
    seen = []
    for move in (BOARD_ONE, X, X, Action.THROW_NONE, Action.THROW_LOW):
        seen.append(state.kind)
        state = state.apply(move) if isinstance(move, Action) else state.apply_chance(move)
    # P1 threw one, so the deck owes exactly one replacement before board 2.
    assert seen == [NodeKind.CHANCE] + [NodeKind.DECISION] * 4
    assert state.kind is NodeKind.CHANCE
    state = state.apply_chance(state.chance_outcomes()[0][0])
    assert state.kind is NodeKind.CHANCE  # board 2
    state = state.apply_chance(BOARD_TWO).apply(X).apply(X)
    assert state.kind is NodeKind.TERMINAL

def test_the_root_deal_is_uniform_over_the_hundred_thousand_hands():
    # The one chance event the state machine does not hold as a node: 100,100
    # ordered deals is too many to read off a single `chance_outcomes()`, so a
    # sampler is handed the deal instead, exactly as rung 2 hands in its 30.
    rng = np.random.default_rng(20260918)
    deals = [random_deal(rng).holes for _ in range(400)]
    for first, second in deals:
        assert len(first) == len(second) == 3
        assert not set(first) & set(second)
        assert list(first) == sorted(first) and list(second) == sorted(second)
    dealt = {card for deal in deals for hole in deal for card in hole}
    assert dealt == set(DECK)
    # Distinct enough to be a deal rather than a fixture, and reproducible
    # enough that a seeded solve can be re-run.
    assert len(set(deals)) > 390
    assert deals[0] == random_deal(np.random.default_rng(20260918)).holes

# ---------------------------------------------------------------------------
# Betting: the plan's enumerated tables
# ---------------------------------------------------------------------------

# §3, written out action by action: `x` check, `p` pot, `c` call, `f` fold.
ROUND_ONE_LINES = (
    "xx",
    "pf",
    "pc",
    "xpf",
    "xpc",
    "ppf",
    "ppc",
    "xppf",
    "xppc",
    "pppf",
    "pppc",
    "xpppf",
    "xpppc",
)

def test_round_one_has_exactly_the_plan_s_thirteen_lines():
    lines, _ = betting_subtree(play(BOARD_ONE))
    assert sorted(lines) == sorted(ROUND_ONE_LINES)

def test_round_one_has_four_two_wide_and_four_three_wide_decisions():
    _, widths = betting_subtree(play(BOARD_ONE))
    assert widths == {2: 4, 3: 4}

# §3's round-2 table: each (pot, behind) round 1 can leave, and what the round
# looks like when it opens there.
ROUND_TWO_STATES = {
    (2, (25, 25)): (13, {2: 4, 3: 4}),
    (6, (23, 23)): (9, {2: 4, 3: 2}),
    (18, (17, 17)): (5, {2: 4}),
    (52, (0, 0)): (1, {}),
}

def round_two_openings():
    """Every state round 2 can open in, reached by actually playing round 1."""
    openings = {}
    for line in ROUND_ONE_LINES:
        if line.endswith("f"):
            continue
        actions = [{"x": X, "c": X, "p": P}[symbol] for symbol in line]
        state = play(BOARD_ONE, *actions, Action.THROW_NONE, Action.THROW_NONE, BOARD_TWO)
        chips = chip_state(state.betting)
        openings[(chips.pot, chips.behind)] = state
    return openings

def test_round_two_opens_in_the_plan_s_four_states():
    assert set(round_two_openings()) == set(ROUND_TWO_STATES)

@pytest.mark.parametrize(("opening", "expected"), sorted(ROUND_TWO_STATES.items()))
def test_each_round_two_state_has_the_shape_the_plan_measured(opening, expected):
    lines, widths = betting_subtree(round_two_openings()[opening])
    assert (len(lines), dict(widths)) == expected

def test_the_raise_war_commits_two_then_eight_then_an_all_in_at_twenty_five():
    # §3: uncapped, pot-limit from a 2-chip pot would commit 2 -> 8 -> 26. The
    # 25 behind the ante truncates the third to an all-in, which is exactly what
    # the 26-chip stack was chosen to do.
    ladder = []
    state = play(BOARD_ONE)
    for _ in range(3):
        state = state.apply(P)
        ladder.append(max(chip_state(state.betting).in_round))
    assert ladder == [2, 8, 25]
    assert chip_state(state.betting).behind == (0, 17)

def test_a_fourth_raise_is_never_offered_because_the_stack_ran_out():
    state = play(BOARD_ONE, P, P, P)
    assert state.legal_actions() == (F, X)
    assert legal_actions_for(state) == (F, X)

def test_folding_is_not_offered_when_checking_is_free():
    assert play(BOARD_ONE).legal_actions() == (X, P)
    assert play(BOARD_ONE, X).legal_actions() == (X, P)
    assert play(BOARD_ONE, P).legal_actions() == (F, X, P)

def test_both_all_in_leaves_round_two_with_nothing_to_decide():
    state = play(BOARD_ONE, P, P, P, X)
    assert state.is_draw_decision()  # all-in players still draw
    state = play(BOARD_ONE, P, P, P, X, Action.THROW_NONE, Action.THROW_NONE, BOARD_TWO)
    chips = chip_state(state.betting)
    assert (chips.pot, chips.behind) == (2 * STACK, (0, 0))
    assert state.kind is NodeKind.TERMINAL

def test_pot_limit_sizing_matches_the_stated_formula():
    # bet = min(pot, behind); raise = to_call + min(pot + to_call, behind - to_call).
    after_bet = chip_state(play(BOARD_ONE, P).betting)
    assert (after_bet.pot, after_bet.behind) == (4, (23, 25))
    after_raise = chip_state(play(BOARD_ONE, P, P).betting)
    assert (after_raise.pot, after_raise.behind) == (12, (23, 17))
    assert after_raise.owed_by(0) == 6

def test_the_shorthand_and_the_labels_read_the_two_jobs_of_check_call():
    assert line_symbol((X, P, X)) == "xpc"
    assert action_label(X, ()) == "check"
    assert action_label(X, (P,)) == "call"
    assert action_label(P, ()) == "bet"
    assert action_label(P, (P,)) == "raise"
    assert action_label(Action.THROW_NONE) == "stand pat"
    assert action_label(Action.THROW_TOP) == "throw top"

# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------

def test_the_draw_offers_four_throws_of_at_most_one_card():
    # The cap is one, so the draw node is 4 wide rather than 7: stand pat, or
    # throw exactly one of the three. A player can improve a card here, never a
    # pair of them.
    state = play(BOARD_ONE, X, X)
    assert state.is_draw_decision()
    assert state.legal_actions() == DRAW_ACTIONS
    assert THROW_CAP == 1
    assert len(DRAW_ACTIONS) == 4
    assert {throw_count(action) for action in DRAW_ACTIONS} == {0, 1}
    assert all(throw_count(action) <= THROW_CAP for action in DRAW_ACTIONS)
    assert len(set(DISCARD_MASK.values())) == 4

def test_standing_pat_does_not_wake_the_deck_and_a_throw_does():
    draw = play(BOARD_ONE, X, X)
    assert draw.apply(Action.THROW_NONE).kind is NodeKind.DECISION
    thrown = draw.apply(Action.THROW_MID)
    assert thrown.kind is NodeKind.CHANCE
    outcomes = thrown.chance_outcomes()
    # One replacement, drawn from the eight cards the deal and the first board
    # card left, and each is equally likely.
    assert all(len(cards) == 1 for cards, _ in outcomes)
    assert len(outcomes) == 8
    assert sum(probability for _, probability in outcomes) == pytest.approx(1.0)

def test_the_deck_deals_the_replacement_and_the_hole_fills_back_up():
    state = play(BOARD_ONE, X, X, Action.THROW_LOW)
    assert len(state.holes[0]) == 2
    assert state.draws == (DrawSignal(1),)
    replacement = state.chance_outcomes()[0][0]
    filled = state.apply_chance(replacement)
    assert len(filled.holes[0]) == 3
    assert set(replacement) <= set(filled.holes[0])
    assert filled.is_draw_decision() and filled.current_player == 1

def test_a_throw_names_its_cards_by_canonical_position():
    draw = play(BOARD_ONE, X, X)
    ordered = draw_order(hole=draw.holes[0], discarded=(), board=draw.board)
    assert ordered == parse_cards("2c 3c 6d")
    thrown = draw.apply(Action.THROW_TOP)
    assert thrown.discards[0] == parse_cards("6d")
    assert thrown.holes[0] == parse_cards("2c 3c")

def test_a_relabelled_position_throws_the_relabelled_card():
    # The trap `draw_order` exists for. These two positions are one position
    # under a suit swap, so they share an infoset — and a single strategy plays
    # them both. If the draw mask read physical suits, "throw the lowest" would
    # throw the board-suited card in one and the offsuit card in the other.
    swap = (1, 0, 2)
    mirror = (relabel(swap, HOLES[0]), relabel(swap, HOLES[1]))
    here = play(BOARD_ONE, X, X)
    there = play(relabel(swap, BOARD_ONE), X, X, holes=mirror)
    assert here.infoset() == there.infoset()
    here_order = draw_order(hole=here.holes[0], discarded=(), board=here.board)
    there_order = draw_order(hole=there.holes[0], discarded=(), board=there.board)
    assert relabel(swap, here_order) == there_order
    assert relabel(swap, here.apply(Action.THROW_LOW).discards[0]) == (
        there.apply(Action.THROW_LOW).discards[0]
    )

def test_every_throw_maps_a_whole_infoset_to_one_infoset():
    """The draw mask's soundness, checked exhaustively rather than by example.

    A ledger's four columns are shared by every concrete position in an
    infoset, so an action has to mean one decision across all of them: throwing
    column k must land every member of the class in the *same* child class. It
    does — over all 970 canonical (hole, board-1) classes and the 5,460 concrete
    pictures behind them, zero classes split.

    The second half is why `draw_order` exists. Order the hole by physical suit
    numbers instead of canonical labels and 700 (class, action) pairs split
    apart: one strategy, two different cards thrown, and a solve that is wrong
    with no traceback. That is the §7.3 trap in its most concrete form.
    """

    def child_classes(order_by, mask):
        classes = defaultdict(set)
        for hole in combinations(DECK, 3):
            for card in DECK:
                if card in hole:
                    continue
                board = (card,)
                ordered = order_by(hole, board)
                thrown = tuple(c for i, c in enumerate(ordered) if mask >> i & 1)
                kept = tuple(c for c in hole if c not in thrown)
                classes[canonical(hole, (), board)].add(canonical(kept, thrown, board))
        return classes

    def canonically(hole, board):
        return draw_order(hole=hole, discarded=(), board=board)

    def physically(hole, board):
        return tuple(sorted(hole))

    def splits(order_by, mask):
        return sum(len(images) > 1 for images in child_classes(order_by, mask).values())

    split_canonical = sum(splits(canonically, mask) for mask in DISCARD_MASK.values())
    split_physical = sum(splits(physically, mask) for mask in DISCARD_MASK.values())
    assert split_canonical == 0
    assert split_physical == 700

def test_the_cap_of_one_costs_eleven_thousand_post_draw_keys():
    """The number `THROW_CAP`'s comment claims, re-derived rather than asserted.

    A post-draw key is (what I hold, what I threw, the board so far) up to suit
    relabelling — the private position a ledger is allocated against once the
    draw is over. At a cap of one there are 11,140 of them: 970 for standing pat
    and 10,170 for the three one-card throws. At a cap of two the same count is
    61,590, which is why the cap is a bigger lever on this rung's cost than the
    deck size is.

    This is not PR 3's census, which measures *infosets* — these differ by the
    opponent's draw signal, the second board card and the betting lines. It
    pins only the term the cap actually moves.
    """
    stood_pat, drew = set(), set()
    for hole in combinations(DECK, HOLE_CARDS):
        for board_card in (card for card in DECK if card not in hole):
            board = (board_card,)
            stood_pat.add(canonical(hole, (), board))
            ordered = draw_order(hole=hole, discarded=(), board=board)
            for mask in DISCARD_MASK.values():
                thrown = tuple(card for bit, card in enumerate(ordered) if mask >> bit & 1)
                if not thrown:
                    continue
                kept = tuple(card for card in hole if card not in thrown)
                for drawn in DECK:
                    if drawn not in hole and drawn != board_card:
                        drew.add(canonical(tuple(sorted(kept + (drawn,))), thrown, board))
    assert (len(stood_pat), len(drew)) == (970, 10_170)
    assert len(stood_pat | drew) == 11_140

def test_a_discard_is_gone_from_the_deck_and_never_comes_back():
    state = play(BOARD_ONE, X, X, Action.THROW_LOW)
    discarded = set(state.discards[0])
    assert discarded and not discarded & set(state.remaining_deck())
    assert all(not discarded & set(cards) for cards, _ in state.chance_outcomes())
    filled = state.apply_chance(state.chance_outcomes()[0][0])
    assert not discarded & set(filled.holes[0] + filled.holes[1])

@pytest.mark.parametrize("first", DRAW_ACTIONS)
@pytest.mark.parametrize("second", DRAW_ACTIONS)
def test_the_deck_never_runs_dry_whatever_both_players_throw(first, second):
    state = play(BOARD_ONE, X, X, first)
    if state.is_chance_node():
        state = state.apply_chance(state.chance_outcomes()[0][0])
    state = state.apply(second)
    if state.is_chance_node() and state._pending_draw is not None:
        state = state.apply_chance(state.chance_outcomes()[0][0])
    # 15 cards, minus 6 in hands, minus the board card, minus both discards.
    thrown = throw_count(first) + throw_count(second)
    assert len(state.remaining_deck()) == 8 - thrown
    assert state.kind is NodeKind.CHANCE
    board_two = state.chance_outcomes()
    assert len(board_two) == 8 - thrown
    assert len(state.apply_chance(board_two[0][0]).remaining_deck()) == 7 - thrown

def test_only_the_count_of_a_draw_is_public():
    signal = DrawSignal(1)
    assert signal.count == 1
    assert [field for field in DrawSignal.__dataclass_fields__] == ["count"]
    with pytest.raises(ValueError, match="throws 0 to 1"):
        DrawSignal(2)

# ---------------------------------------------------------------------------
# Infosets
# ---------------------------------------------------------------------------

def test_an_infoset_never_holds_the_opponent_s_cards():
    # P0 is to act, so the two deals below differ only in what P1 was dealt —
    # and that indistinguishability is the game.
    mine = play(BOARD_ONE, holes=HOLES).infoset()
    others = (HOLES[0], parse_cards("3d 4d 5d"))
    assert mine == play(BOARD_ONE, holes=others).infoset()

def test_perfect_recall_keeps_what_this_player_threw():
    # The plan's example, isolated: two histories that reach the same hole and
    # the same public draw counts, having thrown different cards. Drop
    # `discarded` from the key and they collide — two positions the player can
    # plainly tell apart, with different cards missing from the deck.
    def round_two(dealt, thrown_replacement):
        return play(
            BOARD_ONE,
            X,
            X,
            Action.THROW_LOW,
            thrown_replacement,
            Action.THROW_NONE,
            BOARD_TWO,
            holes=(parse_cards(dealt), HOLES[1]),
        ).infoset()

    threw_club = round_two("2c 3c 6d", parse_cards("6c"))
    threw_heart = round_two("2h 3c 6d", parse_cards("6c"))
    def everything_else(key):
        return (key.player, key.hole, key.board, key.draws, key.betting)

    assert everything_else(threw_club) == everything_else(threw_heart)
    assert threw_club.discarded != threw_heart.discarded
    assert threw_club != threw_heart

def test_which_cards_were_kept_and_which_drawn_is_merged():
    # Lossless, not an abstraction: both players hold the same three cards, have
    # the same cards missing from the deck and pay out identically — so the two
    # histories are not merely one infoset, they are one *state*, and nothing
    # downstream can tell them apart.
    def after_drawing(dealt, replacement):
        return play(
            BOARD_ONE, X, X, Action.THROW_LOW, holes=(parse_cards(dealt), HOLES[1])
        ).apply_chance(parse_cards(replacement))

    drew_the_club = after_drawing("2c 3c 6d", "6c")
    drew_the_diamond = after_drawing("2c 3c 6c", "6d")
    assert drew_the_club.holes[0] == drew_the_diamond.holes[0] == parse_cards("3c 6c 6d")
    assert drew_the_club.discards[0] == drew_the_diamond.discards[0] == parse_cards("2c")
    assert drew_the_club == drew_the_diamond

def test_the_opponent_s_draw_count_is_part_of_the_key():
    stood_pat = play(BOARD_ONE, X, X, Action.THROW_NONE)
    drew_one = play(BOARD_ONE, X, X, Action.THROW_LOW).apply_chance(parse_cards("6c"))
    assert stood_pat.current_player == drew_one.current_player == 1
    assert stood_pat.infoset().draws == (DrawSignal(0),)
    assert drew_one.infoset().draws == (DrawSignal(1),)
    assert stood_pat.infoset() != drew_one.infoset()

def test_a_ledger_is_as_wide_as_the_legal_actions_at_every_spot():
    for moves in (
        (BOARD_ONE,),
        (BOARD_ONE, P),
        (BOARD_ONE, X, X),
        (BOARD_ONE, X, X, Action.THROW_NONE),
    ):
        state = play(*moves)
        assert state.infoset().legal_actions() == state.legal_actions()
        assert state.infoset().player == state.current_player

def test_an_infoset_refuses_a_key_the_rules_cannot_produce():
    good = play(BOARD_ONE, X).infoset()
    with pytest.raises(ValueError, match="canonical form"):
        InfoSet(
            player=good.player,
            hole=parse_cards("2h 3h 6d"),
            discarded=(),
            board=parse_cards("4h"),
            draws=(),
            betting=good.betting,
        )
    with pytest.raises(ValueError, match="acts"):
        InfoSet(
            player=0,
            hole=good.hole,
            discarded=(),
            board=good.board,
            draws=(),
            betting=good.betting,
        )
    with pytest.raises(ValueError, match="has not made yet"):
        InfoSet(
            player=1,
            hole=good.hole,
            discarded=parse_cards("2h"),
            board=good.board,
            draws=(),
            betting=good.betting,
        )

# ---------------------------------------------------------------------------
# The split pot
# ---------------------------------------------------------------------------

# Four deals, one per way the pot can break. Every one of them is a genuine
# showdown driven through the state machine, not a hand-built terminal.
SCOOP = (parse_cards("6c 6d 6h"), parse_cards("2c 3d 4h"))
HALF_EACH = (parse_cards("6c 6d 6h"), parse_cards("5h 2c 3d"))
QUARTER = (parse_cards("2c 3c 4h"), parse_cards("2d 3h 4c"))

def showdown(holes, board_one, board_two):
    return play(
        board_one,
        X,
        X,
        Action.THROW_NONE,
        Action.THROW_NONE,
        board_two,
        X,
        X,
        holes=holes,
    )

def test_a_scoop_takes_the_whole_pot():
    state = showdown(SCOOP, parse_cards("5c"), parse_cards("5d"))
    assert pot_shares(state.holes, state.board) == (1.0, 0.0)
    assert state.returns() == (1.0, -1.0)  # nothing but the antes went in

def test_winning_one_half_each_is_a_wash():
    state = showdown(HALF_EACH, parse_cards("5c"), parse_cards("5d"))
    holes, board = state.holes, state.board
    assert inner_score(holes[0]) > inner_score(holes[1])
    assert outer_score(holes[1], board) > outer_score(holes[0], board)
    assert pot_shares(holes, board) == (0.5, 0.5)
    assert state.returns() == (0.0, 0.0)

def test_a_chopped_half_pays_a_quarter():
    state = showdown(QUARTER, parse_cards("5c"), parse_cards("6c"))
    holes, board = state.holes, state.board
    assert inner_score(holes[0]) == inner_score(holes[1])  # both a straight
    assert outer_score(holes[0], board) > outer_score(holes[1], board)  # a flush
    assert pot_shares(holes, board) == (0.75, 0.25)
    assert state.returns() == (0.5, -0.5)

def test_the_pot_pays_in_halves_of_whatever_went_in():
    # A bet called in each round: 1 ante + 2 in round 1, then a pot-sized 6 into
    # the 6-chip pot round 1 left. The scoop collects the opponent's 9.
    checked = showdown(SCOOP, parse_cards("5c"), parse_cards("5d"))
    bet_both_rounds = play(
        parse_cards("5c"),
        P,
        X,
        Action.THROW_NONE,
        Action.THROW_NONE,
        parse_cards("5d"),
        P,
        X,
        holes=SCOOP,
    )
    assert checked.contributions() == (1, 1)
    assert bet_both_rounds.contributions() == (9, 9)
    assert bet_both_rounds.returns() == (9.0, -9.0)

def test_a_fold_costs_the_folder_exactly_what_they_had_matched():
    # Folding to a re-raise after betting costs 3, not the ante: 1 ante plus the
    # 2 the bet had put in. The rest of the stack goes home.
    folded = play(BOARD_ONE, P, P, F)
    assert folded.contributions() == (3, 9)
    assert folded.returns() == (-3.0, 3.0)
    assert play(BOARD_ONE, P, F).returns() == (1.0, -1.0)

def test_returns_sum_to_zero_across_random_hands():
    for state in playouts(200):
        assert sum(state.returns()) == pytest.approx(0.0)

def test_swapping_the_two_holes_at_a_showdown_negates_the_returns():
    # Only at showdowns: a fold is decided by the betting, and swapping cards
    # that nobody saw cannot change what a fold pays.
    showdowns = 0
    for state in playouts(80):
        if len(state.betting) != 2 or state.betting[-1][-1:] == (F,):
            continue
        mirrored = MiniState(
            holes=(state.holes[1], state.holes[0]),
            discards=(state.discards[1], state.discards[0]),
            board=state.board,
            draws=tuple(reversed(state.draws)),
            betting=state.betting,
        )
        forward, backward = state.returns(), mirrored.returns()
        assert backward == pytest.approx((-forward[0], -forward[1]))
        showdowns += 1
    assert showdowns > 10

def test_a_relabelled_hand_pays_the_same():
    swap = (2, 0, 1)
    for state in playouts(40):
        mirrored = MiniState(
            holes=(relabel(swap, state.holes[0]), relabel(swap, state.holes[1])),
            discards=(relabel(swap, state.discards[0]), relabel(swap, state.discards[1])),
            board=relabel(swap, state.board),
            draws=state.draws,
            betting=state.betting,
        )
        assert mirrored.returns() == state.returns()

# ---------------------------------------------------------------------------
# Every accessor belongs to one node kind, and impossible states are refused
# ---------------------------------------------------------------------------

def test_an_accessor_refuses_the_two_kinds_it_does_not_belong_to():
    chance = MiniState(holes=HOLES)
    decision = play(BOARD_ONE)
    terminal = play(BOARD_ONE, P, F)
    with pytest.raises(ValueError, match="deck is to act"):
        chance.returns()
    with pytest.raises(ValueError, match="deck is to act"):
        chance.legal_actions()
    with pytest.raises(ValueError, match="player is to act"):
        decision.returns()
    with pytest.raises(ValueError, match="player is to act"):
        decision.chance_outcomes()
    with pytest.raises(ValueError, match="hand is over"):
        terminal.legal_actions()
    with pytest.raises(ValueError, match="hand is over"):
        terminal.infoset()

def test_an_illegal_action_is_refused_rather_than_applied():
    with pytest.raises(ValueError, match="not legal"):
        play(BOARD_ONE, F)
    with pytest.raises(ValueError, match="not legal"):
        play(BOARD_ONE, X, X, P)  # the draw takes throws, not bets
    with pytest.raises(ValueError, match="not legal"):
        play(BOARD_ONE, P, Action.THROW_LOW)

@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"holes": (parse_cards("2c 3c 6d"), parse_cards("2c 4h 5h"))}, "dealt twice"),
        ({"holes": HOLES, "board": BOARD_ONE}, "round 1 opens"),
        ({"holes": HOLES, "betting": ((X,),)}, "round 1 opens"),
        (
            {"holes": HOLES, "board": BOARD_ONE, "betting": ((X, X, P),)},
            "goes on after the round ended",
        ),
        ({"holes": HOLES, "board": BOARD_ONE, "betting": ((F,),)}, "not legal"),
        ({"holes": HOLES, "board": BOARD_ONE, "betting": ((1, 1),)}, "Action members"),
        (
            {
                "holes": (parse_cards("3c 6d"), HOLES[1]),
                "discards": (parse_cards("2c"), ()),
                "board": BOARD_ONE,
                "betting": ((X, X),),
            },
            "has not drawn",
        ),
        (
            {
                "holes": HOLES,
                "board": BOARD_ONE + BOARD_TWO,
                "betting": ((X, X), ()),
            },
            "both draws finish",
        ),
    ],
)
def test_a_position_the_rules_cannot_produce_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MiniState(**kwargs)
