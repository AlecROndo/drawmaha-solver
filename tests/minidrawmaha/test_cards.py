"""The 15-card deck and the one collapse rung 3 is allowed: relabelling suits.

The census figures asserted here are the plan's §4.7 table — 455 hands, 95
classes, 970 and 5,160 canonical pairs. They are re-derived by enumeration
rather than read from a fixture, because the enumeration is cheap at this deck
size and a fixture would only record what this module already computes.
"""

from itertools import combinations, permutations

import pytest

from drawmaha_solver.minidrawmaha.cards import (
    DECK,
    N_RANKS,
    N_SUITS,
    RANKS,
    SUITS,
    Card,
    canonical,
    canonical_relabelling,
    hand_symbol,
    parse_cards,
    relabel,
)

# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------

def test_the_deck_is_fifteen_cards_five_ranks_by_three_suits():
    assert (N_RANKS, N_SUITS) == (5, 3)
    assert len(DECK) == 15
    assert len(set(DECK)) == 15

def test_every_rank_comes_in_every_suit():
    assert {(rank, suit) for rank in RANKS for suit in SUITS} == set(DECK)

def test_three_suits_is_what_makes_trips_possible():
    # §4.1: with two suits a rank holds two cards and the inner half loses its
    # top category. Three cards of a rank exist exactly because there are three.
    for rank in RANKS:
        assert len([card for card in DECK if card.rank == rank]) == 3

def test_the_deck_floor_leaves_five_cards_of_slack():
    # §4.1, at a throw cap of one: 6 private + board 1 + both players throwing
    # 1 + board 2 = 10 cards out of 15. No line can exhaust the stub, so the
    # last board card always has several outcomes and no reshuffle rule is
    # needed. (At a cap of two the floor was 12 and the slack 3.)
    assert len(DECK) - (2 * 3 + 1 + 2 * 1 + 1) == 5

def test_cards_render_and_parse_back():
    assert hand_symbol(DECK[:4]) == "2c 2d 2h 3c"
    assert parse_cards("6h 2c") == (Card(4, 2), Card(0, 0))
    assert parse_cards(hand_symbol(DECK)) == DECK

def test_parsing_refuses_a_card_outside_the_deck():
    with pytest.raises(ValueError, match="7s"):
        parse_cards("7s")

# ---------------------------------------------------------------------------
# Canonical relabelling
# ---------------------------------------------------------------------------

HOLE = parse_cards("2c 3c 6d")
BOARD = parse_cards("4c")

def test_relabelling_suits_never_moves_a_rank():
    for perm in permutations(SUITS):
        assert [card.rank for card in relabel(perm, HOLE)] == [card.rank for card in HOLE]

def test_canonical_is_idempotent():
    once = canonical(HOLE, BOARD)
    assert canonical(*once) == once

def test_canonical_sorts_within_each_group():
    for group in canonical(parse_cards("6d 2c 3c"), BOARD):
        assert list(group) == sorted(group)

def test_one_relabelling_applied_to_the_whole_picture_leaves_the_key_fixed():
    # The whole point of the collapse: which physical suit is which carries no
    # information, so every joint relabelling of a position is one position.
    groups = (HOLE, parse_cards("3d"), BOARD)
    key = canonical(*groups)
    for perm in permutations(SUITS):
        assert canonical(*(relabel(perm, group) for group in groups)) == key

def test_relabelling_the_hole_alone_changes_the_key():
    # §7.3's correctness trap, stated as its own fact: the hole's suits mean
    # something only relative to the board's. Here the hole holds two cards of
    # the board's suit; after the swap it holds one, and that is a different
    # position — so the key must move.
    swap = (1, 0, 2)
    assert canonical(relabel(swap, HOLE), BOARD) != canonical(HOLE, BOARD)

def test_canonicalising_the_groups_separately_merges_distinct_positions():
    # The bug the joint call exists to prevent. One suited hole, two boards: a
    # board card of the hole's own suit is a three-flush, one of another suit is
    # nothing. Each group on its own is the same picture in both positions — a
    # suited pair, a lone card — so separate canonicalisation cannot tell them
    # apart and the two would silently share one ledger.
    hole = parse_cards("2c 3c")
    with_flush, without = parse_cards("4c"), parse_cards("4d")
    assert canonical(hole, with_flush) != canonical(hole, without)
    assert canonical(with_flush) == canonical(without)

def test_the_relabelling_is_the_one_that_produces_the_key():
    # `canonical_relabelling` is what game.py orders a hole by, so it must be
    # the very permutation behind the key and not merely one that ties it.
    groups = (HOLE, parse_cards("3d"), BOARD)
    perm = canonical_relabelling(*groups)
    assert tuple(tuple(sorted(relabel(perm, g))) for g in groups) == canonical(*groups)

def test_the_relabelling_is_stable_across_equivalent_pictures():
    # Two pictures related by a swap must canonicalise *through* that swap, so
    # that position k of one hole and position k of the other are the same card
    # under relabelling. game.py's draw mask rides on exactly this.
    swap = (1, 0, 2)
    perm = canonical_relabelling(HOLE, BOARD)
    swapped = canonical_relabelling(relabel(swap, HOLE), relabel(swap, BOARD))
    assert [perm[suit] for suit in SUITS] == [swapped[swap[suit]] for suit in SUITS]

# ---------------------------------------------------------------------------
# The census: how much the collapse actually buys (§4.7)
# ---------------------------------------------------------------------------

HANDS = tuple(combinations(DECK, 3))

def test_four_hundred_and_fifty_five_hands_collapse_to_ninety_five_classes():
    assert len(HANDS) == 455
    classes = {canonical(hand) for hand in HANDS}
    assert len(classes) == 95
    assert len(HANDS) / len(classes) == pytest.approx(4.79, abs=0.005)

def test_hole_and_one_board_card_make_nine_hundred_and_seventy_classes():
    pairs = {
        canonical(hand, (board,))
        for hand in HANDS
        for board in DECK
        if board not in hand
    }
    assert len(pairs) == 970

def test_hole_and_both_board_cards_make_five_thousand_one_hundred_and_sixty():
    pairs = {
        canonical(hand, board)
        for hand in HANDS
        for board in combinations([c for c in DECK if c not in hand], 2)
    }
    assert len(pairs) == 5_160
