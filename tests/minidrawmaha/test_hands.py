"""Both pot halves, graded against the frequency tables in the plan's §4.4.

The two tables are the answer sheet for this module, and they are *measured in
this deck* rather than imported from 52-card convention. Two rows look wrong to
a 52-card eye and both are pinned deliberately: a four-card straight is COMMON
here (a run of four out of five ranks), so it sits below two pair and trips; and
outer high card is the rarest result of all at 0.74%, yet ranks last, because
the floor is the absence of a made hand rather than a rarity claim.

Every frequency below is re-derived by brute force — 455 dealt 3-card hands and
30,030 (hole, board) deals — so a change to the scorers fails here rather than
silently moving the game.
"""

from collections import Counter
from itertools import combinations
from random import Random

import pytest

from drawmaha_solver.minidrawmaha.cards import DECK, parse_cards
from drawmaha_solver.minidrawmaha.hands import (
    BOARD_CARDS,
    HOLE_CARDS,
    OUTER_HOLE_CARDS,
    InnerCategory,
    OuterCategory,
    four_card_score,
    inner_score,
    outer_score,
)

HANDS = tuple(combinations(DECK, HOLE_CARDS))

def deals():
    """Every (hole, board) the outer scorer must rank — all 30,030 of them."""
    for hole in HANDS:
        rest = [card for card in DECK if card not in hole]
        for board in combinations(rest, BOARD_CARDS):
            yield hole, board

# ---------------------------------------------------------------------------
# The inner half: a dealt 3-card hand, over all 455
# ---------------------------------------------------------------------------

# §4.4, with the counts the percentages are rounded from.
INNER_TABLE = (
    (InnerCategory.TRIPS, 5, 1.10),
    (InnerCategory.STRAIGHT_FLUSH, 9, 1.98),
    (InnerCategory.FLUSH, 21, 4.62),
    (InnerCategory.STRAIGHT, 72, 15.82),
    (InnerCategory.PAIR, 180, 39.56),
    (InnerCategory.HIGH_CARD, 168, 36.92),
)

def test_the_inner_half_reproduces_its_frequency_table():
    assert len(HANDS) == 455
    counts = Counter(inner_score(hand).category for hand in HANDS)
    assert counts == {category: count for category, count, _ in INNER_TABLE}
    for category, count, percent in INNER_TABLE:
        assert 100 * count / len(HANDS) == pytest.approx(percent, abs=0.005)

def test_the_inner_ranking_is_rarity_order_with_high_card_pinned_to_the_floor():
    # Every category is rarer than the one below it, except the one stated
    # exception: high card (36.92%) is marginally rarer than a pair (39.56%) and
    # still ranks last, because it is the absence of a hand. That inversion
    # costs 2.6 points of rarity here against 27 in the rejected 12-card deck.
    ranked = [category for category, _, _ in INNER_TABLE]
    assert ranked == sorted(InnerCategory, reverse=True)
    frequencies = {category: count for category, count, _ in INNER_TABLE}
    rarity_ordered = ranked[:-1]
    assert rarity_ordered == sorted(rarity_ordered, key=frequencies.__getitem__)
    assert frequencies[InnerCategory.HIGH_CARD] < frequencies[InnerCategory.PAIR]

def test_three_cards_cannot_make_two_pair_so_there_are_six_categories():
    assert len(InnerCategory) == 6
    assert {score.category for score in map(inner_score, HANDS)} == set(InnerCategory)

@pytest.mark.parametrize(
    ("hand", "category"),
    [
        ("2c 2d 2h", InnerCategory.TRIPS),
        ("2c 3c 4c", InnerCategory.STRAIGHT_FLUSH),
        ("2c 3c 5c", InnerCategory.FLUSH),
        ("2c 3d 4h", InnerCategory.STRAIGHT),
        ("2c 2d 6h", InnerCategory.PAIR),
        ("2c 3d 5h", InnerCategory.HIGH_CARD),
    ],
)
def test_inner_categories_by_hand(hand, category):
    assert inner_score(parse_cards(hand)).category is category

def test_the_top_rank_does_not_wrap_round_to_the_bottom():
    # Rank 4 is the top of this deck and rank 0 the floor; they are not
    # neighbours. There is no wheel to make 6-2-3 a straight, which is why the
    # ranks are numbered rather than given an ace's two faces.
    assert inner_score(parse_cards("6c 2d 3h")).category is InnerCategory.HIGH_CARD
    assert inner_score(parse_cards("4c 5d 6h")).category is InnerCategory.STRAIGHT

def test_inner_kickers_read_the_paired_rank_first():
    pair_of_sixes = inner_score(parse_cards("6c 6d 2h"))
    pair_of_twos = inner_score(parse_cards("2c 2d 6h"))
    assert pair_of_sixes > pair_of_twos
    assert inner_score(parse_cards("2c 2d 6h")) > inner_score(parse_cards("2c 2d 5h"))

def test_suits_never_break_an_inner_tie():
    assert inner_score(parse_cards("2c 3d 5h")) == inner_score(parse_cards("2d 3h 5c"))

# ---------------------------------------------------------------------------
# The outer half: exactly 2 hole cards plus both board cards, over all 30,030
# ---------------------------------------------------------------------------

OUTER_TABLE = (
    (OuterCategory.STRAIGHT_FLUSH, 387, 1.29),
    (OuterCategory.FLUSH, 543, 1.81),
    (OuterCategory.TRIPS, 2_970, 9.89),
    (OuterCategory.TWO_PAIR, 4_920, 16.38),
    (OuterCategory.STRAIGHT, 7_542, 25.11),
    (OuterCategory.PAIR, 13_446, 44.78),
    (OuterCategory.HIGH_CARD, 222, 0.74),
)

def test_the_outer_half_reproduces_its_frequency_table():
    counts = Counter(outer_score(hole, board).category for hole, board in deals())
    total = sum(counts.values())
    assert total == 30_030
    assert counts == {category: count for category, count, _ in OUTER_TABLE}
    for category, count, percent in OUTER_TABLE:
        assert 100 * count / total == pytest.approx(percent, abs=0.005)

def test_all_seven_outer_categories_occur_which_is_why_the_deck_has_five_ranks():
    # At four ranks, "four distinct ranks" IS the whole rank set, so high card
    # and plain flush have nowhere to exist and two of these seven rows would be
    # empty. The fifth rank is bought precisely to fill them.
    assert len(OuterCategory) == 7
    assert {category for category, _, _ in OUTER_TABLE} == set(OuterCategory)
    assert all(count > 0 for _, count, _ in OUTER_TABLE)

def test_a_four_card_straight_is_common_here_so_two_pair_and_trips_beat_it():
    # The row that looks wrong to a 52-card eye. A run of four out of five ranks
    # is easy to make (25.11%), two pair is not (16.38%) — and ranking by
    # measured rarity is the same rule that puts straights above both in a
    # 52-card deck. Do not "fix" this by importing poker convention.
    assert OuterCategory.TWO_PAIR > OuterCategory.STRAIGHT
    assert OuterCategory.TRIPS > OuterCategory.STRAIGHT

def test_outer_high_card_is_the_rarest_result_and_still_ranks_last():
    # The second row that looks wrong. The scorer takes the best of three
    # two-card choices, so ending with nothing at all is a near-miracle at
    # 0.74% — rarer than a straight flush. It is still the floor, by the one
    # stated convention: the floor means "no made hand", not "hardest to make".
    counts = {category: count for category, count, _ in OUTER_TABLE}
    assert min(counts, key=counts.__getitem__) is OuterCategory.HIGH_CARD
    assert min(OuterCategory) is OuterCategory.HIGH_CARD

@pytest.mark.parametrize(
    ("hole", "board", "category"),
    [
        ("2c 3c 6h", "4c 6c", OuterCategory.FLUSH),
        ("2c 3c 6h", "4c 5c", OuterCategory.STRAIGHT_FLUSH),
        ("2c 2d 6h", "2h 3c", OuterCategory.TRIPS),
        ("2c 2d 6h", "3c 3d", OuterCategory.TWO_PAIR),
        ("2c 3d 6h", "4c 5d", OuterCategory.STRAIGHT),
        ("2c 6d 6h", "3c 4c", OuterCategory.PAIR),
        ("3c 4c 5c", "2c 6d", OuterCategory.HIGH_CARD),
    ],
)
def test_outer_categories_by_deal(hole, board, category):
    assert outer_score(parse_cards(hole), parse_cards(board)).category is category

def test_the_outer_half_takes_exactly_two_hole_cards():
    # Dealt trips, the outer half can still only bring two of them, so the
    # strongest hand available is a pair. This is the shrunken Omaha rule and
    # the whole reason the two halves disagree about which hand is good.
    trips, board = parse_cards("2c 2d 2h"), parse_cards("3c 4c")
    assert OUTER_HOLE_CARDS == 2
    assert inner_score(trips).category is InnerCategory.TRIPS
    assert outer_score(trips, board).category is OuterCategory.PAIR

def test_the_outer_half_takes_the_best_of_its_three_choices():
    # Two of the three pairings make nothing; the scorer must not settle for the
    # first one it is handed.
    hole, board = parse_cards("2c 6d 6h"), parse_cards("3c 4c")
    choices = combinations(hole, OUTER_HOLE_CARDS)
    made = [four_card_score(chosen + board) for chosen in choices]
    assert sorted(score.category for score in made) == [
        OuterCategory.HIGH_CARD,
        OuterCategory.HIGH_CARD,
        OuterCategory.PAIR,
    ]
    assert outer_score(hole, board) == max(made)

def test_four_of_a_kind_cannot_happen_in_three_suits():
    # The outer half ranks four cards, but a rank owns only three of them, so
    # quads are not a category this deck can produce.
    assert not hasattr(OuterCategory, "QUADS")
    assert all(
        max(Counter(rank for rank in outer_score(hole, board).ranks).values()) <= 3
        for hole, board in deals()
    )

# ---------------------------------------------------------------------------
# Both scorers order hands totally, and tie only on genuine ties
# ---------------------------------------------------------------------------

def test_an_inner_tie_means_the_same_category_and_the_same_ranks():
    # Suits are unordered, so two hands tie exactly when nothing but their suits
    # differs. Checked as a bijection over all 455 hands rather than sampled:
    # every score maps to one (category, ranks) fact and back.
    by_score, by_fact = {}, {}
    for hand in HANDS:
        score = inner_score(hand)
        fact = (score.category, tuple(sorted((card.rank for card in hand), reverse=True)))
        by_score.setdefault(score, fact)
        by_fact.setdefault(fact, score)
        assert by_score[score] == fact
        assert by_fact[fact] == score

@pytest.mark.parametrize("half", ["inner", "outer"])
def test_comparison_is_antisymmetric_and_transitive(half):
    rng = Random(20260917)
    if half == "inner":
        sample = [inner_score(hand) for hand in rng.sample(HANDS, 40)]
    else:
        board = parse_cards("4c 5d")
        holes = [hand for hand in HANDS if not set(hand) & set(board)]
        sample = [outer_score(hole, board) for hole in rng.sample(holes, 40)]
    for left in sample:
        for right in sample:
            assert (left < right) + (right < left) + (left == right) == 1
    for _ in range(500):
        a, b, c = (sample[rng.randrange(len(sample))] for _ in range(3))
        if a <= b <= c:
            assert a <= c

# ---------------------------------------------------------------------------
# Both scorers refuse a hand of the wrong size
# ---------------------------------------------------------------------------

def test_the_scorers_refuse_a_hand_of_the_wrong_size():
    with pytest.raises(ValueError, match="three"):
        inner_score(parse_cards("2c 3d"))
    with pytest.raises(ValueError, match="three"):
        outer_score(parse_cards("2c 3d"), parse_cards("4c 5d"))
    with pytest.raises(ValueError, match="both board cards"):
        outer_score(parse_cards("2c 3d 6h"), parse_cards("4c"))
