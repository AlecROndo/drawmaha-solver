"""The census: how big mini-drawmaha is, and the committed referee that pins it.

Rung 2 could ask OpenSpiel how big Leduc was. Nothing implements Drawmaha, so
rung 3 manufactures its own referee: `scripts/generate_minidrawmaha_census.py`
counts what the enumerator yields and writes `census.json`, and this file
re-derives every number in that file from the rules and demands agreement. A
change to the deck, the throw cap or the betting tree therefore fails here
loudly, instead of quietly shipping a solver for a different game.

The two derivations are deliberately different:

* the fixture's totals were produced by **consuming the generator** — 3.1
  million `InfoSet` objects;
* the tests below rebuild them as a **sum of products** over the enumerator's
  two building blocks, in about twenty-five seconds, and check a 150,000-key
  prefix of the generator against that product.

Agreement between the two is the real check: a generator that skipped a public
point, or yielded one twice, would match neither. The whole generator count is
here as well, behind `MINIDRAWMAHA_FULL_CENSUS=1`, so the default run stays
short — the fast tests already move whenever the slow one would.

The grain is then proved in **both** directions, because a census counted at the
wrong grain counts a different game from the one the solver will play:
*coverage* — every infoset random playouts reach is one the enumerator yields —
and *constructibility* — a sampled enumerated key can be dealt back into a
`MiniState` whose `.infoset()` is that key.
"""

import json
import os
from dataclasses import replace
from functools import cache
from itertools import islice
from pathlib import Path
from random import Random

import pytest

from drawmaha_solver.minidrawmaha.cards import DECK, canonical
from drawmaha_solver.minidrawmaha.enumeration import (
    all_infosets,
    merged,
    private_keys,
    public_decision_points,
    public_tree,
)
from drawmaha_solver.minidrawmaha.game import (
    DISCARD_MASK,
    DRAW_ACTIONS,
    THROW_CAP,
    InfoSet,
    MiniState,
    NodeKind,
    draw_order,
)
from drawmaha_solver.minidrawmaha.hands import HOLE_CARDS

CENSUS = json.loads((Path(__file__).parent / "census.json").read_text())

FULL_RUN = os.environ.get("MINIDRAWMAHA_FULL_CENSUS") == "1"

# How many cards a player may already have thrown, at whichever cap the rules
# are set to. Read from `THROW_CAP` rather than written out, because the cap is
# this rung's declared cost lever (`game.py:104`) and a referee that spelled out
# its own key shapes would pass a cap change while silently refereeing the
# game it was written for.
THROW_RANGE = tuple(range(THROW_CAP + 1))

# Plan §4.3's measured post-draw key counts for the 15-card deck, one per cap.
# This is the one cross-check that comes from OUTSIDE the repo: a design-time
# census walked draw histories to get them, while the enumerator builds the
# same set straight in key space. Keyed by cap so moving the lever meets a
# number that was already predicted, instead of a pin that has to be re-typed.
PLAN_POST_DRAW_KEYS = {1: 11_140, 2: 61_590, 3: 212_440}

# Enough of the generator to cross several public points and both board sizes
# in about a second. The prefix test is not the global claim — the signature
# test below is — it is the check that the generator's loop agrees with the
# product on keys it actually emitted.
PREFIX = 150_000

# ---------------------------------------------------------------------------
# The public tree: 141 decision points that do not depend on a single card
# ---------------------------------------------------------------------------

def test_the_public_tree_is_the_shape_the_census_recorded():
    counts = {kind.value: 0 for kind in NodeKind}
    for point in public_tree():
        counts[point.kind.value] += 1
    assert counts == {
        "decision": CENSUS["public_tree"]["decision_points"],
        "chance": CENSUS["public_tree"]["chance_points"],
        "terminal": CENSUS["public_tree"]["terminal_points"],
    }

def test_the_public_decision_points_split_by_stage_as_recorded():
    stages = {"round_one": 0, "draw": 0, "round_two": 0}
    for point in public_decision_points():
        stages[_stage(point.is_draw_decision, len(point.betting))] += 1
    assert stages == CENSUS["public_tree"]["decision_points_by_stage"]

def test_round_one_still_has_the_plan_s_eight_decision_points():
    # Four two-wide and four three-wide, which is `test_game.py`'s round-1
    # shape read off the public tree instead of a betting subtree — the same
    # 13 lines seen from the side the census counts them on.
    widths = sorted(
        point.width
        for point in public_decision_points()
        if not point.is_draw_decision and len(point.betting) == 1
    )
    assert widths == [2, 2, 2, 2, 3, 3, 3, 3]

def test_the_cap_widens_the_draw_twice_over_once_per_seat():
    # The draw is the only place a ledger is wider than three, and the cap is
    # the lever on it. It pulls TWICE, which is why the cap dominates the
    # census: it sets the ledger width at every draw point, and it sets how
    # many draw points P1 has, because P0's count is public by the time P1
    # acts. P0 draws once per round-1 closing line — 7, whatever the cap.
    draws = [point for point in public_decision_points() if point.is_draw_decision]
    assert {point.width for point in draws} == {len(DRAW_ACTIONS)}
    seats = [point.player for point in draws]
    assert (seats.count(0), seats.count(1)) == (7, 7 * (THROW_CAP + 1))

def test_two_public_points_can_never_share_an_infoset():
    # This is why counting the generator is enough and no 3.1-million-entry
    # set has to be built: `player`, `draws` and `betting` are all carried in
    # the key itself, so two distinct public points cannot produce one key,
    # and within a point the private keys are distinct by construction.
    points = public_decision_points()
    signatures = {(point.player, point.draws, point.betting) for point in points}
    assert len(signatures) == len(points)

def test_two_routes_to_one_public_point_add_their_concrete_histories():
    # Three of the four throws take one card, so the walk arrives at the same
    # public point three times over and the weights must accumulate rather than
    # overwrite — otherwise the concrete count below is silently short.
    point = next(p for p in public_decision_points() if p.is_draw_decision)
    assert merged(point, point).concrete_nodes == 2 * point.concrete_nodes

def test_two_arrivals_that_are_not_one_spot_refuse_to_merge():
    # The guard, exercised. The public record deciding who acts is a property
    # of the rules, not of the enumerator, so a rules change could break it —
    # and a merge that went ahead anyway would report a smaller game with
    # nothing failing.
    point = public_decision_points()[0]
    impostor = replace(point, player=1 - point.player)
    with pytest.raises(ValueError, match="two different decisions"):
        merged(point, impostor)

def test_the_concrete_tree_is_the_one_nobody_walks():
    # The ratio that forces the whole factorisation: one concrete node per
    # (deal, board, actions, replacements) history, against one public point
    # per position anybody has to reason about. Counted exactly, by weighting
    # the public walk with the chance outcomes it stands in for. Asserted as a
    # ratio rather than an absolute, because the absolute moves with the deck
    # and the cap while the reason for factorising does not: a walk that is
    # eight orders of magnitude wider than its own public tree is unwalkable at
    # any setting this rung would consider.
    concrete = {kind.value: 0 for kind in NodeKind}
    for point in public_tree():
        concrete[point.kind.value] += point.concrete_nodes
    assert concrete == CENSUS["concrete_tree"]
    assert concrete["decision"] > 10**8 * len(public_decision_points())

# ---------------------------------------------------------------------------
# The private keys: what one player can see, once the suits are relabelled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", sorted(CENSUS["private_keys"]["by_shape"]))
def test_each_private_key_shape_is_the_recorded_size(shape):
    board_cards, discards = (int(part) for part in shape.split("_"))
    keys = private_keys(board_cards=board_cards, discards=discards)
    assert len(keys) == CENSUS["private_keys"]["by_shape"][shape]

def test_the_post_draw_key_count_is_the_one_the_plan_predicted_for_this_cap():
    # §4.3's headline column: canonical (hand, thrown, board 1) keys. The
    # design-time census walked draw histories to count them; the enumerator
    # builds the same set straight in key space, and must land on the same
    # number at whichever cap the rules are set to.
    assert len(DECK) == 15, "PLAN_POST_DRAW_KEYS is measured for the 15-card deck"
    total = sum(len(private_keys(board_cards=1, discards=k)) for k in THROW_RANGE)
    assert total == PLAN_POST_DRAW_KEYS[THROW_CAP]
    assert CENSUS["private_keys"]["post_draw_board_one_total"] == total

def test_the_two_pinned_pair_counts_come_back_out_of_the_key_sets():
    # `test_cards.py` pins 970 and 5,160 from the deck alone. A key with
    # nothing thrown is exactly such a pair, so the two files must agree.
    assert len(private_keys(board_cards=1, discards=0)) == 970
    assert len(private_keys(board_cards=2, discards=0)) == 5_160

def test_every_private_key_is_already_canonical_and_disjoint():
    # Canonical because `InfoSet` refuses anything else, disjoint because a
    # card in two places is the one mistake this rung cannot make. Five hundred
    # per shape rather than all 343,400: the property is a fact about how the
    # keys are built, so a sample from every shape catches a broken build just
    # as well and leaves the suite short.
    rng = Random(20260918)
    for board_cards in (1, 2):
        for discards in THROW_RANGE:
            keys = private_keys(board_cards=board_cards, discards=discards)
            for index in rng.sample(range(len(keys)), min(len(keys), 500)):
                hole, thrown, board = keys[index]
                assert canonical(hole, thrown, board) == (hole, thrown, board)
                cards = hole + thrown + board
                assert len(set(cards)) == len(cards)

# ---------------------------------------------------------------------------
# The infoset count: the gate's own number
# ---------------------------------------------------------------------------

def _stage(is_draw: bool, rounds: int) -> str:
    return "draw" if is_draw else ("round_one" if rounds == 1 else "round_two")

@cache
def _key_sets() -> dict[tuple[int, int], frozenset]:
    """Every shape of private key as a set, built once for the whole module.

    Membership is the question both grain tests ask, and the six key sets cost
    about twenty-five seconds to build — rebuilding one per lookup is the
    difference between a test that takes a second and one that never finishes.
    """
    return {
        (board_cards, discards): frozenset(
            private_keys(board_cards=board_cards, discards=discards)
        )
        for board_cards in (1, 2)
        for discards in THROW_RANGE
    }

def _product_census() -> dict[str, dict[str, int]]:
    """The infoset count as a sum of products, which is how the gate is read.

    One public point contributes exactly as many ledgers as there are private
    keys of its shape, so the whole count is a 141-term sum and needs no walk.
    """
    by_player = {"0": 0, "1": 0}
    by_stage = {"round_one": 0, "draw": 0, "round_two": 0}
    by_width: dict[str, int] = {}
    for point in public_decision_points():
        keys = len(private_keys(board_cards=point.board_cards, discards=point.discards))
        by_player[str(point.player)] += keys
        by_stage[_stage(point.is_draw_decision, len(point.betting))] += keys
        by_width[str(point.width)] = by_width.get(str(point.width), 0) + keys
    return {"by_player": by_player, "by_stage": by_stage, "by_ledger_width": by_width}

def test_the_infoset_totals_are_the_public_points_times_their_private_keys():
    product = _product_census()
    assert product["by_player"] == CENSUS["infosets"]["by_player"]
    assert product["by_stage"] == CENSUS["infosets"]["by_stage"]
    assert product["by_ledger_width"] == CENSUS["infosets"]["by_ledger_width"]
    assert sum(product["by_player"].values()) == CENSUS["infosets"]["total"]

def test_the_deck_gate_verdict_follows_from_the_numbers_it_was_read_on():
    gate = CENSUS["gate"]
    worst = max(CENSUS["infosets"]["by_player"].values())
    assert gate["measured_worst_player"] == worst
    over = worst > gate["budget_infosets_per_player"]
    assert gate["verdict"] == ("over_budget" if over else "within_budget")

def test_the_generator_agrees_with_the_product_on_the_keys_it_emits():
    points = {
        (point.player, point.draws, point.betting): point
        for point in public_decision_points()
    }
    shapes = _key_sets()
    seen = set()
    for key in islice(all_infosets(), PREFIX):
        seen.add(key)
        point = points[(key.player, key.draws, key.betting)]
        assert key.legal_actions() == point.actions
        shape = shapes[point.board_cards, point.discards]
        assert (key.hole, key.discarded, key.board) in shape
    assert len(seen) == PREFIX

@pytest.mark.skipif(
    not FULL_RUN,
    reason="builds all 3.1M keys; set MINIDRAWMAHA_FULL_CENSUS=1 to run it",
)
def test_the_generator_yields_exactly_the_committed_census():
    # The fixture's own derivation, repeated: count what comes out, and trust
    # nothing about how many that should be.
    by_player = {"0": 0, "1": 0}
    for key in all_infosets():
        by_player[str(key.player)] += 1
    assert by_player == CENSUS["infosets"]["by_player"]

# ---------------------------------------------------------------------------
# The grain, direction 1: everything a real hand reaches is enumerated
# ---------------------------------------------------------------------------

def playouts(count, *, seed=20260918):
    """Random complete hands: uniform over legal actions and over chance outcomes."""
    rng = Random(seed)
    for _ in range(count):
        deck = list(DECK)
        rng.shuffle(deck)
        dealt = deck[: 2 * HOLE_CARDS]
        state = MiniState(
            holes=(tuple(sorted(dealt[:HOLE_CARDS])), tuple(sorted(dealt[HOLE_CARDS:])))
        )
        while not state.is_terminal():
            if state.is_chance_node():
                state = state.apply_chance(rng.choice(state.chance_outcomes())[0])
            else:
                yield state.infoset()
                state = state.apply(rng.choice(state.legal_actions()))

def test_every_infoset_a_random_hand_reaches_is_one_the_enumerator_yields():
    points = {
        (point.player, point.draws, point.betting): point
        for point in public_decision_points()
    }
    shapes = _key_sets()
    reached = 0
    for key in playouts(400):
        reached += 1
        assert (key.player, key.draws, key.betting) in points
        point = points[(key.player, key.draws, key.betting)]
        shape = shapes[point.board_cards, point.discards]
        assert (key.hole, key.discarded, key.board) in shape
    # A run that reached nothing would pass every assertion above.
    assert reached > 1_000

# ---------------------------------------------------------------------------
# The grain, direction 2: everything enumerated can actually be dealt
# ---------------------------------------------------------------------------

def _throw(state: MiniState, *, exactly=None, count=None) -> MiniState:
    """Make the player to act throw exactly `exactly`, or any `count` cards.

    A throw names its cards by canonical position, so the action that discards
    a chosen pair is found by asking `draw_order` where those cards sit — the
    same lookup `apply` does in reverse.
    """
    player = state.current_player
    if exactly is None:
        mask = (1 << count) - 1
    else:
        order = draw_order(hole=state.holes[player], discarded=(), board=state.board)
        mask = sum(1 << index for index, card in enumerate(order) if card in exactly)
    action = next(a for a, m in DISCARD_MASK.items() if m == mask)
    return state.apply(action)

def realise(key: InfoSet) -> MiniState:
    """Deal and play a hand that arrives at `key`.

    The key names only what its own player can see, so everything else is a
    free choice: which of the three cards in hand were the replacements, what
    the opponent holds, and which board card came first. This picks the first
    of each — the point is that *some* history reaches the key, since that is
    what makes a ledger for it worth allocating.
    """
    player, opponent = key.player, 1 - key.player
    replacements = key.hole[: len(key.discarded)]
    shown = set(key.hole) | set(key.discarded) | set(key.board)
    spare = [card for card in DECK if card not in shown]
    opponent_throws = key.draws[opponent].count if len(key.draws) > opponent else 0
    holes = {
        player: tuple(sorted(key.hole[len(key.discarded) :] + key.discarded)),
        opponent: tuple(sorted(spare[:HOLE_CARDS])),
    }

    state = MiniState(holes=(holes[0], holes[1])).apply_chance((key.board[0],))
    for action in key.betting[0]:
        state = state.apply(action)
    if len(key.betting) == 1 and not key.is_draw_decision():
        return state
    for drawer in (0, 1):
        if key.is_draw_decision() and drawer == key.player:
            return state
        if drawer == player:
            state = _throw(state, exactly=key.discarded)
            if key.discarded:
                state = state.apply_chance(tuple(sorted(replacements)))
        else:
            state = _throw(state, count=opponent_throws)
            if opponent_throws:
                taken = spare[HOLE_CARDS : HOLE_CARDS + opponent_throws]
                state = state.apply_chance(tuple(sorted(taken)))
    state = state.apply_chance((key.board[1],))
    for action in key.betting[1]:
        state = state.apply(action)
    return state

def test_every_public_point_can_be_dealt_into_with_a_sample_of_its_keys():
    # Every one of the 141 public points, three keys each: the first, the last
    # and one drawn at random. Structural enumeration is only a census of this
    # game if each thing it counts is a position the rules can actually reach.
    rng = Random(20260918)
    checked = 0
    for point in public_decision_points():
        keys = private_keys(board_cards=point.board_cards, discards=point.discards)
        for hole, thrown, board in (keys[0], keys[-1], rng.choice(keys)):
            key = InfoSet(
                player=point.player,
                hole=hole,
                discarded=thrown,
                board=board,
                draws=point.draws,
                betting=point.betting,
            )
            state = realise(key)
            assert state.kind is NodeKind.DECISION
            assert state.infoset() == key
            checked += 1
    assert checked == 3 * len(public_decision_points())

def test_a_realised_hand_deals_every_card_once():
    # The trap `realise` would fall into silently: handing a player a card that
    # is already on the board reads as a perfectly good history until a
    # showdown scores it. `MiniState` refuses it, so this pins that the sample
    # above is exercising that refusal rather than dodging it.
    point = next(p for p in public_decision_points() if p.discards == THROW_CAP)
    keys = private_keys(board_cards=point.board_cards, discards=point.discards)
    hole, thrown, board = keys[0]
    state = realise(
        InfoSet(
            player=point.player,
            hole=hole,
            discarded=thrown,
            board=board,
            draws=point.draws,
            betting=point.betting,
        )
    )
    piles = (*state.holes, *state.discards, state.board)
    dealt = [card for group in piles for card in group]
    assert len(dealt) == len(set(dealt))
