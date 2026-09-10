"""Hold rung 2 to an outside referee's answers, with no OpenSpiel installed.

Rung 1 could be checked against a closed form written on paper. Leduc has
none, so the ground truth is a second independent implementation — OpenSpiel's
`leduc_poker` — plus an exact solve of it. `scripts/generate_leduc_referee.py`
interrogates that referee and writes `referee.json`; every test here
re-derives the same fact from our own code and demands agreement.

The re-derivation is the point. A test that merely read a number out of the
fixture and asserted it equalled itself would pass forever while the solver
rotted. Each test below walks our tree, or runs our solver, and only then
looks at what the referee said.

`open_spiel` is deliberately not a project dependency (large wheels; the
Vercel deployment build must not pull them), so nothing in this file imports
it — the fixture is the referee's testimony, taken once and committed.
"""

from __future__ import annotations

import pytest

from drawmaha_solver.leduc.cfr import run_iteration
from drawmaha_solver.leduc.exploitability import expected_value, exploitability
from drawmaha_solver.leduc.game import (
    DEAL_PROBABILITY,
    DEALS,
    LP_VALUE_P0,
    LeducState,
    all_infosets,
)
from drawmaha_solver.leduc.infoset_table import average_strategy, new_infoset_table

# Our updates are SIMULTANEOUS and the referee's ALTERNATE, so the two curves
# are never expected to agree iterate for iterate. Measured across the seven
# checkpoints the ratio runs 0.43 to 1.23 — we lead early and trail slightly
# late. This band admits that whole range with room either side, while still
# catching the failure that matters: a solver that has stopped converging, or
# converges an order of magnitude slower than a correct one.
PACE_BAND = (0.25, 2.5)

# ---------------------------------------------------------------------------
# The shape of the game: does our tree have the same nodes as the referee's?
# ---------------------------------------------------------------------------


def _census() -> dict[str, int]:
    """Count our own tree's nodes by kind, from the 30 post-deal roots."""
    counts = {"decision_nodes": 0, "board_nodes": 0, "terminal_nodes": 0}

    def walk(state: LeducState) -> None:
        if state.is_terminal():
            counts["terminal_nodes"] += 1
            return
        if state.is_chance_node():
            counts["board_nodes"] += 1
            for card, _ in state.chance_outcomes():
                walk(state.apply_chance(card))
            return
        counts["decision_nodes"] += 1
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(LeducState(cards=deal))
    return counts


def test_our_tree_has_exactly_the_referees_nodes(referee):
    # A miscounted tree changes every number downstream and no internal
    # consistency check can catch it — only a second implementation can.
    assert _census() == referee["census"]


def test_every_terminal_pays_one_of_the_referees_eight_magnitudes(referee):
    magnitudes = set()

    def walk(state: LeducState) -> None:
        if state.is_terminal():
            magnitudes.update(abs(int(p)) for p in state.returns())
            return
        if state.is_chance_node():
            for card, _ in state.chance_outcomes():
                walk(state.apply_chance(card))
            return
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(LeducState(cards=deal))
    # Pins the whole betting ladder — ante, both bet sizes, the raise cap —
    # in a single set. A wrong cap or bet size shows up as a new magnitude.
    assert sorted(magnitudes) == referee["payoff_magnitudes"]


def test_one_deal_and_then_one_board_are_worth_what_the_referee_says(referee):
    assert DEAL_PROBABILITY == pytest.approx(referee["path_probability"]["after_deal"])

    # Walk to the first board turn and read the deck's own probability there.
    state = LeducState(cards=DEALS[0])
    while not state.is_chance_node():
        state = state.apply(state.legal_actions()[0])
    board_probability = state.chance_outcomes()[0][1]
    assert DEAL_PROBABILITY * board_probability == pytest.approx(
        referee["path_probability"]["after_board"]
    )


# ---------------------------------------------------------------------------
# The infoset collapse: our rank key against the referee's suited one
# ---------------------------------------------------------------------------


def test_the_referees_suited_spots_collapse_onto_ours_without_loss(referee):
    # The referee distinguishes the two jacks; we do not, because they are the
    # same hand. Re-derive BOTH counts from our own tree — the suited one by
    # keying on the card the player holds instead of its rank — so the
    # collapse is checkable here rather than asserted on the referee's word.
    suited, ranked = set(), set()

    def walk(state: LeducState) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for card, _ in state.chance_outcomes():
                walk(state.apply_chance(card))
            return
        spot = state.infoset()
        ranked.add(spot)
        suited.add((state.cards[state.current_player], state.board, spot.betting))
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(LeducState(cards=deal))

    assert len(suited) == referee["infosets"]["openspiel_information_states"]
    assert len(ranked) == referee["infosets"]["rank_keyed_spots"]


def test_the_solver_allocates_one_ledger_per_spot_the_referee_reaches(referee):
    # `all_infosets` builds the 288 by construction, from rules; the walk above
    # finds them by exploration. Both must land on the referee's count, or the
    # table is allocating spots the game cannot reach (or missing some).
    assert len(all_infosets()) == referee["infosets"]["rank_keyed_spots"]
    assert len(new_infoset_table()) == referee["infosets"]["rank_keyed_spots"]


# ---------------------------------------------------------------------------
# The answer: does our solve go where the exact solve says it should?
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def curve(referee) -> dict[int, tuple[float, float]]:
    """Our exploitability and game value at each of the referee's checkpoints.

    One cumulative solve sampled in place, rather than a fresh solve per
    checkpoint. Vanilla CFR enumerates the whole tree, so this is
    deterministic — no seed, and two runs agree bit for bit.
    """
    table = new_infoset_table()
    measured, done = {}, 0
    for point in referee["cfr_exploitability"]:
        while done < point["iterations"]:
            run_iteration(table)
            done += 1
        strategies = average_strategy(table)
        measured[done] = (exploitability(strategies), expected_value(strategies)[0])
    return measured


def test_our_solve_approaches_the_exact_value_of_the_game(curve, lp_value):
    # The number the whole rung aims at, and until now a hand-typed literal:
    # the sequence-form LP's value to P0. Measured distance at 300 iterations
    # is 0.003, so 0.01 is a band with real headroom that would still catch a
    # solver drifting to the wrong answer.
    _, game_value = curve[300]
    assert game_value == pytest.approx(lp_value, abs=0.01)


def test_more_training_moves_the_game_value_toward_the_exact_one(curve, lp_value):
    target = lp_value
    early = abs(curve[1][1] - target)
    late = abs(curve[800][1] - target)
    assert late < early


def test_our_exploitability_keeps_the_referees_pace(curve, referee):
    low, high = PACE_BAND
    for point in referee["cfr_exploitability"]:
        ours = curve[point["iterations"]][0]
        ratio = ours / point["exploitability"]
        assert low <= ratio <= high, (point["iterations"], ours, ratio)


def test_exploitability_falls_at_every_checkpoint(curve, referee):
    readings = [curve[p["iterations"]][0] for p in referee["cfr_exploitability"]]
    assert readings == sorted(readings, reverse=True), readings
    # Near-optimal play is where a sign or bookkeeping error in the
    # best-response recursion would surface as a small negative reading.
    assert min(readings) >= -1e-12, readings


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------


def test_the_fixture_records_where_it_came_from(referee):
    # It is committed data with no runtime provenance, so a reader who finds a
    # surprising number has to be able to tell what produced it.
    assert "generate_leduc_referee.py" in referee["_source"]
    assert referee["openspiel_version"]


def test_the_game_modules_copy_of_the_exact_value_matches_the_referees(referee):
    # `game.LP_VALUE_P0` is what analysis.py and play.py report against; the
    # fixture is what the LP solver actually said. They are two copies of one
    # number, so something has to hold them together.
    assert LP_VALUE_P0 == pytest.approx(
        referee["lp_value_to_p0"], abs=referee["lp_value_precision"]
    )


def test_the_exact_value_is_recorded_to_more_precision_than_anything_asserts_on_it(referee):
    # The LP is solved numerically, so its last digits move between solver
    # versions; every assertion above must sit well outside that noise.
    assert referee["lp_value_precision"] <= 1e-9
