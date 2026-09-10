"""Regenerate `tests/leduc/referee.json` from OpenSpiel, the outside referee.

Rung 2 has no closed form. Kuhn's equilibrium could be written down and
compared against; Leduc's cannot, so the only ground truth available is a
second, independently written implementation of the same game — OpenSpiel's
`leduc_poker` — plus an exact solve of it. This script interrogates that
referee once and writes down what it says; `tests/leduc/test_referee.py` then
re-derives every one of those facts from our own code and demands agreement,
with no OpenSpiel present.

That split is the whole point. OpenSpiel's wheels are large and the Vercel
deployment build must not pull them, so `open_spiel` is deliberately NOT a
project dependency and nothing under `src/` or `tests/` imports it. This
script is the only file that does, it is never imported by anything, and it
runs by hand:

    uv run --with open_spiel --with cvxpy --with ecos --no-project \\
        python3 scripts/generate_leduc_referee.py

The facts it records, and why each is worth pinning:

- **Census** — how many decision, board and terminal nodes the game has. A
  miscounted tree is the failure mode that silently changes every number
  downstream, and it cannot be caught by any internal consistency check.
- **Infoset collapse** — OpenSpiel keys information states by suited card;
  we key by rank, since Leduc's two suits are strategically identical. 936
  of theirs must collapse onto 288 of ours with no referee infoset straddling
  two of ours, or our key is throwing away something that matters.
- **Payoff magnitudes** — the set of distinct |payoff| values reachable.
  Pins the betting ladder (ante, two rounds, raise caps) end to end.
- **Path probabilities** — what one deal, and then one board, is worth.
- **LP value** — the exact value of the game to the first player, from the
  sequence-form linear program. This is the number the whole rung is aimed
  at, and until now it lived in our tests as a hand-typed literal.
- **CFR curve** — OpenSpiel's own solver's exploitability at checkpoints.
  Our convergence bounds were justified against this curve in a code comment;
  committing it makes that justification checkable.

Its updates ALTERNATE and ours are SIMULTANEOUS, so the curve is a pace
reference, not an iterate-for-iterate target — see `test_referee.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyspiel
from open_spiel.python.algorithms import cfr, exploitability, sequence_form_lp

# Our own tree, walked in lockstep with the referee's below. `src` is on the
# path explicitly because this script runs with --no-project (the project is
# not installed into the throwaway OpenSpiel environment).
sys.path.insert(0, "src")

from drawmaha_solver.leduc.game import DEALS, LeducState  # noqa: E402

FIXTURE = Path("tests/leduc/referee.json")

# Where OpenSpiel's own CFR gets sampled. Spread log-like from the first
# iteration to the point where the curve has flattened, so a reader can see
# both the early collapse and the long tail.
CFR_CHECKPOINTS = (1, 20, 50, 120, 300, 500, 800)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Interrogate the referee and write the fixture.

    Steps:
      1. Walk our tree and OpenSpiel's in lockstep, counting nodes.
      2. Collapse OpenSpiel's suited infosets onto our rank-keyed ones.
      3. Solve the game exactly with the sequence-form LP.
      4. Run OpenSpiel's own CFR, sampling exploitability at checkpoints.
      5. Write it all down.
    """
    game = pyspiel.load_game("leduc_poker")

    # Step 1: census, by walking both trees at once.
    census = _lockstep_census(game)
    print(f"census: {census}")

    # Step 2: the 936 -> 288 collapse.
    collapse = _infoset_collapse(game)
    print(f"collapse: {collapse}")

    # Step 3: the exact value of the game.
    lp_value = float(sequence_form_lp.solve_zero_sum_game(game)[0])
    print(f"LP value to P0: {lp_value:.12f}")

    # Step 4: the referee's own convergence pace.
    curve = _cfr_curve(game)
    for point in curve:
        print(f"  T={point['iterations']:<4} {point['exploitability']:.6f}")

    # Step 5.
    _write(
        census=census,
        collapse=collapse,
        lp_value=lp_value,
        curve=curve,
        payoff_magnitudes=_payoff_magnitudes(),
    )


# ---------------------------------------------------------------------------
# Step 1: the census, verified node-for-node against the referee
# ---------------------------------------------------------------------------


def _lockstep_census(game) -> dict[str, int]:
    """Walk both trees together, asserting agreement, and count what we pass.

    Counts are OUR tree's, but every node is checked against the referee's
    before it is counted, so a count that survives this walk is a count both
    implementations agree on. Our roots are the 30 post-deal states, so our
    chance nodes are only the board turns; OpenSpiel deals the two private
    cards one at a time and therefore has 7 chance nodes we do not model
    (1 root + 6 second-card), which `board_nodes` deliberately excludes.
    """
    counts = {"decision_nodes": 0, "board_nodes": 0, "terminal_nodes": 0}

    def compare(mine, theirs) -> None:
        assert mine.is_terminal() == theirs.is_terminal()
        if mine.is_terminal():
            assert list(mine.returns()) == list(theirs.returns()), (
                mine.betting,
                mine.returns(),
                theirs.returns(),
            )
            counts["terminal_nodes"] += 1
            return

        assert mine.is_chance_node() == theirs.is_chance_node()
        if mine.is_chance_node():
            ours = sorted((int(c), round(p, 12)) for c, p in mine.chance_outcomes())
            refs = sorted((int(a), round(p, 12)) for a, p in theirs.chance_outcomes())
            assert ours == refs, (ours, refs)
            counts["board_nodes"] += 1
            for card, _ in mine.chance_outcomes():
                compare(mine.apply_chance(card), theirs.child(int(card)))
            return

        assert mine.current_player == theirs.current_player()
        assert [int(a) for a in mine.legal_actions()] == list(theirs.legal_actions()), (
            mine.betting,
        )
        counts["decision_nodes"] += 1
        for action in mine.legal_actions():
            compare(mine.apply(action), theirs.child(int(action)))

    for state, root in _paired_roots(game):
        compare(state, root)
    return counts


# ---------------------------------------------------------------------------
# Step 2: does our rank key throw away anything the referee keeps?
# ---------------------------------------------------------------------------


def _infoset_collapse(game) -> dict[str, int]:
    """Map every referee information state onto the rank-keyed spots it covers.

    A referee infoset that straddles two of ours would mean our key merges
    two situations the referee can tell apart — i.e. we are blind to
    something strategically real. The reverse (several of theirs inside one
    of ours) is exactly the suit merge we intend.
    """
    referee_to_ours: dict[str, set] = {}
    ours: set = set()

    def collapse(mine, ref) -> None:
        if mine.is_terminal():
            return
        if mine.is_chance_node():
            for card, _ in mine.chance_outcomes():
                collapse(mine.apply_chance(card), ref.child(int(card)))
            return
        key = mine.infoset()
        ours.add(key)
        referee_to_ours.setdefault(ref.information_state_string(), set()).add(key)
        for action in mine.legal_actions():
            collapse(mine.apply(action), ref.child(int(action)))

    for state, root in _paired_roots(game):
        collapse(state, root)

    straddling = {k: v for k, v in referee_to_ours.items() if len(v) > 1}
    assert not straddling, f"{len(straddling)} referee infosets straddle two of ours"
    return {
        "openspiel_information_states": len(referee_to_ours),
        "rank_keyed_spots": len(ours),
    }


def _paired_roots(game):
    """Our 30 post-deal states, each beside the referee state holding the same cards."""
    for deal in DEALS:
        root = game.new_initial_state()
        root.apply_action(int(deal[0]))
        root.apply_action(int(deal[1]))
        yield LeducState(cards=deal), root


# ---------------------------------------------------------------------------
# Step 4: the referee's own convergence pace
# ---------------------------------------------------------------------------


def _cfr_curve(game) -> list[dict[str, float]]:
    """OpenSpiel CFR's exploitability at each checkpoint, cumulatively trained.

    `CFRSolver` updates the two players in alternation, one per call to
    `evaluate_and_update_policy`. Ours updates both simultaneously, so these
    numbers describe the same destination reached at a similar pace, never
    the same iterate.
    """
    solver = cfr.CFRSolver(game)
    curve = []
    done = 0
    for mark in CFR_CHECKPOINTS:
        while done < mark:
            solver.evaluate_and_update_policy()
            done += 1
        curve.append(
            {
                "iterations": mark,
                "exploitability": float(
                    exploitability.exploitability(game, solver.average_policy())
                ),
            }
        )
    return curve


# ---------------------------------------------------------------------------
# The payoff ladder, read off our own terminals
# ---------------------------------------------------------------------------


def _payoff_magnitudes() -> list[int]:
    """Every distinct |payoff| a terminal can pay, ascending.

    Read from our tree rather than the referee's because step 1 has already
    proved the two agree at every terminal; this is a summary of a verified
    surface, not a second opinion.
    """
    seen: set[int] = set()

    def walk(state) -> None:
        if state.is_terminal():
            seen.update(abs(int(p)) for p in state.returns())
            return
        if state.is_chance_node():
            for card, _ in state.chance_outcomes():
                walk(state.apply_chance(card))
            return
        for action in state.legal_actions():
            walk(state.apply(action))

    for deal in DEALS:
        walk(LeducState(cards=deal))
    return sorted(seen)


# ---------------------------------------------------------------------------
# Writing the fixture
# ---------------------------------------------------------------------------


def _board_probability() -> float:
    """What one board card is worth, read off a board node rather than assumed.

    Writing `1 / 4` here would hard-code the size of the remaining deck into a
    file whose whole job is provenance — and it is the deck, not this script,
    that decides how many cards are left once two are in hands. Step 1 has
    already checked this node's outcomes against the referee's, so reading it
    is reading a verified surface.
    """
    state = LeducState(cards=DEALS[0])
    while not state.is_chance_node():
        state = state.apply(state.legal_actions()[0])
    return state.chance_outcomes()[0][1]


def _write(*, census, collapse, lp_value, curve, payoff_magnitudes) -> None:
    """Serialize the referee's answers, with the provenance a reader needs."""
    fixture = {
        "_source": (
            "OpenSpiel leduc_poker, interrogated by "
            "scripts/generate_leduc_referee.py. Do not hand-edit: regenerate."
        ),
        "openspiel_version": pyspiel.__version__,
        "census": census,
        "infosets": collapse,
        "payoff_magnitudes": payoff_magnitudes,
        "path_probability": {
            "after_deal": 1 / len(DEALS),
            "after_board": (1 / len(DEALS)) * _board_probability(),
        },
        "lp_value_to_p0": lp_value,
        # The LP is solved numerically (cvxpy/ECOS), so its last digits move
        # between solver versions. Anything asserting against `lp_value_to_p0`
        # must allow at least this much slack.
        "lp_value_precision": 1e-9,
        "cfr_exploitability": curve,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE.resolve()}")


if __name__ == "__main__":
    main()
