"""Regenerate `tests/minidrawmaha/census.json` — rung 3's stand-in for a referee.

Rung 2 could ask OpenSpiel how big Leduc was, and
`scripts/generate_leduc_referee.py` wrote the answer down. No engine implements
Drawmaha, so rung 3 has no outside opinion to interrogate; what it has instead
is an enumeration of its own rules, and this script writes down what that
enumeration says. `tests/minidrawmaha/test_census.py` then rebuilds every
number here from the rules a second, different way and demands agreement, so a
change to the deck, the throw cap or the betting tree fails loudly rather than
quietly shipping a solver for a different game.

    uv run python3 scripts/generate_minidrawmaha_census.py

It takes a few minutes, which is the point of committing the answer. The cost
is not the arithmetic — it is that this script **counts what the enumerator
yields**, one `InfoSet` at a time, all 23.7 million of them. It deliberately
does not compute the total any cheaper way: the test does that, and two
derivations that agree are worth more than one that is fast.

The facts it records, and why each is worth pinning:

- **The public tree** — 288 decision points, 78 chance nodes, 393 terminals.
  The shape of the game with the cards taken out, and the thing a mis-stated
  betting rule moves first.
- **The concrete tree** — what the same tree costs per (deal, board, actions,
  replacements) history: 8.7 x 10^11 decision nodes. Nobody walks it. It is
  recorded because it is the reason the rest of this file is measured at the
  grain it is measured at.
- **The private keys** — how many canonical `(hole, discarded, board)` pictures
  exist per shape. The 61,590 at one board card is the plan's §4.3 number,
  reached here from key space rather than from draw histories.
- **The infosets** — the ledger count, split by player, by stage and by ledger
  width. This is the deck gate's number.
- **The gate verdict** — measured against §8's budget of ~5 million infosets
  per player, recorded in the file so the fixture states its own answer.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")

from drawmaha_solver.minidrawmaha.cards import DECK, N_RANKS, N_SUITS  # noqa: E402
from drawmaha_solver.minidrawmaha.enumeration import (  # noqa: E402
    ROOT_DEALS,
    all_infosets,
    private_keys,
    public_decision_points,
    public_tree,
)
from drawmaha_solver.minidrawmaha.game import (  # noqa: E402
    ANTE,
    STACK,
    THROW_CAP,
    NodeKind,
)
from drawmaha_solver.minidrawmaha.hands import BOARD_CARDS, HOLE_CARDS  # noqa: E402

FIXTURE = Path("tests/minidrawmaha/census.json")

# Plan §8: "under ~5 million infosets per player", a tunable target encoding
# "fits on a laptop without ceremony" rather than a derived constant. It lives
# here, beside the count it judges, so the fixture records what the number was
# measured against instead of leaving it in a review comment.
GATE_BUDGET_PER_PLAYER = 5_000_000

# Every shape of private key the game can ask for: one or two board cards, and
# a player who has thrown nothing, one card or two.
KEY_SHAPES = tuple(
    (board_cards, discards)
    for board_cards in range(1, BOARD_CARDS + 1)
    for discards in range(THROW_CAP + 1)
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Count the game, then write the count down.

    Steps:
      1. Walk the public tree and weigh it against the concrete one.
      2. Build the private key sets.
      3. Consume the enumerator, counting every key it yields.
      4. Read the gate.
      5. Write it all down.
    """
    public = _public_census()
    print(f"public tree: {public}")

    keys = _private_key_census()
    print(f"private keys: {keys}")

    infosets = _infoset_census()
    print(f"infosets: {infosets['total']:,}  by player {infosets['by_player']}")

    gate = _gate(infosets)
    print(f"gate: {gate['verdict']} — {gate['measured_worst_player']:,} vs "
          f"{gate['budget_infosets_per_player']:,} ({gate['ratio_to_budget']:.2f}x)")

    _write(public=public, keys=keys, infosets=infosets, gate=gate)

# ---------------------------------------------------------------------------
# Step 1: the public tree, and the concrete one it stands in for
# ---------------------------------------------------------------------------

def _public_census() -> dict:
    """The 759 public nodes, and how many concrete histories each stands for."""
    by_kind: Counter[str] = Counter()
    concrete: Counter[str] = Counter()
    for point in public_tree():
        by_kind[point.kind.value] += 1
        concrete[point.kind.value] += point.concrete_nodes
    stages: Counter[str] = Counter()
    players: Counter[str] = Counter()
    for point in public_decision_points():
        stages[_stage(point.is_draw_decision, len(point.betting))] += 1
        players[str(point.player)] += 1
    return {
        "decision_points": by_kind[NodeKind.DECISION.value],
        "chance_points": by_kind[NodeKind.CHANCE.value],
        "terminal_points": by_kind[NodeKind.TERMINAL.value],
        "decision_points_by_stage": dict(sorted(stages.items())),
        "decision_points_by_player": dict(sorted(players.items())),
        "concrete": {kind: concrete[kind] for kind in sorted(concrete)},
    }

def _stage(is_draw: bool, rounds: int) -> str:
    """Which of the three things a decision is: a bet, a throw, or a bet again."""
    if is_draw:
        return "draw"
    return "round_one" if rounds == 1 else "round_two"

# ---------------------------------------------------------------------------
# Step 2: the private keys
# ---------------------------------------------------------------------------

def _private_key_census() -> dict:
    """How many canonical pictures exist per shape, and §4.3's post-draw total."""
    by_shape = {
        f"{board_cards}_{discards}": len(
            private_keys(board_cards=board_cards, discards=discards)
        )
        for board_cards, discards in KEY_SHAPES
    }
    return {
        "by_shape": by_shape,
        "post_draw_board_one_total": sum(
            count for shape, count in by_shape.items() if shape.startswith("1_")
        ),
    }

# ---------------------------------------------------------------------------
# Step 3: the count itself, taken off the enumerator one key at a time
# ---------------------------------------------------------------------------

def _infoset_census() -> dict:
    """Consume `all_infosets()` and count what comes out.

    Every classification here is read off the yielded key rather than off the
    public point that produced it — the seat, the stage and the ledger width
    all come from `InfoSet` itself. That is the difference between counting
    what the solver will allocate and counting what this script expected.
    """
    by_player: Counter[str] = Counter()
    by_stage: Counter[str] = Counter()
    by_width: Counter[str] = Counter()
    for count, key in enumerate(all_infosets(), start=1):
        by_player[str(key.player)] += 1
        by_stage[_stage(key.is_draw_decision(), len(key.betting))] += 1
        by_width[str(len(key.legal_actions()))] += 1
        if count % 5_000_000 == 0:
            print(f"  {count:,} keys...")
    return {
        "total": sum(by_player.values()),
        "by_player": dict(sorted(by_player.items())),
        "by_stage": dict(sorted(by_stage.items())),
        "by_ledger_width": dict(sorted(by_width.items(), key=lambda item: int(item[0]))),
    }

# ---------------------------------------------------------------------------
# Step 4: the gate
# ---------------------------------------------------------------------------

def _gate(infosets: dict) -> dict:
    """Read §8's budget against the worse of the two seats.

    The worse seat, not the average: a table holds both, and the budget is
    stated per player because that is the unit the exact best-response walk
    pays in.
    """
    worst = max(infosets["by_player"].values())
    return {
        "budget_infosets_per_player": GATE_BUDGET_PER_PLAYER,
        "measured_worst_player": worst,
        "ratio_to_budget": worst / GATE_BUDGET_PER_PLAYER,
        "verdict": "over_budget" if worst > GATE_BUDGET_PER_PLAYER else "within_budget",
    }

# ---------------------------------------------------------------------------
# Step 5: writing the fixture
# ---------------------------------------------------------------------------

def _write(*, public: dict, keys: dict, infosets: dict, gate: dict) -> None:
    """Serialize the census, with the configuration that produced it."""
    concrete = public.pop("concrete")
    fixture = {
        "_source": (
            "drawmaha_solver.minidrawmaha.enumeration, counted by "
            "scripts/generate_minidrawmaha_census.py. Do not hand-edit: regenerate."
        ),
        # Every number below moves if any of these do, which is the whole
        # reason the file is committed.
        "configuration": {
            "deck_cards": len(DECK),
            "ranks": N_RANKS,
            "suits": N_SUITS,
            "hole_cards": HOLE_CARDS,
            "board_cards": BOARD_CARDS,
            "throw_cap": THROW_CAP,
            "ante": ANTE,
            "stack": STACK,
            "root_deals": ROOT_DEALS,
        },
        "public_tree": public,
        "concrete_tree": concrete,
        "private_keys": keys,
        "infosets": infosets,
        "gate": gate,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE.resolve()}")

if __name__ == "__main__":
    main()
