"""The deal pack: hands dealt in advance for the browser, with the strategy at every node.

`/rung3` plays the frozen strategy in a browser, and the browser cannot look
the strategy up. The table is 6,220,050 rows, its key is the joint canonical
suit relabelling of hole, discards and ordered board (`canonical_picture`),
and the site runs no Python on Vercel. Rung 2 could transcribe its 288-row
key; this one is the routine the repo most wants to exist exactly once.

So the lookup is done here, ahead of time. A deal is a **fixed deck order**:
P0's three cards, P1's three, board card 1, then the stub in the order it will
be consumed — a replacement for whoever draws, in draw order, then board card
2. For one deck order the hand has a finite tree of decisions — 8 in round 1,
21 at the draw, 28 per throw combination in round 2 — and this module walks
every one of them for BOTH players and records the strategy's mix there. The
browser then only has to follow the path it is on and read the vector. It
transcribes the betting grammar and nothing about the key.

What goes in a deal, beside the vectors:

* `order` — each player's three cards in the canonical low/mid/top order
  `draw_order` lays them out, so a throw names a physical card without the
  browser knowing what canonical means.
* `show` — the showdown of each of the 16 post-draw worlds: both final
  holdings, the board, who takes each half (`2` is a chop), and the category
  names. Hand ranking is not transcribed either.

Vectors are per-mille integers summing to 1000 (largest remainder), in
`legal_actions()` order. `d1` is keyed by P0's draw **count**, which is all
P1's infoset knows. A pack is written in chunks, with a manifest that records
the seed, the strategy's digest and provenance, so the browser can say what it
is playing and a re-export cannot silently swap it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from drawmaha_solver.minidrawmaha.cards import DECK, Card
from drawmaha_solver.minidrawmaha.game import (
    DRAW_ACTIONS,
    Action,
    MiniState,
    draw_order,
    line_symbol,
    throw_count,
)
from drawmaha_solver.minidrawmaha.hands import Score, inner_score, outer_score
from drawmaha_solver.minidrawmaha.strategy import (
    StrategyInfo,
    fetch_strategy,
    load_strategy,
    sha256_of,
)

PER_MILLE = 1000

# One letter per throw, the order `DRAW_ACTIONS` lists them.
THROW_CODE = {
    Action.THROW_NONE: "n",
    Action.THROW_LOW: "l",
    Action.THROW_MID: "m",
    Action.THROW_TOP: "t",
}

# Where the stub starts: two holes and the first board card are dealt first.
STUB = 7

# A side of the showdown: P0 won it, P1 won it, or it was chopped.
CHOP = 2

Deal = dict[str, object]

# ---------------------------------------------------------------------------
# Cards as the browser names them
# ---------------------------------------------------------------------------

def card_index(card: Card) -> int:
    """A card's position in `DECK`, which is `rank * 3 + suit`."""
    return DECK.index(card)

def indices(cards: tuple[Card, ...]) -> list[int]:
    return [card_index(card) for card in cards]

# ---------------------------------------------------------------------------
# Quantising a row
# ---------------------------------------------------------------------------

def per_mille(row: np.ndarray) -> list[int]:
    """`row` as integers summing to exactly 1000, by largest remainder.

    Rounding each entry separately can sum to 999 or 1001, and a sampler that
    reads the vector as a partition of 1000 would then either never choose the
    last action or choose it a thousandth too often. The floors are taken
    first and the shortfall is handed to the entries that lost the most.
    """
    values = np.asarray(row, dtype=np.float64)
    values = values / values.sum()
    scaled = values * PER_MILLE
    floors = np.floor(scaled).astype(int)
    short = PER_MILLE - int(floors.sum())
    for index in np.argsort(-(scaled - floors), kind="stable")[:short]:
        floors[index] += 1
    return floors.tolist()

# ---------------------------------------------------------------------------
# One deal
# ---------------------------------------------------------------------------

def deal_from_deck(deck: tuple[Card, ...]) -> MiniState:
    """The state after the private deal and board card 1, off `deck`'s top seven."""
    if sorted(deck) != sorted(DECK):
        raise ValueError("a deck order is a permutation of the fifteen cards")
    root = MiniState(holes=(tuple(sorted(deck[0:3])), tuple(sorted(deck[3:6]))))
    return root.apply_chance((deck[6],))

def export_deal(deck: tuple[Card, ...], strategies: Mapping) -> Deal:
    """Every reachable decision of the hand dealt from `deck`, with `strategies`' mix at each.

    The round-1 walk banks each node under its line; where round 1 closes it
    hands over to the draw, which enumerates P0's four throws, then P1's, each
    consuming the stub in order, then deals board card 2 and walks round 2.
    """
    root = deal_from_deck(deck)
    deal: Deal = {
        "deck": indices(deck),
        "order": [
            indices(draw_order(hole=root.holes[player], discarded=(), board=root.board))
            for player in (0, 1)
        ],
        "r1": {},
        "d0": {},
        "d1": {},
        "r2": {},
        "show": {},
    }

    def after_round_one(state: MiniState) -> None:
        line = line_symbol(state.betting[0])
        deal["d0"][line] = per_mille(strategies[state.infoset()])
        deal["d1"][line] = {}
        deal["r2"][line] = {}
        for first in DRAW_ACTIONS:
            after_first, stub = _draw(state, first, deck, STUB)
            count = str(throw_count(first))
            if count not in deal["d1"][line]:
                deal["d1"][line][count] = per_mille(strategies[after_first.infoset()])
            for second in DRAW_ACTIONS:
                after_second, stub2 = _draw(after_first, second, deck, stub)
                round_two = after_second.apply_chance((deck[stub2],))
                combo = THROW_CODE[first] + THROW_CODE[second]
                banked: dict[str, list[int]] = {}
                _walk_round(round_two, strategies, banked, on_close=None)
                deal["r2"][line][combo] = banked
                # The showdown depends on the cards alone, not on the line.
                deal["show"].setdefault(combo, showdown(round_two))

    _walk_round(root, strategies, deal["r1"], on_close=after_round_one)
    return deal

def _draw(
    state: MiniState, throw: Action, deck: tuple[Card, ...], stub: int
) -> tuple[MiniState, int]:
    """Apply one player's throw and, if it threw, deal `deck[stub]` as the replacement.

    Whether the deck owes a replacement is read off the hole, not off the node
    kind: after the second player stands pat the next node is a chance node
    too, but it is board card 2, which the caller deals.
    """
    player = state.current_player
    after = state.apply(throw)
    if len(after.holes[player]) < len(state.holes[player]):
        return after.apply_chance((deck[stub],)), stub + 1
    return after, stub

def _walk_round(
    state: MiniState,
    strategies: Mapping,
    banked: dict[str, list[int]],
    *,
    on_close: Callable[[MiniState], None] | None,
) -> None:
    """Bank every decision of the betting round `state` is in, under its line.

    A fold ends the hand; a round that closes is handed to `on_close` (round 1,
    which leads to the draw) or is the showdown (round 2, nothing to bank).
    """
    if state.is_terminal():
        return
    if state.is_draw_decision():
        if on_close is not None:
            on_close(state)
        return
    banked[line_symbol(state.betting[-1])] = per_mille(strategies[state.infoset()])
    for action in state.legal_actions():
        _walk_round(state.apply(action), strategies, banked, on_close=on_close)

def showdown(state: MiniState) -> dict[str, object]:
    """Both final holdings, the board, and who takes each half of the pot."""
    inner = [inner_score(hole) for hole in state.holes]
    outer = [outer_score(hole, state.board) for hole in state.holes]
    return {
        "holes": [indices(hole) for hole in state.holes],
        "board": indices(state.board),
        "inner": _side(inner),
        "outer": _side(outer),
        "cats": [
            [_category(inner[player]), _category(outer[player])] for player in (0, 1)
        ],
    }

def _side(scores: list[Score]) -> int:
    if scores[0] > scores[1]:
        return 0
    if scores[1] > scores[0]:
        return 1
    return CHOP

def _category(score: Score) -> str:
    return score.category.name.lower().replace("_", " ")

# ---------------------------------------------------------------------------
# A pack
# ---------------------------------------------------------------------------

def random_decks(count: int, rng: np.random.Generator) -> list[tuple[Card, ...]]:
    """`count` deck orders, each a uniform permutation of the fifteen cards."""
    return [tuple(DECK[index] for index in rng.permutation(len(DECK))) for _ in range(count)]

def write_pack(
    deals: list[Deal],
    out: Path,
    *,
    chunk: int,
    seed: int,
    info: StrategyInfo,
    sha256: str,
) -> dict[str, object]:
    """Write `deals` in chunks of `chunk` under `out`, with the manifest; returns the manifest.

    `out` is emptied of earlier `pack-*.json` files first, so a shorter export
    cannot leave a stale chunk behind that the manifest no longer names.
    """
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("pack-*.json"):
        stale.unlink()
    names = []
    for number, start in enumerate(range(0, len(deals), chunk)):
        name = f"pack-{number:02d}.json"
        (out / name).write_text(json.dumps(deals[start : start + chunk], separators=(",", ":")))
        names.append(name)
    manifest = {
        "seed": seed,
        "deals": len(deals),
        "chunk": chunk,
        "chunks": names,
        "strategy": {
            "sha256": sha256,
            "rule": info.rule,
            "column": info.column,
            "iteration": info.iteration,
            "workers": info.workers,
            "seed": info.seed,
        },
    }
    (out / "index.json").write_text(json.dumps(manifest, indent=1))
    return manifest

def main(argv: list[str] | None = None) -> None:
    """Export a deal pack for the browser from the release strategy, or `--strategy PATH`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("web/rung3-viz/public/deals"))
    parser.add_argument("--deals", type=int, default=600)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", type=Path, help="a frozen strategy .npz")
    args = parser.parse_args(argv)

    path = args.strategy or fetch_strategy()
    strategies = load_strategy(path)
    rng = np.random.default_rng(args.seed)
    deals = [export_deal(deck, strategies) for deck in random_decks(args.deals, rng)]
    manifest = write_pack(
        deals,
        args.out,
        chunk=args.chunk,
        seed=args.seed,
        info=strategies.info,
        sha256=sha256_of(path),
    )
    print(f"wrote {manifest['deals']} deals in {len(manifest['chunks'])} chunks to {args.out}")

if __name__ == "__main__":
    main()
