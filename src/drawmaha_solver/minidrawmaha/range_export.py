"""The range export: the frozen strategy, cut by public decision point, for `/solver`.

`/rung3` is handed a deal pack — a few hundred hands dealt in advance with the
strategy's mix at every node of each. `/solver` asks the opposite question:
not "what does the bot do with THIS hand" but "what does it do with EVERY
hand at this spot" — the actor's whole range at one point of the game, and
how that range is split across the actions. The browser can only answer that
if it holds the strategy for every private state of the spot at once, so
this module writes the table out by spot.

The table factorises the way `enumeration.py` builds it, and the export keeps
that factorisation on disk:

* a **public point** is one of the 141 spots a player acts at: the betting so
  far, both draw counts, how many board cards are out, whose turn it is. It
  has nothing to do with cards, so one point is a *column* of the table — the
  browser picks the point from the line it is on and reads the whole column.
* a **private key** is what the actor holds, suits relabelled jointly
  (`canonical_picture`): the hole, the discards, the board in dealt order.
  Every point of one **shape** — `b{board cards}d{discards}` — stands on the
  same sorted list of keys, so a shape's key list is written ONCE and every
  point of that shape is a plain array in that order. Three shapes carry a
  decision: 970 keys at `b1d0`, 10,170 at `b2d0`, 100,400 at `b2d1`.
* the **scale** turns a float32 row into bytes: each probability becomes
  `round(p * scale)` by largest remainder, so a row sums to exactly `scale`
  and a byte holds it. At 250 the resolution is 0.4%, finer than the solve's
  own convergence, and the whole strategy is 14 MB raw, ~4 MB gzipped —
  against 57 MB of float32.

The second thing written is `fixtures.json`: a few hundred random pictures
with Python's answers beside them — the canonical key, the relabelling behind
it, the draw order, both hand scores. The browser ports exactly those four
pieces of `game.py`/`hands.py` and nothing else, and the fixtures are what
pin the port to the original. Every number the page shows is read off the
export; the port only has to NAME things the same way.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drawmaha_solver.minidrawmaha.cards import (
    DECK,
    RANK_SYMBOL,
    SUIT_SYMBOL,
    Card,
    canonical_relabelling,
    hand_symbol,
    parse_cards,
)
from drawmaha_solver.minidrawmaha.deal_pack import THROW_CODE, indices
from drawmaha_solver.minidrawmaha.enumeration import (
    PrivateKey,
    PublicPoint,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.game import (
    THROW_CAP,
    Action,
    InfoSet,
    canonical_picture,
    draw_order,
    line_symbol,
)
from drawmaha_solver.minidrawmaha.hands import (
    BOARD_CARDS,
    HOLE_CARDS,
    Score,
    inner_score,
    outer_score,
)
from drawmaha_solver.minidrawmaha.strategy import (
    StrategyInfo,
    fetch_strategy,
    load_strategy,
    sha256_of,
)

# What a probability is multiplied by before it is rounded to a byte.
SCALE = 250

# A byte holds 0..255, so a row can sum to at most that.
MAX_SCALE = 255

# The chunks are committed and fetched once per visit; the slowest level buys
# a few percent on files that are mostly runs of near-pure rows.
GZIP_LEVEL = 9

# The ledger's columns as single letters. `c` is CHECK_CALL whichever job it
# is doing — the browser knows from the line whether anything was owed.
ACTION_CODE = {
    Action.FOLD: "f",
    Action.CHECK_CALL: "c",
    Action.POT: "p",
    **THROW_CODE,
}

# The three shapes a fixture picture can have: the ones a decision stands on.
# `b1d1` never carries a decision — both players' throws are in before board
# card 2, and nobody acts between the second throw and that card.
FIXTURE_SHAPES = ((1, 0), (2, 0), (2, 1))

# Pictures whose suits matter — a flush in one half or the other, the hole
# suited WITH the board, suited AGAINST it, every suit on the table — as
# `(hole, thrown, board)` symbols. The first group are the pictures the rung's
# own tests name in `test_hands.py`, `test_game.py` and `test_cards.py`; the
# rest are chosen so the browser's port of the joint relabelling and both
# rankings meets every suit relationship with Python's answer beside it.
FLUSH_PICTURES = (
    ("2c 3c 4c", "", "5d"),
    ("2c 3c 5c", "", "4d"),
    ("2c 3c 6h", "", "4c 6c"),
    ("2c 3c 6h", "", "4c 5c"),
    ("2c 2d 2h", "", "3c 4c"),
    ("2c 6d 6h", "", "3c 4c"),
    ("2c 3c 6d", "", "4c"),
    ("2c 3c 6d", "", "4c 5c"),
    ("2d 4h 5h", "", "4c 5c"),
    ("2c 3c 4h", "", "5c 6c"),
    ("2d 3h 4c", "", "5c 6c"),
    ("2c 3c 6d", "3d", "4c 5h"),
    ("2c 3c 5c", "", "4d 6h"),
    ("2d 3d 6d", "", "4c"),
    ("2h 3h 4h", "", "5c 6d"),
    ("2c 3c 6c", "4c", "5c 2d"),
    ("2d 4d 6d", "3h", "5d 2c"),
    ("3h 4h 5h", "6c", "2h 6d"),
    ("2c 3d 4h", "", "5c"),
    ("2c 3d 4h", "5c", "6c 2d"),
    ("3c 4c 5d", "2h", "6d 6h"),
    ("6c 6d 6h", "", "5c 5d"),
    ("2h 3h 4d", "", "5h 6h"),
    ("2h 3h 4d", "4h", "5h 6h"),
    ("2c 4c 6c", "", "3c 5c"),
    ("2d 3d 4d", "", "5d 6d"),
    ("5c 6c 2d", "", "3c 4c"),
    ("2c 3c 6h", "6c", "4c 5c"),
    ("2c 3d 6d", "2d", "4d 5d"),
    ("4c 5c 6c", "", "2c"),
    ("4c 5c 6c", "3d", "2c 3h"),
)

# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ShapeMeta:
    """One key table: its name, its file, how many keys, how many bytes per key."""

    name: str
    file: str
    n: int
    stride: int

    def as_json(self) -> dict[str, object]:
        return {"file": self.file, "n": self.n, "stride": self.stride}

@dataclass(frozen=True, slots=True)
class PointMeta:
    """One public decision point as the manifest describes it.

    `lines` is one `line_symbol` string per betting round begun, `draws` the
    public draw counts so far, `actions` the chunk's columns in
    `legal_actions()` order — everything the browser needs to find the point
    from the spot it is on and to read the chunk's row. `draw` says the point
    chooses a discard rather than a bet.
    """

    id: int
    player: int
    board_cards: int
    draws: tuple[int, ...]
    lines: tuple[str, ...]
    discards: int
    draw: bool
    actions: tuple[str, ...]
    shape: str
    file: str
    n: int

    def as_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "player": self.player,
            "boardCards": self.board_cards,
            "draws": list(self.draws),
            "lines": list(self.lines),
            "discards": self.discards,
            "draw": self.draw,
            "actions": list(self.actions),
            "shape": self.shape,
            "file": self.file,
            "n": self.n,
        }

@dataclass(frozen=True, slots=True)
class Manifest:
    """`index.json`: where the strategy came from, the deck, the scale, the tables."""

    info: StrategyInfo
    sha256: str
    scale: int
    shapes: tuple[ShapeMeta, ...]
    points: tuple[PointMeta, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "strategy": {
                "sha256": self.sha256,
                "rule": self.info.rule,
                "column": self.info.column,
                "iteration": self.info.iteration,
                "workers": self.info.workers,
                "seed": self.info.seed,
            },
            "deck": {"ranks": RANK_SYMBOL, "suits": SUIT_SYMBOL},
            "scale": self.scale,
            "shapes": {shape.name: shape.as_json() for shape in self.shapes},
            "points": [point.as_json() for point in self.points],
        }

def shape_name(board_cards: int, discards: int) -> str:
    """The name a key table and a point share: `b1d0`, `b2d0`, `b2d1`."""
    return f"b{board_cards}d{discards}"

def point_meta(point: PublicPoint, index: int) -> PointMeta:
    """`point` as the manifest's entry `index`, naming the chunk `point-<index>.bin.gz`."""
    return PointMeta(
        id=index,
        player=point.player,
        board_cards=point.board_cards,
        draws=tuple(signal.count for signal in point.draws),
        lines=tuple(line_symbol(line) for line in point.betting),
        discards=point.discards,
        draw=point.is_draw_decision,
        actions=tuple(ACTION_CODE[action] for action in point.actions),
        shape=shape_name(point.board_cards, point.discards),
        file=f"point-{index:03d}.bin.gz",
        n=len(private_keys(board_cards=point.board_cards, discards=point.discards)),
    )

# ---------------------------------------------------------------------------
# Quantising
# ---------------------------------------------------------------------------

def quantise(rows: np.ndarray, *, scale: int) -> np.ndarray:
    """`rows` `(n, width)` as uint8 with every row summing to exactly `scale`, by largest remainder.

    `deal_pack.per_mille` for a whole block at once. Rounding each entry on
    its own leaves a row at `scale - 1` or `scale + 1`, and the browser reads
    a row as a partition of `scale` — the share of a region that takes each
    action — so a row off by one would show a sliver of an action nobody
    takes. The floors are taken first and the shortfall goes to the entries
    that lost the most; a tie goes to the earlier column, every time.

    A row that is not a distribution — negative, non-finite, all zero — is
    refused by its index rather than written as garbage bytes.
    """
    if not 1 <= scale <= MAX_SCALE:
        raise ValueError(f"the scale must fit a byte, 1..{MAX_SCALE}, got {scale}")
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"quantise takes a block of rows, got shape {values.shape}")
    if not np.isfinite(values).all() or (values < 0.0).any():
        bad = int(np.flatnonzero(~(np.isfinite(values) & (values >= 0.0)).all(axis=1))[0])
        raise ValueError(f"row {bad} is not a probability distribution: {values[bad]}")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0.0).any():
        bad = int(np.flatnonzero(totals[:, 0] <= 0.0)[0])
        raise ValueError(f"row {bad} sums to {totals[bad, 0]}, not to anything positive")
    scaled = values / totals * scale
    floors = np.floor(scaled)
    short = np.rint(scale - floors.sum(axis=1)).astype(np.int64)
    # Each entry's standing among its row's remainders, 0 for the largest;
    # the stable sort is what sends a tie to the earlier column.
    order = np.argsort(-(scaled - floors), axis=1, kind="stable")
    standing = np.empty_like(order)
    np.put_along_axis(
        standing, order, np.broadcast_to(np.arange(order.shape[1]), order.shape), axis=1
    )
    return (floors + (standing < short[:, None])).astype(np.uint8)

# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

def encode_keys(keys: Sequence[PrivateKey]) -> np.ndarray:
    """`keys` as uint8 `(n, stride)`: hole, then discards, then board, each card `rank*3+suit`.

    The hole is in the canonical order `private_keys` sorts it in, the board in
    the order it was dealt — the row is the key read left to right, so the
    browser's `keyString` of a canonical picture is a byte-for-byte match.
    """
    return np.array(
        [[index for group in key for index in indices(group)] for key in keys],
        dtype=np.uint8,
    )

def point_rows(strategy: Mapping[InfoSet, np.ndarray], point: PublicPoint) -> np.ndarray:
    """`strategy`'s mix at every private key of `point`, `(n, width)` in `private_keys` order.

    Read one infoset at a time through the `Mapping` — about 1.5 µs a row
    against a frozen strategy, ten seconds for the whole game — rather than
    through the packed table's layout, so a profile that is not a
    `FrozenStrategy` exports the same way and nothing here knows how the
    probabilities are stored.
    """
    keys = private_keys(board_cards=point.board_cards, discards=point.discards)
    rows = np.empty((len(keys), point.width))
    for position, (hole, thrown, board) in enumerate(keys):
        rows[position] = strategy[
            InfoSet(
                player=point.player,
                hole=hole,
                discarded=thrown,
                board=board,
                draws=point.draws,
                betting=point.betting,
            )
        ]
    return rows

def export_ranges(
    strategy: Mapping[InfoSet, np.ndarray],
    *,
    out: Path,
    scale: int = SCALE,
    info: StrategyInfo,
    sha256: str,
) -> Manifest:
    """Write `index.json`, one key table per shape and one chunk per point under `out`.

    `out` is emptied of earlier `point-*.bin.gz` files first, as the deal pack
    empties its chunks, so a point that moved cannot leave a stale chunk the
    manifest no longer names. Only the shapes a decision point stands on get
    a key table; the chunks are gzipped at `GZIP_LEVEL` with no timestamp, so
    the same strategy exports to the same bytes and a re-export shows up in
    git only when the numbers moved.
    """
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("point-*.bin.gz"):
        stale.unlink()
    points = public_decision_points()

    shapes = []
    for board_cards, discards in sorted({(p.board_cards, p.discards) for p in points}):
        name = shape_name(board_cards, discards)
        table = encode_keys(private_keys(board_cards=board_cards, discards=discards))
        file = f"keys-{name}.bin"
        (out / file).write_bytes(table.tobytes())
        shapes.append(ShapeMeta(name=name, file=file, n=table.shape[0], stride=table.shape[1]))

    metas = []
    for index, point in enumerate(points):
        meta = point_meta(point, index)
        block = quantise(point_rows(strategy, point), scale=scale)
        (out / meta.file).write_bytes(
            gzip.compress(block.tobytes(), compresslevel=GZIP_LEVEL, mtime=0)
        )
        metas.append(meta)

    manifest = Manifest(
        info=info, sha256=sha256, scale=scale, shapes=tuple(shapes), points=tuple(metas)
    )
    (out / "index.json").write_text(json.dumps(manifest.as_json(), indent=1))
    return manifest

# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Picture:
    """What one player sees, in physical suits: the hole, what they threw, the board as dealt.

    Refuses a picture no deal produces — a repeated card, a hole that is not
    three, more throws than the cap, a board that is not one or two cards —
    because a fixture that pinned an impossible picture would pin the port to
    an answer the game never asks for.
    """

    hole: tuple[Card, ...]
    thrown: tuple[Card, ...]
    board: tuple[Card, ...]

    def __post_init__(self) -> None:
        dealt = (*self.hole, *self.thrown, *self.board)
        if len(set(dealt)) != len(dealt):
            raise ValueError(f"a picture holds each card once: {hand_symbol(dealt)}")
        if len(self.hole) != HOLE_CARDS:
            raise ValueError(f"a hole is {HOLE_CARDS} cards, got {hand_symbol(self.hole)}")
        if len(self.thrown) > THROW_CAP:
            raise ValueError(f"a draw throws at most {THROW_CAP}, got {hand_symbol(self.thrown)}")
        if not 1 <= len(self.board) <= BOARD_CARDS:
            raise ValueError(f"the board is 1 or 2 cards, got {hand_symbol(self.board)}")

    def groups(self) -> tuple[tuple[Card, ...], ...]:
        """The groups `canonical` relabels, as `game._picture` builds them: hole, discards, one per board card.

        Each board card on its own so its dealt position survives the sort
        inside `canonical` — the same split `canonical_picture` makes, and
        `test_range_export.py` holds the two to the same key.
        """
        return (self.hole, self.thrown, *((card,) for card in self.board))

def random_picture(rng: np.random.Generator, *, board_cards: int, discards: int) -> Picture:
    """A uniform picture of this shape: the top cards of a shuffled deck, dealt hole, throw, board."""
    deal = [DECK[index] for index in rng.permutation(len(DECK))]
    hole = HOLE_CARDS
    return Picture(
        hole=tuple(deal[:hole]),
        thrown=tuple(deal[hole : hole + discards]),
        board=tuple(deal[hole + discards : hole + discards + board_cards]),
    )

def _score_json(score: Score) -> list[int]:
    return [int(score.category), *score.ranks]

def fixture_case(picture: Picture) -> dict[str, object]:
    """`picture` with Python's four answers beside it, cards as `rank*3+suit`.

    `key` is `canonical_picture`'s, `relabelling` the permutation behind it
    (`new_suit = relabelling[old_suit]`), `drawOrder` the physical hole as
    `THROW_LOW/MID/TOP` index it, `inner` and `outer` the two `Score`s as
    `[category, *ranks]` — `outer` is `None` while only one board card is out,
    because the outer half needs both.
    """
    hole, thrown, board = canonical_picture(picture.hole, picture.thrown, picture.board)
    order = draw_order(hole=picture.hole, discarded=picture.thrown, board=picture.board)
    outer = (
        _score_json(outer_score(picture.hole, picture.board))
        if len(picture.board) == BOARD_CARDS
        else None
    )
    return {
        "hole": indices(picture.hole),
        "thrown": indices(picture.thrown),
        "board": indices(picture.board),
        "key": {"hole": indices(hole), "thrown": indices(thrown), "board": indices(board)},
        "relabelling": list(canonical_relabelling(*picture.groups())),
        "drawOrder": indices(order),
        "inner": _score_json(inner_score(picture.hole)),
        "outer": outer,
    }

def write_fixtures(path: Path, *, seed: int = 7, n: int = 400) -> int:
    """Write `n` seeded random pictures, cycling the three shapes, then `FLUSH_PICTURES`; returns the count.

    The random pictures come first and the named ones after, so a test can
    find the flush pictures at a fixed offset; the seed makes two runs write
    identical bytes.
    """
    rng = np.random.default_rng(seed)
    pictures = [
        random_picture(rng, board_cards=board_cards, discards=discards)
        for board_cards, discards in (FIXTURE_SHAPES[i % len(FIXTURE_SHAPES)] for i in range(n))
    ]
    pictures += [
        Picture(hole=parse_cards(hole), thrown=parse_cards(thrown), board=parse_cards(board))
        for hole, thrown, board in FLUSH_PICTURES
    ]
    cases = [fixture_case(picture) for picture in pictures]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}, separators=(",", ":")))
    return len(cases)

# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Export the ranges and the fixtures from the release strategy, or `--strategy PATH`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("web/solver-viz/public/mini"))
    parser.add_argument(
        "--fixtures", type=Path, default=Path("web/solver-viz/src/mini/fixtures.json")
    )
    parser.add_argument("--strategy", type=Path, help="a frozen strategy .npz")
    parser.add_argument("--scale", type=int, default=SCALE)
    args = parser.parse_args(argv)

    path = args.strategy or fetch_strategy()
    strategy = load_strategy(path)
    manifest = export_ranges(
        strategy, out=args.out, scale=args.scale, info=strategy.info, sha256=sha256_of(path)
    )
    cases = write_fixtures(args.fixtures)

    total = sum(file.stat().st_size for file in args.out.iterdir())
    largest = max(args.out.glob("point-*.bin.gz"), key=lambda file: file.stat().st_size)
    print(
        f"wrote {len(manifest.points)} points over {len(manifest.shapes)} shapes to {args.out}: "
        f"{total / 1e6:.2f} MB, largest chunk {largest.name} at "
        f"{largest.stat().st_size / 1e3:.0f} kB gzipped"
    )
    print(f"wrote {cases} fixture cases to {args.fixtures} ({args.fixtures.stat().st_size / 1e3:.0f} kB)")

if __name__ == "__main__":
    main()
