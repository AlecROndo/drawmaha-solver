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
  and a byte holds it. At 250 each entry is within 1/250 (0.4%) of the
  strategy's own number — an action taken under 0.2% of the time rounds to
  a zero byte — and the whole strategy is 14 MB raw, ~4 MB gzipped, against
  57 MB of float32. What the browser builds from the bytes is exact to that
  rounding and no further: a state's reach is a product of up to five
  entries, so the rounding compounds along the line.

The second thing written is `fixtures.json`: a few hundred random pictures
with Python's answers beside them — the canonical key, the relabelling behind
it, the draw order, both hand scores. The browser ports exactly those four
pieces of `game.py`/`hands.py` and nothing else, and the fixtures are what
pin the port to the original. Every number the page shows is read off the
export; the port only has to NAME things the same way.

The site ships three strategies of the same game, one export each
(`RELEASES`): rung 3's tabular LCFR average, and rung 4's two Deep CFR
answers at iteration 500 — Π, the policy net, which is the average a game
too big for tables can keep, and the exact running average, which
mini-drawmaha is small enough to hold beside it. Each is a GitHub release
asset fetched by digest, and each `index.json` carries its grade copied from
the committed grades file, so the page's status line quotes a number with a
committed source and the tests hold the copy to the file. The three exports
share their key tables and their point walk byte for byte — only the chunks
differ — which is what lets the page switch strategy under a spot without
changing what the spot's states are. Every export holds a row for every
state, reached or not, and the three differ in what an unvisited row holds:
the two averages show a flat mix there (an empty row averages to uniform, by
`average_rows`' convention), while Π shows the net's generalisation from the
rows it did see — a real mix, and the reason a net is interesting. So under
the switch an unvisited row is never missing, and it is where the two kinds
of answer differ most. The fixtures depend on the rules alone,
so exporting another strategy must leave them untouched: `settle_fixtures`
refuses to overwrite a committed file that differs.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drawmaha_solver.minidrawmaha.rules.cards import (
    DECK,
    RANK_SYMBOL,
    SUIT_SYMBOL,
    Card,
    canonical_relabelling,
    hand_symbol,
    parse_cards,
)
from drawmaha_solver.minidrawmaha.answer.deal_pack import THROW_CODE, indices
from drawmaha_solver.minidrawmaha.rules.enumeration import (
    PrivateKey,
    PublicPoint,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.rules.game import (
    THROW_CAP,
    Action,
    InfoSet,
    canonical_picture,
    draw_order,
    line_symbol,
)
from drawmaha_solver.minidrawmaha.rules.hands import (
    BOARD_CARDS,
    HOLE_CARDS,
    Score,
    inner_score,
    outer_score,
)
from drawmaha_solver.minidrawmaha.answer.strategy import (
    RELEASE_TAG,
    STRATEGY_FILE,
    STRATEGY_SHA256,
    StrategyInfo,
    fetch_strategy,
    load_strategy,
    sha256_of,
)

# What a probability is multiplied by before it is rounded to a byte.
SCALE = 250

# A byte holds 0..255, so a row can sum to at most that.
MAX_SCALE = 255

# What an export owns under `--out`, and so deletes before writing: a point's
# chunk and a shape's key table. `index.json` is overwritten, not globbed.
STALE_PATTERNS = ("point-*.bin.gz", "keys-*.bin")

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
# The releases: the strategies the site ships
# ---------------------------------------------------------------------------

# The repo root, for the grades files a release names. The package is
# installed editable, so this file sits under `src/` in the checkout.
REPO = Path(__file__).resolve().parents[4]

# Where the exports land, relative to the repo root the CLI is run from.
PUBLIC = Path("web/solver-viz/public")

GITHUB_RELEASES = "https://github.com/AlecROndo/drawmaha-solver/releases/download"

@dataclass(frozen=True, slots=True)
class Release:
    """One strategy the site ships: where its file is published, and where its grade is recorded.

    `name` is what `--release` takes and `directory` where the export lands
    under `web/solver-viz/public/`. The asset is fetched by `tag` and `asset`
    and refused unless it hashes to `sha256`, so the export is reproducible
    from a clean checkout. `grade_file` is the committed record the page
    quotes and `grade_key` the object in it that holds the strategy's
    `exploitability`; the exporter copies the number into `index.json`, and
    `test_range_export.py` holds the committed copy to the file.
    """

    name: str
    directory: str
    tag: str
    asset: str
    sha256: str
    grade_file: str
    grade_key: str

    @property
    def url(self) -> str:
        return f"{GITHUB_RELEASES}/{self.tag}/{self.asset}"

    @property
    def out(self) -> Path:
        return PUBLIC / self.directory

RELEASES: dict[str, Release] = {
    release.name: release
    for release in (
        Release(
            name="tabular",
            directory="mini",
            tag=RELEASE_TAG,
            asset=STRATEGY_FILE,
            sha256=STRATEGY_SHA256,
            grade_file="figures/rung3/answer_sheet.json",
            grade_key="frozen_strategy",
        ),
        Release(
            name="deep-pi",
            directory="mini-deep-pi",
            tag="rung4-deep-t500",
            asset="deep-256x4-s16000-lr5e-4-t500-pi.npz",
            sha256="7aa08383a7f6389bc60633904b050793ed7a3d42cb017c877c3d26ca5a2058a5",
            grade_file="figures/rung4/sweep_256x4-s16000-lr5e-4.json",
            grade_key="deep/00500/pi",
        ),
        Release(
            name="deep-exact",
            directory="mini-deep-exact",
            tag="rung4-deep-t500",
            asset="deep-256x4-s16000-lr5e-4-t500-exact.npz",
            sha256="117c936030b0271c73db516ec4bcedaf83162952ee2a9740e8fd4832424115aa",
            grade_file="figures/rung4/sweep_256x4-s16000-lr5e-4.json",
            grade_key="deep/00500/exact",
        ),
    )
}

def release_of(sha256: str) -> Release | None:
    """The release whose asset hashes to `sha256`, so a local copy of one is treated as it."""
    return next((release for release in RELEASES.values() if release.sha256 == sha256), None)

@dataclass(frozen=True, slots=True)
class Grade:
    """The strategy's exact exploitability as the page quotes it, and where the number was read from."""

    exploitability: float
    hands_per_seat: int
    file: str
    key: str

    def as_json(self) -> dict[str, object]:
        return {
            "exploitability": self.exploitability,
            "handsPerSeat": self.hands_per_seat,
            "file": self.file,
            "key": self.key,
        }

def read_grade(release: Release, info: StrategyInfo, *, root: Path = REPO) -> Grade:
    """`release`'s grade from its committed file under `root`.

    The entry's `hands_per_seat` is kept when it is there (rung 4's grades
    carry it). Rung 3's answer sheet predates the field, and lockstep plays
    one hand per seat per worker per iteration, so there it is
    iteration × workers. The fallback is taken only when the field is absent
    — a recorded 0 is a recorded 0, not a missing field — and neither route
    may come to zero: a deep strategy records `workers=0` (its hands are
    counted by the run, not by lockstep), so a deep entry without the field
    would fall through to 0 × iteration, and the page would print "0 hands".
    A missing file, a file without the key, a count of zero and a count that
    is not an integer are each refused by name rather than read as a number
    the page would then quote.
    """
    file = root / release.grade_file
    if not file.exists():
        raise FileNotFoundError(
            f"{release.name}'s grades file {release.grade_file} is not under {root}; the exporter "
            "reads it from the checkout (the package installed editable), not from site-packages"
        )
    recorded = json.loads(file.read_text())
    if release.grade_key not in recorded:
        raise KeyError(f"{release.grade_file} has no {release.grade_key}; it holds {sorted(recorded)}")
    entry = recorded[release.grade_key]
    recorded_hands = "hands_per_seat" in entry
    # `type(...) is int`, not `isinstance`: a bool is an int to isinstance, and
    # a JSON `true` would otherwise be read as one hand.
    if recorded_hands and type(entry["hands_per_seat"]) is not int:
        raise ValueError(
            f"{release.grade_file}'s {release.grade_key} records hands_per_seat = "
            f"{entry['hands_per_seat']!r}, not an integer"
        )
    hands = entry["hands_per_seat"] if recorded_hands else info.iteration * info.workers
    if hands <= 0:
        counted = (
            f"records hands_per_seat = {hands}"
            if recorded_hands
            else f"records no hands_per_seat and the strategy's iteration × workers is {info.iteration} × {info.workers}"
        )
        raise ValueError(
            f"{release.grade_file}'s {release.grade_key} {counted}; a grade the page quotes "
            "must say how many hands it was trained on"
        )
    return Grade(
        exploitability=float(entry["exploitability"]),
        hands_per_seat=hands,
        file=release.grade_file,
        key=release.grade_key,
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
    """`index.json`: where the strategy came from and how it grades, the deck, the scale, the tables.

    `release` and `grade` are `None` for a file that is not a shipped
    strategy (a stand-in in the tests, or an experiment exported with
    `--strategy --out`): the page then has provenance but no number to quote,
    and says so rather than inventing one.
    """

    info: StrategyInfo
    sha256: str
    asset: str
    release: Release | None
    grade: Grade | None
    scale: int
    shapes: tuple[ShapeMeta, ...]
    points: tuple[PointMeta, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "strategy": {
                "sha256": self.sha256,
                "release": self.release.tag if self.release else None,
                "asset": self.asset,
                "rule": self.info.rule,
                "column": self.info.column,
                "iteration": self.info.iteration,
                "workers": self.info.workers,
                "seed": self.info.seed,
            },
            "grade": self.grade.as_json() if self.grade else None,
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
    asset: str,
    release: Release | None = None,
    grade: Grade | None = None,
) -> Manifest:
    """Write `index.json`, one key table per shape and one chunk per point under `out`.

    `out` is emptied of earlier `point-*.bin.gz` and `keys-*.bin` files first,
    as the deal pack empties its chunks, so a point that moved or a shape no
    longer written cannot leave a stale file the manifest does not name. Only
    the shapes a decision point stands on get a key table; the chunks are
    gzipped at `GZIP_LEVEL` with no timestamp, so
    the same strategy exports to the same bytes and a re-export shows up in
    git only when the numbers moved. `asset` is the strategy file's name,
    `release` and `grade` its provenance when it is a shipped strategy.
    """
    out.mkdir(parents=True, exist_ok=True)
    for pattern in STALE_PATTERNS:
        for stale in out.glob(pattern):
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
        info=info,
        sha256=sha256,
        asset=asset,
        release=release,
        grade=grade,
        scale=scale,
        shapes=tuple(shapes),
        points=tuple(metas),
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

def settle_fixtures(path: Path, *, write: bool = False, seed: int = 7, n: int = 400) -> int:
    """Write the fixtures to `path` if absent, leave them if identical, refuse if they differ; returns the count.

    The fixtures depend on the rules alone, never on the strategy, so a
    second export must not change them: a difference means the key, the draw
    order or a ranking moved, and the browser's port needs re-pinning on
    purpose — `--write-fixtures` — not as a side effect of exporting another
    strategy. The fresh file is written beside `path` and compared byte for
    byte, so the committed file is untouched unless the caller said so, and
    the fresh one is removed whichever way it goes.
    """
    fresh = path.with_name(path.name + ".fresh")
    try:
        count = write_fixtures(fresh, seed=seed, n=n)
        if path.exists() and not write and fresh.read_bytes() != path.read_bytes():
            raise ValueError(
                f"the fixtures at {path} differ from the rules' answers; the key, the draw "
                "order or a ranking moved, so re-pin the browser's port on purpose with "
                "--write-fixtures and re-export every strategy"
            )
        if not path.exists() or write:
            fresh.replace(path)
    finally:
        fresh.unlink(missing_ok=True)
    return count

# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Export one shipped strategy's ranges (`--release`, default tabular) or a local file's (`--strategy`), and settle the fixtures.

    A local file that hashes to a release's asset is treated as that release
    — its export lands in the release's directory with its grade — so a copy
    already on this machine needs no download. Any other file is an
    experiment: it gets no grade, and `--out` must say where it goes, so an
    experiment can never overwrite a shipped export by default.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release",
        choices=sorted(RELEASES),
        help="which shipped strategy to fetch and export (default: tabular)",
    )
    parser.add_argument(
        "--strategy",
        type=Path,
        help="a frozen strategy .npz instead of the download; a release's own file is known by its digest",
    )
    parser.add_argument(
        "--out", type=Path, help="the export directory; a release's own under web/solver-viz/public/"
    )
    parser.add_argument(
        "--fixtures", type=Path, default=Path("web/solver-viz/src/mini/fixtures.json")
    )
    parser.add_argument(
        "--write-fixtures",
        action="store_true",
        help="overwrite fixtures that differ (the rules changed on purpose)",
    )
    parser.add_argument("--scale", type=int, default=SCALE)
    args = parser.parse_args(argv)

    if args.strategy:
        path = args.strategy
        digest = sha256_of(path)
        release = release_of(digest)
        if args.release and (release is None or release.name != args.release):
            raise ValueError(
                f"{path} hashes to {digest}, not to release {args.release}'s "
                f"{RELEASES[args.release].sha256}"
            )
    else:
        release = RELEASES[args.release or "tabular"]
        path = fetch_strategy(url=release.url, sha256=release.sha256)
        digest = release.sha256
    if release is None and args.out is None:
        parser.error(f"{path} is not a shipped strategy; say where its export goes with --out")
    out = args.out or release.out

    # The fixtures first: a rules change fails here, before the walk writes anything.
    cases = settle_fixtures(args.fixtures, write=args.write_fixtures)
    strategy = load_strategy(path)
    grade = read_grade(release, strategy.info) if release else None
    manifest = export_ranges(
        strategy,
        out=out,
        scale=args.scale,
        info=strategy.info,
        sha256=digest,
        asset=path.name,
        release=release,
        grade=grade,
    )

    total = sum(file.stat().st_size for file in out.iterdir())
    largest = max(out.glob("point-*.bin.gz"), key=lambda file: file.stat().st_size)
    source = f"release {release.name} ({release.tag}/{release.asset})" if release else str(path)
    graded = f", exploitability {grade.exploitability:.4f}" if grade else ", no grade"
    print(f"strategy: {source}{graded}")
    print(
        f"wrote {len(manifest.points)} points over {len(manifest.shapes)} shapes to {out}: "
        f"{total / 1e6:.2f} MB, largest chunk {largest.name} at "
        f"{largest.stat().st_size / 1e3:.0f} kB gzipped"
    )
    print(f"settled {cases} fixture cases at {args.fixtures} ({args.fixtures.stat().st_size / 1e3:.0f} kB)")

if __name__ == "__main__":
    main()
