"""The range export: what `/solver` is handed, checked against the game it stands in for.

A stand-in frozen strategy plays here — uniform everywhere except a few rows
pinned to known mixes — so nothing needs the 18 MB release file. The one
export over it is shared by the module: the walk reads all 6.2 million rows
and that is the cost worth paying once, not per test.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.cards import (
    DECK,
    canonical_relabelling,
    parse_cards,
    relabel,
)
from drawmaha_solver.minidrawmaha.deal_pack import card_index
from drawmaha_solver.minidrawmaha.enumeration import (
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.game import (
    DRAW_ACTIONS,
    Action,
    InfoSet,
    canonical_picture,
    draw_order,
    line_symbol,
)
from drawmaha_solver.minidrawmaha.hands import (
    InnerCategory,
    OuterCategory,
    inner_score,
    outer_score,
)
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.range_export import (
    ACTION_CODE,
    FLUSH_PICTURES,
    SCALE,
    encode_keys,
    main,
    point_meta,
    quantise,
    shape_name,
    write_fixtures,
)
from drawmaha_solver.minidrawmaha.strategy import (
    STRATEGY_SHA256,
    FrozenStrategy,
    StrategyInfo,
    average_rows,
    save_strategy,
)

INFO = StrategyInfo(rule="test", column="linear", iteration=1, seed=0, workers=1)

# The export of the release strategy as committed for `/solver`.
COMMITTED = Path(__file__).resolve().parents[2] / "web" / "solver-viz" / "public" / "mini"

# The rows the stand-in pins away from uniform: (which point, which key, the mix).
ROOT_MIX = (0.3, 0.7)
THREE_WIDE_MIX = (0.1, 0.2, 0.7)
PINNED_KEY = 5

EXPECTED_SHAPES = {
    "b1d0": {"n": 970, "stride": 4},
    "b2d0": {"n": 10170, "stride": 5},
    "b2d1": {"n": 100400, "stride": 6},
}


def _infoset(point, key):
    hole, thrown, board = key
    return InfoSet(
        player=point.player,
        hole=hole,
        discarded=thrown,
        board=board,
        draws=point.draws,
        betting=point.betting,
    )


def _three_wide_point():
    return next(point for point in public_decision_points() if point.width == 3)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One run of the CLI over the stand-in: the export dir, the fixtures path, the strategy."""
    root = tmp_path_factory.mktemp("ranges")
    table = PackedTable.whole_game()
    widths = table.widths()
    rows = average_rows(np.zeros(int(widths.sum())), widths)
    points = public_decision_points()
    for point, mix in ((points[0], ROOT_MIX), (_three_wide_point(), THREE_WIDE_MIX)):
        keys = private_keys(board_cards=point.board_cards, discards=point.discards)
        _, start, width = table.slot_of(_infoset(point, keys[PINNED_KEY]))
        rows[start : start + width] = mix
    strategy = FrozenStrategy(rows, widths, INFO)
    saved = root / "standin.npz"
    save_strategy(strategy, saved)

    out = root / "mini"
    out.mkdir()
    (out / "point-999.bin.gz").write_bytes(b"stale")
    (out / "keys-b1d1.bin").write_bytes(b"stale")
    fixtures = root / "fixtures.json"
    main(["--strategy", str(saved), "--out", str(out), "--fixtures", str(fixtures)])
    return out, fixtures, strategy


@pytest.fixture(scope="module")
def manifest(exported):
    out, _, _ = exported
    return json.loads((out / "index.json").read_text())


# ---------------------------------------------------------------------------
# Quantising
# ---------------------------------------------------------------------------


def test_quantise_rows_sum_to_the_scale_by_largest_remainder():
    rows = np.array([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0], [0.3, 0.7, 0.0]])
    assert quantise(rows, scale=250).tolist() == [[84, 83, 83], [125, 125, 0], [75, 175, 0]]
    # four equal remainders: the first two entries take the shortfall, every time
    assert quantise(np.full((1, 4), 0.25), scale=250).tolist() == [[63, 63, 62, 62]]
    # a float32 row that sums to 1 only within 1e-6 is renormalised first
    row = np.array([[0.2, 0.3, 0.5]], dtype=np.float32) * np.float32(1.000001)
    assert quantise(row, scale=250).sum() == 250
    assert quantise(rows, scale=250).dtype == np.uint8


def test_a_scale_a_byte_cannot_hold_is_refused():
    with pytest.raises(ValueError, match="256"):
        quantise(np.array([[1.0]]), scale=256)
    with pytest.raises(ValueError, match="0"):
        quantise(np.array([[1.0]]), scale=0)


def test_a_row_that_is_not_a_distribution_is_refused_by_its_index():
    good = [0.5, 0.5]
    with pytest.raises(ValueError, match="row 1 is not a probability distribution"):
        quantise(np.array([good, [1.2, -0.2]]), scale=250)
    with pytest.raises(ValueError, match="row 2 is not a probability distribution"):
        quantise(np.array([good, good, [np.nan, 1.0]]), scale=250)
    with pytest.raises(ValueError, match="row 0 is not a probability distribution"):
        quantise(np.array([[np.inf, 0.0], good]), scale=250)
    with pytest.raises(ValueError, match="row 1 sums to 0.0"):
        quantise(np.array([good, [0.0, 0.0]]), scale=250)
    with pytest.raises(ValueError, match="block of rows"):
        quantise(np.array(good), scale=250)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_the_manifest_carries_provenance_the_deck_and_the_scale(manifest):
    assert manifest["strategy"]["rule"] == "test"
    assert manifest["strategy"]["column"] == "linear"
    assert manifest["strategy"]["iteration"] == 1
    assert manifest["strategy"]["workers"] == 1
    assert manifest["strategy"]["seed"] == 0
    assert len(manifest["strategy"]["sha256"]) == 64
    assert manifest["deck"] == {"ranks": "23456", "suits": "cdh"}
    assert manifest["scale"] == SCALE == 250


def test_the_manifest_lists_the_141_points_in_walk_order(manifest):
    points = public_decision_points()
    assert len(manifest["points"]) == len(points) == 141
    for index, (meta, point) in enumerate(zip(manifest["points"], points, strict=True)):
        assert meta["id"] == index
        assert meta["player"] == point.player
        assert meta["boardCards"] == point.board_cards
        assert meta["draws"] == [signal.count for signal in point.draws]
        assert meta["lines"] == [line_symbol(line) for line in point.betting]
        assert meta["discards"] == point.discards
        assert meta["draw"] == point.is_draw_decision
        assert meta["actions"] == [ACTION_CODE[action] for action in point.actions]
        assert meta["shape"] == shape_name(point.board_cards, point.discards)
        assert meta["file"] == f"point-{index:03d}.bin.gz"
        assert meta["n"] == len(
            private_keys(board_cards=point.board_cards, discards=point.discards)
        )


def test_the_action_symbols_are_the_contracts(manifest):
    assert ACTION_CODE == {
        Action.FOLD: "f",
        Action.CHECK_CALL: "c",
        Action.POT: "p",
        Action.THROW_NONE: "n",
        Action.THROW_LOW: "l",
        Action.THROW_MID: "m",
        Action.THROW_TOP: "t",
    }
    draw_points = [meta for meta in manifest["points"] if meta["draw"]]
    assert len(draw_points) == 7 + 7 * 2  # P0 at each closed line, P1 at each line x count
    for meta in draw_points:
        assert meta["actions"] == ["n", "l", "m", "t"]
        assert len(DRAW_ACTIONS) == 4


def test_the_round_two_opener_after_a_called_bet_reads_xpc_and_empty(manifest):
    points = public_decision_points()
    betting = ((Action.CHECK_CALL, Action.POT, Action.CHECK_CALL), ())
    ids = [index for index, point in enumerate(points) if point.betting == betting]
    assert len(ids) == 4, "one opener per pair of draw counts"
    for index in ids:
        assert manifest["points"][index]["lines"] == ["xpc", ""]
        assert manifest["points"][index]["player"] == 0
        assert manifest["points"][index]["actions"] == ["c", "p"]


# ---------------------------------------------------------------------------
# The key tables
# ---------------------------------------------------------------------------


def test_encode_keys_lays_hole_discard_board_out_as_deck_indices():
    key = (parse_cards("2c 3c 6d"), parse_cards("4h"), parse_cards("5c 3d"))
    encoded = encode_keys((key,))
    assert encoded.dtype == np.uint8
    assert encoded.tolist() == [[0, 3, 13, 8, 9, 4]]
    assert encoded.tolist()[0] == [card_index(card) for group in key for card in group]


def test_only_the_three_decision_shapes_are_written_with_their_sizes(exported, manifest):
    out, _, _ = exported
    assert set(manifest["shapes"]) == set(EXPECTED_SHAPES)
    for name, expected in EXPECTED_SHAPES.items():
        shape = manifest["shapes"][name]
        assert shape["file"] == f"keys-{name}.bin"
        assert shape["n"] == expected["n"] and shape["stride"] == expected["stride"]
        assert (out / shape["file"]).stat().st_size == shape["n"] * shape["stride"]
    assert sorted(path.name for path in out.glob("keys-*.bin")) == sorted(
        f"keys-{name}.bin" for name in EXPECTED_SHAPES
    )


def test_each_key_table_is_private_keys_in_sorted_order(exported, manifest):
    out, _, _ = exported
    for name, shape in manifest["shapes"].items():
        board_cards, discards = int(name[1]), int(name[3])
        keys = private_keys(board_cards=board_cards, discards=discards)
        table = np.frombuffer((out / shape["file"]).read_bytes(), dtype=np.uint8)
        table = table.reshape(shape["n"], shape["stride"])
        assert table[0].tolist() == encode_keys(keys[:1])[0].tolist()
        assert table[-1].tolist() == encode_keys(keys[-1:])[0].tolist()
        # every card index is a real card, and no row repeats a card
        assert table.max() < len(DECK)
        assert all(len(set(row)) == len(row) for row in table[:: max(1, shape["n"] // 50)].tolist())


# ---------------------------------------------------------------------------
# The point chunks
# ---------------------------------------------------------------------------


def _chunk(out, meta):
    data = gzip.decompress((out / meta["file"]).read_bytes())
    return np.frombuffer(data, dtype=np.uint8).reshape(meta["n"], len(meta["actions"]))


def test_every_chunk_inflates_to_n_by_width_with_rows_summing_to_the_scale(exported, manifest):
    out, _, _ = exported
    for meta in manifest["points"]:
        data = gzip.decompress((out / meta["file"]).read_bytes())
        assert len(data) == meta["n"] * len(meta["actions"]), meta["file"]
        sums = _chunk(out, meta).sum(axis=1, dtype=np.int64)
        assert (sums == manifest["scale"]).all(), meta["file"]


def test_stale_chunks_and_key_tables_are_deleted_and_only_the_named_remain(exported):
    out, _, _ = exported
    assert not (out / "point-999.bin.gz").exists()
    assert not (out / "keys-b1d1.bin").exists()
    assert len(list(out.glob("point-*.bin.gz"))) == 141
    assert len(list(out.glob("keys-*.bin"))) == len(EXPECTED_SHAPES)


def test_the_pinned_rows_round_to_the_expected_bytes(exported, manifest):
    out, _, strategy = exported
    root = manifest["points"][0]
    assert root["actions"] == ["c", "p"]
    rows = _chunk(out, root)
    assert rows[PINNED_KEY].tolist() == [75, 175]
    assert rows[PINNED_KEY + 1].tolist() == [125, 125]

    three = _three_wide_point()
    index = public_decision_points().index(three)
    rows = _chunk(out, manifest["points"][index])
    assert rows[PINNED_KEY].tolist() == [25, 50, 175]
    assert rows[0].tolist() == [84, 83, 83]

    # the chunk's row k IS the strategy's row for private key k of that point
    keys = private_keys(board_cards=three.board_cards, discards=three.discards)
    expected = quantise(strategy[_infoset(three, keys[PINNED_KEY])][None, :], scale=250)
    assert rows[PINNED_KEY].tolist() == expected[0].tolist()


def test_a_draw_point_s_uniform_rows_quantise_to_63_63_62_62(exported, manifest):
    out, _, _ = exported
    draw = next(meta for meta in manifest["points"] if meta["draw"])
    rows = _chunk(out, draw)
    assert (rows == np.array([63, 63, 62, 62], dtype=np.uint8)).all()


# ---------------------------------------------------------------------------
# The committed export
# ---------------------------------------------------------------------------
#
# Everything about the committed export that does not depend on the strategy's
# numbers is pinned here, so a change to the manifest's layout, the point
# walk, the key order or the byte format cannot drift away from the files
# under `web/solver-viz/public/mini/` without a re-export. The numbers
# themselves need the 18 MB release file and are not re-derived.


@pytest.fixture(scope="module")
def committed():
    assert COMMITTED.is_dir(), f"the committed export is missing at {COMMITTED}"
    return json.loads((COMMITTED / "index.json").read_text())


def test_the_committed_manifest_names_the_release_strategy_at_this_scale(committed):
    assert committed["strategy"]["sha256"] == STRATEGY_SHA256
    assert committed["scale"] == SCALE
    assert committed["deck"] == {"ranks": "23456", "suits": "cdh"}


def test_the_committed_manifest_is_the_current_point_walk(committed):
    points = public_decision_points()
    assert committed["points"] == [
        point_meta(point, index).as_json() for index, point in enumerate(points)
    ]


def test_the_committed_key_tables_are_the_current_private_keys(committed):
    assert set(committed["shapes"]) == set(EXPECTED_SHAPES)
    for name, shape in committed["shapes"].items():
        board_cards, discards = int(name[1]), int(name[3])
        expected = encode_keys(private_keys(board_cards=board_cards, discards=discards))
        assert (shape["n"], shape["stride"]) == expected.shape
        assert (COMMITTED / shape["file"]).read_bytes() == expected.tobytes(), name
    assert sorted(path.name for path in COMMITTED.glob("keys-*.bin")) == sorted(
        shape["file"] for shape in committed["shapes"].values()
    )


def test_every_committed_chunk_is_well_formed_and_none_is_stale(committed):
    for meta in committed["points"]:
        sums = _chunk(COMMITTED, meta).sum(axis=1, dtype=np.int64)
        assert (sums == committed["scale"]).all(), meta["file"]
    assert sorted(path.name for path in COMMITTED.glob("point-*.bin.gz")) == sorted(
        meta["file"] for meta in committed["points"]
    )


# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------


def _cards(indices):
    return tuple(DECK[index] for index in indices)


def _groups(hole, thrown, board):
    return (hole, thrown, *((card,) for card in board))


def test_fixtures_hold_400_seeded_pictures_plus_the_flush_pictures(exported):
    _, fixtures, _ = exported
    cases = json.loads(fixtures.read_text())["cases"]
    assert len(cases) == 400 + len(FLUSH_PICTURES)
    shapes = {(len(case["board"]), len(case["thrown"])) for case in cases}
    assert shapes == {(1, 0), (2, 0), (2, 1)}


def test_fixtures_are_deterministic_in_the_seed(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_fixtures(first, seed=3, n=30)
    write_fixtures(second, seed=3, n=30)
    assert first.read_bytes() == second.read_bytes()
    write_fixtures(second, seed=4, n=30)
    assert first.read_bytes() != second.read_bytes()


def test_every_fixture_case_is_a_legal_picture_with_python_s_answers(exported):
    _, fixtures, _ = exported
    for case in json.loads(fixtures.read_text())["cases"]:
        hole, thrown, board = _cards(case["hole"]), _cards(case["thrown"]), _cards(case["board"])
        dealt = (*hole, *thrown, *board)
        assert len(hole) == 3 and len(thrown) <= 1 and 1 <= len(board) <= 2
        assert len(set(dealt)) == len(dealt), case
        assert not (len(board) == 1 and thrown), "no decision stands on b1d1"

        key = canonical_picture(hole, thrown, board)
        assert case["key"] == {
            "hole": [card_index(c) for c in key[0]],
            "thrown": [card_index(c) for c in key[1]],
            "board": [card_index(c) for c in key[2]],
        }
        relabelling = tuple(case["relabelling"])
        assert relabelling == canonical_relabelling(*_groups(hole, thrown, board))
        assert tuple(sorted(relabel(relabelling, hole))) == key[0]

        order = draw_order(hole=hole, discarded=thrown, board=board)
        assert case["drawOrder"] == [card_index(c) for c in order]
        assert case["drawOrder"] == sorted(
            case["hole"], key=lambda i: (DECK[i].rank, relabelling[DECK[i].suit])
        )

        inner = inner_score(hole)
        assert case["inner"] == [int(inner.category), *inner.ranks]
        if len(board) == 1:
            assert case["outer"] is None
        else:
            outer = outer_score(hole, board)
            assert case["outer"] == [int(outer.category), *outer.ranks]


def test_the_flush_pictures_pin_every_suit_relationship(exported):
    _, fixtures, _ = exported
    cases = json.loads(fixtures.read_text())["cases"][400:]
    assert len(cases) == len(FLUSH_PICTURES) >= 20
    inner = {case["inner"][0] for case in cases}
    outer = {case["outer"][0] for case in cases if case["outer"] is not None}
    assert {InnerCategory.FLUSH, InnerCategory.STRAIGHT_FLUSH} <= inner
    assert {OuterCategory.FLUSH, OuterCategory.STRAIGHT_FLUSH} <= outer
    suits_seen = set()
    for case in cases:
        hole_suits = {DECK[i].suit for i in case["hole"]}
        board_suits = {DECK[i].suit for i in case["board"]}
        suits_seen.add(
            "with" if hole_suits & board_suits else "against"
        )
        if len(hole_suits | board_suits | {DECK[i].suit for i in case["thrown"]}) == 3:
            suits_seen.add("three")
    assert suits_seen == {"with", "against", "three"}
    # the pictures named by the rung's own tests are among them
    holes = {tuple(case["hole"]) for case in cases}
    assert tuple(card_index(c) for c in parse_cards("2c 3c 6h")) in holes
    assert tuple(card_index(c) for c in parse_cards("2c 3c 4h")) in holes
