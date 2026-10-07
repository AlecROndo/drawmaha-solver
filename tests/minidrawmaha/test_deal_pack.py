"""The deal pack: what the browser is handed, checked against the game it stands in for.

A stand-in profile plays here (uniform, or a pure one), so nothing needs the
18 MB release file; the frozen strategy itself is pinned in `test_strategy.py`.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha import deal_pack
from drawmaha_solver.minidrawmaha.cards import DECK, parse_cards
from drawmaha_solver.minidrawmaha.deal_pack import (
    CHOP,
    PER_MILLE,
    STUB,
    THROW_CODE,
    card_index,
    deal_from_deck,
    export_deal,
    per_mille,
    random_decks,
    write_pack,
)
from drawmaha_solver.minidrawmaha.game import (
    DRAW_ACTIONS,
    Action,
    draw_order,
    pot_shares,
)
from drawmaha_solver.minidrawmaha.strategy import StrategyInfo


class Uniform(Mapping):
    """Every infoset plays its legal actions uniformly."""

    def __getitem__(self, key):
        width = len(key.legal_actions())
        return np.full(width, 1 / width)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class Pot(Uniform):
    """Bets and raises whenever it can; otherwise uniform. Draw nodes stay uniform."""

    def __getitem__(self, key):
        legal = key.legal_actions()
        if Action.POT not in legal:
            return super().__getitem__(key)
        return np.array([1.0 if a is Action.POT else 0.0 for a in legal])


class Drifting(Uniform):
    """Answers differently every time it is asked, so a node read twice disagrees with itself."""

    def __init__(self):
        self.calls = 0

    def __getitem__(self, key):
        self.calls += 1
        legal = key.legal_actions()
        vector = np.zeros(len(legal))
        vector[self.calls % len(legal)] = 1.0
        return vector


# A deck order chosen so P0's draw order is not the sorted order: the board is
# a heart and P0 holds two fours of which one is a heart, so the canonical
# labels, not the physical suit numbers, decide which four is "mid".
DECK_ORDER = parse_cards("4h 2d 4c  6d 3h 5c  2h  3c 5d 6h 2c 3d 4d 5h 6c")

INFO = StrategyInfo(rule="test", column="linear", iteration=1, seed=0, workers=1)

# The eight round-1 spots a player acts at, and the seven lines that reach the draw.
ROUND_ONE_NODES = {"", "x", "p", "xp", "pp", "xpp", "ppp", "xppp"}
DRAW_LINES = {"xx", "xpc", "xppc", "xpppc", "pc", "ppc", "pppc"}


@pytest.fixture(scope="module")
def deal():
    return export_deal(DECK_ORDER, Uniform())


def test_per_mille_sums_to_exactly_a_thousand_by_largest_remainder():
    assert per_mille(np.array([1 / 3, 1 / 3, 1 / 3])) == [334, 333, 333]
    assert per_mille(np.array([0.5, 0.5])) == [500, 500]
    assert per_mille(np.array([0.0004, 0.9996])) == [0, 1000]
    # a tie in the remainders goes to the first entry, every time
    assert per_mille(np.array([0.0005, 0.9995])) == [1, 999]
    # a float32 row that sums to 1 only within 1e-6 is renormalised first
    row = np.array([0.2, 0.3, 0.5], dtype=np.float32) * np.float32(1.000001)
    assert sum(per_mille(row)) == PER_MILLE


def test_the_deck_is_the_deal_order_and_a_permutation_of_fifteen(deal):
    assert deal["deck"] == [card_index(card) for card in DECK_ORDER]
    assert sorted(deal["deck"]) == list(range(len(DECK)))


def test_a_deck_that_is_not_a_permutation_is_refused():
    with pytest.raises(ValueError):
        deal_from_deck(DECK_ORDER[:14] + (DECK_ORDER[0],))


def test_round_one_banks_the_eight_spots_and_the_draw_the_seven_lines(deal):
    assert set(deal["r1"]) == ROUND_ONE_NODES
    assert set(deal["d0"]) == DRAW_LINES
    assert set(deal["d1"]) == DRAW_LINES
    assert set(deal["r2"]) == DRAW_LINES
    for line in DRAW_LINES:
        assert set(deal["d1"][line]) == {"0", "1"}
        assert set(deal["r2"][line]) == {a + b for a in "nlmt" for b in "nlmt"}


def test_every_vector_has_its_spot_s_width_and_sums_to_a_thousand(deal):
    def check(vector, width):
        assert len(vector) == width
        assert sum(vector) == PER_MILLE

    root = deal_from_deck(DECK_ORDER)
    for line, vector in deal["r1"].items():
        state = root
        for letter in line:
            state = state.apply(Action.POT if letter == "p" else Action.CHECK_CALL)
        check(vector, len(state.legal_actions()))
    # the shape that walk implies: fold/call/raise facing a bet, only
    # fold/call facing the all-in, check/bet otherwise
    assert {line: len(v) for line, v in deal["r1"].items()} == {
        "": 2, "x": 2, "p": 3, "xp": 3, "pp": 3, "xpp": 3, "ppp": 2, "xppp": 2,
    }
    for line in DRAW_LINES:
        check(deal["d0"][line], len(DRAW_ACTIONS))
        for vector in deal["d1"][line].values():
            check(vector, len(DRAW_ACTIONS))


def test_round_two_has_no_decisions_after_an_all_in_and_eight_after_a_check_through(deal):
    for combo in deal["r2"]["xx"]:
        assert len(deal["r2"]["xx"][combo]) == 8
        assert len(deal["r2"]["pc"][combo]) == 6
        assert len(deal["r2"]["ppc"][combo]) == 4
        assert deal["r2"]["pppc"][combo] == {}
        assert deal["r2"]["xpppc"][combo] == {}


def test_the_draw_order_is_canonical_not_the_dealt_order(deal):
    root = deal_from_deck(DECK_ORDER)
    for player in (0, 1):
        expected = draw_order(hole=root.holes[player], discarded=(), board=root.board)
        assert deal["order"][player] == [card_index(card) for card in expected]
    # P0 holds 2d 4c 4h against a 2h board. Sorted by physical suit the club
    # four comes first; under the canonical labels the heart four — the one
    # that shares the board's suit — is "mid" and the club four is "top".
    assert [DECK[i] for i in deal["order"][0]] == list(parse_cards("2d 4h 4c"))
    assert [DECK[i] for i in deal["order"][0]] != sorted(root.holes[0])


def test_p1_draw_infoset_sees_how_many_cards_p0_threw_and_not_which():
    """What `d1` keyed by count rests on: the three single throws are one P1 infoset."""
    closed = deal_from_deck(DECK_ORDER).apply(Action.CHECK_CALL).apply(Action.CHECK_CALL)
    after = {}
    for throw in DRAW_ACTIONS:
        state = closed.apply(throw)
        if len(state.holes[0]) < 3:
            state = state.apply_chance((DECK_ORDER[STUB],))
        after[throw] = state.infoset()
    singles = {after[t] for t in (Action.THROW_LOW, Action.THROW_MID, Action.THROW_TOP)}
    assert len(singles) == 1
    assert after[Action.THROW_NONE] not in singles


def test_a_profile_whose_p1_draw_depends_on_the_thrown_card_is_refused():
    with pytest.raises(ValueError, match="which card P0 threw"):
        export_deal(DECK_ORDER, Drifting())


def test_a_showdown_that_changes_between_lines_is_refused(monkeypatch):
    """The export banks each combo's showdown once and checks every later line against
    it. The game cannot make them differ, so the guard is exercised by a showdown that
    answers differently once the first line's sixteen worlds are banked."""
    real = deal_pack.showdown
    calls = []

    def drifting(state):
        calls.append(state)
        shown = real(state)
        return shown if len(calls) <= 16 else {**shown, "inner": (shown["inner"] + 1) % 3}

    monkeypatch.setattr(deal_pack, "showdown", drifting)
    with pytest.raises(ValueError, match="differs between lines"):
        export_deal(DECK_ORDER, Uniform())


@pytest.mark.parametrize("line", sorted(DRAW_LINES))
def test_the_showdown_consumes_the_stub_in_draw_order_and_matches_pot_shares(deal, line):
    """Replayed by hand off the stub from every line that reaches the draw: the
    showdown is the cards' alone, so each line must land on the one banked."""
    deck = DECK_ORDER
    closed = deal_from_deck(deck)
    for letter in line:
        closed = closed.apply(Action.POT if letter == "p" else Action.CHECK_CALL)
    for first in DRAW_ACTIONS:
        for second in DRAW_ACTIONS:
            combo = THROW_CODE[first] + THROW_CODE[second]
            shown = deal["show"][combo]
            state = closed.apply(first)
            stub = STUB
            if len(state.holes[0]) < 3:
                state = state.apply_chance((deck[stub],))
                stub += 1
            state = state.apply(second)
            if len(state.holes[1]) < 3:
                state = state.apply_chance((deck[stub],))
                stub += 1
            state = state.apply_chance((deck[stub],))
            assert shown["holes"] == [[card_index(c) for c in hole] for hole in state.holes]
            assert shown["board"] == [card_index(c) for c in state.board]
            shares = pot_shares(state.holes, state.board)
            won = {0: (0.5, 0.0), 1: (0.0, 0.5), CHOP: (0.25, 0.25)}
            inner, outer = won[shown["inner"]], won[shown["outer"]]
            assert shares == (inner[0] + outer[0], inner[1] + outer[1])
            assert all(isinstance(name, str) and name for pair in shown["cats"] for name in pair)


def test_a_pure_profile_shows_up_as_a_pure_vector(deal):
    pure = export_deal(DECK_ORDER, Pot())
    assert pure["r1"][""] == [0, 1000]
    assert pure["r1"]["xp"] == [0, 0, 1000]
    assert pure["r1"]["xppp"] == [500, 500]  # facing the all-in no raise is legal: uniform
    assert pure["d0"]["xx"] == deal["d0"]["xx"]  # draws untouched


def test_random_decks_are_distinct_permutations_and_seeded():
    decks = random_decks(5, np.random.default_rng(3))
    assert len({tuple(d) for d in decks}) == 5
    for deck in decks:
        assert sorted(deck) == sorted(DECK)
    assert random_decks(5, np.random.default_rng(3)) == decks


def test_the_pack_is_chunked_with_a_manifest_and_stale_chunks_are_removed(tmp_path, deal):
    out = tmp_path / "deals"
    out.mkdir()
    (out / "pack-07.json").write_text("[]")
    manifest = write_pack([deal] * 5, out, chunk=2, seed=9, info=INFO, sha256="ab" * 32)
    assert manifest["chunks"] == ["pack-00.json", "pack-01.json", "pack-02.json"]
    assert manifest["deals"] == 5 and manifest["chunk"] == 2 and manifest["seed"] == 9
    assert manifest["strategy"]["sha256"] == "ab" * 32
    assert manifest["strategy"]["rule"] == "test"
    assert not (out / "pack-07.json").exists()
    assert not list(out.glob("*.tmp"))
    assert json.loads((out / "index.json").read_text()) == manifest
    chunks = [json.loads((out / name).read_text()) for name in manifest["chunks"]]
    assert [len(c) for c in chunks] == [2, 2, 1]
    assert chunks[0][0] == deal


def test_an_export_that_dies_while_staging_leaves_the_previous_pack_whole(tmp_path, deal, monkeypatch):
    out = tmp_path / "deals"
    before = write_pack([deal] * 5, out, chunk=2, seed=9, info=INFO, sha256="ab" * 32)
    (out / "pack-09.json.tmp").write_text("left by an earlier crash")
    real = Path.write_text

    def dies_on_the_manifest(self, text, *args, **kwargs):
        if self.name == "index.json.tmp":
            raise OSError("disk full")
        return real(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", dies_on_the_manifest)
    with pytest.raises(OSError, match="disk full"):
        write_pack([deal], out, chunk=1, seed=10, info=INFO, sha256="cd" * 32)
    assert json.loads((out / "index.json").read_text()) == before
    assert sorted(p.name for p in out.iterdir()) == ["index.json", "pack-00.json", "pack-01.json", "pack-02.json"]
    assert len(json.loads((out / "pack-00.json").read_text())) == 2


def test_a_shorter_export_keeps_the_chunks_it_names_and_removes_the_rest(tmp_path, deal):
    out = tmp_path / "deals"
    write_pack([deal] * 5, out, chunk=2, seed=9, info=INFO, sha256="ab" * 32)
    manifest = write_pack([deal] * 3, out, chunk=2, seed=9, info=INFO, sha256="ab" * 32)
    assert manifest["chunks"] == ["pack-00.json", "pack-01.json"]
    assert sorted(p.name for p in out.iterdir()) == ["index.json", "pack-00.json", "pack-01.json"]
    assert len(json.loads((out / "pack-01.json").read_text())) == 1
