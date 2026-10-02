"""The terminal table: what the human may type, and what they are told.

The bot here plays a stand-in profile (uniform, or a pure one) so nothing
needs the 18 MB release file; the frozen strategy itself is pinned in
`test_strategy.py`.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.cards import CARD_SYMBOL, parse_cards
from drawmaha_solver.minidrawmaha.game import Action, MiniState
from drawmaha_solver.minidrawmaha.play import (
    RELEASE_EXPLOITABILITY,
    RELEASE_SELF_PLAY_VALUE_P0,
    bot_action,
    narrate,
    parse_action,
    play_hand,
    play_session,
    prompt_for,
    report,
    showdown_line,
    thrown_card,
)
from drawmaha_solver.play_session import QuitGame, Scoreboard

GRADES = Path(__file__).parents[2] / "figures" / "rung3" / "grades.json"

class Uniform(Mapping):
    """Every infoset plays its legal actions uniformly."""

    def __getitem__(self, key):
        width = len(key.legal_actions())
        return np.full(width, 1 / width)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

class Always(Uniform):
    """Every infoset plays `action` whenever it is legal, else uniformly."""

    def __init__(self, action):
        self.action = action

    def __getitem__(self, key):
        legal = key.legal_actions()
        if self.action not in legal:
            return super().__getitem__(key)
        return np.array([1.0 if a is self.action else 0.0 for a in legal])

def opened(p0="2c 4d 6h", p1="3c 3d 5h", board="4h") -> MiniState:
    """Round 1, P0 to act, nothing bet."""
    deal = MiniState(holes=(parse_cards(p0), parse_cards(p1)))
    return deal.apply_chance(parse_cards(board))

def at_draw() -> MiniState:
    """Round 1 checked through: P0 to draw."""
    return opened().apply(Action.CHECK_CALL).apply(Action.CHECK_CALL)

def test_fold_is_refused_when_nothing_is_owed():
    state = opened()
    assert parse_action("f", state) is None
    assert parse_action("c", state) is Action.CHECK_CALL
    assert parse_action("bet", state) is Action.POT

def test_facing_a_bet_the_words_change():
    state = opened().apply(Action.POT)
    assert parse_action("f", state) is Action.FOLD
    assert parse_action("call", state) is Action.CHECK_CALL
    assert parse_action("r", state) is Action.POT
    assert parse_action("bet", state) is None
    assert "2 to call" in prompt_for(state)

def test_a_throw_can_be_named_by_position_or_by_card():
    state = at_draw()
    low, mid, top = (thrown_card(state, a) for a in
                     (Action.THROW_LOW, Action.THROW_MID, Action.THROW_TOP))
    assert {low, mid, top} == set(state.holes[0])
    assert parse_action("s", state) is Action.THROW_NONE
    assert parse_action("t", state) is Action.THROW_TOP
    assert parse_action(CARD_SYMBOL[mid], state) is Action.THROW_MID
    assert parse_action("c", state) is None

def test_the_draw_prompt_shows_each_throws_card_and_its_key():
    prompt = prompt_for(at_draw())
    assert "[s]tand pat" in prompt and "throw [t]op" in prompt and "throw [l]ow" in prompt
    for card in at_draw().holes[0]:
        assert CARD_SYMBOL[card] in prompt

@pytest.mark.parametrize("typed", ["q", "quit", " EXIT "])
def test_quitting_raises(typed):
    with pytest.raises(QuitGame):
        parse_action(typed, opened())

def test_the_bot_follows_its_strategy_at_its_own_spot():
    state = opened().apply(Action.CHECK_CALL)  # P1 to act
    rng = np.random.default_rng(0)
    assert {bot_action(state, Always(Action.POT), rng=rng) for _ in range(20)} == {Action.POT}

def test_the_bot_is_deterministic_given_a_seed():
    state = at_draw()
    picks = [
        [bot_action(state, Uniform(), rng=np.random.default_rng(7)) for _ in range(5)]
        for _ in range(2)
    ]
    assert picks[0] == picks[1]

def test_the_bots_draw_is_narrated_without_its_card():
    state = at_draw().apply(Action.THROW_NONE)  # P1 to draw
    line = narrate(state, Action.THROW_LOW, mine=False)
    assert line == "bot draws one"
    assert not any(symbol in line for symbol in CARD_SYMBOL.values())
    assert narrate(at_draw(), Action.THROW_TOP, mine=True) == (
        f"you throw {CARD_SYMBOL[thrown_card(at_draw(), Action.THROW_TOP)]}"
    )

def test_a_bet_is_narrated_with_its_chips_and_the_pot():
    assert narrate(opened(), Action.POT, mine=False) == "bot bets 2   (pot 4)"
    assert narrate(opened(), Action.CHECK_CALL, mine=True) == "you check"

def most_aggressive(state: MiniState) -> Action:
    return state.legal_actions()[-1]

def quitting_after(moves: int):
    """A human who plays `most_aggressive` for `moves` decisions, then types q."""
    made = []

    def ask(state):
        if len(made) == moves:
            raise QuitGame
        made.append(state.current_player)
        return most_aggressive(state)

    return ask

@pytest.mark.parametrize("seed", range(6))
def test_a_whole_hand_plays_out_and_is_scored(seed, capsys):
    board = Scoreboard()
    final = play_hand(
        Uniform(), np.random.default_rng(seed), human_seat=seed % 2, board=board,
        ask=most_aggressive,
    )
    assert final.is_terminal() and board.hands == 1
    assert sum(final.returns()) == pytest.approx(0.0)
    assert board.chips == pytest.approx(final.returns()[seed % 2])
    assert "running" in capsys.readouterr().out

def test_a_showdown_names_both_halves():
    state = at_draw().apply(Action.THROW_NONE).apply(Action.THROW_NONE)
    state = state.apply_chance(parse_cards("5c"))
    state = state.apply(Action.CHECK_CALL).apply(Action.CHECK_CALL)
    line = showdown_line(state, human_seat=0)
    assert "inner:" in line and "outer:" in line and "share" in line

# ---------------------------------------------------------------------------
# A session
# ---------------------------------------------------------------------------

def test_seats_alternate_from_p0_and_a_quit_mid_hand_is_not_scored(capsys):
    board = play_session(Uniform(), np.random.default_rng(3), ask=quitting_after(25))
    seats = [int(line.split("you are P")[1][0])
             for line in capsys.readouterr().out.splitlines() if line.startswith("--- hand")]
    assert seats == [hand % 2 for hand in range(len(seats))]
    assert board.hands == len(seats) - 1, "the hand the human quit in is abandoned"
    assert board.seat_hands == [(board.hands + 1) // 2, board.hands // 2]

def test_a_seeded_session_replays_exactly(capsys):
    def session():
        board = play_session(Uniform(), np.random.default_rng(11), ask=quitting_after(30))
        return board.seat_chips, capsys.readouterr().out

    assert session() == session()

# ---------------------------------------------------------------------------
# The closing report
# ---------------------------------------------------------------------------

def one_hand_each_seat() -> Scoreboard:
    board = Scoreboard()
    board.record(human_seat=0, returns=(-1.0, 1.0))
    board.record(human_seat=1, returns=(-1.0, 1.0))
    return board

def test_the_release_numbers_the_report_quotes_are_the_graded_ones():
    graded = json.loads(GRADES.read_text())["lcfr/002000000/linear"]
    assert RELEASE_SELF_PLAY_VALUE_P0 == pytest.approx(graded["value_p0"], abs=1e-6)
    assert RELEASE_EXPLOITABILITY == pytest.approx(graded["exploitability"], abs=1e-6)

def test_against_the_release_each_seat_is_read_against_its_self_play_value(capsys):
    report(one_hand_each_seat(), release=True)
    out = capsys.readouterr().out
    assert "as P0: -1 over 1 hands (-1.000 per hand; this seat is worth -0.091" in out
    assert "as P1: +1 over 1 hands (+1.000 per hand; this seat is worth +0.091" in out
    assert "+0.061" in out

def test_another_strategy_is_not_read_against_the_release_numbers(capsys):
    # `--strategy PATH` can be any frozen file; quoting the release's seat
    # values and exploitability beside it would state numbers nobody measured.
    report(one_hand_each_seat(), release=False)
    out = capsys.readouterr().out
    assert "as P0: -1 over 1 hands (-1.000 per hand)" in out
    assert "0.091" not in out and "0.061" not in out
