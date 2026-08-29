import numpy as np
import pytest

from drawmaha_solver.rps.game import Action
from drawmaha_solver.rps.players import (
    FixedStrategyPlayer,
    HumanPlayer,
    QuitGame,
    RegretMatchingPlayer,
    play_match,
)

def test_fixed_strategy_player_plays_its_pure_strategy():
    rng = np.random.default_rng(0)
    player = FixedStrategyPlayer([0.0, 1.0, 0.0])
    assert all(player.act(rng) == Action.PAPER for _ in range(50))

@pytest.mark.parametrize("bad", [[0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 1.0, 0.5]])
def test_fixed_strategy_player_rejects_non_distributions(bad):
    with pytest.raises(ValueError):
        FixedStrategyPlayer(bad)

def test_play_match_returns_one_payoff_per_round():
    rng = np.random.default_rng(0)
    payoffs = play_match(RegretMatchingPlayer(), RegretMatchingPlayer(), 100, rng)
    assert payoffs.shape == (100,)
    assert set(np.unique(payoffs)) <= {-1.0, 0.0, 1.0}

@pytest.mark.parametrize(
    "typed,expected",
    [("r", Action.ROCK), ("ROCK", Action.ROCK), ("paper", Action.PAPER), ("s", Action.SCISSORS)],
)
def test_human_player_parses_input(monkeypatch, typed, expected):
    monkeypatch.setattr("builtins.input", lambda _: typed)
    assert HumanPlayer().act(np.random.default_rng(0)) == expected

def test_human_player_reprompts_on_garbage_then_accepts(monkeypatch, capsys):
    answers = iter(["banana", "p"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert HumanPlayer().act(np.random.default_rng(0)) == Action.PAPER
    assert "didn't understand" in capsys.readouterr().out

def test_human_player_quits(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "q")
    with pytest.raises(QuitGame):
        HumanPlayer().act(np.random.default_rng(0))
