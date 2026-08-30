# Rung 0 — RPS Regret Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rung 0 of the validation ladder: regret matching on rock-paper-scissors, verified against the known uniform Nash equilibrium, with convergence figures and a human-vs-bot CLI.

**Architecture:** A `drawmaha_solver.rps` package with four layers, each swappable/testable alone: pure game rules (`game.py`), the generic regret-matching ledger (`regret_matching.py`, knows nothing about RPS), players + match runner (`players.py`, the only glue), and two entry points (`analysis.py` figures, `play.py` CLI). The ledger uses the **expected-utility update** — regret += u − ⟨σ, u⟩ — which is exactly the counterfactual-regret form CFR uses at every infoset in rung 1, and needs no "which action was sampled" argument.

**Tech Stack:** Python 3.12, NumPy, matplotlib (Agg), uv, pytest. No other dependencies.

**Spec:** `README.md` (game + ladder). Success criteria from the kickoff goal: every core behavior test-first; self-play average strategy within 0.02 of uniform and exploitability < 0.02 chips/round at 50k iterations; vs a biased opponent the average converges to the best response; figures regenerate from one command; merge-ready PR superseding #1.

## Global Constraints

- Python `>=3.12`; runtime deps exactly `numpy>=2.0`, `matplotlib>=3.9`; dev dep `pytest>=8.0`; managed by uv (`uv sync`, `uv run pytest`).
- Commit subjects match `^(feat|fix|docs|chore|perf|refactor)(\([a-z0-9_-]+\))?: .+` — imperative, lowercase after colon, no trailing period. No Claude co-author attribution.
- Figures: light-mode reference palette only — series `#2a78d6` (rock), `#eb6834` (paper), `#1baf7a` (scissors); ink `#0b0b0b`, secondary `#52514e`, muted `#898781`, grid `#e1e0d9`, baseline `#c3c2b7`, surface `#fcfcfb`. Titles must state the finding, never a placeholder. Never a dual axis.
- Test files run with `uv run pytest tests/rps/<file> -q` from the repo root.

---

### Task 1: Project scaffold + game rules

**Files:**
- Create: `pyproject.toml`, `src/drawmaha_solver/__init__.py`, `src/drawmaha_solver/rps/__init__.py`, `src/drawmaha_solver/rps/game.py`
- Test: `tests/rps/test_game.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Action(IntEnum)` with `ROCK=0, PAPER=1, SCISSORS=2`; `N_ACTIONS: int`; `PAYOFF: np.ndarray` (3×3 float, row player's payoff); `payoff(a: Action, b: Action) -> float`; `winner(a: Action, b: Action) -> int | None`; `action_values(strategy: np.ndarray) -> np.ndarray`; `best_response_value(strategy: np.ndarray) -> float`.

- [ ] **Step 1: Write `pyproject.toml` (scaffolding folded into this task)**

```toml
[project]
name = "drawmaha-solver"
version = "0.1.0"
description = "Heads-up pot-limit Drawmaha solver (Deep CFR) with a GTOWizard-style dashboard"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "matplotlib>=3.9",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[project.scripts]
rps-analysis = "drawmaha_solver.rps.analysis:main"
rps-play = "drawmaha_solver.rps.play:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/drawmaha_solver"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Also create empty `src/drawmaha_solver/__init__.py` and `src/drawmaha_solver/rps/__init__.py`, then run `uv sync`.

- [ ] **Step 2: Write the failing tests** — `tests/rps/test_game.py`:

```python
import numpy as np
import pytest

from drawmaha_solver.rps.game import (
    PAYOFF,
    Action,
    action_values,
    best_response_value,
    payoff,
    winner,
)

R, P, S = Action.ROCK, Action.PAPER, Action.SCISSORS

# (p0 action, p1 action, expected winner, expected p0 payoff)
OUTCOMES = [
    (R, R, None, 0.0),
    (R, P, 1, -1.0),
    (R, S, 0, 1.0),
    (P, R, 0, 1.0),
    (P, P, None, 0.0),
    (P, S, 1, -1.0),
    (S, R, 1, -1.0),
    (S, P, 0, 1.0),
    (S, S, None, 0.0),
]

@pytest.mark.parametrize("a,b,expected_winner,expected_payoff", OUTCOMES)
def test_all_nine_outcomes(a, b, expected_winner, expected_payoff):
    assert winner(a, b) == expected_winner
    assert payoff(a, b) == expected_payoff

def test_payoff_matrix_is_zero_sum():
    assert np.array_equal(PAYOFF, -PAYOFF.T)

def test_action_values_against_pure_rock():
    values = action_values(np.array([1.0, 0.0, 0.0]))
    assert values[R] == 0.0
    assert values[P] == 1.0
    assert values[S] == -1.0

def test_uniform_strategy_is_unexploitable():
    assert best_response_value(np.full(3, 1 / 3)) == pytest.approx(0.0)

def test_pure_strategy_is_fully_exploitable():
    assert best_response_value(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)

def test_slightly_biased_strategy_is_slightly_exploitable():
    # 40/30/30 rock-heavy: the counter (paper) earns 0.4*1 + 0.3*0 + 0.3*(-1) = 0.1
    assert best_response_value(np.array([0.4, 0.3, 0.3])) == pytest.approx(0.1)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/rps/test_game.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'drawmaha_solver.rps.game'`

- [ ] **Step 4: Write the implementation** — `src/drawmaha_solver/rps/game.py`:

```python
"""Rock-paper-scissors rules: actions, payoffs, winner determination.

Rung 0 of the validation ladder. RPS is the smallest game with a
mixed-strategy Nash equilibrium — uniform (1/3, 1/3, 1/3) — which gives the
regret-matching ledger a known exact answer to converge to before any game
tree exists.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

class Action(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

N_ACTIONS = len(Action)

# PAYOFF[a, b] = row player's chips when row plays a and column plays b:
# +1 win, 0 tie, -1 loss. Zero-sum and symmetric: PAYOFF == -PAYOFF.T.
PAYOFF = np.array(
    [
        [0.0, -1.0, 1.0],  # rock:     ties rock, loses to paper, beats scissors
        [1.0, 0.0, -1.0],  # paper:    beats rock, ties paper, loses to scissors
        [-1.0, 1.0, 0.0],  # scissors: loses to rock, beats paper, ties scissors
    ]
)

def payoff(a: Action, b: Action) -> float:
    """Player a's payoff when a meets b."""
    return float(PAYOFF[a, b])

def winner(a: Action, b: Action) -> int | None:
    """0 if action `a` wins, 1 if action `b` wins, None on a tie."""
    p = PAYOFF[a, b]
    if p > 0:
        return 0
    if p < 0:
        return 1
    return None

def action_values(strategy: np.ndarray) -> np.ndarray:
    """Expected payoff of each pure action against a mixed strategy."""
    return PAYOFF @ np.asarray(strategy, dtype=np.float64)

def best_response_value(strategy: np.ndarray) -> float:
    """One-shot exploitability: what a best-responding adversary earns per
    round against `strategy`.

    Exactly 0 at the Nash equilibrium (uniform), positive everywhere else.
    Because RPS is symmetric zero-sum, the adversary's payoff matrix is the
    same PAYOFF, so this is just the best entry of action_values.
    """
    return float(np.max(action_values(strategy)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/rps/test_game.py -q`
Expected: 14 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/drawmaha_solver tests/rps/test_game.py
git commit -m "feat(rps): scaffold package and add RPS rules with winner and exploitability"
```

---

### Task 2: Regret-matching ledger

**Files:**
- Create: `src/drawmaha_solver/rps/regret_matching.py`
- Test: `tests/rps/test_regret_matching.py` (unit-math tests only; convergence tests arrive in Task 3)

**Interfaces:**
- Consumes: nothing from Task 1 (the ledger is game-agnostic).
- Produces: `RegretMatcher(n_actions: int)` with attributes `cumulative_regret: np.ndarray`, `strategy_sum: np.ndarray` and methods `strategy() -> np.ndarray`, `average_strategy() -> np.ndarray`, `update(utilities: np.ndarray) -> None`. **Note the update signature: it takes only the utility vector** — regret is measured against the current strategy's expected utility ⟨σ, u⟩, not against a sampled action.

- [ ] **Step 1: Write the failing tests** — `tests/rps/test_regret_matching.py`:

```python
import numpy as np

from drawmaha_solver.rps.regret_matching import RegretMatcher

UNIFORM = np.full(3, 1 / 3)

def test_initial_strategy_is_uniform():
    assert np.allclose(RegretMatcher(3).strategy(), UNIFORM)
    assert np.allclose(RegretMatcher(3).average_strategy(), UNIFORM)

def test_all_negative_regret_falls_back_to_uniform():
    m = RegretMatcher(3)
    m.cumulative_regret = np.array([-1.0, -2.0, -0.5])
    assert np.allclose(m.strategy(), UNIFORM)

def test_strategy_proportional_to_positive_regret_only():
    m = RegretMatcher(3)
    m.cumulative_regret = np.array([3.0, 1.0, -5.0])
    assert np.allclose(m.strategy(), [0.75, 0.25, 0.0])

def test_update_accumulates_regret_against_expected_utility():
    m = RegretMatcher(3)
    # Opponent showed paper: utilities (rock -1, paper 0, scissors +1).
    # Current strategy is uniform, so expected utility = 0 and the regret
    # increment is the utility vector itself.
    m.update(np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(m.cumulative_regret, [-1.0, 0.0, 1.0])
    # The strategy in effect for the round was banked into the average.
    assert np.allclose(m.strategy_sum, UNIFORM)

def test_update_measures_regret_relative_to_current_strategy():
    m = RegretMatcher(3)
    m.cumulative_regret = np.array([1.0, 1.0, 0.0])  # current strategy (.5, .5, 0)
    # Utilities (2, 0, 1): expected utility = .5*2 + .5*0 = 1,
    # so increments are (2-1, 0-1, 1-1) = (1, -1, 0).
    m.update(np.array([2.0, 0.0, 1.0]))
    assert np.allclose(m.cumulative_regret, [2.0, 0.0, 0.0])
    assert np.allclose(m.strategy_sum, [0.5, 0.5, 0.0])

def test_average_strategy_is_mean_of_played_strategies():
    m = RegretMatcher(3)
    m.strategy_sum = np.array([1.0, 3.0, 0.0])
    assert np.allclose(m.average_strategy(), [0.25, 0.75, 0.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rps/test_regret_matching.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'drawmaha_solver.rps.regret_matching'`

- [ ] **Step 3: Write the implementation** — `src/drawmaha_solver/rps/regret_matching.py`:

```python
"""Regret matching (Hart & Mas-Colell 2000): the ledger at the heart of CFR.

The entire algorithm:

1. After each round, for every action a, record regret(a) = u(a) − ⟨σ, u⟩:
   how much better always-a would have done than the mixed strategy σ we
   actually played, against the opponent's revealed action.
2. Next round, play each action with probability proportional to its
   accumulated POSITIVE regret (uniform when nothing is positive).
3. The running AVERAGE of the strategies played converges to equilibrium —
   for two-player zero-sum games, a Nash equilibrium. The current strategy
   never converges (it cycles forever); only the average does.

CFR (rung 1) is this same ledger run at every information set of a game
tree at once — and the u − ⟨σ, u⟩ form here is exactly its counterfactual
regret, so nothing about the update rule changes when the game grows a tree.
"""

from __future__ import annotations

import numpy as np

class RegretMatcher:
    """One regret-matching ledger over a fixed set of actions."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.cumulative_regret = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions)

    def strategy(self) -> np.ndarray:
        """Current strategy: positive regrets normalized; uniform fallback."""
        positive = np.maximum(self.cumulative_regret, 0.0)
        total = positive.sum()
        if total <= 0.0:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return positive / total

    def average_strategy(self) -> np.ndarray:
        """Mean of all strategies played so far — the thing that converges."""
        total = self.strategy_sum.sum()
        if total <= 0.0:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return self.strategy_sum / total

    def update(self, utilities: np.ndarray) -> None:
        """Record one finished round.

        `utilities[a]` is the payoff action a would have earned against what
        the opponent actually did. Regret is measured against the current
        strategy's expected utility, and that same strategy is banked into
        the average — so call this exactly once per round, after acting.
        """
        current = self.strategy()
        self.strategy_sum += current
        self.cumulative_regret += utilities - current @ utilities
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rps/test_regret_matching.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/rps/regret_matching.py tests/rps/test_regret_matching.py
git commit -m "feat(rps): add regret-matching ledger with expected-utility update"
```

---

### Task 3: Players, match runner, and convergence-to-Nash tests

**Files:**
- Create: `src/drawmaha_solver/rps/players.py`
- Test: `tests/rps/test_players.py`, and append convergence tests to `tests/rps/test_regret_matching.py`

**Interfaces:**
- Consumes: Task 1's `Action`, `N_ACTIONS`, `PAYOFF`; Task 2's `RegretMatcher` (with `update(utilities)`).
- Produces: `QuitGame(Exception)`; abstract `Player` with `act(rng: np.random.Generator) -> Action` and `observe(own: Action, opp: Action) -> None` (default no-op); `FixedStrategyPlayer(strategy)` (raises `ValueError` on a non-distribution); `RegretMatchingPlayer()` with attribute `learner: RegretMatcher`; `HumanPlayer()` parsing r/p/s/full words, `q` raises `QuitGame`; `play_match(p0, p1, n_rounds, rng) -> np.ndarray` of p0's per-round payoffs.

- [ ] **Step 1: Write the failing tests** — `tests/rps/test_players.py`:

```python
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
```

Append to `tests/rps/test_regret_matching.py` (these need `play_match`, so they land in this task, not Task 2 — add the imports shown):

```python
from drawmaha_solver.rps.game import best_response_value
from drawmaha_solver.rps.players import FixedStrategyPlayer, RegretMatchingPlayer, play_match

def test_self_play_average_strategy_converges_to_nash():
    rng = np.random.default_rng(0)
    p0, p1 = RegretMatchingPlayer(), RegretMatchingPlayer()
    play_match(p0, p1, 50_000, rng)
    for player in (p0, p1):
        avg = player.learner.average_strategy()
        assert np.allclose(avg, UNIFORM, atol=0.02), avg
        assert best_response_value(avg) < 0.02

def test_vs_biased_opponent_average_converges_to_best_response():
    # Against 50/25/25 rock-heavy, the unique best response is pure paper
    # (worth +0.25/round).
    rng = np.random.default_rng(0)
    learner = RegretMatchingPlayer()
    payoffs = play_match(learner, FixedStrategyPlayer([0.5, 0.25, 0.25]), 20_000, rng)
    avg = learner.learner.average_strategy()
    assert avg[1] > 0.9, avg
    assert payoffs.mean() > 0.1

def test_match_is_deterministic_given_seed():
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        runs.append(play_match(RegretMatchingPlayer(), RegretMatchingPlayer(), 1_000, rng))
    assert np.array_equal(runs[0], runs[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rps/test_players.py tests/rps/test_regret_matching.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'drawmaha_solver.rps.players'`

- [ ] **Step 3: Write the implementation** — `src/drawmaha_solver/rps/players.py`:

```python
"""Player interface, concrete players, and the match runner."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from drawmaha_solver.rps.game import N_ACTIONS, PAYOFF, Action
from drawmaha_solver.rps.regret_matching import RegretMatcher

class QuitGame(Exception):
    """Raised by a human player who wants to stop."""

class Player(ABC):
    @abstractmethod
    def act(self, rng: np.random.Generator) -> Action:
        """Choose an action for the next round."""

    def observe(self, own: Action, opp: Action) -> None:
        """See the finished round (default: ignore it)."""

class FixedStrategyPlayer(Player):
    """Plays a fixed mixed strategy forever."""

    def __init__(self, strategy):
        strategy = np.asarray(strategy, dtype=np.float64)
        if strategy.shape != (N_ACTIONS,) or np.any(strategy < 0) or not np.isclose(strategy.sum(), 1.0):
            raise ValueError(f"not a distribution over {N_ACTIONS} actions: {strategy}")
        self.strategy = strategy

    def act(self, rng: np.random.Generator) -> Action:
        return Action(rng.choice(N_ACTIONS, p=self.strategy))

class RegretMatchingPlayer(Player):
    """Samples from the regret matcher's current strategy and feeds every
    finished round back into the ledger."""

    def __init__(self):
        self.learner = RegretMatcher(N_ACTIONS)

    def act(self, rng: np.random.Generator) -> Action:
        return Action(rng.choice(N_ACTIONS, p=self.learner.strategy()))

    def observe(self, own: Action, opp: Action) -> None:
        # PAYOFF[:, opp] = what each of my actions pays against the opponent's
        # revealed action — computable because RPS reveals both moves.
        self.learner.update(PAYOFF[:, opp])

class HumanPlayer(Player):
    """Reads r/p/s (or full words) from stdin; q raises QuitGame."""

    PROMPT = "[r]ock / [p]aper / [s]cissors / [q]uit > "
    PARSE = {
        "r": Action.ROCK,
        "rock": Action.ROCK,
        "p": Action.PAPER,
        "paper": Action.PAPER,
        "s": Action.SCISSORS,
        "scissors": Action.SCISSORS,
    }

    def act(self, rng: np.random.Generator) -> Action:
        while True:
            raw = input(self.PROMPT).strip().lower()
            if raw in ("q", "quit", "exit"):
                raise QuitGame
            if raw in self.PARSE:
                return self.PARSE[raw]
            print(f"  didn't understand {raw!r}")

def play_match(p0: Player, p1: Player, n_rounds: int, rng: np.random.Generator) -> np.ndarray:
    """Run n_rounds; both players observe each round. Returns p0's payoffs."""
    payoffs = np.empty(n_rounds)
    for i in range(n_rounds):
        a0 = p0.act(rng)
        a1 = p1.act(rng)
        p0.observe(a0, a1)
        p1.observe(a1, a0)
        payoffs[i] = PAYOFF[a0, a1]
    return payoffs
```

- [ ] **Step 4: Run the full suite to verify it passes**

Run: `uv run pytest -q`
Expected: all tests pass (game 14 + ledger 6 + players 9 + convergence 3 = 32)

- [ ] **Step 5: Commit**

```bash
git add src/drawmaha_solver/rps/players.py tests/rps/test_players.py tests/rps/test_regret_matching.py
git commit -m "feat(rps): add players and match runner; verify convergence to nash"
```

---

### Task 4: Convergence analysis + figures  *(parallel with Task 5 — do NOT run git commands; the coordinator commits)*

**Files:**
- Create: `src/drawmaha_solver/rps/analysis.py`
- Test: `tests/rps/test_analysis.py`

**Interfaces:**
- Consumes: Task 1's `PAYOFF`, `Action`, `best_response_value`; Task 2's `RegretMatcher` (with `update(utilities)`).
- Produces: `run_self_play(n_iters, seed) -> dict` with keys `current0`, `average0` (each `(n_iters, 3)`), `exploitability0` (`(n_iters,)`), `final_average1`; `run_vs_fixed(n_iters, opponent, seed) -> dict` with keys `average`, `payoffs`; `main()` (argparse: `--iters` default 100_000, `--seed` default 7, `--out` default `figures/rung0`) writing four PNGs and a stdout summary.

- [ ] **Step 1: Write the failing tests** — `tests/rps/test_analysis.py`:

```python
import numpy as np

from drawmaha_solver.rps.analysis import run_self_play, run_vs_fixed

def test_run_self_play_shapes_and_convergence():
    traj = run_self_play(5_000, seed=0)
    assert traj["current0"].shape == (5_000, 3)
    assert traj["average0"].shape == (5_000, 3)
    assert traj["exploitability0"].shape == (5_000,)
    # rows are distributions
    assert np.allclose(traj["average0"].sum(axis=1), 1.0)
    # late average is closer to Nash than early average
    assert traj["exploitability0"][-1] < traj["exploitability0"][100]

def test_run_vs_fixed_learns_the_counter():
    traj = run_vs_fixed(5_000, np.array([0.5, 0.25, 0.25]), seed=0)
    assert traj["average"].shape == (5_000, 3)
    assert traj["payoffs"].shape == (5_000,)
    assert traj["average"][-1, 1] > 0.8  # paper

def test_main_writes_four_figures(tmp_path, monkeypatch):
    import sys

    from drawmaha_solver.rps import analysis

    monkeypatch.setattr(
        sys, "argv", ["rps-analysis", "--iters", "2000", "--out", str(tmp_path)]
    )
    analysis.main()
    written = {p.name for p in tmp_path.glob("*.png")}
    assert written == {
        "self_play_average_strategy.png",
        "self_play_current_vs_average.png",
        "self_play_exploitability.png",
        "vs_biased_average_strategy.png",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rps/test_analysis.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'drawmaha_solver.rps.analysis'`

- [ ] **Step 3: Write the implementation** — `src/drawmaha_solver/rps/analysis.py`:

```python
"""Convergence analysis for rung 0: does the ledger find the Nash equilibrium?

Runs two experiments and renders the figures that certify rung 0:

1. Self-play (learner vs learner): the average strategy must converge to the
   uniform Nash (1/3, 1/3, 1/3) and its exploitability must fall like 1/sqrt(T),
   while the current strategy visibly cycles forever.
2. Learner vs a biased fixed opponent: the average strategy must converge to
   the best response (all paper against a rock-heavy opponent).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drawmaha_solver.rps.game import PAYOFF, Action, best_response_value
from drawmaha_solver.rps.regret_matching import RegretMatcher

# Reference dataviz palette, light mode (first three categorical slots are
# validated all-pairs for exactly three series).
SERIES = {"rock": "#2a78d6", "paper": "#eb6834", "scissors": "#1baf7a"}
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

def run_self_play(n_iters: int, seed: int) -> dict[str, np.ndarray]:
    """Two regret matchers play each other; returns per-iteration trajectories."""
    rng = np.random.default_rng(seed)
    p0, p1 = RegretMatcher(3), RegretMatcher(3)
    current0 = np.empty((n_iters, 3))
    average0 = np.empty((n_iters, 3))
    exploitability0 = np.empty(n_iters)
    for i in range(n_iters):
        s0, s1 = p0.strategy(), p1.strategy()
        a0 = rng.choice(3, p=s0)
        a1 = rng.choice(3, p=s1)
        p0.update(PAYOFF[:, a1])
        p1.update(PAYOFF[:, a0])
        current0[i] = s0
        average0[i] = p0.average_strategy()
        exploitability0[i] = best_response_value(average0[i])
    return {
        "current0": current0,
        "average0": average0,
        "exploitability0": exploitability0,
        "final_average1": p1.average_strategy(),
    }

def run_vs_fixed(n_iters: int, opponent: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Learner vs a fixed mixed strategy; returns trajectories and payoffs."""
    rng = np.random.default_rng(seed)
    learner = RegretMatcher(3)
    average = np.empty((n_iters, 3))
    payoffs = np.empty(n_iters)
    for i in range(n_iters):
        a0 = rng.choice(3, p=learner.strategy())
        a1 = rng.choice(3, p=opponent)
        learner.update(PAYOFF[:, a1])
        average[i] = learner.average_strategy()
        payoffs[i] = PAYOFF[a0, a1]
    return {"average": average, "payoffs": payoffs}

def _log_indices(n: int, k: int = 500) -> np.ndarray:
    return np.unique(np.geomspace(1, n, k).astype(int)) - 1

def _new_axes(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)
    return fig, ax

def _legend(ax: plt.Axes) -> None:
    leg = ax.legend(frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(SECONDARY)

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.resolve()}")

def fig_average_strategy(traj: dict, out: Path) -> None:
    idx = _log_indices(len(traj["average0"]))
    fig, ax = _new_axes("Self-play average strategy converges to the uniform Nash (1/3, 1/3, 1/3)")
    ax.axhline(1 / 3, color=MUTED, linewidth=1, linestyle="--")
    ax.text(1.2, 1 / 3 + 0.012, "Nash 1/3", color=MUTED, fontsize=8.5)
    for action in Action:
        name = action.name.lower()
        ax.plot(idx + 1, traj["average0"][idx, action], color=SERIES[name], linewidth=1.8, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("average P(action)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 0.8)
    _legend(ax)
    _save(fig, out)

def fig_current_vs_average(traj: dict, out: Path, window: int = 5000) -> None:
    fig, ax = _new_axes("The current strategy cycles forever — only the running average converges")
    window = min(window, len(traj["current0"]))
    t = np.arange(1, window + 1)
    ax.plot(t, traj["current0"][:window, Action.ROCK], color=SERIES["rock"], linewidth=0.7, alpha=0.7, label="current P(rock)")
    ax.plot(t, traj["average0"][:window, Action.ROCK], color=SERIES["paper"], linewidth=2.2, label="average P(rock)")
    ax.axhline(1 / 3, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("P(rock)", color=MUTED, fontsize=9)
    _legend(ax)
    _save(fig, out)

def fig_exploitability(traj: dict, out: Path) -> None:
    expl = traj["exploitability0"]
    # The first iterations' averages are exactly/near uniform, which reads as
    # ~0 exploitability and spikes off the bottom of a log axis — skip them.
    idx = _log_indices(len(expl))
    idx = idx[idx >= 9]
    fig, ax = _new_axes("Exploitability of the average strategy falls like 1/sqrt(T)")
    anchor = 100
    c = expl[anchor - 1] * np.sqrt(anchor)
    ax.plot(idx + 1, c / np.sqrt(idx + 1), color=MUTED, linewidth=1, linestyle="--", label="c/sqrt(T) reference")
    ax.plot(idx + 1, expl[idx], color=SERIES["rock"], linewidth=1.8, label="best-response value vs average strategy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("exploitability (chips/round)", color=MUTED, fontsize=9)
    _legend(ax)
    _save(fig, out)

def fig_vs_biased(traj: dict, opponent: np.ndarray, out: Path) -> None:
    idx = _log_indices(len(traj["average"]))
    opp_desc = ", ".join(f"{int(round(p * 100))}% {a.name.lower()}" for a, p in zip(Action, opponent))
    fig, ax = _new_axes(f"Against a biased opponent ({opp_desc}), the average converges to the counter: paper")
    for action in Action:
        name = action.name.lower()
        ax.plot(idx + 1, traj["average"][idx, action], color=SERIES[name], linewidth=1.8, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("average P(action)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1.05)
    _legend(ax)
    _save(fig, out)

def main() -> None:
    parser = argparse.ArgumentParser(description="Rung-0 convergence analysis")
    parser.add_argument("--iters", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("figures/rung0"))
    args = parser.parse_args()

    self_play = run_self_play(args.iters, args.seed)
    opponent = np.array([0.5, 0.25, 0.25])
    biased = run_vs_fixed(min(args.iters, 20_000), opponent, args.seed)

    fig_average_strategy(self_play, args.out / "self_play_average_strategy.png")
    fig_current_vs_average(self_play, args.out / "self_play_current_vs_average.png")
    fig_exploitability(self_play, args.out / "self_play_exploitability.png")
    fig_vs_biased(biased, opponent, args.out / "vs_biased_average_strategy.png")

    final_avg = self_play["average0"][-1]
    print(f"self-play {args.iters:,} iters:")
    print(f"  p0 average strategy  {np.array2string(final_avg, precision=4)}")
    print(f"  p1 average strategy  {np.array2string(self_play['final_average1'], precision=4)}")
    print(f"  exploitability of p0 average  {self_play['exploitability0'][-1]:.5f} chips/round")
    print(f"vs fixed {np.array2string(opponent, precision=2)} over {len(biased['payoffs']):,} iters:")
    print(f"  average strategy  {np.array2string(biased['average'][-1], precision=4)}")
    print(f"  mean payoff {biased['payoffs'].mean():+.4f} chips/round (best response earns +0.25)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rps/test_analysis.py -q`
Expected: 3 passed

- [ ] **Step 5: Generate the real figures and report the numbers**

Run: `uv run rps-analysis`
Expected: four PNGs under `figures/rung0/`, self-play average within 0.01 of uniform, exploitability < 0.01, vs-biased paper > 0.99. Include the printed summary in your report.

*(No commit — the coordinator reviews the figures visually, then commits.)*

---

### Task 5: Human-vs-bot CLI  *(parallel with Task 4 — do NOT run git commands; the coordinator commits)*

**Files:**
- Create: `src/drawmaha_solver/rps/play.py`
- Test: `tests/rps/test_play.py`

**Interfaces:**
- Consumes: Task 1's `Action`, `payoff`, `winner`; Task 3's `HumanPlayer`, `QuitGame`, `RegretMatchingPlayer`.
- Produces: `main()` — interactive loop, exits cleanly on `q`, prints running score and, at the end, the bot's average strategy.

- [ ] **Step 1: Write the failing tests** — `tests/rps/test_play.py`:

```python
from drawmaha_solver.rps import play

def test_main_plays_rounds_and_quits_cleanly(monkeypatch, capsys):
    answers = iter(["r", "p", "s", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    play.main()  # must not raise
    out = capsys.readouterr().out
    assert "3 rounds" in out
    assert "average strategy" in out.lower()

def test_main_immediate_quit_prints_no_summary(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "q")
    play.main()
    assert "rounds" not in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rps/test_play.py -q`
Expected: FAIL — `ImportError: cannot import name 'play'` (module missing)

- [ ] **Step 3: Write the implementation** — `src/drawmaha_solver/rps/play.py`:

```python
"""Play rock-paper-scissors against the regret-matching learner in the terminal."""

from __future__ import annotations

import numpy as np

from drawmaha_solver.rps.game import Action, payoff, winner
from drawmaha_solver.rps.players import HumanPlayer, QuitGame, RegretMatchingPlayer

def main() -> None:
    print("Rock-paper-scissors vs the regret-matching ledger.")
    print("It starts uniform and learns from every round you play. q to quit.\n")
    rng = np.random.default_rng()
    human = HumanPlayer()
    bot = RegretMatchingPlayer()
    score = {0: 0, 1: 0, None: 0}
    chips = 0.0
    rounds = 0

    while True:
        try:
            a_h = human.act(rng)
        except QuitGame:
            break
        a_b = bot.act(rng)
        human.observe(a_h, a_b)
        bot.observe(a_b, a_h)
        w = winner(a_h, a_b)
        chips += payoff(a_h, a_b)
        score[w] += 1
        rounds += 1
        verdict = {0: "you win", 1: "bot wins", None: "tie"}[w]
        print(
            f"  you: {a_h.name.lower():<8} bot: {a_b.name.lower():<8} -> {verdict}   "
            f"(you {score[0]} – {score[1]} bot, {score[None]} ties)"
        )

    if rounds:
        avg = bot.learner.average_strategy()
        print(f"\n{rounds} rounds. You net {chips:+.0f} chips.")
        print(
            "Bot's average strategy so far: "
            + ", ".join(f"{a.name.lower()} {avg[a]:.0%}" for a in Action)
        )
        print("(Against a non-uniform human it drifts toward the counter of your habits.)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rps/test_play.py -q`
Expected: 2 passed

*(No commit — the coordinator commits after review.)*

---

### Task 6: README, figure review, and PR (coordinator, not a subagent)

**Files:**
- Modify: `README.md` (add the evaluation-protocol section, rung-0 usage/results section with two embedded figures, and update Status)
- Add: `figures/rung0/*.png` (generated in Task 4)

Steps: visually inspect all four PNGs; commit Task 4 (`feat(rps): add convergence analysis with figures`) and Task 5 (`feat(rps): add human-vs-bot play cli`) separately; update README with the measured numbers; commit (`docs: add evaluation protocol and rung-0 results to readme`); push; open the PR via the open-pr skill; run the merge-readiness gate; close PR #1 with a comment pointing at the new PR.

---

## Self-review notes

- Spec coverage: game framework ✓ (Task 1), player inputs ✓ (Task 3 HumanPlayer + Task 5 CLI), regret matching ✓ (Task 2), analysis ✓ (Task 4), README/eval protocol ✓ (Task 6). Success criteria all have owning tasks.
- Type consistency: `update(utilities)` single-argument everywhere (Tasks 2, 3, 4); `RegretMatchingPlayer.learner` used by Tasks 4–5 test/impl; `play_match` signature identical in Tasks 3–4.
- Placeholder scan: none — every step carries full file contents or exact commands.
