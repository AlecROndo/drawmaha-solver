"""Play mini-drawmaha against rung 3's solved strategy in the terminal.

Rungs 1 and 2 solved their game before the first hand; this one cannot, because
the strategy took 2,000,000 lockstep iterations on Modal. So the opponent is the
frozen LCFR average (`strategy.py`), downloaded once from the repo's release
and checked against its pinned digest, or read from `--strategy PATH`. It is
exploitable for about 0.061 chips a hand by a perfect adversary — 6 chips per
hundred hands, at an ante of 1 — which no human will collect by feel.

What is new at this table, against rung 2's:

* **The draw.** After round 1 each player may throw one of their three cards
  for a fresh one, face down. The prompt names the cards by the position the
  solver's ledgers use — low, mid, top in canonical order — and shows which
  physical card each one is, so "throw low" is never a guess. The bot's draw
  is reported the way the table sees it: whether it drew, never what.
* **Two pots in one.** Half the pot goes to the best three cards held (inner),
  half to the best two held plus both board cards (outer). The showdown names
  both winners, so a split shows up as the quarter it is.
* **Pot-limit chips.** A bet is the pot and the 26-chip stack caps the raise
  war, so the prompt always shows the pot and what a call costs.

The bot reads only its own infoset, and seats alternate every hand because the
game is not symmetric: in the frozen strategy's self-play P0 is worth about
−0.091 a hand. That number comes from the strategy, not from an exact solve, so
it is a target to read the per-seat lines against, not a law.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from drawmaha_solver.leduc.play import QuitGame, Scoreboard, verdict
from drawmaha_solver.minidrawmaha.cards import CARD_SYMBOL, hand_symbol
from drawmaha_solver.minidrawmaha.game import (
    ANTE,
    STACK,
    Action,
    MiniState,
    action_label,
    chip_state,
    draw_order,
    pot_shares,
    random_deal,
    throw_count,
)
from drawmaha_solver.minidrawmaha.hands import inner_score, outer_score
from drawmaha_solver.minidrawmaha.strategy import fetch_strategy, load_strategy

Profile = Mapping  # InfoSet -> probability row, e.g. a FrozenStrategy

# P0's value per hand when the frozen LCFR strategy plays itself, from
# `exploitability.expected_value` on the release file.
SELF_PLAY_VALUE_P0 = -0.0906

Ask = Callable[[MiniState], Action]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Load the frozen strategy, then play hands against it until quit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", type=Path, help="a frozen strategy .npz")
    parser.add_argument("--seed", type=int, help="seed the deal and the bot")
    args = parser.parse_args(argv)

    print("Mini-drawmaha against rung 3's solved strategy.")
    path = args.strategy or fetch_strategy()
    strategies = load_strategy(path)
    info = strategies.info
    print(
        f"Loaded {info.rule.upper()} after {info.iteration:,} iterations "
        f"({info.workers} hands per seat each), {info.column} average.\n"
    )
    print(f"Both ante {ANTE} from {STACK}-chip stacks. Fifteen cards: ranks 2-6 in")
    print("three suits. Three cards each, a board card, a betting round, one draw")
    print("(throw at most one card, face down), a second board card, a second")
    print("betting round. Half the pot to the best three cards held, half to the")
    print("best two held plus both board cards. Bets are pot-sized. q to quit.\n")

    rng = np.random.default_rng(args.seed)
    board = Scoreboard()
    while True:
        try:
            play_hand(strategies, rng, human_seat=board.hands % 2, board=board)
        except QuitGame:
            break
    report(board)


# ---------------------------------------------------------------------------
# One hand
# ---------------------------------------------------------------------------


def play_hand(
    strategies: Profile,
    rng: np.random.Generator,
    *,
    human_seat: int,
    board: Scoreboard,
    ask: Ask | None = None,
) -> MiniState:
    """Deal, play both rounds and the draw, score, narrate; returns the final state."""
    ask = ask or ask_action
    state = random_deal(rng)
    print(f"--- hand {board.hands + 1}   you are P{human_seat}, "
          f"holding {hand_symbol(state.holes[human_seat])}")

    while not state.is_terminal():
        if state.is_chance_node():
            state = deal_chance(state, rng, human_seat)
            continue
        player = state.current_player
        if player == human_seat:
            action = ask(state)
        else:
            action = bot_action(state, strategies, rng)
        print("  " + narrate(state, action, mine=player == human_seat))
        state = state.apply(action)

    returns = state.returns()
    board.record(human_seat=human_seat, returns=returns)
    # Always reveal, a fold included: whether the bot was bluffing is the lesson.
    print("  " + showdown_line(state, human_seat))
    print(f"  {verdict(returns[human_seat])}   "
          f"(running {board.chips:+.0f} over {board.hands})\n")
    return state


def deal_chance(state: MiniState, rng: np.random.Generator, human_seat: int) -> MiniState:
    """Turn a board card or deal a draw's replacement, at the deck's own odds."""
    outcomes, probabilities = zip(*state.chance_outcomes())
    drawing = [len(hole) < 3 for hole in state.holes]
    after = state.apply_chance(outcomes[rng.choice(len(outcomes), p=probabilities)])
    if drawing[human_seat]:
        new = set(after.holes[human_seat]) - set(state.holes[human_seat])
        print(f"  you draw {hand_symbol(tuple(sorted(new)))}"
              f" -> {hand_symbol(after.holes[human_seat])}")
    elif not any(drawing):
        print(f"  board: {hand_symbol(after.board)}   pot {chip_state(after.betting).pot}")
    return after


def bot_action(state: MiniState, strategies: Profile, rng: np.random.Generator) -> Action:
    """Sample the frozen strategy at the bot's own infoset.

    The sampled index is a POSITION in the spot's legal actions, not an
    `Action` value, exactly as in rung 2.
    """
    spot = state.infoset()
    probabilities = np.asarray(strategies[spot], dtype=np.float64)
    probabilities /= probabilities.sum()  # float32 rows land a hair off 1
    return spot.legal_actions()[rng.choice(len(probabilities), p=probabilities)]


def narrate(state: MiniState, action: Action, *, mine: bool) -> str:
    """One line for one decision, saying only what the table would see of the bot's."""
    who = "you" if mine else "bot"
    if state.is_draw_decision():
        if mine:
            if action is Action.THROW_NONE:
                return "you stand pat"
            return f"you throw {CARD_SYMBOL[thrown_card(state, action)]}"
        return "bot draws one" if throw_count(action) else "bot stands pat"
    line = state.betting[-1]
    before = chip_state(state.betting)
    after = chip_state(state.betting[:-1] + (line + (action,),))
    player = state.current_player
    paid = after.committed[player] - before.committed[player]
    word = action_label(action, line)
    verb = word if mine else {"check": "checks", "call": "calls", "fold": "folds",
                              "bet": "bets", "raise": "raises"}[word]
    if paid:
        return f"{who} {verb} {paid}   (pot {after.pot})"
    return f"{who} {verb}"


def thrown_card(state: MiniState, action: Action):
    """The physical card a one-card throw discards, by its canonical position."""
    player = state.current_player
    ordered = draw_order(
        hole=state.holes[player], discarded=state.discards[player], board=state.board
    )
    return ordered[(Action.THROW_LOW, Action.THROW_MID, Action.THROW_TOP).index(action)]


def showdown_line(state: MiniState, human_seat: int) -> str:
    """Both hands, the board, and who took each half — or just the cards on a fold."""
    bot = 1 - human_seat
    shown = (f"you {hand_symbol(state.holes[human_seat])}   "
             f"bot {hand_symbol(state.holes[bot])}   board {hand_symbol(state.board)}")
    if len(state.board) < 2 or state.betting[-1][-1:] == (Action.FOLD,):
        return shown
    inner = [inner_score(hole) for hole in state.holes]
    outer = [outer_score(hole, state.board) for hole in state.holes]
    halves = []
    for name, scores in (("inner", inner), ("outer", outer)):
        mine, theirs = scores[human_seat], scores[bot]
        who = "you" if mine > theirs else "bot" if theirs > mine else "chop"
        halves.append(f"{name}: {who} ({category(mine)} v {category(theirs)})")
    shares = pot_shares(state.holes, state.board)
    return f"{shown}\n  {'; '.join(halves)}   share {shares[human_seat]:.2f}"


def category(score) -> str:
    return score.category.name.lower().replace("_", " ")


# ---------------------------------------------------------------------------
# Reading the human's move
# ---------------------------------------------------------------------------


def ask_action(state: MiniState) -> Action:
    """Prompt until the human types a move that is legal here."""
    while True:
        typed = input(prompt_for(state))
        action = parse_action(typed, state)
        if action is not None:
            return action
        print(f"  didn't understand {typed.strip()!r} here")


def choices(state: MiniState) -> dict[str, Action]:
    """Every word that names a legal action here: its key, its label, and for a throw its card."""
    named: dict[str, Action] = {}
    for action in state.legal_actions():
        label, key = _label_and_key(state, action)
        named[key] = named[label] = action
        if action in (Action.THROW_LOW, Action.THROW_MID, Action.THROW_TOP):
            named[CARD_SYMBOL[thrown_card(state, action)].lower()] = action
    return named


def prompt_for(state: MiniState) -> str:
    """The prompt, naming exactly the legal actions; a throw also shows its card."""
    words = []
    for action in state.legal_actions():
        label, key = _label_and_key(state, action)
        at = label.index(key, label.find(" ") + 1 if label.startswith("throw ") else 0)
        word = f"{label[:at]}[{key}]{label[at + 1:]}"
        if action in (Action.THROW_LOW, Action.THROW_MID, Action.THROW_TOP):
            word += f" {CARD_SYMBOL[thrown_card(state, action)]}"
        words.append(word)
    if not state.is_draw_decision():
        player = state.current_player
        chips = chip_state(state.betting)
        owed = chips.owed_by(player)
        status = f"pot {chips.pot}" + (f", {owed} to call" if owed else "")
        return f"  {status} | " + " / ".join([*words, "[q]uit"]) + " > "
    return "  " + " / ".join([*words, "[q]uit"]) + " > "


def parse_action(typed: str, state: MiniState) -> Action | None:
    """The action `typed` names here, or None. Raises QuitGame on q.

    Context-strict, as in rung 2: "fold" with nothing to call is refused
    rather than read as a check.
    """
    word = typed.strip().lower()
    if word in ("q", "quit", "exit"):
        raise QuitGame
    return choices(state).get(word)


def _label_and_key(state: MiniState, action: Action) -> tuple[str, str]:
    """An action's word here and the one letter that selects it.

    Betting words take their first letter (only one of check and call is ever
    legal at once). Throws take low/mid/top's, and standing pat its `s`.
    """
    line = () if state.is_draw_decision() else state.betting[-1]
    label = action_label(action, line)
    if action is Action.THROW_NONE:
        return label, "s"
    if label.startswith("throw "):
        return label, label.removeprefix("throw ")[0]
    return label, label[0]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def report(board: Scoreboard) -> None:
    """The session's net, and each seat's rate beside what that seat earns in self-play.

    The per-seat target is the value of the seat the human sat in when the
    strategy plays itself, so a human matching it in both seats has played as
    well as the bot. Over alternating seats those targets cancel to zero; a
    perfect adversary can push the average up to +0.061, and a weaker player
    can land anywhere below. A hand's result swings by several chips, so the
    rates are noise until the hands run into the thousands.
    """
    if board.per_hand is None:
        return
    print(f"\n{board.hands} hands. You net {board.chips:+.0f} chips "
          f"({board.per_hand:+.3f} per hand).")
    for seat in (0, 1):
        rate = board.per_hand_in_seat(seat)
        if rate is None:
            continue
        target = SELF_PLAY_VALUE_P0 if seat == 0 else -SELF_PLAY_VALUE_P0
        print(f"  as P{seat}: {board.seat_chips[seat]:+.0f} over "
              f"{board.seat_hands[seat]} hands ({rate:+.3f} per hand; "
              f"this seat is worth {target:+.3f} in the bot's self-play)")
    print("Alternating seats, matching the bot averages zero and a perfect")
    print("adversary at most +0.061 a hand. A hand swings by several chips, so")
    print("short sessions are mostly noise.\n")


if __name__ == "__main__":
    main()
