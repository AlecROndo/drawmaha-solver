"""Kuhn poker rules: deck, betting, the game tree, information sets, payoffs.

Rung 1 of the validation ladder, and the first game with a *tree*. Both
players ante 1 chip, each is dealt one card from a three-card deck (J < Q < K,
the third card unseen), and a single round of betting follows in which the
only bet size is 1. Each player faces at most one decision, and at every
decision there are exactly two choices, written PASS and BET. Their meaning
depends on what came before — a PASS with nothing to call is a check, a PASS
facing a bet is a fold, a BET facing a bet is a call — so `action_label`
renders the poker word while the ledger keeps a uniform two-wide action index.

The vocabulary a solver needs from this module: a **history** is the sequence
of actions so far (the tree has 4 decision nodes and 5 terminals); an
**infoset** is what one player actually knows when acting — their own card
plus the history, never the opponent's card — and is the unit CFR keeps a
regret ledger for; there are exactly 12. **Returns** are in chips, with the
ante as the unit, from each player's own perspective.

Chance appears exactly once, at the root: `DEALS` lists the six equally likely
(P0 card, P1 card) orderings. Nothing here samples or learns — a state is
immutable and `apply` returns a new one, so a tree walker recurses without
having to undo anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# ---------------------------------------------------------------------------
# Cards and the chance node
# ---------------------------------------------------------------------------

class Card(IntEnum):
    # IntEnum, ordered low to high, so "the higher card wins" is a plain `>`.
    JACK = 0
    QUEEN = 1
    KING = 2

DECK = (Card.JACK, Card.QUEEN, Card.KING)

CARD_SYMBOL = {Card.JACK: "J", Card.QUEEN: "Q", Card.KING: "K"}

# The game's only chance event: one card each, dealt without replacement, so
# the six ordered pairs are equally likely. The third card stays in the deck.
DEALS = tuple((a, b) for a in DECK for b in DECK if a != b)
DEAL_PROBABILITY = 1.0 / len(DEALS)

# ---------------------------------------------------------------------------
# Actions and betting
# ---------------------------------------------------------------------------

class Action(IntEnum):
    # Two actions at every decision node, so regret and strategy vectors are
    # uniformly 2-wide and these values index them directly.
    PASS = 0
    BET = 1

N_ACTIONS = len(Action)

ACTION_SYMBOL = {Action.PASS: "p", Action.BET: "b"}

ANTE = 1
BET_SIZE = 1

def action_label(action: Action, history: tuple[Action, ...]) -> str:
    """The poker word for `action` at a node reached by `history`.

    PASS is a check with nothing to call and a fold facing a bet; BET is a
    bet into an unraised pot and a call facing a bet. Display only — the
    game logic never branches on these strings.
    """
    facing_bet = bool(history) and history[-1] is Action.BET
    if action is Action.PASS:
        return "fold" if facing_bet else "check"
    return "call" if facing_bet else "bet"

# ---------------------------------------------------------------------------
# The game tree
# ---------------------------------------------------------------------------

# Every history in the game, listed rather than derived — the tree is nine
# nodes and writing them down is the clearest statement of the rules.
DECISION_HISTORIES = (
    (),                          # P0 opens
    (Action.PASS,),              # P0 checked; P1 may check behind or bet
    (Action.BET,),               # P0 bet; P1 may fold or call
    (Action.PASS, Action.BET),   # P0 checked and was bet into; fold or call
)

TERMINAL_HISTORIES = frozenset(
    {
        (Action.PASS, Action.PASS),                # checked down
        (Action.PASS, Action.BET, Action.PASS),    # P0 folds
        (Action.PASS, Action.BET, Action.BET),     # P0 calls
        (Action.BET, Action.PASS),                 # P1 folds
        (Action.BET, Action.BET),                  # P1 calls
    }
)

REACHABLE_HISTORIES = frozenset(DECISION_HISTORIES) | TERMINAL_HISTORIES

# ---------------------------------------------------------------------------
# Information sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InfoSet:
    """What the player to act knows: their own card and the public history.

    Deliberately does NOT carry the opponent's card — two deals that differ
    only in the opponent's card must produce one equal, equally hashing
    infoset, because that indistinguishability is the game. Frozen and
    hashable so it can key a regret table directly.
    """

    card: Card
    history: tuple[Action, ...]

    @property
    def player(self) -> int:
        """Whose decision this is. P0 acts on even-length histories."""
        return len(self.history) % 2

    def __str__(self) -> str:
        # The literature's notation: "K", "Jpb", "Qb".
        path = "".join(ACTION_SYMBOL[a] for a in self.history)
        return f"{CARD_SYMBOL[self.card]}{path}"

def all_infosets() -> tuple[InfoSet, ...]:
    """Every infoset in the game: 4 decision nodes x 3 possible own cards.

    Twelve entries — the full set of ledgers a tabular solver allocates.
    """
    return tuple(
        InfoSet(card=card, history=history)
        for history in DECISION_HISTORIES
        for card in DECK
    )

# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KuhnState:
    """One node of the game tree: the deal, plus the actions taken so far.

    Immutable — `apply` returns a new state, so a recursive walker never has
    to undo a move. Constructing a state validates that it is a position the
    rules can actually produce; every accessor that only makes sense at one
    kind of node (`returns` at a terminal, `legal_actions` at a decision)
    raises rather than returning a plausible wrong answer at the other.
    """

    cards: tuple[Card, Card]
    history: tuple[Action, ...] = ()

    def __post_init__(self) -> None:
        # Enum members, not the ints behind them. `Card` and `Action` are
        # IntEnums, so a raw (0, 1) hashes and compares equal to (JACK, QUEEN)
        # and would slip past every check below — but `returns` and
        # `action_label` ask `is Action.BET`, which a plain 1 fails, so a raw
        # history scores the wrong winner in silence. A solver that indexes
        # deals or actions by number converts here, at the boundary.
        if len(self.cards) != 2 or not all(isinstance(c, Card) for c in self.cards):
            raise ValueError(f"cards must be two Card members, got {self.cards}")
        # One card each from a three-card deck: a repeated card is not a deal
        # the game can produce.
        if self.cards[0] == self.cards[1]:
            raise ValueError(f"cards must be two distinct cards, got {self.cards}")
        if not all(isinstance(a, Action) for a in self.history):
            raise ValueError(f"history must hold Action members, got {self.history}")
        # Betting stops the moment a hand is decided, so sequences like
        # (PASS, PASS, PASS) name no node in the tree.
        if self.history not in REACHABLE_HISTORIES:
            raise ValueError(f"{self.history} is not a reachable history")

    def is_terminal(self) -> bool:
        """True once the hand is decided by a fold or a showdown."""
        return self.history in TERMINAL_HISTORIES

    @property
    def current_player(self) -> int:
        """Whose turn it is. Players alternate, P0 first."""
        if self.is_terminal():
            raise ValueError(f"{self.history} is terminal; nobody is to act")
        return len(self.history) % 2

    def legal_actions(self) -> tuple[Action, ...]:
        """Both actions, always — Kuhn never restricts the choice."""
        if self.is_terminal():
            raise ValueError(f"{self.history} is terminal; no actions remain")
        return (Action.PASS, Action.BET)

    def apply(self, action: Action) -> KuhnState:
        """The state after the player to act takes `action`."""
        if self.is_terminal():
            raise ValueError(f"{self.history} is terminal; cannot apply {action.name}")
        return KuhnState(cards=self.cards, history=self.history + (action,))

    def infoset(self) -> InfoSet:
        """What the player to act can see."""
        return InfoSet(card=self.cards[self.current_player], history=self.history)

    def returns(self) -> tuple[float, float]:
        """Chips won by (P0, P1), in units of the ante. Sums to zero.

        A fold hands the ante over and returns the bet; a showdown is won by
        the higher card, for the ante alone if the hand was checked down and
        for the ante plus the bet if a bet was called.
        """
        if not self.is_terminal():
            raise ValueError(f"{self.history} is not terminal; no payoff yet")

        contested = Action.BET in self.history
        # A PASS with a bet outstanding is a fold — the only way the hand ends
        # without a showdown.
        if contested and self.history[-1] is Action.PASS:
            folder = (len(self.history) - 1) % 2
            winner, stake = 1 - folder, float(ANTE)
        else:
            winner = 0 if self.cards[0] > self.cards[1] else 1
            stake = float(ANTE + BET_SIZE) if contested else float(ANTE)

        return (stake, -stake) if winner == 0 else (-stake, stake)
