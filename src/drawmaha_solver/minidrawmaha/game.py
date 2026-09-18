"""Mini-drawmaha rules: a bomb pot, one draw, and a pot that pays twice.

Rung 3's game, and the first one on this ladder with **no referee anywhere** —
no engine implements Drawmaha, so every rule here is a decision, and the ones
that are not simply inherited from real Drawmaha are argued in §3–§4 of the
rung-3 plan. The hand, start to finish:

1. Both players ante **1** from a **26**-chip stack. No blinds, no position
   asymmetry, and **no preflop betting** — it is a bomb pot, so the money is in
   before anybody has seen a board card.
2. Three private cards each, then **board card 1**, face up.
3. **Betting round 1.**
4. **The draw**, sequential: P0 throws **at most one** of their three cards,
   then P1 does, knowing *whether* P0 drew but never which card. The
   replacement is dealt **face down** — the full game's face-up draw-one house
   rule is not played at this rung.
5. **Board card 2**, then **betting round 2**, then the showdown.
6. **The pot splits in half.** The **inner** half goes to the best 3-card hand
   made of the three private cards alone; the **outer** half to the best 4-card
   hand made of exactly two private cards plus both board cards. Each half is
   won outright or chopped; winning both is a scoop.

Three things here are genuinely new against rung 2, and each has a trap:

* **The chance node depends on the action.** Leduc's board came out the same way
  whatever anyone did. Here the draw decides whether a chance node happens at
  all: stand pat and the deck never acts, throw a card and it deals one. A
  throw is therefore two nodes — the decision, then the replacement — and in
  between, the actor's hole is *short*, which is exactly how `kind` knows the
  deck owes them a card. The machinery deals `k` at a chance node rather than
  one, because the cap is a constant and the rest of the file does not know
  what it is.
* **Thrown cards leave the deck for good, and only their owner knows which.**
  `remaining_deck()` is derived from the state, never counted down from 15, so a
  discard can never be redealt to anybody.
* **A draw action names a card by position, and the positions are canonical.**
  `THROW_LOW` throws the lowest of the actor's three cards *ordered by the
  canonical suit labels* of their whole visible picture, not by the physical
  suit numbers. Under physical numbers, two positions that differ only by a suit
  swap sort their equal-ranked cards in opposite orders, so one ledger's "throw
  the lowest" would mean the board-suited card in one and the offsuit card in
  the other — the same infoset pointing at two different decisions. See
  `draw_order`.

Betting is pot-limit with the action set `{fold, check/call, pot}`, one
pot-sized bet, and the stack is what caps the escalation: from the 2-chip ante
pot an *uncapped* raise war would commit each actor 2 → 8 → 26 of their own
chips, so a 26-chip stack buys a bet, a raise and an all-in shove, and nothing
deeper. Those are the uncapped numbers, and only the first two are what a stack
of 26 actually pays: 25 is all there is behind the ante, so the third level
truncates to an all-in and `chip_state().in_round` tops out at 25, never 26.
That is the whole reason for the number. Round 1 has 13 complete lines; round 2
opens in whichever of four (pot, stacks) states round 1 left behind, so the
infoset key needs no explicit chip counts — the lines determine them.

A node is one of three kinds — `DECISION`, `CHANCE`, `TERMINAL` — and that
trichotomy is what a tree walker branches on. States are frozen: `apply` and
`apply_chance` return new ones, so a walker never has to undo a move.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from itertools import combinations
from typing import TYPE_CHECKING

from drawmaha_solver.minidrawmaha.cards import (
    DECK,
    Card,
    canonical,
    canonical_relabelling,
    hand_symbol,
)
from drawmaha_solver.minidrawmaha.hands import (
    BOARD_CARDS,
    HOLE_CARDS,
    inner_score,
    outer_score,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# The deck as a membership test. Every state construction asks "is this a real
# card?" of up to twelve cards, and `DECK` is an ordered tuple because dealing
# and rendering want the order — so the set is kept beside it rather than
# rebuilt per check.
_DECK_SET = frozenset(DECK)

# ---------------------------------------------------------------------------
# The shape of a hand
# ---------------------------------------------------------------------------

ANTE = 1

# 26 chips, so 25 behind the ante. Pot-limit from a 2-chip pot commits
# 2 -> 8 -> 26 in an uncapped raise war, each roughly 3x the last; 25 behind
# truncates the third to an all-in and there is no fourth level. Moving this
# number changes the betting tree, not just the money.
STACK = 26

N_ROUNDS = 2

# Board card 1 comes before the betting, board card 2 after the draw.
BOARD_STREETS = (1, 1)

# At most ONE of your three cards, realised by the four masks in `DISCARD_MASK`
# — this constant states the cap the masks obey rather than generating them.
#
# The cap is the biggest single lever on this rung's cost, bigger than the deck:
# at two, the draw node is 7 actions wide and the post-draw key count is 61,590;
# at one it is 4 wide and 11,140, which is fewer keys than the whole 12-card deck
# would cost at a cap of two. Both ranking tables are untouched either way —
# they depend on the deck, not on the cap. What one buys instead of two is the
# loss of the two-card draw: a player can improve a card, never a pair of them,
# so no line ever reshapes a hand wholesale. Restoring it is this constant plus
# the three two-card masks; nothing else in the file counts to two.
THROW_CAP = 1

class Action(IntEnum):
    """Every decision in the game, betting and drawing, in one enum.

    The values are NOT ledger indices. A ledger is as wide as
    `legal_actions()` at its infoset, and entry k belongs to
    `legal_actions()[k]` — which at a betting node is 2 or 3 of the first three
    members and at a draw node is all four throws.

    The throws are a 3-bit mask over the actor's three cards with popcount at
    most `THROW_CAP`, and the names read low/mid/top in `draw_order` — the
    canonical order, not the order the cards happened to be dealt in.
    """

    FOLD = 0
    CHECK_CALL = 1  # check when nothing is owed, call when something is
    POT = 2  # bet or raise the pot, capped at the stack

    THROW_NONE = 3  # 0b000 — stand pat, and the deck does not act at all
    THROW_LOW = 4  # 0b001
    THROW_MID = 5  # 0b010
    THROW_TOP = 6  # 0b100

BETTING_ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.POT)

DISCARD_MASK = {
    Action.THROW_NONE: 0b000,
    Action.THROW_LOW: 0b001,
    Action.THROW_MID: 0b010,
    Action.THROW_TOP: 0b100,
}

# Ascending by value, which is the order a draw ledger's columns are in.
DRAW_ACTIONS = tuple(DISCARD_MASK)

def throw_count(action: Action) -> int:
    """How many cards `action` throws away — the public half of a draw."""
    return DISCARD_MASK[action].bit_count()

# The betting shorthand, and only the betting: a draw never appears inside a
# betting line, so a throw has no symbol here. It reads as a word through
# `action_label` and as a count through `InfoSet.__str__`, which is the whole
# of what the table learns about it anyway.
#
# CHECK_CALL is absent for a different reason: it renders as `x` or `c`
# depending on whether anything is owed, which is context a dict does not have.
# See `line_symbol`.
ACTION_SYMBOL = {Action.FOLD: "f", Action.POT: "p"}

def _facing_bet(line: tuple[Action, ...]) -> bool:
    """True when the player to act has chips to match.

    A pattern, not an amount: only POT leaves anything outstanding, and a
    CHECK_CALL either checked or matched. The amount owed needs the pot and the
    stacks, and `betting_legal_actions` reads it off `chip_state` instead.
    """
    return bool(line) and line[-1] is Action.POT

def action_label(action: Action, line: tuple[Action, ...] = ()) -> str:
    """The poker word for `action`, for display only.

    CHECK_CALL is a check with nothing owed and a call facing a bet; POT is a
    bet into an unraised pot and a raise otherwise — so the label needs the
    round's line, exactly as in rung 2. A throw names the cards it discards by
    their canonical position, because that is what the action means.
    """
    if action not in BETTING_ACTIONS:
        thrown = [
            name
            for bit, name in enumerate(("low", "mid", "top"))
            if DISCARD_MASK[action] >> bit & 1
        ]
        return "stand pat" if not thrown else "throw " + "+".join(thrown)
    if action is Action.FOLD:
        return "fold"
    if action is Action.CHECK_CALL:
        return "call" if _facing_bet(line) else "check"
    return "raise" if _facing_bet(line) else "bet"

def line_symbol(line: tuple[Action, ...]) -> str:
    """One betting round as `xppc`-style shorthand: check, bet, bet, call.

    `x` and `c` are the same `CHECK_CALL` action doing two different jobs, and
    which job depends on whether anything was owed — so the shorthand replays
    the round rather than mapping actions one to one. The plan's line tables are
    written in this notation, and `test_game.py` compares against them.
    """
    text = ""
    for turn, action in enumerate(line):
        if action is Action.CHECK_CALL:
            text += "c" if _facing_bet(line[:turn]) else "x"
        else:
            text += ACTION_SYMBOL[action]
    return text

# ---------------------------------------------------------------------------
# The shape of one betting round
# ---------------------------------------------------------------------------
#
# Three predicates are the whole betting rule set, and both rounds obey them:
# what may be done, what ends the hand, what closes the round. No list of legal
# lines exists in this file — the 13 of round 1 are grown from these in
# `test_game.py` and compared against the plan's table there, which is the only
# place a hand-written list can actually catch anything.

def _is_fold(line: tuple[Action, ...]) -> bool:
    """True once someone has given up the hand, in either round."""
    return bool(line) and line[-1] is Action.FOLD

def _is_closed(line: tuple[Action, ...], behind: tuple[int, int]) -> bool:
    """True once the stakes are level and nobody is left to act.

    The opening CHECK_CALL is a check, not a close — `(check,)` is still the
    other player's turn — so a closing call needs something in front of it.
    The second clause is the all-in case: after `pppc` both stacks are empty, so
    round 2 opens with no decision in it at all and closes at length zero.
    """
    if behind == (0, 0):
        return True
    return len(line) >= 2 and line[-1] is Action.CHECK_CALL

def _legal_betting(owed: int, behind: int) -> tuple[Action, ...]:
    """The rule itself, stated in the only two quantities it depends on.

    Fold is not offered when checking is free — mirroring rung 2, where an
    opening player gets `{call, raise}` only — and POT needs chips beyond the
    call, so the stack ends an escalation with no raise cap anywhere. The
    one-wide answer is unreachable at a 26-chip stack (a player with nothing
    behind and nothing owed has already closed the round); it is here because
    the stack is a constant somebody will change.
    """
    if owed == 0:
        if behind > 0:
            return (Action.CHECK_CALL, Action.POT)
        return (Action.CHECK_CALL,)
    if behind > owed:
        return (Action.FOLD, Action.CHECK_CALL, Action.POT)
    return (Action.FOLD, Action.CHECK_CALL)

# ---------------------------------------------------------------------------
# The money
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ChipState:
    """Where the chips are: the pot, what each player has put in, what is left.

    `committed` counts the whole hand including the ante, `in_round` only the
    round now in progress — the difference is what a call has to match. Derived
    from the betting lines alone, which is why no infoset carries chip counts.
    """

    pot: int
    committed: tuple[int, int]
    behind: tuple[int, int]
    in_round: tuple[int, int]

    def owed_by(self, player: int) -> int:
        """What `player` must put in to match the other, this round."""
        return self.in_round[1 - player] - self.in_round[player]

def _pot_size(*, pot: int, owed: int, behind: int) -> int:
    """Chips a pot-sized bet or raise puts in, all-in capped.

    Facing nothing, a pot bet is the pot. Facing `owed`, the raise is the pot
    *after* the call plus the call itself — `min(pot + owed, behind - owed)` on
    top of matching — which is the standard pot-limit rule and the source of the
    2 -> 8 -> 26 ladder. The cap is the stack, so there is no explicit raise
    count anywhere in this file.
    """
    return owed + min(pot + owed, behind - owed)

def chip_state(betting: tuple[tuple[Action, ...], ...]) -> ChipState:
    """Replay the betting and report where the chips ended up.

    One pass per round. A folder stops paying where they stood, so their
    contribution is whatever they had already matched — which is what a fold
    hands to the winner.
    """
    return _replay(betting, validate=False)

def _replay(betting: tuple[tuple[Action, ...], ...], *, validate: bool) -> ChipState:
    """Walk the betting once, optionally checking each action as it goes.

    The chips and the rules are the same walk: what a player may do depends on
    what is owed and what is behind, and both of those are only known by having
    replayed the money up to that point. So validation rides along on the
    replay — one pass — rather than re-deriving the chip state per action, which
    is what `MiniState.__post_init__` used to cost on every construction. Rung 2
    could check a line against a precomputed set of reachable lines instead,
    because its bet sizes were constants; here legality depends on the stacks.
    """
    pot = 2 * ANTE
    committed = [ANTE, ANTE]
    behind = [STACK - ANTE, STACK - ANTE]
    in_round = [0, 0]
    for index, line in enumerate(betting):
        in_round = [0, 0]
        for turn, action in enumerate(line):
            player = turn % 2
            owed = in_round[1 - player] - in_round[player]
            if validate:
                done = line[:turn]
                if _is_fold(done) or _is_closed(done, (behind[0], behind[1])):
                    raise ValueError(f"{line} goes on after the round ended")
                if action not in _legal_betting(owed, behind[player]):
                    history = (*betting[:index], done)
                    raise ValueError(f"{action!r} is not legal after {history}")
            if action is Action.FOLD:
                break
            if action is Action.CHECK_CALL:
                put = min(owed, behind[player])
            else:
                put = _pot_size(pot=pot, owed=owed, behind=behind[player])
            in_round[player] += put
            committed[player] += put
            behind[player] -= put
            pot += put
    return ChipState(
        pot=pot,
        committed=(committed[0], committed[1]),
        behind=(behind[0], behind[1]),
        in_round=(in_round[0], in_round[1]),
    )

def betting_legal_actions(betting: tuple[tuple[Action, ...], ...]) -> tuple[Action, ...]:
    """What the player to act may do at a betting node, ascending by value."""
    state = chip_state(betting)
    player = len(betting[-1]) % 2
    return _legal_betting(state.owed_by(player), state.behind[player])

# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DrawSignal:
    """What the table learns when a player draws: the count, and nothing else.

    *Which* card was thrown is private and *what* replaced it is dealt face
    down, so this one number — 0 or 1 at this cap — is the entire public record
    of a draw. The full game's face-up draw-one rule would add a card here; at
    this rung it does not exist.
    """

    count: int

    def __post_init__(self) -> None:
        if not 0 <= self.count <= THROW_CAP:
            raise ValueError(f"a draw throws 0 to {THROW_CAP} cards, got {self.count}")

def draw_order(
    *, hole: tuple[Card, ...], discarded: tuple[Card, ...], board: tuple[Card, ...]
) -> tuple[Card, ...]:
    """The actor's three cards as `THROW_LOW`, `THROW_MID`, `THROW_TOP` index them.

    Sorted by rank, and ties within a rank broken by the **canonical** suit
    label of the actor's whole visible picture rather than by the physical suit
    number. That is not decoration, it is what keeps the draw actions meaningful
    across an infoset.

    Take a hole of two low cards and a high one, with a board card sharing a
    suit with exactly one of the low pair. Two such positions can differ by a
    suit swap alone — the same position twice, one ledger. Ordered by physical
    suit, "the lowest card" is the board-suited card in one and the offsuit card
    in the other, so a single strategy would be throwing two different cards.
    Ordered by canonical label, position k is the same card in both, and the
    ledger's four columns mean one thing each.
    """
    relabelling = canonical_relabelling(hole, discarded, board)
    return tuple(sorted(hole, key=lambda card: (card.rank, relabelling[card.suit])))

# ---------------------------------------------------------------------------
# The showdown
# ---------------------------------------------------------------------------

def pot_shares(
    holes: tuple[tuple[Card, ...], tuple[Card, ...]], board: tuple[Card, ...]
) -> tuple[float, float]:
    """Each player's share of the pot at a showdown, as fractions summing to 1.

    Two independent comparisons, half the pot on each. A scoop is `(1, 0)`, one
    half each is `(0.5, 0.5)`, and winning one half while chopping the other is
    `(0.75, 0.25)`. Quarters are real outcomes here, which is why shares are
    fractions and `returns()` is in floats: a split-pot game does not always pay
    in whole chips.
    """
    shares = [0.0, 0.0]
    for scores in (
        (inner_score(holes[0]), inner_score(holes[1])),
        (outer_score(holes[0], board), outer_score(holes[1], board)),
    ):
        if scores[0] > scores[1]:
            shares[0] += 0.5
        elif scores[1] > scores[0]:
            shares[1] += 0.5
        else:
            shares[0] += 0.25
            shares[1] += 0.25
    return (shares[0], shares[1])

def _chips(winner: int, stake: float) -> tuple[float, float]:
    """The zero-sum payoff pair: `winner` collects `stake`, the other pays it.

    One place the sign is decided. Written inline at each call site it is two
    mirrored ternaries that read the same and mean opposite things.
    """
    amount = float(stake)
    return (amount, -amount) if winner == 0 else (-amount, amount)

# ---------------------------------------------------------------------------
# Information sets
# ---------------------------------------------------------------------------

class NodeKind(StrEnum):
    """Which of the three things a node can be — the trichotomy a walk turns on.

    Every accessor on `MiniState` belongs to exactly one kind and refuses at the
    other two, so `returns()` can never report a pot nobody has won.
    """

    DECISION = "decision"
    CHANCE = "chance"
    TERMINAL = "terminal"

_WHO_ACTS = {
    NodeKind.DECISION: "a player is to act",
    NodeKind.CHANCE: "the deck is to act",
    NodeKind.TERMINAL: "the hand is over",
}

@dataclass(frozen=True, slots=True)
class InfoSet:
    """What the player to act knows, with the suits relabelled to canonical form.

    Four private-to-public facts and two public ones. `hole` is always three
    cards — the draw replaces what it throws, so a player never acts holding
    fewer. `discarded` is what **this** player threw away, and it is mandatory:
    the discard is the player's own action, so perfect recall forbids dropping
    it, and both CFR's guarantees and the exact best-response recursion assume
    perfect recall. Without it, a player who threw a low card and drew a high
    one would share a ledger with one who was simply dealt that high card —
    two histories they can plainly tell apart, with different cards missing
    from the deck.

    What is deliberately **not** stored is which of the three cards were kept
    and which were drawn. Merging those is lossless rather than an abstraction:
    both histories hold the same cards, have removed the same cards from the
    deck, and pay out identically, so their continuation games are the same
    game.

    `hole`, `discarded` and `board` are canonicalised **jointly, in one suit
    relabelling** — separately would forget whether the hole shares the board's
    suit, which is what a flush is. `draws` carries counts only and has no suits
    to relabel. The opponent's cards appear nowhere; that indistinguishability
    is the game.
    """

    player: int
    hole: tuple[Card, ...]
    discarded: tuple[Card, ...]
    board: tuple[Card, ...]
    draws: tuple[DrawSignal, ...]
    betting: tuple[tuple[Action, ...], ...]

    def __post_init__(self) -> None:
        # An infoset is a ledger key, so a malformed one does not crash — it
        # quietly allocates a second ledger for a position that already has
        # one, or shares one between positions that must not share. Every
        # check here exists because its failure mode is silent.
        if self.player not in (0, 1):
            raise ValueError(f"player is 0 or 1, got {self.player!r}")
        # A player never acts holding fewer than three: the draw's replacements
        # arrive before anybody's next decision.
        if len(self.hole) != HOLE_CARDS:
            raise ValueError(f"a player acts on {HOLE_CARDS} cards, got {self.hole}")
        if len(self.discarded) > THROW_CAP:
            raise ValueError(f"a draw throws at most {THROW_CAP}, got {self.discarded}")
        # Board card 1 is dealt before the first decision and card 2 before the
        # last, so nobody ever acts on an empty or a three-card board.
        if not 1 <= len(self.board) <= sum(BOARD_STREETS):
            raise ValueError(f"the board is 1 or 2 cards while anyone acts: {self.board}")
        if len(self.draws) > 2:
            raise ValueError(f"two players draw once each, got {self.draws}")
        # Canonical form is what makes two suit-equivalent positions ONE key.
        # `MiniState.infoset()` has already applied it, and this is not the same
        # check twice: `InfoSet` is public, a solver's table builds keys
        # directly, and a hand-built key in physical suits would hash as a
        # perfectly ordinary ledger nobody ever reaches. The relabelling search
        # is memoised, so asking again costs a dict lookup.
        if canonical(self.hole, self.discarded, self.board) != (
            self.hole,
            self.discarded,
            self.board,
        ):
            raise ValueError(
                f"{self} is not in canonical form; build infosets with MiniState.infoset()"
            )
        if self.player != self._actor():
            raise ValueError(f"{self} says P{self.player} acts, but P{self._actor()} does")
        # A player's own draw count is public, so the signal and the discards
        # are two views of one fact and must agree.
        if len(self.draws) > self.player:
            if self.draws[self.player].count != len(self.discarded):
                raise ValueError(f"{self} threw {self.discarded}, signalled {self.draws}")
        elif self.discarded:
            raise ValueError(f"{self} holds discards from a draw it has not made yet")

    def _actor(self) -> int:
        """Who the position says is to act: the drawer, or the next to bet."""
        if self.is_draw_decision():
            return len(self.draws)
        return len(self.betting[-1]) % 2

    def is_draw_decision(self) -> bool:
        """True when this ledger chooses a discard rather than a bet.

        The draw sits between the rounds, so it is the spot where round 1 has
        closed and the second board card has not come: one betting line, both
        draws not yet in.
        """
        return (
            len(self.betting) == 1
            and len(self.board) == 1
            and _is_closed(self.betting[0], chip_state(self.betting).behind)
            and len(self.draws) < 2
        )

    @property
    def round_index(self) -> int:
        """0 before the second board card, 1 after it."""
        return len(self.betting) - 1

    def legal_actions(self) -> tuple[Action, ...]:
        """This ledger's width and its column order, in one tuple."""
        if self.is_draw_decision():
            return DRAW_ACTIONS
        return betting_legal_actions(self.betting)

    def __str__(self) -> str:
        # "2c 3c 6d|4c:xp" is a hole and a board card with a bet to call;
        # "/2c|t2" adds the discards and the public draw counts.
        parts = [hand_symbol(self.hole)]
        if self.discarded:
            parts.append("/" + hand_symbol(self.discarded))
        parts.append("|" + hand_symbol(self.board))
        if self.draws:
            parts.append("|" + "".join(f"t{signal.count}" for signal in self.draws))
        parts.append(":" + "|".join(line_symbol(line) for line in self.betting))
        return "".join(parts)

# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MiniState:
    """One node of the game tree: both holes, the discards, the board, the betting.

    Constructed with the private deal already done, as in rung 2 — the deal is
    the root chance event and there are 100,100 of them, so a solver samples it
    (`random_deal`) or enumerates it outside the state machine rather than
    reading 100,100 outcomes off one node.

    Everything else the rules do is inside here. The fields are exactly the
    history: `discards[p]` is what player p threw, `draws` is the public record
    of the draws *decided* so far, and `betting` holds one line per round begun.
    A hole shorter than three cards is a draw awaiting its replacements, and
    that is what makes the deck's turn action-dependent.

    Immutable, so `apply` and `apply_chance` return new states and a recursive
    walker never has to undo a move. Constructing a state validates that it is a
    position the rules can produce, because a silently-accepted impossible one
    becomes a ledger entry or a payoff that no traceback ever points at.
    """

    holes: tuple[tuple[Card, ...], tuple[Card, ...]]
    discards: tuple[tuple[Card, ...], tuple[Card, ...]] = ((), ())
    board: tuple[Card, ...] = ()
    draws: tuple[DrawSignal, ...] = ()
    betting: tuple[tuple[Action, ...], ...] = ()

    def __post_init__(self) -> None:
        # Cards, then betting, then the draw — each stage may lean on the one
        # before it, and the draw's rules are stated against a betting history
        # already known to be legal.
        self._validate_cards()
        self._validate_betting()
        self._validate_draw()

    # -- validation ---------------------------------------------------------

    def _validate_cards(self) -> None:
        """One deck, dealt once: every card accounted for and nothing dealt twice."""
        if len(self.holes) != 2 or len(self.discards) != 2:
            raise ValueError("a heads-up state holds two holes and two discard piles")
        dealt = [card for group in self._piles for card in group]
        # A raw `(0, 1)` tuple hashes and compares equal to `Card(0, 1)`, so it
        # would pass every check below and then fail in `hands.py`, which reads
        # `.rank` — a showdown crash far from the mistake.
        if not all(isinstance(card, Card) for card in dealt):
            raise ValueError(f"every card must be a Card member, got {dealt}")
        # The same card in two places is the bug this whole rung has to not
        # have: a discard silently redealt, or a board card also in a hand,
        # pays out a showdown that could never happen.
        if len(set(dealt)) != len(dealt):
            raise ValueError(f"the same card was dealt twice: {hand_symbol(tuple(dealt))}")
        if any(card not in _DECK_SET for card in dealt):
            raise ValueError(f"a card is not in the deck: {hand_symbol(tuple(dealt))}")
        if len(self.board) > sum(BOARD_STREETS):
            most = sum(BOARD_STREETS)
            raise ValueError(f"the board is {most} cards at most, got {self.board}")
        piles = zip(self.holes, self.discards, strict=True)
        for player, (hole, discarded) in enumerate(piles):
            if len(discarded) > THROW_CAP:
                raise ValueError(f"P{player} threw {len(discarded)}, cap is {THROW_CAP}")
            # Three cards normally, and one or two only while the deck owes
            # replacements — a hole of none is not a position, and a hole of
            # four is a deal that never happened.
            if not HOLE_CARDS - THROW_CAP <= len(hole) <= HOLE_CARDS:
                raise ValueError(f"P{player} holds {len(hole)} cards, which no rule makes")

    def _validate_draw(self) -> None:
        """The draws, the discards and the short hole are three views of one fact."""
        if len(self.draws) > 2:
            raise ValueError(f"two players draw once each, got {self.draws}")
        if self.draws and not self._round_one_closed:
            raise ValueError("the draw comes after round 1 closes")
        # P0 draws first, so `draws[p]` is P0's signal and P1 has drawn only
        # once both are in. A pile of discards without the matching signal, or
        # a signal whose count disagrees with the pile, is a history where the
        # public record and the private one have come apart.
        for player, discarded in enumerate(self.discards):
            drawn = len(self.draws) > player
            if not drawn and discarded:
                raise ValueError(f"P{player} has discards but has not drawn")
            if drawn and self.draws[player].count != len(discarded):
                signal = self.draws[player]
                raise ValueError(f"P{player} threw {len(discarded)}, signalled {signal}")
        # The draw is sequential and its replacements arrive in the very next
        # node, so at most one hole is ever mid-draw.
        short = [player for player, hole in enumerate(self.holes) if len(hole) < HOLE_CARDS]
        if len(short) > 1:
            raise ValueError("only one draw is ever awaiting replacements")
        if short and len(self.draws) != short[0] + 1:
            raise ValueError(f"P{short[0]}'s hole is short but it is not their draw")

    def _validate_betting(self) -> None:
        """Replay the betting once, checking each action against the rule that allowed it.

        No list of reachable lines to fall out of date: an action is legal iff
        `_legal_betting` offered it at that point, which is the same authority
        `legal_actions()` answers with.
        """
        if len(self.betting) > N_ROUNDS:
            raise ValueError(f"a hand has {N_ROUNDS} rounds, got {len(self.betting)}")
        if bool(self.board) != bool(self.betting):
            raise ValueError("round 1 opens exactly when the first board card is dealt")
        if (len(self.board) == sum(BOARD_STREETS)) != (len(self.betting) == N_ROUNDS):
            raise ValueError("round 2 opens exactly when the second board card is dealt")
        for line in self.betting:
            # `Action` is an IntEnum, so a raw 1 compares equal to CHECK_CALL
            # and would pass the legality check inside the replay — then fail
            # the `is Action.FOLD` identity test in `returns`, scoring a fold as
            # a showdown. Reject the ints first.
            if not all(isinstance(action, Action) for action in line):
                raise ValueError(f"betting must hold Action members, got {line}")
        _replay(self.betting, validate=True)
        if len(self.betting) == N_ROUNDS:
            if not self._round_one_closed:
                raise ValueError("round 1 must close before round 2 opens")
            if len(self.draws) != 2 or any(len(hole) < HOLE_CARDS for hole in self.holes):
                raise ValueError("both draws finish before the second board card")

    # -- which of the three kinds of node this is ---------------------------

    @property
    def _piles(self) -> tuple[tuple[Card, ...], ...]:
        """Every card this position accounts for: both holes, both discard piles, the board.

        The five groups are what "already dealt" means, and both the duplicate
        check and `remaining_deck()` are statements about exactly this list —
        so a sixth group added later cannot be forgotten by one of them.
        """
        return (*self.holes, *self.discards, self.board)

    @property
    def _line(self) -> tuple[Action, ...]:
        """The betting inside the round now in progress."""
        return self.betting[-1]

    @property
    def _round_one_closed(self) -> bool:
        """True once round 1's stakes are level — the gate the draw sits behind."""
        if not self.betting:
            return False
        first = self.betting[0]
        return not _is_fold(first) and _is_closed(first, chip_state((first,)).behind)

    @property
    def _pending_draw(self) -> int | None:
        """The player whose thrown cards have not been replaced yet, if any."""
        for player, hole in enumerate(self.holes):
            if len(hole) < HOLE_CARDS:
                return player
        return None

    @property
    def round_index(self) -> int:
        """0 before the second board card, 1 after it."""
        return max(len(self.betting) - 1, 0)

    @property
    def kind(self) -> NodeKind:
        """Decision, chance, or terminal — read off the cards and the lines.

        In order: the deck opens the hand; a fold ends it wherever it happens; a
        round still open is somebody's decision; a closed round 1 hands over to
        the draw, and the draw to the deck; a closed round 2 is the showdown.
        """
        if not self.board:
            return NodeKind.CHANCE
        if _is_fold(self._line):
            return NodeKind.TERMINAL
        if not _is_closed(self._line, chip_state(self.betting).behind):
            return NodeKind.DECISION
        if self.round_index == N_ROUNDS - 1:
            return NodeKind.TERMINAL
        if self._pending_draw is not None:
            return NodeKind.CHANCE
        return NodeKind.DECISION if len(self.draws) < 2 else NodeKind.CHANCE

    def is_terminal(self) -> bool:
        """True once a fold or a showdown has decided the hand."""
        return self.kind is NodeKind.TERMINAL

    def is_chance_node(self) -> bool:
        """True when the deck owes a board card or a set of replacements."""
        return self.kind is NodeKind.CHANCE

    def is_draw_decision(self) -> bool:
        """True when the player to act chooses a discard rather than a bet."""
        if self.kind is not NodeKind.DECISION:
            return False
        return self._round_one_closed and len(self.draws) < 2

    def _require(self, kind: NodeKind) -> None:
        """Refuse an accessor that does not belong to this node's kind."""
        if self.kind is not kind:
            raise ValueError(
                f"this is a {self.kind} node, not a {kind} node; {_WHO_ACTS[self.kind]}"
            )

    # -- chance -------------------------------------------------------------

    def remaining_deck(self) -> tuple[Card, ...]:
        """The cards nobody can be dealt: the deck minus everything already out.

        Derived, never counted down from 15, and that is a correctness rule
        rather than a style: a thrown card is gone from the stub too, and only
        its owner knows which card it was. A hard-coded count would eventually
        redeal somebody's discard.
        """
        seen = {card for group in self._piles for card in group}
        return tuple(card for card in DECK if card not in seen)

    def chance_outcomes(self) -> tuple[tuple[tuple[Card, ...], float], ...]:
        """Every set of cards the deck could produce here, with its probability.

        One signature for both of the deck's jobs: a board card is a set of one,
        and a draw of `k` is a set of `k` dealt in a single node — the order they
        arrive in is not part of the position, so the outcomes are combinations
        and each is equally likely.
        """
        self._require(NodeKind.CHANCE)
        stub = self.remaining_deck()
        pending = self._pending_draw
        size = 1 if pending is None else HOLE_CARDS - len(self.holes[pending])
        outcomes = tuple(combinations(stub, size))
        return tuple((cards, 1.0 / len(outcomes)) for cards in outcomes)

    def apply_chance(self, cards: tuple[Card, ...]) -> MiniState:
        """The state after the deck produces `cards` — a board card or replacements.

        Which of the two is read off the position, not passed in: a short hole
        means the cards are that player's replacements, otherwise they are the
        next board card, and a board card opens the betting round behind it.
        """
        self._require(NodeKind.CHANCE)
        pending = self._pending_draw
        if pending is not None:
            holes = list(self.holes)
            holes[pending] = tuple(sorted(holes[pending] + tuple(cards)))
            return MiniState(
                holes=(holes[0], holes[1]),
                discards=self.discards,
                board=self.board,
                draws=self.draws,
                betting=self.betting,
            )
        return MiniState(
            holes=self.holes,
            discards=self.discards,
            board=self.board + tuple(cards),
            draws=self.draws,
            betting=(*self.betting, ()),
        )

    # -- decisions ----------------------------------------------------------

    @property
    def current_player(self) -> int:
        """Whose turn it is. P0 opens every betting round and draws first."""
        self._require(NodeKind.DECISION)
        if self.is_draw_decision():
            return len(self.draws)
        return len(self._line) % 2

    def legal_actions(self) -> tuple[Action, ...]:
        """What the player to act may do, in the fixed order ledgers index by."""
        self._require(NodeKind.DECISION)
        if self.is_draw_decision():
            return DRAW_ACTIONS
        return betting_legal_actions(self.betting)

    def apply(self, action: Action) -> MiniState:
        """The state after the player to act takes `action`.

        A throw removes the masked cards from the actor's hole and records the
        public count, leaving the hole short — so the very next node is the deck
        dealing exactly that many replacements. Standing pat records a count of
        zero and produces no chance node at all, which is the action-dependent
        chance branch this rung exists to model.
        """
        self._require(NodeKind.DECISION)
        if action not in self.legal_actions():
            raise ValueError(f"{action!r} is not legal here: {self.legal_actions()}")
        player = self.current_player
        if not self.is_draw_decision():
            line = self._line + (action,)
            return MiniState(
                holes=self.holes,
                discards=self.discards,
                board=self.board,
                draws=self.draws,
                betting=self.betting[:-1] + (line,),
            )
        mask = DISCARD_MASK[action]
        ordered = draw_order(
            hole=self.holes[player], discarded=self.discards[player], board=self.board
        )
        thrown = tuple(card for index, card in enumerate(ordered) if mask >> index & 1)
        holes, discards = list(self.holes), list(self.discards)
        holes[player] = tuple(card for card in holes[player] if card not in thrown)
        discards[player] = tuple(sorted(discards[player] + thrown))
        return MiniState(
            holes=(holes[0], holes[1]),
            discards=(discards[0], discards[1]),
            board=self.board,
            draws=(*self.draws, DrawSignal(len(thrown))),
            betting=self.betting,
        )

    def infoset(self) -> InfoSet:
        """What the player to act can see, with the suits relabelled to canonical form.

        The three card groups this player can see go through `canonical` in one
        call, so the relabelling that hides which physical suit is which cannot
        forget how they relate. The opponent's hole and discards are not passed
        in at all.
        """
        self._require(NodeKind.DECISION)
        player = self.current_player
        hole, discarded, board = canonical(
            self.holes[player], self.discards[player], self.board
        )
        return InfoSet(
            player=player,
            hole=hole,
            discarded=discarded,
            board=board,
            draws=self.draws,
            betting=self.betting,
        )

    # -- payoffs ------------------------------------------------------------

    def contributions(self) -> tuple[int, int]:
        """Chips (P0, P1) have put in across the whole hand, ante included."""
        return chip_state(self.betting).committed

    def returns(self) -> tuple[float, float]:
        """Net chips to (P0, P1). Sums to zero. Half the pot rides on each hand.

        A fold hands the folder's own stake over and returns the rest, so it
        costs what they had matched rather than a flat ante. A showdown pays
        each half on its own comparison — the inner half on the three cards
        held, the outer on the best two of them with both board cards — so the
        four outcomes are a scoop, a half each, and either quarter-split in
        between. Returns are net of each player's own contributions, which is
        what makes them sum to zero.
        """
        self._require(NodeKind.TERMINAL)
        chips = chip_state(self.betting)
        if _is_fold(self._line):
            loser = (len(self._line) - 1) % 2
            return _chips(winner=1 - loser, stake=chips.committed[loser])
        # Only a matched call or a shared all-in reaches a showdown, so the
        # stakes are level and the pot is twice either contribution.
        if chips.committed[0] != chips.committed[1]:
            raise AssertionError(f"showdown with unmatched stakes {chips.committed}")
        shares = pot_shares(self.holes, self.board)
        return (
            shares[0] * chips.pot - chips.committed[0],
            shares[1] * chips.pot - chips.committed[1],
        )

# ---------------------------------------------------------------------------
# What a tree walker calls
# ---------------------------------------------------------------------------

def legal_actions_for(state: MiniState) -> tuple[Action, ...]:
    """What may be done at `state` — betting actions, or all four throws.

    The free function the rest of the package calls, so a walker never has to
    know whether it is looking at a bet or a draw.
    """
    return state.legal_actions()

# ---------------------------------------------------------------------------
# The root chance event
# ---------------------------------------------------------------------------

def random_deal(rng: np.random.Generator) -> MiniState:
    """Deal three cards to each player, uniformly over all 100,100 deals.

    The private deal is the root chance event and it is the one the state
    machine does not model as a node: there are too many outcomes to read off a
    single `chance_outcomes()`, and rung 2 set the precedent of handing the deal
    in. Sampling it is exactly what external-sampling MCCFR wants.
    """
    order = rng.permutation(len(DECK))
    dealt = [DECK[index] for index in order[: 2 * HOLE_CARDS]]
    return MiniState(
        holes=(tuple(sorted(dealt[:HOLE_CARDS])), tuple(sorted(dealt[HOLE_CARDS:]))),
    )
