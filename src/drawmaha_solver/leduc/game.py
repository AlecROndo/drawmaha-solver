"""Leduc poker rules: deck, two betting rounds, the board, information sets, payoffs.

Rung 2 of the validation ladder, and the first game where **chance acts in the
middle of the tree**. Both players ante 1 chip and are dealt one card each from
a six-card deck of two jacks, two queens and two kings. A betting round runs; if
nobody folded, one public card — the *board* — is turned from what is left of the
deck, and a second betting round runs at double the stake. Rung 1's single chance
node at the root is still here, but a second one now sits between the rounds, and
that is the whole reason this rung exists.

Three things change from `kuhn/game.py`, and each of them is a rule the referee
(OpenSpiel's `leduc_poker`) pins exactly:

* **Three actions, but never three choices.** FOLD, CALL, RAISE — yet folding is
  illegal with nothing to call, and raising is illegal once the round's two-raise
  cap is spent. So a node is 2- or 3-wide and `legal_actions()` is the authority.
  A ledger is sized and indexed by *position within that tuple*, never by the
  `Action` value, which is why the tuple's order is documented and fixed.
* **Two rounds of one shape.** Round 2 repeats round 1 exactly — P0 opens, the
  cap resets, the same six decision lines — with the bet size doubled from 2 to
  4. So the betting rules are three predicates (`legal_actions_for`,
  `_is_fold`, `_is_closed`) that both rounds obey, and the fifteen lines a
  round can produce are *grown* from them rather than listed. `betting` is a
  tuple of one line per round begun.
* **The board decides the showdown.** A private card that matches the board
  outranks every unpaired card, so on a jack board a jack beats a king.

A node is one of three kinds — `NodeKind.DECISION`, `CHANCE`, `TERMINAL` — and
that trichotomy is what a tree walker branches on. It is a single computed
value here, not a pair of predicates each method re-derives, and every accessor
declares the kind it belongs to and refuses at the other two.

The vocabulary a solver needs: a **betting line** is the actions inside one
round; a **history** is the pair of lines plus the cards; an **infoset** is what
one player knows when acting — their own card's *rank*, the board's rank, and
both lines, never the opponent's card and never a suit. There are 288 of them,
the 936 suit-distinct positions collapsed by rank, and 96 are 3-wide.

Chance appears twice: `DEALS` lists the thirty equally likely (P0 card, P1 card)
orderings, and `chance_outcomes()` reads the four survivors of the deck off the
state when the board is due. Nothing here samples or learns — a state is
immutable and `apply` returns a new one, so a tree walker recurses without having
to undo anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

# ---------------------------------------------------------------------------
# Cards, ranks, and the private deal
# ---------------------------------------------------------------------------

class Rank(IntEnum):
    # IntEnum, ordered low to high, so "the higher rank wins" is a plain `>`.
    JACK = 0
    QUEEN = 1
    KING = 2

class Card(IntEnum):
    """One of six physical cards: two of each rank, in two suits.

    Numbered 0..5 the way the referee numbers them, so `rank` is `card // 2`
    and the twin of a card is its neighbour. Ordering comparisons raise: card 1
    is a jack and card 2 is a queen, but card 0 and card 1 are the same hand,
    so `<` between two Cards is never the comparison anyone means. Compare
    `rank`, or better, `hand_strength`.
    """

    JACK_A = 0
    JACK_B = 1
    QUEEN_A = 2
    QUEEN_B = 3
    KING_A = 4
    KING_B = 5

    @property
    def rank(self) -> Rank:
        return Rank(int(self) // 2)

    @property
    def suit(self) -> int:
        return int(self) % 2

    # Rung 1's deck held one card per rank, so `<` on a card *was* the hand
    # comparison. A two-suit deck breaks that: JACK_A < JACK_B evaluates True
    # while the two hands are identical, and the reverse of the same expression
    # is False, so an accidental `<` silently awards a pot to whoever's suit
    # sorts lower. That is a wrong showdown with no traceback — the exact class
    # of bug this module fails loudly on elsewhere — so the ordering is removed
    # rather than merely warned about in the docstring.
    def _unordered(self, other: object) -> bool:
        raise TypeError(
            "Cards are not ordered — JACK_A and JACK_B are the same hand. "
            "Compare .rank, or hand_strength(private, board) for a showdown."
        )

    __lt__ = _unordered
    __le__ = _unordered
    __gt__ = _unordered
    __ge__ = _unordered

RANKS = (Rank.JACK, Rank.QUEEN, Rank.KING)

DECK = tuple(Card)

RANK_SYMBOL = {Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K"}

CARD_SYMBOL = {card: RANK_SYMBOL[card.rank] + "ab"[card.suit] for card in DECK}

# The first chance event: one card each, dealt without replacement, so the
# thirty ordered pairs are equally likely. Walking these thirty *concrete*
# deals rather than the nine rank pairs is what keeps every chance probability
# uniform; the suits collapse later, in the infoset key, and only there.
DEALS = tuple((a, b) for a in DECK for b in DECK if a != b)
DEAL_PROBABILITY = 1.0 / len(DEALS)

# ---------------------------------------------------------------------------
# Actions and betting
# ---------------------------------------------------------------------------

class Action(IntEnum):
    # Numbered as the referee numbers them. The values are NOT ledger indices:
    # a ledger is as wide as `legal_actions()` at its infoset, and entry k of a
    # regret or strategy vector belongs to `legal_actions()[k]`.
    FOLD = 0
    CALL = 1
    RAISE = 2

N_ACTIONS = len(Action)

ACTION_SYMBOL = {Action.FOLD: "f", Action.CALL: "c", Action.RAISE: "r"}

ANTE = 1

# One bet size per round: round 2's is double round 1's, which is why the
# largest pot is 1 + 2*2 + 2*4 = 13 a side rather than Kuhn's 2.
BET_SIZES = (2, 4)

N_ROUNDS = len(BET_SIZES)

# Two raises per round, counting both players — after a raise and a re-raise
# the only answers left are fold and call.
RAISE_CAP = 2

# A private card that matches the board outranks every unpaired card, so pairs
# need a strength above KING. Only one player can ever hold it: a rank has two
# cards, and if both are in hands there is none left for the board.
PAIR_STRENGTH = int(Rank.KING) + 1

def action_label(action: Action, line: tuple[Action, ...]) -> str:
    """The poker word for `action` at a decision reached by `line`.

    CALL is a check with nothing to call and a call facing a raise; RAISE is a
    bet into an unraised pot and a raise otherwise. Display only — the game
    logic never branches on these strings.
    """
    facing_raise = bool(line) and line[-1] is Action.RAISE
    if action is Action.FOLD:
        return "fold"
    if action is Action.CALL:
        return "call" if facing_raise else "check"
    return "raise" if facing_raise else "bet"

# ---------------------------------------------------------------------------
# The shape of one betting round
# ---------------------------------------------------------------------------
#
# Three predicates are the whole betting rule set, and both rounds obey them
# identically: what may be done, what ends the hand, what closes the round.
# Which lines exist, and which of them are decisions, folds or closes, is grown
# from those three below rather than written out — so a betting rule is stated
# in exactly one place and cannot go stale against a list.

def legal_actions_for(line: tuple[Action, ...]) -> tuple[Action, ...]:
    """What the player to act may do, given the round's line so far.

    Ascending by `Action` value, always, because a ledger's entry k means
    `legal_actions()[k]` and nothing else keeps the two in step. Fold needs a
    raise to fold to; raise needs the cap to have room.
    """
    raises = line.count(Action.RAISE)
    if raises == 0:
        return (Action.CALL, Action.RAISE)
    if raises < RAISE_CAP:
        return (Action.FOLD, Action.CALL, Action.RAISE)
    return (Action.FOLD, Action.CALL)

def _is_fold(line: tuple[Action, ...]) -> bool:
    """True once someone has given up the hand, in either round."""
    return bool(line) and line[-1] is Action.FOLD

def _is_closed(line: tuple[Action, ...]) -> bool:
    """True once the stakes are level and the round is over.

    The opening CALL is a check, not a close — `(call,)` is still P1's turn —
    so a closing call needs something in front of it. After round 1 these are
    the five lines that hand over to the deck; after round 2 they are showdowns.
    """
    return len(line) >= 2 and line[-1] is Action.CALL

def _grow_round_lines() -> tuple[tuple[tuple[Action, ...], ...], ...]:
    """Every line one round can produce, grown from the three rules above.

    Breadth-first from the empty line: a fold or a closing call is a leaf,
    anything else is a decision whose children are its legal actions. Fifteen
    lines fall out — six decisions, four folds, five closes. Rung 1 listed its
    nine histories by hand because there was no rule to derive them from: every
    action was legal at every Kuhn node. Here `legal_actions_for` *is* that
    rule, so a hand-written list would be a second, silently divergent copy of
    it. The independent statement lives in `test_game.py`, as a table typed out
    from the referee — which is the only place it can actually catch anything.
    """
    decisions: list[tuple[Action, ...]] = []
    folds: list[tuple[Action, ...]] = []
    closes: list[tuple[Action, ...]] = []
    frontier: list[tuple[Action, ...]] = [()]
    while frontier:
        line = frontier.pop(0)
        if _is_fold(line):
            folds.append(line)
        elif _is_closed(line):
            closes.append(line)
        else:
            decisions.append(line)
            frontier.extend(line + (action,) for action in legal_actions_for(line))
    return tuple(decisions), tuple(folds), tuple(closes)

ROUND_DECISION_LINES, ROUND_FOLD_LINES, ROUND_CLOSING_LINES = _grow_round_lines()

REACHABLE_ROUND_LINES = frozenset(
    ROUND_DECISION_LINES + ROUND_FOLD_LINES + ROUND_CLOSING_LINES
)

# ---------------------------------------------------------------------------
# Validation at the boundary
# ---------------------------------------------------------------------------

def _validate_betting(
    betting: tuple[tuple[Action, ...], ...], board: Rank | Card | None
) -> None:
    """Check a (betting, board) pair against the rules — the only place it happens.

    `InfoSet` and `LeducState` describe the same position at two grains, ranks
    and cards, so they agree on everything except which cards are which; this
    is the single owner of the facts they share. A position the rules cannot
    produce must not be constructible, because a silently-accepted one becomes
    a ledger entry or a payoff that no traceback ever points at.
    """
    for line in betting:
        # `Action` is an IntEnum, so a raw (1, 1) compares equal to
        # (CALL, CALL) and would pass the reachability test below — then fail
        # the `is Action.FOLD` identity test in `returns`, scoring a fold as a
        # showdown. Reject the ints before the membership check, not after.
        if not all(isinstance(a, Action) for a in line):
            raise ValueError(f"betting must hold Action members, got {line}")
        if line not in REACHABLE_ROUND_LINES:
            raise ValueError(f"{line} is not a reachable betting line")
    if not 1 <= len(betting) <= N_ROUNDS:
        raise ValueError(f"a hand has {N_ROUNDS} rounds at most, got {len(betting)}")

    # The board and the round count are two views of one fact: round 2 exists
    # exactly when the deck has turned a card, and it opens only once round 1
    # has closed with the stakes level.
    opened_round_two = len(betting) == N_ROUNDS
    if opened_round_two and not _is_closed(betting[0]):
        raise ValueError(f"round 1 {betting[0]} must be closed before round 2 opens")
    if opened_round_two and board is None:
        raise ValueError("round 2 needs a board; deal it before betting again")
    if not opened_round_two and board is not None:
        raise ValueError(f"round 2 has not opened, so there is no board yet: {board!r}")

# ---------------------------------------------------------------------------
# Information sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InfoSet:
    """What the player to act knows: their own RANK, the board's rank, both lines.

    Keyed on rank rather than card, which is the one deliberate lossy step in
    this file. The two jacks are the same hand, so they must share one ledger
    instead of training two half-sized copies; that collapse is what turns the
    referee's 936 suit-distinct positions into 288 ledgers. It does NOT carry
    the opponent's card — that indistinguishability is the game — and it does
    carry round 1's line into round 2, because a player who raised and got
    called remembers doing it (perfect recall).

    Frozen and hashable so it can key a regret table directly. Whose decision
    it is falls out of the current line's length and is not stored.
    """

    rank: Rank
    board: Rank | None
    betting: tuple[tuple[Action, ...], ...]

    def __post_init__(self) -> None:
        # Enum members, not the ints behind them. Nothing reads an infoset by
        # identity, so a raw key would render and hash as if it were real — an
        # invisible duplicate ledger entry rather than a crash, which is
        # strictly worse for a regret table than an exception.
        if not isinstance(self.rank, Rank):
            raise ValueError(f"rank must be a Rank member, got {self.rank!r}")
        if self.board is not None and not isinstance(self.board, Rank):
            raise ValueError(f"board must be a Rank member or None, got {self.board!r}")
        _validate_betting(self.betting, self.board)
        # An infoset is what the player *to act* knows, so unlike a state it
        # exists only at a decision: there is no ledger for a finished hand.
        if self.betting[-1] not in ROUND_DECISION_LINES:
            raise ValueError(f"{self.betting[-1]} is not a decision node; nobody is to act")

    @property
    def player(self) -> int:
        """Whose decision this is. P0 opens every round, so it is parity."""
        return len(self.betting[-1]) % 2

    @property
    def round_index(self) -> int:
        """0 while the board is face down, 1 once it is up."""
        return len(self.betting) - 1

    def legal_actions(self) -> tuple[Action, ...]:
        """This ledger's width and its column order, in one tuple."""
        return legal_actions_for(self.betting[-1])

    def __str__(self) -> str:
        # "K:" is the king in the opening seat; "J:cc|Q:r" is a jack that
        # checked round 1 down and now faces a queen board and a bet.
        def render(rank: Rank, line: tuple[Action, ...]) -> str:
            return RANK_SYMBOL[rank] + ":" + "".join(ACTION_SYMBOL[a] for a in line)

        text = render(self.rank, self.betting[0])
        if self.board is not None:
            text += "|" + render(self.board, self.betting[1])
        return text

def all_infosets() -> tuple[InfoSet, ...]:
    """Every infoset in the game: 18 in round 1 and 270 in round 2.

    Round 1 is 3 own ranks x 6 decision lines. Round 2 multiplies that by the
    3 board ranks and by the 5 round-1 lines that reach a board, giving
    3 x 3 x 5 x 6 = 270. The 288 total is the number of ledgers a tabular
    solver allocates, and 96 of them are 3-wide.
    """
    round_one = tuple(
        InfoSet(rank=rank, board=None, betting=(line,))
        for rank in RANKS
        for line in ROUND_DECISION_LINES
    )
    round_two = tuple(
        InfoSet(rank=rank, board=board, betting=(opening, line))
        for rank in RANKS
        for board in RANKS
        for opening in ROUND_CLOSING_LINES
        for line in ROUND_DECISION_LINES
    )
    return round_one + round_two

# ---------------------------------------------------------------------------
# The pot and the showdown
# ---------------------------------------------------------------------------

def _round_contributions(line: tuple[Action, ...], bet_size: int) -> tuple[int, int]:
    """Chips (P0, P1) put into the pot during one round.

    After k raises the price of staying in is k * bet_size, so a CALL matches
    that price and a RAISE sets a new one. A folder stops paying where they
    stood, which is why folding to a re-raise still costs the raise already
    made — the source of the +/-3 payoff the referee reports.
    """
    paid = [0, 0]
    raises = 0
    for turn, action in enumerate(line):
        if action is Action.FOLD:
            break
        if action is Action.RAISE:
            raises += 1
        paid[turn % 2] = raises * bet_size
    return (paid[0], paid[1])

def _chips(winner: int, stake: int) -> tuple[float, float]:
    """The zero-sum payoff pair: `winner` collects `stake`, the other pays it.

    One place the sign is decided. Written inline at each call site it is two
    mirrored ternaries that read the same and mean opposite things, and getting
    one backwards flips a payoff without failing any type check.
    """
    amount = float(stake)
    return (amount, -amount) if winner == 0 else (-amount, amount)

def hand_strength(private: Card, board: Card) -> int:
    """How good one private card is against a given board.

    A card that pairs the board beats every unpaired card, so the ladder is
    J < Q < K < any pair. Both players pairing is impossible — a rank owns two
    cards, and if a player holds one and the board shows the other, there is
    nothing left for the opponent — so ties only ever happen between two equal
    unpaired ranks.
    """
    if private.rank == board.rank:
        return PAIR_STRENGTH
    return int(private.rank)

# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class NodeKind(StrEnum):
    """Which of the three things a node can be — the trichotomy the walk turns on.

    A tree walker asks this once per node and branches three ways: recurse on
    the deck's outcomes, recurse on the player's actions, or bank a payoff.
    Every accessor on `LeducState` belongs to exactly one kind and refuses at
    the other two, so the kind is a value rather than something each method
    re-derives from a pair of predicates.
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
class LeducState:
    """One node of the game tree: the deal, the board, and the betting so far.

    `betting` holds one line per round *begun*, so `((call, call), (raise,))`
    reads "checked through round 1, then bet on the board". Immutable — `apply`
    and `apply_chance` return new states, so a recursive walker never has to
    undo a move.

    The state is one of exactly three kinds, and every accessor that only makes
    sense at one kind raises at the others rather than returning a plausible
    wrong answer: `legal_actions`/`apply`/`infoset` at a decision node,
    `chance_outcomes`/`apply_chance` when the deck is due, `returns` at a
    terminal. Constructing a state validates that it is a position the rules
    can actually produce.
    """

    cards: tuple[Card, Card]
    board: Card | None = None
    betting: tuple[tuple[Action, ...], ...] = ((),)

    def __post_init__(self) -> None:
        # `Card` is an IntEnum, so a raw (0, 2) hashes and compares equal to
        # (JACK_A, QUEEN_A) and would slip past the checks below — but
        # `hand_strength` reads `.rank`, which a plain int lacks, so the
        # mistake surfaces at a showdown far from where it was made. A solver
        # that indexes deals by number converts here, at the boundary.
        if len(self.cards) != 2 or not all(isinstance(c, Card) for c in self.cards):
            raise ValueError(f"cards must be two Card members, got {self.cards}")
        if self.cards[0] == self.cards[1]:
            raise ValueError(f"cards must be two distinct cards, got {self.cards}")
        if self.board is not None and not isinstance(self.board, Card):
            raise ValueError(f"board must be a Card member, got {self.board!r}")
        if self.board in self.cards:
            raise ValueError(f"{self.board.name} was already dealt to a player")
        _validate_betting(self.betting, self.board)

    # -- which of the three kinds of node this is ---------------------------

    @property
    def round_index(self) -> int:
        """0 while the board is face down, 1 once it is up."""
        return len(self.betting) - 1

    @property
    def _line(self) -> tuple[Action, ...]:
        """The betting inside the round now in progress."""
        return self.betting[-1]

    @property
    def kind(self) -> NodeKind:
        """Decision, chance, or terminal — read off the current round's line.

        Three rules, in order. A fold ends the hand wherever it happens. A
        round still open is somebody's decision. A round closed with the stakes
        level hands over to the deck after round 1 and to the showdown after
        round 2 — which is the whole structural difference from rung 1, stated
        in one line.
        """
        if _is_fold(self._line):
            return NodeKind.TERMINAL
        if not _is_closed(self._line):
            return NodeKind.DECISION
        return NodeKind.TERMINAL if self.round_index == N_ROUNDS - 1 else NodeKind.CHANCE

    def is_terminal(self) -> bool:
        """True once a fold or a showdown has decided the hand."""
        return self.kind is NodeKind.TERMINAL

    def is_chance_node(self) -> bool:
        """True when round 1 closed without a fold and the board is due."""
        return self.kind is NodeKind.CHANCE

    def _require(self, kind: NodeKind) -> None:
        """Refuse an accessor that does not belong to this node's kind.

        The alternative is answering anyway: `returns()` at a decision node
        would report a pot nobody has won, and a walker would bank it. One
        guard, one message shape, three kinds.
        """
        if self.kind is not kind:
            raise ValueError(
                f"{self.betting} is a {self.kind} node, not a {kind} node; "
                f"{_WHO_ACTS[self.kind]}"
            )

    # -- chance -------------------------------------------------------------

    def remaining_deck(self) -> tuple[Card, ...]:
        """The cards nobody has seen: the deck minus the hands minus the board.

        The chance branch reads its outcomes from here rather than from a
        hard-coded four, so the same walk serves a game that turns more cards.
        """
        # `None` is not a Card, so an undealt board simply matches nothing.
        seen = {*self.cards, self.board}
        return tuple(card for card in DECK if card not in seen)

    def chance_outcomes(self) -> tuple[tuple[Card, float], ...]:
        """Each card the board could be, with its probability — uniform, 1/4."""
        self._require(NodeKind.CHANCE)
        survivors = self.remaining_deck()
        return tuple((card, 1.0 / len(survivors)) for card in survivors)

    def apply_chance(self, card: Card) -> LeducState:
        """The state after the deck turns `card` as the board, opening round 2.

        Dealing a card already in a hand is caught by the new state's own
        validation, not re-checked here.
        """
        self._require(NodeKind.CHANCE)
        return LeducState(cards=self.cards, board=card, betting=self.betting + ((),))

    # -- decisions ----------------------------------------------------------

    @property
    def current_player(self) -> int:
        """Whose turn it is. P0 opens both rounds, and players alternate."""
        self._require(NodeKind.DECISION)
        return len(self._line) % 2

    def legal_actions(self) -> tuple[Action, ...]:
        """What the player to act may do, in the fixed order ledgers index by."""
        self._require(NodeKind.DECISION)
        return legal_actions_for(self._line)

    def apply(self, action: Action) -> LeducState:
        """The state after the player to act takes `action`."""
        self._require(NodeKind.DECISION)
        if action not in self.legal_actions():
            raise ValueError(f"{action!r} is not legal after {self._line}")
        line = self._line + (action,)
        return LeducState(
            cards=self.cards, board=self.board, betting=self.betting[:-1] + (line,)
        )

    def infoset(self) -> InfoSet:
        """What the player to act can see, with the suits dropped."""
        self._require(NodeKind.DECISION)
        return InfoSet(
            rank=self.cards[self.current_player].rank,
            board=self.board.rank if self.board is not None else None,
            betting=self.betting,
        )

    # -- payoffs ------------------------------------------------------------

    def contributions(self) -> tuple[int, int]:
        """Chips (P0, P1) have put in across the whole hand, ante included."""
        paid = [ANTE, ANTE]
        for line, bet_size in zip(self.betting, BET_SIZES, strict=False):
            for player, chips in enumerate(_round_contributions(line, bet_size)):
                paid[player] += chips
        return (paid[0], paid[1])

    def returns(self) -> tuple[float, float]:
        """Chips won by (P0, P1), in units of the ante. Sums to zero.

        The loser pays exactly what they put in — the ante plus every bet they
        matched — which is the referee's +/-(1 + 2*b1 + 4*b2) written as a
        running total instead of a formula. A fold hands over the folder's own
        stake and returns the rest, so folding to a re-raise after raising once
        costs 3, not 1. A showdown has both stakes level by construction, and
        the board decides it: a pair beats every unpaired card, and two equal
        unpaired ranks split.
        """
        self._require(NodeKind.TERMINAL)
        paid = self.contributions()

        if _is_fold(self._line):
            loser = (len(self._line) - 1) % 2
            return _chips(winner=1 - loser, stake=paid[loser])

        # Only a matched call reaches a showdown, so the stakes are level and
        # either player's contribution is the amount at risk.
        if paid[0] != paid[1]:
            raise AssertionError(f"showdown with unmatched stakes {paid}: {self.betting}")
        strengths = tuple(hand_strength(card, self.board) for card in self.cards)
        if strengths[0] == strengths[1]:
            return (0.0, 0.0)
        return _chips(winner=0 if strengths[0] > strengths[1] else 1, stake=paid[0])
