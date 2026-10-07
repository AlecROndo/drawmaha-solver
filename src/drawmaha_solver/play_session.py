"""The bookkeeping every heads-up terminal table shares: quitting, the tally, the verdict.

Rung 2's `leduc-play` and rung 3's `minidraw-play` seat a human against a
fixed strategy for as many hands as they like, alternating seats, and the
games differ in everything except what happens around a hand. This module is
that part. A SESSION is the run of hands from the first deal to the human's
`q`; it ends by raising `QuitGame` from whatever prompt the `q` was typed at,
so the hand in progress is abandoned rather than scored. The `Scoreboard`
tallies the finished hands from the human's side, per seat as well as in
total, because neither game is symmetric: each seat has its own value, and
only the total across alternating seats is read against zero. `verdict` is
the one line a player reads at the end of a hand.

What a seat is worth, and therefore what each per-seat line is read against,
is the game's business, so the reports that print those targets stay with the
games.
"""

from __future__ import annotations

from dataclasses import dataclass, field

class QuitGame(Exception):
    """Raised by a prompt when the human asks to stop; the hand in progress is not scored."""

def verdict(chips: float) -> str:
    """How the hand ended, from the human's seat.

    Three outcomes, not two: both games can deal a tie, and a showdown
    between equal hands splits the pot. Reporting that as "bot wins 0" would
    call a chop a loss on the one line of the hand the player actually reads.
    """
    if chips > 0:
        return f"you win {chips:.0f}"
    if chips < 0:
        return f"bot wins {-chips:.0f}"
    return "split pot"

# Mutable on purpose, unlike the value types elsewhere: it is the session's
# running tally, updated once per finished hand.
@dataclass(slots=True)
class Scoreboard:
    """Chips and hands from the human's seat, across alternating seats.

    Kept per seat as well as in total, because the total is the only number
    that is supposed to approach zero. Each seat separately approaches its own
    value, and seeing the two straddle zero by opposite amounts is what shows
    the alternation is cancelling a seat edge rather than hiding one.
    """

    chips: float = 0.0
    hands: int = 0
    seat_chips: list[float] = field(default_factory=lambda: [0.0, 0.0])
    seat_hands: list[int] = field(default_factory=lambda: [0, 0])

    def record(self, *, human_seat: int, returns: tuple[float, float]) -> None:
        """Bank one finished hand, taking the human's side of the payoff."""
        self.chips += returns[human_seat]
        self.seat_chips[human_seat] += returns[human_seat]
        self.seat_hands[human_seat] += 1
        self.hands += 1

    @property
    def per_hand(self) -> float | None:
        """Chips per hand, or None before any hand has been played."""
        if self.hands == 0:
            return None
        return self.chips / self.hands

    def per_hand_in_seat(self, seat: int) -> float | None:
        """Chips per hand in `seat`, or None before that seat has played."""
        if self.seat_hands[seat] == 0:
            return None
        return self.seat_chips[seat] / self.seat_hands[seat]
