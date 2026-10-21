"""What mini-drawmaha is, with no solver in it.

The deck and the suit relabelling (`cards`), both halves of the pot
(`hands`), the hand as a state machine with its `InfoSet` (`game`), every
infoset the game has (`enumeration`), and the 136-float view of one that
rung 4's nets read (`features`). Nothing here knows a ledger exists; the
grader imports this package and nothing else, so that it cannot share a bug
with the solver it grades.
"""
