"""Mini-drawmaha: rung 3 of the validation ladder.

A shrunken Drawmaha that keeps the two things the full game is made of — a
**draw** and a **split pot** — on a deck small enough that the exact answer is
still computable. Fifteen cards, three private each, two board cards, two
betting rounds, one sequential draw of at most two cards, and a pot that pays
half on the best 3-card hand held and half on the best 4-card hand made with
both board cards.

The rung exists because it is the last one where "how far is this strategy from
optimal?" has an exact answer. Rung 4's neural solver needs a game whose truth
is known to be graded against, and a network here would leave that grader
ungraded.
"""
