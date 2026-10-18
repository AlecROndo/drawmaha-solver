"""Deep CFR on mini-drawmaha: rung 4.

This package replaces rung 3's per-infoset regret ledgers with a network, and
keeps rung 3's walk. PR 3 lands the data side: `reservoir`, Vitter's Algorithm
R over (row, t, target) rows, and `collect`, K lockstep walks for one seat on W
processes that return the rows a phase offers the memories. Neither imports
torch, so the default test suite runs them.
"""
