"""Deep CFR on mini-drawmaha: the network half of rung 4.

Rung 3 solves mini-drawmaha with one ledger per infoset, 6.2 million of them
in `PackedTable`. This package replaces the ledgers' REGRET half with a
network that reads what the player can see — the 136 features of
`minidrawmaha.rules.features` — and writes seven outputs, one per `Action`. The
walk itself does not change: `materialise` writes each net's outputs into
the table's regret slots, so rung 3's `traverse` plays the net's strategy
through the same `RegretMatcher.strategy()` it always read.

Eight modules, one concern each:

- `net`: the MLP, chunked prediction into NumPy, and the save file.
- `materialise`: two seats' nets into `cumulative_regret`, Deep CFR's
  eq. (3) obtained by what is written rather than by changing the ledger.
- `fit`: the from-scratch refit on (row, t, target) arrays.
- `reservoir`: the memories those arrays come from — Algorithm R over
  (row, t, target) rows, a counter-based stream per phase, an exact resume.
- `collect`: K lockstep walks for one seat on W spawned processes, returned
  as the rows a phase offers the memories.
- `train`: the loop that calls them in order, iteration after iteration —
  its resume state, grid marks, progress log and graceful stop — behind a
  `Learner` seam, so the loop never names torch.
- `average`: the exact average strategy (Single Deep CFR) — own reach pushed
  through a predecessor table every phase, `t · reach · σ` summed per column
  at the loop's observer hook, and frozen for the grader.
- `policy`: Π, the original Deep CFR's average — a fresh net fit on the
  policy memory, frozen by running it over every key.

`net`, `materialise`, `fit` and `policy` need torch, which the default
install leaves out: install the `deep` extra (`uv sync --extra deep`).
`reservoir`, `collect`, `train` and `average` are NumPy only (`train` and
`policy` import torch inside their functions, when a real run asks for it).
Nothing outside this package imports torch, so the default test suite runs
without it.
"""
