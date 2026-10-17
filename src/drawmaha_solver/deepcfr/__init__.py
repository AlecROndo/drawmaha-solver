"""Deep CFR on mini-drawmaha: the network half of rung 4.

Rung 3 solves mini-drawmaha with one ledger per infoset, 6.2 million of them
in `PackedTable`. This package replaces the ledgers' REGRET half with a
network that reads what the player can see — the 136 features of
`minidrawmaha.features` — and writes seven outputs, one per `Action`. The
walk itself does not change: `materialise` writes each net's outputs into
the table's regret slots, so rung 3's `traverse` plays the net's strategy
through the same `RegretMatcher.strategy()` it always read.

Three modules, one concern each:

- `net`: the MLP, chunked prediction into NumPy, and the save file.
- `materialise`: two seats' nets into `cumulative_regret`, Deep CFR's
  eq. (3) obtained by what is written rather than by changing the ledger.
- `fit`: the from-scratch refit on (row, t, target) arrays.

Everything here needs torch, which the default install leaves out: install
the `deep` extra (`uv sync --extra deep`). Nothing outside this package
imports it, so the default test suite runs without torch.
"""
