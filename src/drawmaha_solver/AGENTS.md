# `src/drawmaha_solver/` — the solver package

One subpackage per rung (`rps/`, `kuhn/`, `leduc/`, `minidrawmaha/`,
`deepcfr/`) and three shared modules:

- `regret_matching.py` — the ledger every rung trains. The learning rule never
  changes as the games grow; only what feeds it does.
- `play_session.py` — the terminal-table bookkeeping `leduc-play` and
  `minidraw-play` share: quitting, the per-seat tally, the verdict line.
- `plotting.py` — the house figure style, three categorical inks and neutral
  everything else.

## The layering inside a rung

Every rung keeps the same layers, and the README section for the rung explains
each one: the rules (`game.py`, with no solver in it) → the ledgers
(`infoset_table.py`) → the solver (`cfr.py` at rungs 1 and 2, `mccfr.py` and
`lockstep.py` at rung 3) → the meter (`exploitability.py`) → the entry points
(`analysis.py`, `play.py`, `exploit_server.py`). Entry points are registered in
`pyproject.toml` under `[project.scripts]`; a new one is added there in the
same change.

## Rules

- `exploitability.py` imports `game.py` and never the solver. See the root
  file for why; here is where it is enforced by convention, so a review of a
  grader starts with its import block.
- Every loop over a ledger's columns pairs `enumerate(legal_actions())` with
  the ledger; nothing else keeps a column's meaning (column 0 is *call* at the
  open and *fold* facing a raise) in step with the rules.
- Rungs 1 and 2 run unseeded on purpose: vanilla CFR enumerates the whole
  tree, two runs agree bit for bit, and `figures/rung1/`, `figures/rung2/` are
  regression fingerprints. Rungs 3 and 4 sample, so they are seeded, and every
  sampled quantity is a function of (seed, iteration, seat, walk index) and
  never of how many workers ran it.
- The exploit servers (`kuhn/exploit_server.py`, `leduc/exploit_server.py`)
  are stdlib HTTP and import the rung's own modules only; the page POSTs
  locked rows and draws what Python returns. No CFR is ported to the browser.
- Convergence numbers quoted in a docstring or the README are recomputed by a
  test from the committed data, not transcribed from a session.
