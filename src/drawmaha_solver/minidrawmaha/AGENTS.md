# `minidrawmaha/` — rung 3: the game, its tabular solve, and the answer

Three subpackages, and one module that stays alone on purpose:

- `rules/` — what the game is, with no solver in it. `cards` (the 15-card
  deck and the joint canonical suit relabelling), `hands` (both halves of the
  pot as score tuples), `game` (the hand as a state machine, `InfoSet`,
  `canonical_picture`), `enumeration` (every infoset, 6,220,050 of them,
  reached through the 141 public points), and `features` (rung 4's 136-float
  view of an infoset, built from these rules alone).
- `solver/` — the learner and its memory. `regret_rules` (vanilla, LCFR, CFR+,
  DCFR, and the averaging columns), `packed_table` (the only table: four flat
  arrays), `infoset_table` (a ledger window over it), `mccfr` (external
  sampling, one hand per seat), `lockstep` (W hands per seat against one
  frozen σ; the reference), `parallel` (the same in W processes over shared
  memory), `lockstep_run` (resume, grid checkpoints, progress).
- `answer/` — everything built on the solved game. `strategy` (the frozen
  LCFR average, a release asset pinned by SHA-256), `analysis`
  (`minidraw-analysis`, writes `figures/rung3/`), `play` (`minidraw-play`),
  `deal_pack` (`minidraw-deals`, the `/rung3` trainer's pack), `range_export`
  (`minidraw-ranges`, the `/solver` export), `releases` (the catalogue both
  exporters read: every released strategy, its digest, and where its grade
  is recorded).
- `exploitability.py` — the exact meter. It imports `rules/` only, nothing
  from `solver/`, and stays at this level to make that visible.

## Facts the code does not show on sight

- **The key is the joint canonical suit relabelling of (hole, discarded,
  ordered board).** The board order is kept deliberately: round 1 was played
  against the first card alone, and that carries information. A checkpoint
  from the earlier unordered key (3,142,290 rows) is refused on load; do not
  add support for it.
- **The key is still coarser than the responder's real infoset** (with a rank
  held twice, "throw the low one" picks a physical card by the relabelling's
  tie-break). The grader therefore carries vectors over physical states and
  takes its max per physical entry. Never group the responder's max by the
  table's key; it under-reports by about 0.012 chips per hand.
- **A traverser row can be banked twice in one traversal** (about one
  traversal in a thousand, through the relabelling). "At most once per
  iteration" is false here, so DCFR defers its discount to the row's next
  visit and CFR+ floors after every bank. Do not "simplify" either to the
  textbook once-per-iteration form.
- **Lockstep is a determinism contract.** `parallel` must equal `lockstep`
  bit for bit at the same W, and the rows one worker contributes depend on
  (seed, t, seat, k) and never on W. The tests hold both; keep them true.
- **Checkpoints are flat `.npz` files written with `allow_pickle=False`**, and
  record the rule and the averaging columns they banked; a resume refuses a
  different W or rule by name.
- **The release asset** (tag `rung3-lcfr-2m`, file `strategy-lcfr-2m.npz`) is
  the rung's answer. Its digest is pinned in `answer/strategy.py`, catalogued
  in `answer/releases.py` and recorded again in the `/solver` export
  `web/solver-viz/public/mini/index.json`; re-freezing the strategy changes
  all three in one PR. The `/rung3` trainer's pack
  (`web/rung3-viz/public/deals/index.json`) records the digest of the
  strategy it deals against, which since the pot-normalised arm is rung 4's
  Deep CFR exact average, not this one.
- **Sampling is sparse where play is rare.** Of the 6.2 million rows, about
  2.7 million were never banked by the 2M-iteration LCFR run and 3.5 million
  read exactly uniform; nearly all sit in round 2. A human who deviates walks
  the bot into them. More iterations, a different average, or rung 4's nets
  are the remedies, not a patch to the lookup.
- **The rung's spec lives outside the repo**, on Alec's machine at
  `~/Desktop/Claude/poker/rung3-delta/rung3-delta-plan.md` with the census
  scripts beside it. The deck, the throw cap, the dropped face-up draw-one
  rule and the shrink ladder were decided there; when a ranking, a count or a
  rule looks wrong, check it before changing the code.

## Costs to know before running anything

- The packed table is about 0.3 GB; the exact grader holds about 1.5 GB of
  showdown tables and a grade takes about two minutes.
- Streaming all 6.2 million keys through `all_infosets()` takes about fifteen
  seconds. Tests that do it, grade the whole game, or calibrate on Leduc for
  a million iterations are gated by `MINIDRAWMAHA_FULL_TABLE`,
  `MINIDRAWMAHA_FULL_GRADE` and `MINIDRAWMAHA_FULL_CALIBRATION`.
- The race's checkpoints, 555 MB each, live on the Modal volume
  `drawmaha-rung3`, never in git.
