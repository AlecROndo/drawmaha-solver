# `solver-viz/` — the range interface at `/solver`

Two games behind one interface, and they are not equally real:

- **Mini-drawmaha is exact.** `public/mini/` is the export written by
  `uv run minidraw-ranges` from rung 3's frozen strategy: one file per
  public decision point, plus the key tables per shape. The browser rebuilds
  the actor's whole range at a spot from it, weighted by the actor's own
  action probabilities along the line. `src/mini/canonical.ts`,
  `src/mini/ranking.ts`, `src/mini/grammar.ts` and `src/mini/points.ts` are
  ports of the Python key, hand ranking, betting grammar and public tree,
  pinned to `src/mini/fixtures.json`, which the Python side generates and
  checks (`tests/minidrawmaha/test_range_export.py`). When the Python key or
  grammar changes, re-export the data and regenerate the fixtures; never patch
  a port to agree with stale data.
- **Full Drawmaha is the interface ahead of rung 4's network.** The range's
  composition (`src/engine/sample.ts`) is sampled and real; the **action mix
  is illustrative**, a smooth function of each hand's two equities in
  `src/engine/policy.ts`. That file is the one that changes when the network
  exists. Every view drawing those numbers says so in its status line.

`src/ui/game.ts` is all the shared views know about a game: the grid's two
axes, how many cards a hand holds, and how a deck is laid out. The same
grid, ledger, deck and draw board draw both games through it; a view that
needs more than that is asking for a per-game component, not a wider
interface.
