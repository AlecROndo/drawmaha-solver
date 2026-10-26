# `solver-viz/` — the range interface at `/solver`

Two games behind one interface, and they are not equally real:

- **Mini-drawmaha is exact, and three strategies sit behind one switch.**
  `public/mini/`, `public/mini-deep-pi/` and `public/mini-deep-exact/` are
  the exports written by `uv run minidraw-ranges --release tabular|deep-pi|deep-exact`
  from the released strategies (`RELEASES` in
  `src/drawmaha_solver/minidrawmaha/answer/range_export.py`: rung 3's
  tabular LCFR, and rung 4's Deep CFR at iteration 500 as Π and as the
  exact average): one file per public decision point, plus the key tables
  per shape. The three exports share their key tables and point walk byte
  for byte and differ only in the chunks, which is what lets the switch
  (`src/mini/strategies.ts`, the strategy as the hash's third segment in
  `src/ui/hash.ts`) change the export under a spot while the line, the held
  cards and the filters stay. The browser rebuilds the actor's whole range
  at a spot from the export, weighted by the actor's own action
  probabilities along the line. Each `index.json` names its strategy and
  carries its grade, copied from the committed grades file at export time;
  the status line is read from that manifest, never typed in TypeScript, and
  the Python tests hold the copy to the file. `src/mini/canonical.ts`,
  `src/mini/ranking.ts`, `src/mini/grammar.ts` and `src/mini/points.ts` are
  ports of the Python key, hand ranking, betting grammar and public tree,
  pinned to `src/mini/fixtures.json`, which the Python side generates and
  checks (`tests/minidrawmaha/test_range_export.py`). The fixtures depend on
  the rules alone: the exporter refuses to overwrite them unless told
  (`--write-fixtures`). When the Python key or grammar changes, re-pin the
  fixtures on purpose and re-export every strategy; never patch a port to
  agree with stale data.
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
