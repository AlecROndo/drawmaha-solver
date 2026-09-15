# Rung-2 visualizer: the action timeline dashboard

Date: 2026-09-15 · Status: approved (interactive design loop, Option 1 deepened)

## What this is

`/rung2` — the Leduc solver's dashboard. One view: an action timeline you walk
and rewind, over two seat panels showing what each rank's whole range does at
the current node. Everything is infoset-based, range versus range; the page
never shows a specific holding. Every number is output of the Python solver,
delivered by a committed `leduc-analysis --json` export — the page re-implements
no solver math.

The design was chosen from five interactive options and iterated to approval;
the reference prototype (kept outside the repo) is
`/Users/alec/Desktop/Claude/poker/rung2-viz-options/option1-timeline.html`,
built by `build_option1.py` in the same folder. The repo implementation is the
source of truth from this point on.

## Page anatomy (top to bottom)

1. **Header** — rung kicker, serif title, short prose, and the certification
   line: iterations, exploitability of the average strategy (chips/hand), game
   value vs the exact LP value. All read from the export, never hardcoded.
2. **Fig. 1, the timeline panel** — the site's ladder motif applied to a hand:
   - One station per event: `deal`, each action, the board card, and (at the
     frontier) a ghost `to act` slot.
   - Stations are **16px boxes** (2px radius): filled oxblood when played,
     dimmed hollow when explored-but-ahead of the cursor, dashed hollow for the
     ghost. The current station is framed by four **crop marks** (surveyor's
     "you are here"), not a ring.
   - Rail: 2px hairline full-width, 3px solid as far as the cursor, both
     pixel-positioned from the fixed station width.
   - Stations are `flex: 0 0 150px` and never shrink; the strip scrolls
     horizontally inside the panel.
   - Below each station: actor label (P0/P1/round), the action word, the pot
     after it. Board stations render their rank in the serif face.
   - **Next-action strip** under the rail: the legal actions at the cursor,
     round-1 actions weighted by *reach* (how often play arrives there, shown
     as a percentage on the button); board turns show the uniform 1/3. The
     on-path button carries an outline; when the cursor sits behind explored
     stations, a fork note warns "a different choice here replaces the N steps
     ahead."
3. **Two seat panels** (`P0` / `P1`, side by side, stacking under 900px) — one
   row per rank (J, Q, K), each row a full-width **mix bar** of that rank's
   mixed strategy at the relevant infoset: bet/raise in oxblood on the LEFT,
   check/call in sand to its right, fold as a hatched tail. Read-only in this
   PR. Number strip under each bar reads in the same left-to-right order as
   the bar. The acting seat shows its live spot; the waiting seat shows the
   last spot it acted at, labeled "took <action> · N% of the time."
4. **Legend** — swatches in bar order (bet/raise, check/call, fold).

## The state model (the part the user cared most about)

```
PATH: the whole explored line, a list of steps {k:'a'|'b', v}
CUR:  cursor into PATH — where the reader is looking
```

- Clicking a station moves `CUR` only. **Rewinding never deletes stations**;
  walking forward again revisits the same nodes.
- Taking the action already at `PATH[CUR]` advances the cursor (no change to
  PATH). Taking a **different** action truncates at the cursor and appends —
  the only operation that drops explored stations, and it is announced by the
  fork note before it can happen.
- Derived state `{l1, board, l2}` is recomputed from `PATH[0..CUR]` on render.

## Data contract: `leduc-analysis --json`

Mirrors rung 1 (`kuhn-analysis --json` → committed
`web/rung1-viz/src/data/solve.json`). New flag on `leduc/analysis.py`; export
committed as `web/rung2-viz/src/data/solve.json` (~50 KB). Shape:

```jsonc
{
  "iterations": 2000,                 // of this solve
  "exploitabilityAverage": 0.0071,    // chips/hand, final average strategy
  "exploitabilityCurrent": 0.0288,    // the cycling iterate's, for contrast
  "gameValue": -0.0861,               // chips/hand to P0, this run
  "gameValueExact": -0.0856,          // the LP answer already pinned in tests
  "strategy": {                       // all 288 infosets, final average
    "J:": {"c": 0.91, "r": 0.09},     // keys are str(InfoSet): "J:cc|Q:r"
    ...                               // values keyed f/c/r, legal actions only,
  }                                   // each row summing to 1
}
```

No per-checkpoint frames: playback is out of scope (below), and this keeps the
bundle small. The strategy keys are exactly `str(InfoSet)`, so the TypeScript
key-builder must reproduce that format (`rank + ':' + line`, with
`|board + ':' + line2` after the turn) — pinned by a vitest against fixtures
lifted from the export.

## Frontend logic ported from the prototype (all pure, all tested)

`web/rung2-viz/src/leduc.ts`: `legal(line)`, `closed(line)`, `keyFor(...)`,
`potAfter(l1, board, l2)` (per-seat contributions — an uncalled bet is one
seat's chips), `reach(l1)` (product over prefix of mean action probability
across ranks), `stateAt(path, n)`, and the PATH/CUR transition functions
(`jump`, `advance`). React holds `{path, cur}` in one state; components render
from derived state.

Not ported: the prototype's dead `SPOTS` index, its lock/edit UI, its
hardcoded "33%"/"800-iteration" strings (real values come from the export).

## App conventions

`web/rung2-viz` mirrors `web/rung1-viz`: Vite + React + TS, same package
scripts, oxlint config, self-hosted font strategy, `site.tsx` identity rail
with the ladder nav (rung 2 marked current), and the duotone token sheet — with
this page's sharpened radii (5px panels / 3px buttons / 2px bars). Rung 0/1
keep their existing radii; site-wide sharpening is a separate decision.

## Deploy wiring

- `vercel.json`: add `{"source": "/rung2", "destination": "/rung2/index.html"}`.
- `scripts/vercel_build.sh`: `rungs=(rung0 rung1 rung2)`.
- `api/index.py`: Rung 2's ladder station flips from `todo` to done with a
  link to `/rung2`; the "next" pointer moves to rung 3. Existing cover-page
  tests (`test_every_link_goes_somewhere_that_exists`) and the vercel-config
  guard (`test_every_rung_the_build_script_emits_has_a_rewrite`) pick the new
  rung up automatically; extend where they don't.

## Testing

- **pytest** (`tests/leduc/test_analysis_json.py`): export has all 288 keys,
  rows sum to 1 over legal actions only, certification numbers present and
  consistent with the pinned LP value.
- **vitest** (`web/rung2-viz/src/leduc.test.ts`): state model (rewind
  preserves PATH length; same-action walk-on; different-action fork truncates),
  `potAfter` including the uncalled-bet case, `reach` weights, `keyFor` against
  real export keys, `legal`/`closed` tables.
- Suite stays green: `uv run pytest -q` from repo root.

## Non-goals (deferred, deliberately)

- **Lock/edit a row** — honest locking needs downstream re-solving
  (`leduc/exploiter.py` + server, the rung-1 PR #14 equivalent). Ships in a
  later PR with real numbers; no fake row-local lock on the public site.
- **Solve playback / scrubber** and the **Field tab** (all 288 spots at once)
  — later PRs; the export gains per-checkpoint frames only when they land.
- **open_spiel** never enters pyproject.toml (house rule).
