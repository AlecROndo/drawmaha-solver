/**
 * The Learn tab: the range interface over mini-drawmaha, the same view
 * `/solver` shows, hosted here because this is the rung whose strategy it
 * analyses. The component is imported from `web/solver-viz` rather than
 * copied — one source, one stylesheet (`App.tsx` imports it) — and reads the
 * exports from under `/solver/` (`EXPORTS_BASE`), so the data lives once on the site. Under
 * this app's own dev server that path is not served; see the site through
 * `scripts/site/vercel_build.sh` + `scripts/site/serve_site.py` to drive it.
 *
 * The two views, your range and range vs range, are the tabs `/solver` has;
 * the strategy switch (tabular LCFR, Deep CFR Π, Deep CFR exact) is inside
 * the view. Both live in this page's hash (`learn.ts`).
 */

import type { StrategyId } from '../../../solver-viz/src/mini/strategies'
import { MiniRange } from '../../../solver-viz/src/ui/MiniRange'
import type { LearnView } from '../learn'

/** Where `/solver` serves the exports; absolute, so it is the same from `/rung3`. */
export const EXPORTS_BASE = '/solver/'

const VIEWS = [
  { id: 'range', label: 'Your range', blurb: 'the grid, the ribbon, the ledger, the deck, the draw' },
  { id: 'versus', label: 'Range vs range', blurb: 'their range beside yours, and the hand as ranges' },
] as const

export function LearnTab({
  view,
  strategy,
  onView,
  onStrategy,
}: {
  view: LearnView
  strategy: StrategyId
  onView: (v: LearnView) => void
  onStrategy: (s: StrategyId) => void
}) {
  return (
    <>
      <div className="views" role="tablist" aria-label="Learn views">
        {VIEWS.map((entry) => (
          <button key={entry.id} role="tab" aria-selected={view === entry.id} className={view === entry.id ? 'view current' : 'view'} onClick={() => onView(entry.id)}>
            <b>{entry.label}</b>
            <span>{entry.blurb}</span>
          </button>
        ))}
      </div>
      <MiniRange tab={view} strategy={strategy} onStrategy={onStrategy} base={EXPORTS_BASE} />
    </>
  )
}
