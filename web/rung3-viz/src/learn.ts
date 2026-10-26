/**
 * The page's address: which tab, and for the Learn tab which view of the
 * range interface and which strategy — `#research`, `#play`,
 * `#learn/versus/deep-pi`.
 *
 * Parsed leniently, written strictly: a segment that names nothing falls
 * back to the default rather than throwing, so an old `#play` link and a
 * typo both still open the page. The Learn tab's view and strategy are the
 * same ids `/solver` uses in its own hash (`solver-viz/src/ui/hash.ts`), so
 * a spot found here can be opened there by swapping the prefix.
 */

import { DEFAULT_STRATEGY, type StrategyId, isStrategyId } from '../../solver-viz/src/mini/strategies'
import { type TabId as LearnView, isTabId as isLearnView } from '../../solver-viz/src/ui/hash'

export type { LearnView }

export const TAB_IDS = ['research', 'play', 'learn'] as const
export type TabId = (typeof TAB_IDS)[number]

export interface PageView {
  tab: TabId
  /** the Learn tab's view: your range, or range vs range */
  view: LearnView
  strategy: StrategyId
}

export const isTabId = (v: string): v is TabId => (TAB_IDS as readonly string[]).includes(v)

/** The view a hash names, with or without its leading `#`; every unknown segment is the default. */
export function parseHash(hash: string): PageView {
  const [t = '', v = '', s = ''] = hash.replace(/^#/, '').split('/')
  return {
    tab: isTabId(t) ? t : 'research',
    view: isLearnView(v) ? v : 'range',
    strategy: isStrategyId(s) ? s : DEFAULT_STRATEGY,
  }
}

/** The hash for a view: the tab alone, plus `/view/strategy` for the Learn tab, which has both. */
export function formatHash(page: PageView): string {
  return page.tab === 'learn' ? `#learn/${page.view}/${page.strategy}` : `#${page.tab}`
}
