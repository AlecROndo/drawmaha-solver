/**
 * The URL hash as the page's address: which game, which tab, and for
 * mini-drawmaha which strategy — `#mini/range/deep-pi`, `#full/versus`.
 *
 * The hash is parsed leniently and written strictly. A segment that names
 * nothing falls back to the default rather than throwing, because a link
 * written before a strategy existed (or with a typo) should still open the
 * page; and the strategy segment is written only for mini-drawmaha, since
 * the full game has one illustrative policy and no switch.
 */

import { DEFAULT_STRATEGY, type StrategyId, isStrategyId } from '../mini/strategies'

export const GAME_IDS = ['mini', 'full'] as const
export type GameId = (typeof GAME_IDS)[number]
export const TAB_IDS = ['range', 'versus'] as const
export type TabId = (typeof TAB_IDS)[number]

export interface View {
  game: GameId
  tab: TabId
  strategy: StrategyId
}

export const isGameId = (v: string): v is GameId => (GAME_IDS as readonly string[]).includes(v)
export const isTabId = (v: string): v is TabId => (TAB_IDS as readonly string[]).includes(v)

/** The view a hash names, with or without its leading `#`; every unknown segment is the default. */
export function parseHash(hash: string): View {
  const [g = '', t = '', s = ''] = hash.replace(/^#/, '').split('/')
  return {
    game: isGameId(g) ? g : 'mini',
    tab: isTabId(t) ? t : 'range',
    strategy: isStrategyId(s) ? s : DEFAULT_STRATEGY,
  }
}

/** The hash for a view: `#game/tab`, plus `/strategy` when the game has a switch. */
export function formatHash(view: View): string {
  const base = `#${view.game}/${view.tab}`
  return view.game === 'mini' ? `${base}/${view.strategy}` : base
}
