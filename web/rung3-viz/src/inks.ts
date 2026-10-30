/**
 * Colour inside the figures, as CSS variables declared in `index.css` over
 * the theme's series and action tokens: the four regret rules keep four
 * inks across every figure (LCFR, the winner, in the accent), and an
 * action's word picks the action's colour. Kept out of the component files
 * so fast refresh has only components to refresh.
 */

import type { Rule } from './research'

export const RULE_INK: Record<Rule, string> = {
  vanilla: 'var(--rule-vanilla)',
  lcfr: 'var(--rule-lcfr)',
  cfrplus: 'var(--rule-cfrplus)',
  dcfr: 'var(--rule-dcfr)',
}

/**
 * A column's colour by its word, the site's action triplet: the bet and the
 * raise are the accent, the call green, the check and the fold blue. At the
 * draw, standing pat is the quiet choice (blue) and the three throws are the
 * accent at three depths, low to top.
 */
export function inkFor(column: string): string {
  switch (column) {
    case 'bet':
    case 'raise':
      return 'var(--act-bet)'
    case 'call':
      return 'var(--act-call)'
    case 'check':
    case 'fold':
    case 'stand pat':
      return 'var(--act-check)'
    case 'throw low':
      return 'var(--throw-low)'
    case 'throw mid':
      return 'var(--throw-mid)'
    case 'throw top':
      return 'var(--throw-top)'
    default:
      return 'var(--ink-3)'
  }
}
