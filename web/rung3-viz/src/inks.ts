/**
 * Identity colour inside the figures, as CSS variables declared in
 * `index.css`: the four regret rules keep four inks across every figure, and
 * an action's word picks the action's ink. Kept out of the component files so
 * fast refresh has only components to refresh.
 */

import type { Rule } from './research'

export const RULE_INK: Record<Rule, string> = {
  vanilla: 'var(--rule-vanilla)',
  lcfr: 'var(--rule-lcfr)',
  cfrplus: 'var(--rule-cfrplus)',
  dcfr: 'var(--rule-dcfr)',
}

/** A column's ink by its word: the aggressive act is pink, the quiet one yellow, giving up dim. */
export function inkFor(column: string): string {
  switch (column) {
    case 'bet':
    case 'raise':
      return 'var(--act-bet)'
    case 'check':
    case 'call':
    case 'stand pat':
      return 'var(--act-check)'
    case 'fold':
      return 'var(--act-fold)'
    case 'throw low':
      return 'var(--throw-low)'
    case 'throw mid':
      return 'var(--throw-mid)'
    case 'throw top':
      return 'var(--throw-top)'
    default:
      return 'var(--panel-dim)'
  }
}
