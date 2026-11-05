import type { Action } from '../sim/game'
import { actionName } from './format'

/** An action named in ink. Colour on this page belongs to the average and the iterate, not to an action. */
export function Chip({ a }: { a: Action }) {
  return <span className="act">{actionName(a)}</span>
}
