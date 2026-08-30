import type { Action } from '../sim/game'
import { ACTION_COLORS, actionName } from './format'

/** Action name in text ink with a colored dot carrying identity. */
export function Chip({ a, label }: { a: Action; label?: string }) {
  return (
    <span className="chip">
      <span className="dot" style={{ background: ACTION_COLORS[a] }} />
      {label ?? actionName(a)}
    </span>
  )
}
