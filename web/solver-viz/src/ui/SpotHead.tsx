/**
 * The head of the centre column, over the grid: who is acting and on what
 * in one serif line with its detail beside it, then the whole range's mix
 * as one bar with a key whose entries are buttons.
 *
 * The head's controls are the way into the secondary views: the mix bar
 * opens the three panes (one map per action) and each action in the key
 * sorts the ledger by that action. At a draw there is no fold / call / pot
 * mix to draw, so the caller passes `all = null` and its own key as
 * `children`. A load error, when there is one, is printed under the title.
 */

import type { Cell } from '../engine/aggregate'
import type { Node } from '../engine/policy'
import { MixBar, actionWord, pct } from './bars'

export type Opened = null | 'panes'
export type SortKey = null | 'f' | 'c' | 'p'

export function SpotHead({
  title,
  sub,
  error,
  node,
  all,
  opened,
  onOpen,
  sort,
  onSort,
  children,
}: {
  title: string
  sub: React.ReactNode
  error?: string | null
  node: Node
  /** the whole range's cell (mask-free), or null where the decision is not a bet */
  all: Cell | null
  opened: Opened
  onOpen: (o: Opened) => void
  sort: SortKey
  onSort: (k: SortKey) => void
  children?: React.ReactNode
}) {
  const keys = node === 'facing' ? (['f', 'c', 'p'] as const) : (['c', 'p'] as const)
  return (
    <div className="spothead">
      <p className="who">
        {title}
        {sub && <small> · {sub}</small>}
      </p>
      {error && <p className="err">{error}</p>}
      {all ? (
        <div className="mixblock">
          <button
            type="button"
            className={`hbtn mixbtn ${opened === 'panes' ? 'on' : ''}`}
            onClick={() => onOpen(opened === 'panes' ? null : 'panes')}
            aria-expanded={opened === 'panes'}
            aria-label="the whole range's mix; opens one map per action"
          >
            <MixBar mix={all} h={10} />
          </button>
          <div className="acts">
            {keys.map((k) => (
              <button key={k} type="button" className={`hbtn act-${k} ${sort === k ? 'on' : ''}`} onClick={() => onSort(sort === k ? null : k)} aria-pressed={sort === k} title={`sort the ledger by ${actionWord(k, node)}`}>
                <i />
                {actionWord(k, node)} <b>{pct(all[k])}</b>
              </button>
            ))}
          </div>
        </div>
      ) : (
        children
      )}
    </div>
  )
}
