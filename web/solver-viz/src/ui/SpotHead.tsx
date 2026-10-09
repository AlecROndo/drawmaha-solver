/**
 * The fixed header every range view sits under: who is acting and on what,
 * the whole range's mix, the controls that open the secondary views, and
 * how much of the range the current filter has left.
 *
 * The header's controls are the way into the other studies: the mix bar
 * opens the three panes (one map per action), each action button opens the
 * ledger sorted by that action, and the cards button opens the deck. The
 * board field is the one text input on the page; a board that does not
 * parse is an error printed under the field, and the range on screen stays
 * the last good one.
 */

import { useState } from 'react'

import type { Cell } from '../engine/aggregate'
import type { Card } from '../engine/cards'
import type { Node } from '../engine/policy'
import { CardFace, MixBar, MixKey, actionWord, count, pct } from './bars'
import { FULL_GAME } from './game'
import type { Cursor } from './Score'

export type Opened = null | 'panes' | 'deck'
export type SortKey = null | 'f' | 'c' | 'p'

export function SpotHead({
  node,
  cursor,
  all,
  left,
  n,
  exact,
  held,
  boardText,
  boardError,
  onBoard,
  opened,
  onOpen,
  sort,
  onSort,
  onUnhold,
}: {
  node: Node
  cursor: Cursor
  /** the whole range's cell (mask-free) */
  all: Cell
  /** the filtered range's cell: its share is how much is left */
  left: Cell
  n: number
  /** when cards are held, the exact number of holdings that contain them; null when the view is the sample */
  exact: number | null
  held: readonly Card[]
  boardText: string
  boardError: string | null
  onBoard: (text: string) => void
  opened: Opened
  onOpen: (o: Opened) => void
  sort: SortKey
  onSort: (k: SortKey) => void
  onUnhold: (c: Card) => void
}) {
  const [text, setText] = useState(boardText)
  const facing = node === 'facing'
  const title = cursor === 'draw' ? 'Your draw' : facing ? 'You, seat 1, facing a pot bet' : 'You, seat 1, first to act'
  const sub = cursor === 'draw' ? 'the flop is settled · how many do you throw, and which' : facing ? 'pot 4 · 2 to call · a raise is 8' : 'pot 2 · check or bet 2'
  const filtered = left.share < 0.9999 || held.length > 0

  return (
    <div className="spothead">
      <div className="who">
        {title}
        <small>{sub}</small>
        <label className="boardfield">
          <span>board</span>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            onBlur={() => onBoard(text)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') onBoard(text)
            }}
            spellCheck={false}
            aria-label="the flop, three cards like Ks 9s 4d"
          />
          {boardError && <em className="err">{boardError}</em>}
        </label>
      </div>

      <div className="mixblock">
        {cursor === 'you' && (
          <>
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
              {(facing ? (['f', 'c', 'p'] as const) : (['c', 'p'] as const)).map((k) => (
                <button
                  key={k}
                  type="button"
                  className={`hbtn act-${k} ${sort === k ? 'on' : ''}`}
                  onClick={() => onSort(sort === k ? null : k)}
                  aria-pressed={sort === k}
                >
                  <i />
                  {actionWord(k, node)} <b>{pct(all[k])}</b>
                </button>
              ))}
              <span className="hint">{sort ? `ledger sorted by ${actionWord(sort, node)}` : 'click an action to open the ledger'}</span>
            </div>
          </>
        )}
        {cursor === 'draw' && <MixKey mix={all} node={node} />}
      </div>

      <div className="holdblock">
        <button
          type="button"
          className={`hbtn cardsbtn ${opened === 'deck' ? 'on' : ''}`}
          onClick={() => onOpen(opened === 'deck' ? null : 'deck')}
          aria-expanded={opened === 'deck'}
        >
          cards
        </button>
        <span className="holding" aria-label="the cards you have placed">
          {Array.from({ length: 5 }, (_, i) =>
            held[i] !== undefined ? (
              <button key={held[i]} type="button" className="hbtn heldcard" onClick={() => onUnhold(held[i])} aria-label={`remove the card`}>
                <CardFace game={FULL_GAME} card={held[i]} />
              </button>
            ) : (
              <CardFace game={FULL_GAME} key={`g${i}`} ghost />
            ),
          )}
        </span>
        <span className="count">
          {exact !== null ? (
            <>
              <b>{count(Math.round(left.share * exact))}</b> holdings
              <br />
              {left.share < 0.9999 ? `${pct(left.share)} of the ${count(exact)} with your cards · exact` : `every holding with your cards · exact`}
            </>
          ) : (
            <>
              <b>{count(Math.round(left.share * 1906884))}</b> holdings
              <br />
              {filtered ? `${pct(left.share)} of the range · ${count(n)} sampled` : `whole range · ${count(n)} sampled`}
            </>
          )}
        </span>
      </div>
    </div>
  )
}
