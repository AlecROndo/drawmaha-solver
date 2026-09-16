import { useState } from 'react'

import { keyFor, rankWeights, word, RANKS, type Act, type Rank } from '../leduc'
import { SOLVE } from '../solve'
import { boardPending, lineOver, stateAt, type Line, type Walk } from '../timeline'
import { activeStrategy, type Exploit } from './useExploit'
import { Panel } from './site'

/** Bar and number order: aggression leads from the ink side of the duotone. */
const ORDER: Act[] = ['r', 'c', 'f']

const acts = (row: Partial<Record<Act, number>>): Act[] => ORDER.filter((a) => a in row)

const pct = (x: number): string => `${(x * 100).toFixed(0)}%`

const SAME = 1e-4

const rowsDiffer = (
  a: Partial<Record<Act, number>>,
  b: Partial<Record<Act, number>>,
): boolean => (Object.keys(a) as Act[]).some((act) => Math.abs((a[act] ?? 0) - (b[act] ?? 0)) > SAME)

/**
 * One row's whole mixed strategy — and, once a run exists, the Nash baseline
 * as a slim muted lane beneath it, so "where it was" and "where it moved"
 * read in one glance without a separate diff view.
 */
function Mix({
  row,
  base,
  locked,
}: {
  row: Partial<Record<Act, number>>
  base: Partial<Record<Act, number>> | null
  locked: boolean
}) {
  return (
    <div className="mixwrap">
      <div className={locked ? 'mix held' : 'mix'}>
        {acts(row).map((a) => (
          <i key={a} className={a} style={{ width: `${((row[a] ?? 0) * 100).toFixed(2)}%` }} />
        ))}
      </div>
      {base && (
        <div className="mix was" aria-label="the Nash baseline before your edit">
          {acts(base).map((a) => (
            <i key={a} className={a} style={{ width: `${((base[a] ?? 0) * 100).toFixed(2)}%` }} />
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * What the seat likely holds, before a word about what it does. A strategy
 * row can be solid oxblood — "bets every king" — while kings are nearly gone
 * from the range, filtered out by the deal, the board, and the line itself.
 * This bar is that filter, in the ranks' identity colours. It reads the
 * response strategy when a run exists: edited strategies move ranges too.
 */
function Holding({ seat, s, x }: { seat: 0 | 1; s: Line; x: Exploit }) {
  const weights = rankWeights(seat, s, activeStrategy(x.run))
  return (
    <div className="holding">
      <span className="k">likely holding</span>
      <div className="range-bar">
        {RANKS.map(
          (r, i) =>
            weights[i] > 0.0005 && (
              <i key={r} className={r} style={{ width: `${(weights[i] * 100).toFixed(2)}%` }} />
            ),
        )}
      </div>
      <span className="range-nums">
        {RANKS.map((r, i) => (
          <span key={r} className={`rank ${r}`}>
            {r} {pct(weights[i])}
          </span>
        ))}
      </span>
    </div>
  )
}

interface Spot {
  board: Rank | null
  l1: string
  l2: string
  /** the round line the spot's actor is reading */
  ctx: string
  /** set when this is a look BACK at an action already taken */
  took: Act | null
}

/** Seats alternate within a round, so index parity names the actor. */
const lastIndexBySeat = (line: string, seat: number): number => {
  for (let i = line.length - 1; i >= 0; i--) if (i % 2 === seat) return i
  return -1
}

/**
 * The spot a waiting seat last acted at — its own action, not merely the
 * line's last one, which after a call belongs to the other seat. Searches the
 * current round first, then round 1 once the board is out.
 */
function lastActed(seat: number, s: Line): Spot | null {
  if (s.board !== null) {
    const i = lastIndexBySeat(s.l2, seat)
    if (i >= 0) {
      const ctx = s.l2.slice(0, i)
      return { board: s.board, l1: s.l1, l2: ctx, ctx, took: s.l2[i] as Act }
    }
  }
  const i = lastIndexBySeat(s.l1, seat)
  if (i < 0) return null
  const ctx = s.l1.slice(0, i)
  return { board: null, l1: ctx, l2: '', ctx, took: s.l1[i] as Act }
}

/**
 * One seat at the node the cursor is on: what it likely holds, then a row
 * per rank. Each row can be locked (◇→◆) and edited with sliders; a locked
 * row is what the exploit run holds still. After a run, every bar shows the
 * response over its Nash baseline.
 */
export function SeatPanel({ seat, walk, x }: { seat: 0 | 1; walk: Walk; x: Exploit }) {
  const [editing, setEditing] = useState<string | null>(null)
  const s = stateAt(walk.path, walk.cur)
  const inR2 = s.board !== null
  const line = inR2 ? s.l2 : s.l1
  const over = lineOver(s)
  // Between rounds it is the deck's turn, not a seat's: nobody is "to act".
  const acting = over || boardPending(s) ? -1 : line.length % 2
  const isAct = seat === acting

  const spot: Spot | null = isAct
    ? { board: s.board, l1: s.l1, l2: s.l2, ctx: line, took: null }
    : lastActed(seat, s)

  const badge = isAct ? 'to act' : over ? 'hand over' : 'waiting'
  const sub = isAct
    ? 'What each rank does here, right now.'
    : spot
      ? 'Not this seat’s decision — what it did to get here is on the timeline.'
      : 'Hasn’t acted yet — its first read of the hand is still to come.'

  return (
    <Panel className={isAct ? 'seat acting' : 'seat'} label={`Player ${seat}`}>
      <div className="seat-head">
        <span>P{seat}</span>
        <span className={`badge ${isAct ? 'act' : ''}`}>{badge}</span>
      </div>
      <Holding seat={seat} s={s} x={x} />
      <p className="seat-sub">{sub}</p>
      {spot !== null &&
        RANKS.map((r) => {
          const key = keyFor(r, spot.board, spot.l1, spot.l2)
          const nash = SOLVE.strategy[key]
          if (!nash) return null
          const lockedRow = x.locked[key] ?? null
          // Priority: a pending lock edit shows immediately; else the run's
          // response; else Nash. The baseline lane appears whenever what is
          // shown has actually moved off Nash.
          const row = lockedRow ?? (x.run ? (x.run.strategy[key] ?? nash) : nash)
          const showBase = rowsDiffer(row, nash)
          const isEditing = editing === key
          return (
            <div
              key={r}
              className={['rankrow', lockedRow ? 'locked' : '', isEditing ? 'editing' : '']
                .filter(Boolean)
                .join(' ')}
            >
              <span className={`rank ${r}`}>{r}</span>
              <Mix row={row} base={showBase ? nash : null} locked={lockedRow !== null} />
              <span className="nums">
                {spot.took !== null ? (
                  <span>
                    took <b>{word(spot.took, spot.ctx)}</b> {pct(row[spot.took] ?? 0)}
                  </span>
                ) : (
                  acts(row).map((a) => (
                    <span key={a}>
                      {word(a, spot.ctx)} <b>{pct(row[a] ?? 0)}</b>
                    </span>
                  ))
                )}
              </span>
              <button
                className="lockbtn"
                title={lockedRow ? 'clear this lock' : 'lock this row and edit it'}
                aria-pressed={lockedRow !== null}
                onClick={() => {
                  x.toggleLock(key)
                  setEditing(lockedRow ? null : key)
                }}
              >
                {lockedRow ? '◆' : '◇'}
              </button>
              {isEditing && lockedRow && (
                <div className="editor">
                  {acts(lockedRow).map((a) => (
                    <label className="row" key={a}>
                      <span>{word(a, spot.ctx)}</span>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={Math.round((lockedRow[a] ?? 0) * 100)}
                        onChange={(e) => x.setLockValue(key, a, Number(e.target.value) / 100)}
                      />
                      <span>{pct(lockedRow[a] ?? 0)}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          )
        })}
    </Panel>
  )
}
