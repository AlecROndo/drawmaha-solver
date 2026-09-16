import { keyFor, word, RANKS, type Act, type Rank } from '../leduc'
import { SOLVE } from '../solve'
import { boardPending, lineOver, stateAt, type Line, type Walk } from '../timeline'
import { Panel } from './site'

/** Bar and number order: aggression leads from the ink side of the duotone. */
const ORDER: Act[] = ['r', 'c', 'f']

const acts = (row: Partial<Record<Act, number>>): Act[] => ORDER.filter((a) => a in row)

function Mix({ row }: { row: Partial<Record<Act, number>> }) {
  return (
    <div className="mix">
      {acts(row).map((a) => (
        <i key={a} className={a} style={{ width: `${((row[a] ?? 0) * 100).toFixed(2)}%` }} />
      ))}
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
 * One seat's whole range at the node the cursor is on: a row per rank, each
 * bar that rank's full mixed strategy. The acting seat reads at its live
 * spot; the waiting seat re-reads the spot it acted at last, named — a range
 * reading "check 100%" under a timeline saying it bet would look like a
 * contradiction rather than an off-tree line.
 */
export function SeatPanel({ seat, walk }: { seat: 0 | 1; walk: Walk }) {
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
    <Panel label={`Player ${seat}`}>
      <div className="seat-head">
        <span>P{seat}</span>
        <span className={`badge ${isAct ? 'act' : ''}`}>{badge}</span>
      </div>
      <p className="seat-sub">{sub}</p>
      {spot !== null &&
        RANKS.map((r) => {
          const row = SOLVE.strategy[keyFor(r, spot.board, spot.l1, spot.l2)]
          if (!row) return null
          return (
            <div key={r} className="rankrow">
              <span className={`rank ${r}`}>{r}</span>
              <Mix row={row} />
              <span className="nums">
                {spot.took !== null ? (
                  <>
                    took <b>{word(spot.took, spot.ctx)}</b> ·{' '}
                    {((row[spot.took] ?? 0) * 100).toFixed(0)}% of the time
                  </>
                ) : (
                  acts(row).map((a) => (
                    <span key={a}>
                      {word(a, spot.ctx)} <b>{((row[a] ?? 0) * 100).toFixed(0)}%</b>
                    </span>
                  ))
                )}
              </span>
            </div>
          )
        })}
    </Panel>
  )
}
