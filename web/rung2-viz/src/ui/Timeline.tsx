import {
  legal,
  potAfter,
  reach,
  word,
  closed,
  RANKS,
  ANTE,
  type Act,
  type Rank,
} from '../leduc'
import { SOLVE } from '../solve'
import { handOver, stateAt, type Step, type Walk } from '../timeline'

/** Matches `.tl li { flex: 0 0 150px }` — the rail is measured in these. */
const STATION_W = 150

interface Station {
  label: string
  who: string
  pot: number
  board?: Rank
  ghost?: boolean
}

/** One station per event: the deal, each action, the board, a ghost "to act". */
function stations(walk: Walk): Station[] {
  const out: Station[] = [{ label: 'deal', who: 'round 1', pot: 2 * ANTE }]
  for (let i = 0; i < walk.path.length; i++) {
    const step = walk.path[i]
    const before = stateAt(walk.path, i)
    const after = stateAt(walk.path, i + 1)
    const pot = potAfter(after.l1, after.board, after.l2)
    if (step.k === 'b') {
      out.push({ label: step.v, who: 'round 2', board: step.v, pot })
    } else {
      const line = before.board === null ? before.l1 : before.l2
      out.push({ label: word(step.v, line), who: `P${line.length % 2}`, pot })
    }
  }
  // The decision now facing us, shown only when nothing is explored past it.
  if (walk.cur === walk.path.length && !handOver(walk.path, walk.cur)) {
    const s = stateAt(walk.path, walk.cur)
    const line = s.board === null ? s.l1 : s.l2
    out.push({
      label: 'to act',
      ghost: true,
      who: s.board === null && closed(s.l1) ? 'board' : `P${line.length % 2}`,
      pot: potAfter(s.l1, s.board, s.l2),
    })
  }
  return out
}

export function Timeline({ walk, onJump }: { walk: Walk; onJump: (n: number) => void }) {
  const all = stations(walk)
  const mid = STATION_W / 2
  return (
    <div className="tl">
      <div className="scroller">
        <div className="track">
          <div className="rail" style={{ left: mid, width: (all.length - 1) * STATION_W }} />
          <div className="rail done" style={{ left: mid, width: walk.cur * STATION_W }} />
          <ol>
            {all.map((st, i) => {
              const cls = [
                st.ghost ? 'future' : i <= walk.cur ? 'done' : 'ahead',
                i === walk.cur ? 'here' : '',
                st.board ? 'board-stn' : '',
              ]
                .filter(Boolean)
                .join(' ')
              const inner = (
                <>
                  <span className="who">{st.who}</span>
                  <span className="stn-box" />
                  <span className={st.board ? `act rank ${st.board}` : 'act'}>{st.label}</span>
                  <span className="pot">pot {st.pot}</span>
                </>
              )
              return (
                <li key={i} className={cls}>
                  {st.ghost ? (
                    <span className="stn">{inner}</span>
                  ) : (
                    <button
                      className="stn"
                      onClick={() => onJump(i)}
                      aria-current={i === walk.cur ? 'step' : undefined}
                      aria-label={`rewind to ${st.label}, pot ${st.pot}`}
                    >
                      {inner}
                    </button>
                  )}
                </li>
              )
            })}
          </ol>
        </div>
      </div>
    </div>
  )
}

export function NextActions({ walk, onStep }: { walk: Walk; onStep: (step: Step) => void }) {
  const s = stateAt(walk.path, walk.cur)
  const inR2 = s.board !== null
  const line = inR2 ? s.l2 : s.l1
  const onPath = walk.path[walk.cur]
  const ahead = walk.path.length - walk.cur

  let body: React.ReactNode
  if (!inR2 && closed(s.l1)) {
    body = (
      <>
        <span className="lab">the board turns</span>
        {RANKS.map((b) => (
          <button
            key={b}
            className={onPath?.k === 'b' && onPath.v === b ? 'onpath' : ''}
            onClick={() => onStep({ k: 'b', v: b })}
          >
            <span className={`rank ${b}`}>{b}</span>
            {/* Marginally each rank is 2 of the 6 cards: uniform thirds. */}
            <span className="w">{(100 / 3).toFixed(0)}%</span>
          </button>
        ))}
      </>
    )
  } else if (inR2 && closed(s.l2)) {
    body = (
      <span className="done-msg">The hand is over on this line — click a station to go back.</span>
    )
  } else {
    body = (
      <>
        <span className="lab">next</span>
        {legal(line).map((a) => {
          const w = inR2 ? null : reach(s.l1 + a, SOLVE.strategy)
          return (
            <button
              key={a}
              className={onPath?.k === 'a' && onPath.v === a ? 'onpath' : ''}
              onClick={() => onStep({ k: 'a', v: a })}
            >
              {word(a as Act, line)}
              {w !== null && <span className="w">{(w * 100).toFixed(0)}%</span>}
            </button>
          )
        })}
      </>
    )
  }

  return (
    <div className="next">
      {body}
      {ahead > 0 && (
        <span className="fork-note">
          — a different choice here replaces the {ahead} step{ahead > 1 ? 's' : ''} ahead
        </span>
      )}
    </div>
  )
}
