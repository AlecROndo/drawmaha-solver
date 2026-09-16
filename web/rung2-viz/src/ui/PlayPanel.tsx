import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'

import { legal, potAfter, roundInFor, word, BETS, ANTE, type Act, type Rank } from '../leduc'
import { AGGRESSION, nextHand, playAct, roundLine, type PlayRow, type Session } from '../play'
import { SOLVE } from '../solve'
import { Panel } from './site'

/**
 * One hand against the committed solve. The bot samples the final average
 * strategy and never adapts — you are playing the same numbers every figure
 * on this page draws. Seats alternate each hand; both cards are revealed at
 * settlement, a fold included, because the point is to study the line.
 *
 * The session itself (hand, bankroll) lives in App, like the walk state, so
 * flipping tabs does not tear down a hand in progress. The transitions are
 * pure functions in `play.ts`, applied here in updater form: they read only
 * the state they are given, so a double-fired click cannot act on a stale
 * hand, and the banked chips land in the same update as the hand that earned
 * them — nothing for StrictMode's replays to double-count.
 */

const pct = (x: number): string => `${(x * 100).toFixed(0)}`

function MixCaption({ row }: { row: Extract<PlayRow, { kind: 'action' }> }) {
  // AGGRESSION is the roll's own segment order, and row.line is the line the
  // actor actually faced — so a facing-a-bet mix reads "call/raise", not
  // "check/bet".
  return (
    <span className="side-mix">
      {AGGRESSION.filter((a) => a in row.mix).map((a) => (
        <span key={a} className={a === row.act ? 'chosen' : ''}>
          {word(a, row.line)} {pct(row.mix[a] ?? 0)}
        </span>
      ))}
    </span>
  )
}

/**
 * The hand, decision by decision: every node a row, newest at the bottom.
 * Human rows carry the pre-drawn roll — the instruction — and the grade:
 * following the roll is ✓, deviating costs the points the roll sat inside
 * the prescribed action's region.
 */
function SidePanel({
  rows,
  pendingRoll,
  over,
  verdict,
}: {
  rows: PlayRow[]
  pendingRoll: number | null
  over: boolean
  verdict: string | null
}) {
  // Sum the per-row ROUNDED mistakes, so the total equals what a reader adds
  // up from the rows instead of drifting a point on fractional deviations.
  const mistakes = rows.reduce(
    (sum, r) => sum + (r.kind === 'action' && r.human ? Math.round(r.mistake ?? 0) : 0),
    0,
  )
  return (
    <aside className="playside" aria-label="The hand, decision by decision">
      <span className="k">the hand · low rolls play aggressive</span>
      <ol>
        {rows.map((row, i) =>
          row.kind === 'board' ? (
            <li key={i} className="side-row board">
              <span className="side-who">board</span>
              <span className={`side-act rank ${row.rank}`}>{row.rank}</span>
            </li>
          ) : (
            <li key={i} className={`side-row ${row.human ? 'you' : 'bot'}`}>
              <span className="side-who">{row.human ? 'you' : 'solver'}</span>
              <span className="side-main">
                <b>{row.label}</b>
                {/* your rows teach; the solver's stay face down, like its card */}
                {row.human && <MixCaption row={row} />}
              </span>
              {row.human && row.roll !== null && (
                <span className="side-grade">
                  <span className="roll">roll {row.roll}</span>
                  {row.mistake === 0 ? (
                    <span className="ok">✓</span>
                  ) : (
                    <span className="bad">
                      {Math.round(row.mistake ?? 0)}% off — roll said{' '}
                      {word(row.correct as Act, row.line)}
                    </span>
                  )}
                </span>
              )}
            </li>
          ),
        )}
        {!over && pendingRoll !== null && (
          <li className="side-row live">
            <span className="side-who">you</span>
            <span className="side-main">
              <b>your move</b>
              <span className="side-mix">low roll bets, high roll gives up</span>
            </span>
            <span className="side-grade">
              <span className="roll live">roll {pendingRoll}</span>
            </span>
          </li>
        )}
        {over && verdict && (
          <li className="side-row verdict">
            <span className="side-who">result</span>
            <span className="side-main">{verdict}</span>
          </li>
        )}
      </ol>
      <p className="side-total">
        mistakes this hand <b>{mistakes}%</b>
      </p>
    </aside>
  )
}

/** A monoline chip stack: one ellipse per two chips, the amount beside it. */
function Chips({ n, quiet }: { n: number; quiet?: boolean }) {
  if (n <= 0) return null
  const plates = Math.min(4, Math.ceil(n / 2))
  return (
    <span className="chipstack" aria-label={`${n} chips`}>
      <svg viewBox="0 0 26 20" aria-hidden>
        <g fill="none" stroke="currentColor" strokeWidth="1.2">
          {Array.from({ length: plates }, (_, i) => (
            <g key={i}>
              <ellipse cx="13" cy={16 - i * 3.4} rx="9" ry="3" />
            </g>
          ))}
        </g>
      </svg>
      {/* the pile sits beside "pot N", which already speaks the amount */}
      {!quiet && <b>{n}</b>}
    </span>
  )
}

/** The last thing a seat did this hand, spoken at the seat itself. */
const lastActionOf = (rows: PlayRow[], seat: 0 | 1): string | null => {
  for (let i = rows.length - 1; i >= 0; i--) {
    const row = rows[i]
    if (row.kind === 'action' && row.seat === seat) return `${row.label}s`
  }
  return null
}

function CardGlyph({ rank, hidden }: { rank?: Rank; hidden?: boolean }) {
  if (hidden) return <span className="pcard back" aria-label="a card, face down" />
  if (!rank) return <span className="pcard empty" aria-label="no card yet" />
  return <span className={`pcard rank ${rank}`}>{rank}</span>
}

export function PlayPanel({
  session,
  setSession,
}: {
  session: Session
  setSession: Dispatch<SetStateAction<Session>>
}) {
  const { hand, chips, hands } = session

  const over = hand.result !== null
  const line = roundLine(hand)
  const pot = potAfter(hand.l1, hand.board, hand.l2)
  const reveal = over // both cards show at settlement, fold included

  const act = (a: Act) => setSession((s) => playAct(s, a, SOLVE.strategy))
  const next = () => setSession((s) => nextHand(s, SOLVE.strategy))

  // Chips in front of each seat this round, and the pile already in the
  // middle. When a round closes its chips slide to the pile: the slide is a
  // pair of transient tokens spawned by watching the round change.
  const inR2 = hand.board !== null
  const liveInFor: [number, number] = over
    ? [0, 0]
    : inR2
      ? roundInFor(hand.l2, BETS[1])
      : roundInFor(hand.l1, BETS[0])
  const pile = over
    ? pot
    : 2 * ANTE + (inR2 ? ((r) => r[0] + r[1])(roundInFor(hand.l1, BETS[0])) : 0)

  const [sweep, setSweep] = useState<{ amounts: [number, number]; id: number } | null>(null)
  const prev = useRef<{ handId: string; phase: string } | null>(null)
  const handId = `${hands}:${hand.humanSeat}`
  const phase = over ? 'over' : inR2 ? 'r2' : 'r1'
  useEffect(() => {
    const was = prev.current
    prev.current = { handId, phase }
    if (!was || was.handId !== handId) {
      setSweep(null)
      return
    }
    if (was.phase === phase) return
    // The round that just closed is the one that was live last render. Its
    // final chips are read from the committed line, not a render snapshot:
    // the closing call, the bot's reply, and the board reveal all land in one
    // update, so the previous render never saw the caller's chips go in.
    const amounts =
      was.phase === 'r1' ? roundInFor(hand.l1, BETS[0]) : roundInFor(hand.l2, BETS[1])
    if (amounts[0] > 0 || amounts[1] > 0) {
      setSweep((s) => ({ amounts, id: (s?.id ?? 0) + 1 }))
    }
    // hand.l1/l2 are frozen once their round closes; `phase` is the trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handId, phase])

  return (
    <Panel
      className="playpanel"
      k={`Fig. 2 · hand ${hands + 1} — you are P${hand.humanSeat}`}
      title="Play the solver."
      say={`The bot samples the committed ${SOLVE.iterations.toLocaleString()}-iteration average strategy — exploitability ${SOLVE.exploitabilityAverage.toFixed(4)} chips/hand. It never adapts.`}
      label="Play a hand against the solver"
    >
      <div className="playwrap">
        <div className="playmain">
          <div className="table">
        <svg className="felt" viewBox="0 0 560 380" aria-hidden>
          <ellipse cx="280" cy="190" rx="266" ry="176" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <ellipse cx="280" cy="190" rx="246" ry="158" fill="none" stroke="currentColor" strokeWidth="1" strokeOpacity="0.25" />
        </svg>
        <div className="seat-spot villain">
          <span className="who">the solver · P{1 - hand.humanSeat}</span>
          {reveal ? <CardGlyph rank={hand.cards[1 - hand.humanSeat].rank} /> : <CardGlyph hidden />}
          <span className="did">{lastActionOf(hand.rows, (1 - hand.humanSeat) as 0 | 1) ?? '\u00A0'}</span>
          <Chips n={liveInFor[1 - hand.humanSeat]} />
        </div>
        <div className="seat-spot centre">
          <CardGlyph rank={hand.board ?? undefined} />
          <span className="pot">
            <Chips n={pile} quiet /> pot {pot}
          </span>
        </div>
        <div className="seat-spot hero">
          <Chips n={liveInFor[hand.humanSeat]} />
          <span className="did">{lastActionOf(hand.rows, hand.humanSeat) ?? '\u00A0'}</span>
          <CardGlyph rank={hand.cards[hand.humanSeat].rank} />
          <span className="who">you · P{hand.humanSeat}</span>
        </div>
        {sweep && sweep.amounts[1 - hand.humanSeat] > 0 && (
          <span key={`v${sweep.id}`} className="sweep from-villain" onAnimationEnd={() => setSweep(null)}>
            <Chips n={sweep.amounts[1 - hand.humanSeat]} />
          </span>
        )}
        {sweep && sweep.amounts[hand.humanSeat] > 0 && (
          <span key={`h${sweep.id}`} className="sweep from-hero" onAnimationEnd={() => setSweep(null)}>
            <Chips n={sweep.amounts[hand.humanSeat]} />
          </span>
        )}
      </div>

          <div className="play-buttons">
            {over ? (
              <button className="btn" onClick={next}>
                Next hand
              </button>
            ) : (
              legal(line).map((a, i) => (
                <button key={a} className={i === 0 ? 'btn' : 'btn ghost'} onClick={() => act(a)}>
                  {word(a, line)}
                </button>
              ))
            )}
          </div>
        </div>

        <SidePanel
          rows={hand.rows}
          pendingRoll={hand.pendingRoll}
          over={over}
          verdict={over ? (hand.lines.at(-1) ?? null) : null}
        />
      </div>

      <dl className="stat">
        <div>
          <dt>hands</dt>
          <dd>{hands}</dd>
        </div>
        <div>
          <dt>your chips</dt>
          <dd>
            {chips >= 0 ? '+' : ''}
            {chips}
          </dd>
        </div>
        <div>
          <dt>per hand</dt>
          <dd>{hands === 0 ? '—' : (chips / hands).toFixed(3)}</dd>
        </div>
      </dl>
    </Panel>
  )
}
