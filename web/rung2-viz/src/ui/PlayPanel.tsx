import type { Dispatch, SetStateAction } from 'react'

import { legal, potAfter, word, type Act, type Rank } from '../leduc'
import { nextHand, playAct, roundLine, type PlayRow, type Session } from '../play'
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

/** Bar order for the mix captions: aggression first, the roll's own order. */
const ORDER: Act[] = ['r', 'c', 'f']

function MixCaption({ row }: { row: Extract<PlayRow, { kind: 'action' }> }) {
  return (
    <span className="side-mix">
      {ORDER.filter((a) => a in row.mix).map((a) => (
        <span key={a} className={a === row.act ? 'chosen' : ''}>
          {word(a, '')} {pct(row.mix[a] ?? 0)}
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
  const mistakes = rows.reduce(
    (sum, r) => sum + (r.kind === 'action' && r.human ? (r.mistake ?? 0) : 0),
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
                <MixCaption row={row} />
              </span>
              {row.human && row.roll !== null && (
                <span className="side-grade">
                  <span className="roll">roll {row.roll}</span>
                  {row.mistake === 0 ? (
                    <span className="ok">✓</span>
                  ) : (
                    <span className="bad">
                      {Math.round(row.mistake ?? 0)}% off — roll said {word(row.correct as Act, '')}
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
        mistakes this hand <b>{Math.round(mistakes)}%</b>
      </p>
    </aside>
  )
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
        </div>
        <div className="seat-spot centre">
          <CardGlyph rank={hand.board ?? undefined} />
          <span className="pot">pot {pot}</span>
        </div>
        <div className="seat-spot hero">
          <CardGlyph rank={hand.cards[hand.humanSeat].rank} />
          <span className="who">you · P{hand.humanSeat}</span>
        </div>
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
