import type { Dispatch, SetStateAction } from 'react'

import { legal, potAfter, word, type Act, type Rank } from '../leduc'
import { nextHand, playAct, roundLine, type Session } from '../play'
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

      <p className="play-log">{hand.lines.length ? hand.lines.join('\n') : 'Your move.'}</p>

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
