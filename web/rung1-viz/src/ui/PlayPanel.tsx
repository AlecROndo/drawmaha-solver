import { useCallback, useState } from 'react'
import {
  CARD_NAME,
  CARD_SYMBOL,
  DEALS,
  actionLabel,
  actor,
  infosetKey,
  isTerminal,
  returns,
  type Act,
  type Card,
} from '../kuhn'
import { Panel } from './site'
import { SOLVE } from './useSolve'

/**
 * Play hands against the solved equilibrium.
 *
 * The bot plays the average strategy at its own infoset, so it is a genuine
 * Nash equilibrium rather than something adapting to you. Seats alternate,
 * because Kuhn is asymmetric — the first player's game value is −1/18, and a
 * fixed seat would confound your mistakes with the seat's built-in edge.
 *
 * You cannot beat it. The interesting number is how fast you lose.
 */

const FINAL = SOLVE.iterations.length - 1

interface Hand {
  cards: [Card, Card]
  humanSeat: 0 | 1
  history: string
  lines: string[]
  result: number | null
}

/** P(bet) the equilibrium plays at this infoset. */
const betProbability = (card: Card, history: string): number =>
  SOLVE.bet[infosetKey(card, history)][FINAL]

function dealHand(humanSeat: 0 | 1): Hand {
  return {
    cards: DEALS[Math.floor(Math.random() * DEALS.length)],
    humanSeat,
    history: '',
    lines: [],
    result: null,
  }
}

/** Let the bot act while it is its turn, appending each move to the log. */
function runBot(hand: Hand): Hand {
  let { history, lines } = hand
  while (!isTerminal(history) && actor(history) !== hand.humanSeat) {
    const botCard = hand.cards[1 - hand.humanSeat]
    const act: Act = Math.random() < betProbability(botCard, history) ? 'b' : 'p'
    lines = [...lines, `bot ${actionLabel(act, history)}s`]
    history += act
  }
  return { ...hand, history, lines }
}

function settle(hand: Hand): Hand {
  if (!isTerminal(hand.history) || hand.result !== null) return hand
  const chips = returns(hand.history, hand.cards)[hand.humanSeat]
  const you = CARD_SYMBOL[hand.cards[hand.humanSeat]]
  const bot = CARD_SYMBOL[hand.cards[1 - hand.humanSeat]]
  return {
    ...hand,
    result: chips,
    // Reveal even after a fold: knowing whether you were bluffed is the point,
    // and the hand is already over.
    lines: [
      ...hand.lines,
      `you ${you} · bot ${bot} — ${chips > 0 ? `you win ${chips}` : `bot wins ${-chips}`}`,
    ],
  }
}

export function PlayPanel() {
  const [hand, setHand] = useState<Hand>(() => runBot(dealHand(0)))
  const [chips, setChips] = useState(0)
  const [hands, setHands] = useState(0)

  const act = useCallback(
    (a: Act) => {
      if (hand.result !== null) return
      const played = {
        ...hand,
        history: hand.history + a,
        lines: [...hand.lines, `you ${actionLabel(a, hand.history)}`],
      }
      const settled = settle(runBot(played))
      setHand(settled)
      // Banked here rather than in an effect watching `hand`: the hand ending
      // IS this event, and an effect would re-render to discover what we
      // already know — and double-count under StrictMode's remount.
      if (settled.result !== null) {
        setChips((c) => c + settled.result!)
        setHands((n) => n + 1)
      }
    },
    [hand],
  )

  const next = useCallback(() => {
    setHand((h) => runBot(dealHand((1 - h.humanSeat) as 0 | 1)))
  }, [])

  const over = hand.result !== null
  const history = hand.history
  const yourCard = hand.cards[hand.humanSeat]

  return (
    <Panel
      n="03"
      id="play"
      k="Fig. 3 · play it"
      title="You cannot beat it. The number is how fast you lose."
      say={`The bot plays the solved average strategy, so it is a genuine equilibrium and never adapts to you — exploitable for ${SOLVE.exploitabilityAverage[FINAL].toFixed(5)} chips a hand. Seats alternate because Kuhn is asymmetric.`}
      label="Play against the equilibrium"
    >
      <div className="hand">
        <div className="cardglyph" style={{ color: `var(--${CARD_NAME[yourCard]})` }}>
          {CARD_SYMBOL[yourCard]}
        </div>
        <div>
          <span className="k">
            Hand {hands + 1} · you are P{hand.humanSeat}
          </span>
          <p className="play-log">{hand.lines.length ? hand.lines.join('\n') : 'Your move.'}</p>
        </div>
      </div>
      <div className="play-buttons">
        {over ? (
          <button className="btn" onClick={next}>
            Next hand
          </button>
        ) : (
          (['b', 'p'] as Act[]).map((a, i) => (
            <button key={a} className={i === 0 ? 'btn' : 'btn ghost'} onClick={() => act(a)}>
              {actionLabel(a, history)}
            </button>
          ))
        )}
      </div>
      <ul className="fields">
        <li>
          <span>Hands</span>
          <span className="v">{hands}</span>
        </li>
        <li>
          <span>Your chips</span>
          <span className="v">
            {chips >= 0 ? '+' : ''}
            {chips}
          </span>
        </li>
        <li>
          <span>Per hand</span>
          <span className="v">{hands === 0 ? '—' : (chips / hands).toFixed(3)}</span>
        </li>
      </ul>
    </Panel>
  )
}
