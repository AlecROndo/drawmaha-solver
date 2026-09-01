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
    <section className="panel" aria-label="Play against the equilibrium">
      <h2>Fig. 3 · Play it</h2>
      <p className="sub">
        the solved average strategy · seats alternate · exploitable for{' '}
        {SOLVE.exploitabilityAverage[FINAL].toFixed(5)} chips/hand
      </p>
      <p className="play-head">
        You are P{hand.humanSeat}, holding the{' '}
        <b style={{ color: `var(--${CARD_NAME[yourCard]})` }}>{CARD_NAME[yourCard]}</b>.
      </p>
      <p className="play-log">
        {hand.lines.length ? hand.lines.join('\n') : 'Your move.'}
      </p>
      <div className="play-buttons">
        {over ? (
          <button onClick={next}>Next hand</button>
        ) : (
          (['p', 'b'] as Act[]).map((a) => (
            <button key={a} onClick={() => act(a)}>
              {actionLabel(a, history)}
            </button>
          ))
        )}
      </div>
      <p className="tally">
        {hands === 0
          ? 'no hands scored yet'
          : `${hands} hand${hands === 1 ? '' : 's'} · ${chips >= 0 ? '+' : ''}${chips} chips · ${(chips / hands).toFixed(3)} per hand · equilibrium play breaks even at 0.000`}
      </p>
    </section>
  )
}
