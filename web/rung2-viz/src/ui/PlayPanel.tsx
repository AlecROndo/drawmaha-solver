import { useState } from 'react'

import { keyFor, legal, potAfter, word, type Act, type Rank } from '../leduc'
import { sampleAction, settle, shuffled, type Card } from '../play'
import { SOLVE } from '../solve'
import { boardPending, lineOver } from '../timeline'
import { Panel } from './site'

/**
 * One hand against the committed solve. The bot samples the final average
 * strategy and never adapts — you are playing the same numbers every figure
 * on this page draws. Seats alternate each hand; both cards are revealed at
 * settlement, a fold included, because the point is to study the line.
 */

interface Hand {
  cards: [Card, Card]
  boardCard: Card
  humanSeat: 0 | 1
  l1: string
  board: Rank | null
  l2: string
  lines: string[]
  /** chips to the human, set exactly once when the hand ends */
  result: number | null
}

const node = (h: Hand) => ({ l1: h.l1, board: h.board, l2: h.l2 })
const roundLine = (h: Hand): string => (h.board === null ? h.l1 : h.l2)
const actorOf = (h: Hand): 0 | 1 => (roundLine(h).length % 2) as 0 | 1

const appendAct = (h: Hand, act: Act): void => {
  if (h.board === null) h.l1 += act
  else h.l2 += act
}

/**
 * Run the hand forward — the board turning, the bot acting — until it is the
 * human's move or the hand is over, then settle. Mutates its own copy.
 */
function drive(prev: Hand): Hand {
  const h: Hand = { ...prev, lines: [...prev.lines] }
  for (;;) {
    if (lineOver(node(h))) break
    if (boardPending(node(h))) {
      h.board = h.boardCard.rank
      h.lines.push(`the board turns ${h.board}`)
      continue
    }
    if (actorOf(h) === h.humanSeat) return h
    const bot = h.cards[1 - h.humanSeat].rank
    const line = roundLine(h)
    const act = sampleAction(SOLVE.strategy, keyFor(bot, h.board, h.l1, h.l2), legal(line))
    h.lines.push(`the solver ${word(act, line)}s`)
    appendAct(h, act)
  }
  const toP0 = settle(h.l1, h.board, h.l2, h.cards[0].rank, h.cards[1].rank)
  h.result = h.humanSeat === 0 ? toP0 : -toP0
  const you = h.cards[h.humanSeat].rank
  const bot = h.cards[1 - h.humanSeat].rank
  const verdict =
    h.result === 0 ? 'a push' : h.result > 0 ? `you win ${h.result}` : `the solver wins ${-h.result}`
  h.lines.push(`you held ${you} · the solver held ${bot} — ${verdict}`)
  return h
}

function deal(humanSeat: 0 | 1): Hand {
  const deck = shuffled()
  return drive({
    cards: [deck[0], deck[1]],
    boardCard: deck[2],
    humanSeat,
    l1: '',
    board: null,
    l2: '',
    lines: [],
    result: null,
  })
}

function CardGlyph({ rank, hidden }: { rank?: Rank; hidden?: boolean }) {
  if (hidden) return <span className="pcard back" aria-label="a card, face down" />
  if (!rank) return <span className="pcard empty" aria-label="no card yet" />
  return <span className={`pcard rank ${rank}`}>{rank}</span>
}

export function PlayPanel() {
  const [hand, setHand] = useState<Hand>(() => deal(0))
  const [chips, setChips] = useState(0)
  const [hands, setHands] = useState(0)

  const over = hand.result !== null
  const line = roundLine(hand)
  const pot = potAfter(hand.l1, hand.board, hand.l2)
  const reveal = over // both cards show at settlement, fold included

  const act = (a: Act) => {
    if (over) return
    const played: Hand = { ...hand, lines: [...hand.lines, `you ${word(a, line)}`] }
    appendAct(played, a)
    const done = drive(played)
    setHand(done)
    if (done.result !== null) {
      // Bank inside the handler, not an effect: StrictMode replays effects
      // and would count every hand twice.
      setChips((c) => c + (done.result ?? 0))
      setHands((n) => n + 1)
    }
  }

  const next = () => setHand((h) => deal((1 - h.humanSeat) as 0 | 1))

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
