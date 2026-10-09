import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'

import {
  SUIT_GLYPH,
  betWord,
  costOf,
  isThrow,
  rankOf,
  replay,
  suitOf,
  sym,
  type Act,
  type Bet,
  type CardIndex,
  type Throw,
} from '../minidraw'
import {
  AGGRESSION,
  actorAt,
  board1,
  board2,
  fmt,
  holeOf,
  linesOf,
  nextHand,
  phaseOf,
  playAct,
  spotOf,
  thrownCard,
  type Hand,
  type PlayRow,
  type Session,
} from '../play'
import { loadedDeals, type PackState } from '../deals'
import { FROZEN, WORKERS } from '../research'
import { Panel } from './site'

/**
 * One hand against the frozen strategy, on the oval table rung 2 drew, with
 * three cards a seat and two on the board. The bot reads its mix out of the
 * deal pack and never adapts. Seats alternate each hand; both holdings are
 * revealed at settlement, a fold included, because the point is the line.
 *
 * The session (hand, bankroll, the deals still to come) lives in App so
 * flipping tabs does not tear down a hand in progress. The transitions are
 * pure functions in `play.ts`, applied in updater form: they read only the
 * state they are given, so a double-fired click cannot act on a stale hand,
 * and the banked chips land in the same update as the hand that earned them.
 */

const pct = (x: number): string => `${(x * 100).toFixed(0)}`

/** The seat's own value when the strategy plays itself: what matching the bot averages. */
const seatValue = (seat: 0 | 1): number => (seat === 0 ? FROZEN.value_p0 : -FROZEN.value_p0)

const signed = (n: number, digits = 0): string => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(digits)}`

function MixCaption({ row }: { row: Extract<PlayRow, { kind: 'action' }> }) {
  // AGGRESSION is the roll's own segment order; row.words carries each
  // action's word in the context the actor faced.
  return (
    <span className="side-mix">
      {AGGRESSION.filter((a) => a in row.mix).map((a) => (
        <span key={a} className={a === row.act ? 'chosen' : ''}>
          {row.words[a]} {pct(row.mix[a] ?? 0)}
        </span>
      ))}
    </span>
  )
}

/**
 * The hand, decision by decision: every node a row, newest at the bottom.
 * Human rows carry the pre-drawn roll — the instruction — and the grade.
 */
function SidePanel({ rows, pendingRoll, over }: { rows: PlayRow[]; pendingRoll: number | null; over: boolean }) {
  const mistakes = rows.reduce(
    (sum, r) => sum + (r.kind === 'action' && r.human ? Math.round(r.mistake ?? 0) : 0),
    0,
  )
  return (
    <aside className="playside" aria-label="The hand, decision by decision">
      <span className="k">the hand · low rolls play aggressive · low rolls stand pat</span>
      <ol>
        {rows.map((row, i) =>
          row.kind === 'board' ? (
            <li key={i} className="side-row board">
              <span className="side-who">board {row.n}</span>
              <span className="side-act">
                <CardGlyph card={row.card} small />
              </span>
            </li>
          ) : row.kind === 'result' ? (
            <li key={i} className="side-row verdict">
              <span className="side-who">result</span>
              <span className="side-main">{row.text}</span>
            </li>
          ) : (
            <li key={i} className={`side-row ${row.human ? 'you' : 'bot'}`}>
              <span className="side-who">{row.human ? 'you' : 'solver'}</span>
              <span className="side-main">
                <b>
                  {row.label}
                  {row.cost > 0 && <small> {row.cost}</small>}
                  {row.note && <small className="note-drew"> · {row.note}</small>}
                </b>
                {/* your rows teach; the solver's stay face down, like its cards */}
                {row.human && <MixCaption row={row} />}
              </span>
              {row.human && row.roll !== null && (
                <span className="side-grade">
                  <span className="roll">roll {row.roll}</span>
                  {row.mistake === 0 ? (
                    <span className="ok">✓</span>
                  ) : (
                    <span className="bad">
                      {Math.round(row.mistake ?? 0)}% off — roll said {row.words[row.correct as Act]}
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
              <span className="side-mix">the roll picks the strategy’s action for you</span>
            </span>
            <span className="side-grade">
              <span className="roll live">roll {pendingRoll}</span>
            </span>
          </li>
        )}
      </ol>
      <p className="side-total">
        mistakes this hand <b>{mistakes}%</b>
      </p>
    </aside>
  )
}

/** A monoline chip stack: one ellipse per few chips, the amount beside it. */
function Chips({ n, quiet }: { n: number; quiet?: boolean }) {
  if (n <= 0) return null
  const plates = Math.min(5, Math.ceil(n / 4))
  return (
    <span className="chipstack" aria-label={`${n} chips`}>
      <svg viewBox="0 0 26 22" aria-hidden>
        <g fill="none" stroke="currentColor" strokeWidth="1.2">
          {Array.from({ length: plates }, (_, i) => (
            <ellipse key={i} cx="13" cy={18 - i * 3.2} rx="9" ry="3" />
          ))}
        </g>
      </svg>
      {!quiet && <b>{n}</b>}
    </span>
  )
}

/** The last thing a seat did this hand, spoken at the seat itself. */
const lastActionOf = (rows: PlayRow[], seat: 0 | 1): string | null => {
  for (let i = rows.length - 1; i >= 0; i--) {
    const row = rows[i]
    if (row.kind === 'action' && row.seat === seat) {
      if (isThrow(row.act)) return row.human ? (row.act === 'n' ? 'stands pat' : 'draws one') : row.label
      return `${row.label}s${row.cost ? ` ${row.cost}` : ''}`
    }
  }
  return null
}

/** A card face: rank large, suit glyph beneath, the suit's own ink. Face down is oxblood pinstripe. */
export function CardGlyph({ card, hidden, small, thrown }: { card?: CardIndex; hidden?: boolean; small?: boolean; thrown?: boolean }) {
  const size = small ? 'pcard small' : 'pcard'
  if (hidden) return <span className={`${size} back`} aria-label="a card, face down" />
  if (card === undefined) return <span className={`${size} empty`} aria-label="no card yet" />
  return (
    <span className={`${size} face s${suitOf(card)}${thrown ? ' thrown' : ''}`} aria-label={sym(card)}>
      <b>{'23456'[rankOf(card)]}</b>
      <i>{SUIT_GLYPH[suitOf(card)]}</i>
    </span>
  )
}

/** What the buttons under the table say for `act` at this node. */
function buttonWord(h: Hand, act: Act): string {
  if (isThrow(act)) {
    const card = thrownCard(h, act)
    return card === null ? 'stand pat' : `throw ${sym(card)}`
  }
  const lines = linesOf(h)
  const word = betWord(act as Bet, lines[lines.length - 1])
  const cost = costOf(lines, act as Bet)
  return cost > 0 ? `${word} ${cost}` : word
}

export function PlayPanel({
  session,
  setSession,
  pack,
}: {
  session: Session
  setSession: Dispatch<SetStateAction<Session>>
  pack: PackState
}) {
  const { hand, chips, hands } = session
  const [sweep, setSweep] = useState<{ amounts: [number, number]; id: number } | null>(null)
  const prev = useRef<{ handId: string; phase: string } | null>(null)

  const phase = hand ? phaseOf(hand) : 'over'
  const over = phase === 'over'
  const handId = hand ? `${hands}:${hand.humanSeat}` : 'none'
  const roundKey = phase === 'round1' ? 'r1' : phase === 'over' ? 'over' : 'r2'

  // When a round closes its chips slide to the pile: a pair of transient
  // tokens spawned by watching the round change.
  useEffect(() => {
    const was = prev.current
    prev.current = { handId, phase: roundKey }
    if (!hand || !was || was.handId !== handId) {
      setSweep(null)
      return
    }
    if (was.phase === roundKey) return
    const amounts = was.phase === 'r1' ? replay([hand.l1]).inRound : replay(linesOf(hand)).inRound
    if (amounts[0] > 0 || amounts[1] > 0) setSweep((s) => ({ amounts, id: (s?.id ?? 0) + 1 }))
    // hand.l1/l2 are frozen once their round closes; `roundKey` is the trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handId, roundKey])

  if (!hand) {
    return (
      <Panel n="07" className="playpanel" k="Fig. 7 · the table" title="Play the solver." label="Play a hand against the solver">
        <p className="say">{pack.error ? `The deal pack did not load: ${pack.error}` : 'Shuffling the deal pack…'}</p>
      </Panel>
    )
  }

  const you = hand.humanSeat
  const bot = (1 - you) as 0 | 1
  const lines = linesOf(hand)
  const chipsNow = replay(lines)
  const pot = chipsNow.pot
  const liveInFor: [number, number] = over ? [0, 0] : chipsNow.inRound
  const pile = over ? pot : pot - liveInFor[0] - liveInFor[1]
  const reveal = over
  const yourHole = holeOf(hand.deal, you, hand.throws)
  const botHole = holeOf(hand.deal, bot, hand.throws)
  const b1 = board1(hand.deal)
  const b2 = hand.l2 !== null ? board2(hand.deal, hand.throws as [Throw, Throw]) : undefined
  const myTurn = !over && actorAt(hand) === you
  const legal = myTurn ? spotOf(hand).legal : []
  const drawing = phase === 'draw'

  const act = (a: Act) => setSession((s) => playAct(s, a))
  const next = () => setSession((s) => nextHand(s))

  const perSeat = (seat: 0 | 1) => {
    const n = session.seatHands[seat]
    if (n === 0) return '—'
    return `${signed(session.seatChips[seat] / n, 3)} over ${n}`
  }

  return (
    <Panel
      n="07"
      className="playpanel"
      k={`Fig. 7 · hand ${hands + 1} — you are P${you}`}
      live={!over}
      title="Play the solver."
      say={`The bot reads the frozen LCFR average at ${(FROZEN.iteration * WORKERS / 1e6).toFixed(0)}M hands per seat — exploitable for ${FROZEN.exploitability.toFixed(4)} chips/hand by a perfect adversary. It never adapts. Seats alternate; both holdings show at the end.`}
      label="Play a hand against the solver"
    >
      <div className="playwrap">
        <div className="playmain">
          <div className="table">
            <svg className="felt" viewBox="0 0 560 400" aria-hidden>
              <ellipse cx="280" cy="200" rx="266" ry="186" fill="none" stroke="currentColor" strokeWidth="1.5" />
              <ellipse cx="280" cy="200" rx="246" ry="168" fill="none" stroke="currentColor" strokeWidth="1" strokeOpacity="0.25" />
            </svg>
            <div className="seat-spot villain">
              <span className="who">the solver · P{bot}</span>
              <span className="cards">
                {botHole.map((c, i) => (reveal ? <CardGlyph key={c} card={c} /> : <CardGlyph key={i} hidden />))}
              </span>
              <span className="did">{lastActionOf(hand.rows, bot) ?? ' '}</span>
              <Chips n={liveInFor[bot]} />
            </div>
            <div className="seat-spot centre">
              <span className="cards board">
                <CardGlyph card={b1} />
                <CardGlyph card={b2} />
              </span>
              <span className="pot">
                <Chips n={pile} quiet /> pot {pot}
              </span>
            </div>
            <div className="seat-spot you">
              <Chips n={liveInFor[you]} />
              <span className="did">{lastActionOf(hand.rows, you) ?? ' '}</span>
              <span className="cards">
                {(drawing && myTurn ? hand.deal.order[you] : yourHole).map((c) => (
                  <CardGlyph key={c} card={c} />
                ))}
              </span>
              <span className="who">you · P{you}{drawing && myTurn ? ' · low, mid, top' : ''}</span>
            </div>
            {sweep && sweep.amounts[bot] > 0 && (
              <span key={`v${sweep.id}`} className="sweep from-villain" onAnimationEnd={() => setSweep(null)}>
                <Chips n={sweep.amounts[bot]} />
              </span>
            )}
            {sweep && sweep.amounts[you] > 0 && (
              <span key={`h${sweep.id}`} className="sweep from-you" onAnimationEnd={() => setSweep(null)}>
                <Chips n={sweep.amounts[you]} />
              </span>
            )}
          </div>

          <div className="play-buttons">
            {over ? (
              <button className="btn" onClick={next}>
                Next hand
              </button>
            ) : myTurn ? (
              legal.map((a, i) => (
                <button key={a} className={i === legal.length - 1 && !drawing ? 'btn' : i === 0 && drawing ? 'btn' : 'btn ghost'} onClick={() => act(a)}>
                  {buttonWord(hand, a)}
                </button>
              ))
            ) : (
              <span className="waiting">the solver is thinking…</span>
            )}
          </div>
        </div>

        <SidePanel rows={hand.rows} pendingRoll={hand.pendingRoll} over={over} />
      </div>

      <dl className="stat">
        <div>
          <dt>hands</dt>
          <dd>{hands}</dd>
        </div>
        <div>
          <dt>your chips</dt>
          <dd>{chips >= 0 ? `+${fmt(chips)}` : `−${fmt(-chips)}`}</dd>
        </div>
        <div>
          <dt>per hand</dt>
          <dd>{hands === 0 ? '—' : signed(chips / hands, 3)}</dd>
        </div>
        <div>
          <dt>as P0 · seat worth {signed(seatValue(0), 3)}</dt>
          <dd className="seat">{perSeat(0)}</dd>
        </div>
        <div>
          <dt>as P1 · seat worth {signed(seatValue(1), 3)}</dt>
          <dd className="seat">{perSeat(1)}</dd>
        </div>
      </dl>
      <p className="side-total">
        {pack.manifest
          ? `${pack.manifest.deals} pre-dealt hands in the pack (seed ${pack.manifest.seed}) — the same ${pack.manifest.deals} for every visitor, reshuffled each sitting, the seats alternating and the coach's roll drawn live — ${loadedDeals(pack.manifest, pack.loaded)} loaded; the strategy's digest is ${pack.manifest.strategy.sha256.slice(0, 12)}…. `
          : ''}
        {pack.error && pack.manifest
          ? `A later chunk was refused (${pack.error}), so play goes on with the hands already in and comes round to them again. `
          : ''}
        Matching the bot in both seats averages zero; a perfect adversary makes at most {signed(FROZEN.exploitability, 3)} a hand. A hand swings by several chips, so short sessions are mostly noise.
      </p>
    </Panel>
  )
}
