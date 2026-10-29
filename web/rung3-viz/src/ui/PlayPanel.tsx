import { useEffect, useRef, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react'

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
  actorAt,
  board1,
  board2,
  fmt,
  handNumber,
  holeOf,
  linesOf,
  nextHand,
  phaseOf,
  playAct,
  segments,
  spotOf,
  thrownCard,
  type Hand,
  type PlayRow,
  type Session,
} from '../play'
import { loadedDeals, type PackState } from '../deals'
import { describeStrategy } from '../../../solver-viz/src/mini/strategies'

/**
 * One hand against the shipped strategy, in the register of the Learn tab
 * beside it: a numbered window whose header states the spot (the hand, the
 * round, who is to act) the way Learn's range window states its seat and
 * round; the table as three hairline lanes in a well — the solver, the board
 * with the pot, you — rather than a drawn felt; the cards as Learn draws them,
 * rank and suit on one mono line; and the coach's rows carrying the
 * strategy's mix as the same fold / call / pot bands Learn's grid uses, with
 * a tick where your roll fell. Three cards a seat and two on the board. The
 * bot reads its mix out of the deal pack and never adapts. Seats alternate
 * each hand; both holdings are revealed at settlement, a fold included,
 * because the point is the line.
 *
 * Which strategy, and how it grades, is read off the pack's manifest — the
 * same two blocks `/solver`'s manifests carry, named through the same
 * `describeStrategy` — never typed here, so a re-export of the pack against
 * another release changes what the window's status bar says without a code
 * change.
 *
 * The session (hand, bankroll, the deals still to come) lives in App so
 * flipping tabs does not tear down a hand in progress. The transitions are
 * pure functions in `play.ts`, applied in updater form: they read only the
 * state they are given, so a double-fired click cannot act on a stale hand,
 * and the banked chips land in the same update as the hand that earned them.
 */

const pct = (x: number): string => `${(x * 100).toFixed(0)}`

/** The seat's own value when the strategy plays itself: what matching the bot averages. */
const seatValue = (valueP0: number, seat: 0 | 1): number => (seat === 0 ? valueP0 : -valueP0)

const signed = (n: number, digits = 0): string => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(digits)}`

/**
 * Which of the three mix bands an action belongs to, as Learn draws them:
 * going along (check, call, standing pat) is the calm band, pushing (bet,
 * raise, a throw) the hot one, the fold the quiet ink. The three throws
 * share the hot band at three depths — `d1` the low throw, faintest, up to
 * `d3` the top throw at full strength — the order the draw buttons' own
 * palette uses (`--throw-low` / `--throw-mid` / `--throw-top` in
 * `index.css`, the pink at 45%, 70% and full). The order is what is
 * borrowed, not the numbers: the `.mixband .pot.d*` rules key an `opacity`
 * of 0.5, 0.75 and 1 over the pot fill on these classes, and that need not
 * equal the palette's alphas.
 */
const BAND: Record<Act, string> = { f: 'fold', c: 'call', p: 'pot', n: 'call', l: 'pot d1', m: 'pot d2', t: 'pot d3' }

/**
 * The strategy's mix at a spot you played, as a band bar in roll order:
 * aggression owns the low rolls, so the bar reads left to right the way the
 * roll's segments do, and the tick is where your roll fell. Under it, each
 * action's word and share; the one you took in ivory.
 *
 * The bands are `segments` at their exact widths, no gaps and no minimum:
 * the same partition `prescribed` graded the roll against, so the tick at
 * `roll%` stands inside the band of the action the roll prescribed (it is
 * placed at the roll's left edge and extends right, as the segment's
 * half-open interval does). An action the strategy never takes here gets
 * no band and keeps its word below; a sliver for it would push every band
 * right of it off the roll's scale.
 */
function MixCaption({ row }: { row: Extract<PlayRow, { kind: 'action' }> }) {
  // row.words carries each action's word in the context the actor faced.
  const bands = segments(row.mix)
  return (
    <span className="side-mix">
      <span className="mixband" aria-hidden>
        {bands.map(({ act, start, end }) => (
          <i key={act} className={BAND[act]} style={{ width: `${end - start}%` }} />
        ))}
        {row.roll !== null && <em className="tick" style={{ left: `${row.roll}%` }} />}
      </span>
      <span className="mixwords">
        {bands.map(({ act }) => (
          <span key={act} className={act === row.act ? 'chosen' : ''}>
            {row.words[act]} {pct(row.mix[act] ?? 0)}%
          </span>
        ))}
      </span>
    </span>
  )
}

/**
 * The hand, decision by decision: every node a row, newest at the bottom.
 * Human rows carry the pre-drawn roll — the instruction — and the grade.
 */
function SidePanel({ rows }: { rows: PlayRow[] }) {
  const mistakes = rows.reduce(
    (sum, r) => sum + (r.kind === 'action' && r.human ? Math.round(r.mistake ?? 0) : 0),
    0,
  )
  return (
    <aside className="playside" aria-label="The hand, decision by decision">
      <span className="k">the hand, decision by decision</span>
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
                    <span className="ok">✓ the roll agrees</span>
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

/**
 * A card face as Learn draws one: ivory, the rank and the suit glyph on one
 * mono line, the red suits in red. A board card is a transparent frame with
 * ivory ink; a card not yet dealt is a dashed frame; face down is the
 * homepage's oxblood pinstripe.
 */
export function CardGlyph({ card, hidden, small }: { card?: CardIndex; hidden?: boolean; small?: boolean }) {
  const size = small ? 'pcard small' : 'pcard'
  if (hidden) return <span className={`${size} back`} aria-label="a card, face down" />
  if (card === undefined) return <span className={`${size} empty`} aria-label="no card yet" />
  return (
    <span className={`${size} face s${suitOf(card)}`} aria-label={sym(card)}>
      {'23456'[rankOf(card)]}
      {SUIT_GLYPH[suitOf(card)]}
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

/**
 * The trainer's window: the same frame Learn's range window has — a title
 * bar with the window's number and field, the pane, and a status bar along
 * the foot naming the strategy. `PackBoundary` and the loading state render
 * the same frame around a sentence, so the window never changes shape.
 */
export function PlayWindow({
  k,
  live,
  foot,
  children,
}: {
  k: string
  live?: boolean
  foot?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="panel wide playpanel" aria-label="Play a hand against the solver">
      <div className="titlebar">
        <span className="dots">
          <i />
          <i />
          <i />
        </span>
        <span className="k">{k}</span>
        {live && (
          <span className="live">
            <i />
            live
          </span>
        )}
        <span className="no">01</span>
      </div>
      <div className="pane">{children}</div>
      {foot && <div className="statusbar">{foot}</div>}
    </section>
  )
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
      <PlayWindow k="01 · the table">
        <p className="playnote">{pack.error ? `The deal pack did not load: ${pack.error}` : 'Shuffling the deal pack…'}</p>
      </PlayWindow>
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

  // A hand exists only once the pack loaded, so the manifest is here; its grade may still be null.
  const manifest = pack.manifest
  const grade = manifest?.grade ?? null
  const words = manifest ? describeStrategy(manifest) : null
  const seatWorth = (seat: 0 | 1) => (grade ? ` · seat worth ${signed(seatValue(grade.valueP0, seat), 3)}` : '')

  // The header states the spot, as Learn's does: the hand, where in it we
  // are, and whose decision it is. The subline says what to do about it.
  const n = handNumber(session)
  const result = hand.result ?? 0
  const stage = drawing ? 'the draw' : phase === 'round1' ? 'round 1' : over ? 'settled' : 'round 2'
  const headline = over
    ? `Hand ${n} · ${result === 0 ? 'a push' : result > 0 ? `you win ${fmt(result)}` : `the solver wins ${fmt(-result)}`}`
    : `Hand ${n} · ${stage} — ${myTurn ? 'you' : 'the solver'} to act`
  const subline = over
    ? 'Both holdings are up. The next hand swaps seats; the deal pack moves on.'
    : !myTurn
      ? 'The solver reads its mix at this spot from the deal pack and never adapts.'
      : drawing
        ? 'Your three cards are shown low, mid, top. Throw one, or stand pat.'
        : `You are P${you}. Pick an action; the roll grades it against the strategy’s mix once you have.`

  const foot = (
    <>
      <span>
        {words ? (
          <>
            <b>{words.name}</b> · {words.detail}
            {words.grade ? (
              <>
                {' '}
                · exploitable for <b>{words.grade}</b> chips/hand
              </>
            ) : (
              ' · ungraded'
            )}
          </>
        ) : (
          'the strategy in the deal pack'
        )}
      </span>
      {manifest && (
        <span>
          {manifest.deals} deals · seed {manifest.seed} · {loadedDeals(manifest, pack.loaded)} loaded · digest{' '}
          {manifest.strategy.sha256.slice(0, 12)}
          {pack.error ? ` · a later chunk was refused (${pack.error}); play goes on with the hands already in` : ''}
        </span>
      )}
    </>
  )

  return (
    <PlayWindow k={`01 · the table — hand ${n} · you are P${you}`} live={!over} foot={foot}>
      <div className="playhead">
        <div className="who">
          {headline}
          <small>{subline}</small>
        </div>
        <div className="cue">
          <b>{chips >= 0 ? `+${fmt(chips)}` : `−${fmt(-chips)}`}</b>
          <span>
            your chips · {hands} {hands === 1 ? 'hand' : 'hands'}
            {hands > 0 ? ` · ${signed(chips / hands, 3)} a hand` : ''}
          </span>
        </div>
      </div>

      <div className="playwrap">
        <div className="playmain">
          <div className="table">
            <div className="lane villain">
              <span className="who">the solver · P{bot}</span>
              <span className="cards">
                {botHole.map((c, i) => (reveal ? <CardGlyph key={c} card={c} /> : <CardGlyph key={i} hidden />))}
              </span>
              <span className="did">{lastActionOf(hand.rows, bot) ?? ''}</span>
              <Chips n={liveInFor[bot]} />
            </div>
            <div className="lane centre">
              <span className="who">board</span>
              <span className="cards board">
                <CardGlyph card={b1} />
                <CardGlyph card={b2} />
              </span>
              <span className="did" />
              <span className="pot">
                <Chips n={pile} quiet /> pot <b>{pot}</b>
              </span>
            </div>
            <div className="lane you">
              <span className="who">
                you · P{you}
                {drawing && myTurn ? <small>low · mid · top</small> : null}
              </span>
              <span className="cards">
                {(drawing && myTurn ? hand.deal.order[you] : yourHole).map((c) => (
                  <CardGlyph key={c} card={c} />
                ))}
              </span>
              <span className="did">{lastActionOf(hand.rows, you) ?? ''}</span>
              <Chips n={liveInFor[you]} />
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
            {myTurn && hand.pendingRoll !== null && (
              <span className="rollcue">
                <span className="roll">roll {hand.pendingRoll}</span>
                the roll picks the strategy’s action for you
              </span>
            )}
          </div>

          <dl className="stat">
            <div>
              <dt>as P0{seatWorth(0)}</dt>
              <dd className="seat">{perSeat(0)}</dd>
            </div>
            <div>
              <dt>as P1{seatWorth(1)}</dt>
              <dd className="seat">{perSeat(1)}</dd>
            </div>
          </dl>
          <p className="playnote">
            {manifest
              ? `The pack is ${manifest.deals} pre-dealt hands — the same ${manifest.deals} for every visitor, reshuffled each sitting, not fresh draws; the seats alternate and the coach's roll is drawn live, so a deal plays differently each time it comes round. `
              : ''}
            Matching the bot in both seats averages zero{grade ? `; a perfect adversary makes at most ${grade.exploitability.toFixed(3)} a hand` : ''}.
            A hand swings by several chips, so short sessions are mostly noise.
          </p>
        </div>

        <SidePanel rows={hand.rows} />
      </div>
    </PlayWindow>
  )
}
