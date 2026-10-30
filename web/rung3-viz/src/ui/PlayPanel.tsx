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
  type Manifest,
  type PlayRow,
  type Session,
} from '../play'
import { loadedDeals, type PackState } from '../deals'
import { describeStrategy } from '../../../solver-viz/src/mini/strategies'

/**
 * One hand against the shipped strategy, in the register of the solver: the
 * table at left as three hairline lanes (the solver, the board with the pot,
 * you), the actions under it, and the coach at right, where every decision
 * of the hand is a row and each of yours carries the strategy's mix as bands
 * in the action colours (bet and raise oxblood, call green, check and fold
 * blue) with a tick where your roll fell. Under the table and the coach, the
 * sitting's numbers and the ledger of settled hands as a hairline table.
 * Three cards a seat and two on the board. The bot reads its mix out of the
 * deal pack and never adapts. Seats alternate each hand; both holdings are
 * revealed at settlement, a fold included, because the point is the line.
 *
 * Which strategy, and how it grades, is read off the pack's manifest — the
 * same two blocks `/solver`'s manifests carry, named through the same
 * `describeStrategy` — never typed here, so a re-export of the pack against
 * another release changes what the page head says without a code change.
 *
 * The session (hand, bankroll, the deals still to come, the ledger) lives in
 * App so flipping tabs does not tear down a hand in progress. The
 * transitions are pure functions in `play.ts`, applied in updater form: they
 * read only the state they are given, so a double-fired click cannot act on
 * a stale hand, and the banked chips land in the same update as the hand
 * that earned them.
 */

const pct = (x: number): string => `${(x * 100).toFixed(0)}`

/** The seat's own value when the strategy plays itself: what matching the bot averages. */
const seatValue = (valueP0: number, seat: 0 | 1): number => (seat === 0 ? valueP0 : -valueP0)

const signed = (n: number, digits = 0): string => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(digits)}`

/**
 * One sentence naming the bot the pack was dealt against: its release, what
 * kind of strategy it is, and its grade, all read off the manifest. The words
 * come from `/solver`'s `describeStrategy`, less the internal "rung-N" prefix.
 */
export function packWords(manifest: Manifest | null | undefined): string {
  if (!manifest) return 'The deal pack is loading.'
  const words = describeStrategy(manifest)
  const name = words.name.replace(/^rung-\d+\s+/, '')
  const grade = words.grade ? `, exploitable for ${words.grade} chips a hand` : ', not yet graded'
  return `The solver is release ${manifest.strategy.release}: ${name}, ${words.detail}${grade}.`
}

/**
 * Which action colour a choice wears, by what it does: a bet, a raise or a
 * throw pushes (oxblood), a call goes along (green), a check, a fold or
 * standing pat is the quiet choice (blue). The three throws share the
 * oxblood at three depths, low to top. `c` is a check or a call by context,
 * so its word decides.
 */
function bandOf(act: Act, word: string | undefined): string {
  switch (act) {
    case 'p':
      return 'bet'
    case 'c':
      return word?.startsWith('call') ? 'call' : 'check'
    case 'f':
    case 'n':
      return 'check'
    case 'l':
      return 'bet d3'
    case 'm':
      return 'bet d2'
    case 't':
      return 'bet'
  }
}

/**
 * The strategy's mix at a spot you played, as a band bar in roll order:
 * aggression owns the low rolls, so the bar reads left to right the way the
 * roll's segments do, and the tick is where your roll fell. Under it, each
 * action's word and share, the one you took in ink.
 */
function MixCaption({ row }: { row: Extract<PlayRow, { kind: 'action' }> }) {
  // AGGRESSION is the roll's own segment order; row.words carries each
  // action's word in the context the actor faced.
  const acts = AGGRESSION.filter((a) => a in row.mix)
  return (
    <span className="side-mix">
      <span className="mixband" aria-hidden>
        {acts.map((a) => (
          <i key={a} className={bandOf(a, row.words[a])} style={{ flexGrow: Math.max(row.mix[a] ?? 0, 0.002) }} />
        ))}
        {row.roll !== null && <em className="tick" style={{ left: `${row.roll}%` }} />}
      </span>
      <span className="mixwords">
        {acts.map((a) => (
          <span key={a} className={a === row.act ? 'chosen' : ''}>
            <i className={`akey ${bandOf(a, row.words[a])}`} />
            {row.words[a]} {pct(row.mix[a] ?? 0)}%
          </span>
        ))}
      </span>
    </span>
  )
}

/**
 * The coach: the hand, decision by decision, every node a row, newest at the
 * bottom. Your rows carry the strategy's mix, the pre-drawn roll (the
 * instruction) and the grade; the solver's stay face down, like its cards.
 */
function Coach({ rows }: { rows: PlayRow[] }) {
  const mistakes = rows.reduce(
    (sum, r) => sum + (r.kind === 'action' && r.human ? Math.round(r.mistake ?? 0) : 0),
    0,
  )
  return (
    <aside className="coach" aria-label="The coach: the hand, decision by decision">
      <p className="coach-k">Coach · the solver's mix at your spots</p>
      <ol>
        {rows.map((row, i) =>
          row.kind === 'board' ? (
            <li key={i} className="side-row board">
              <span className="side-who">Board {row.n}</span>
              <span className="side-main">
                <CardGlyph card={row.card} small />
              </span>
            </li>
          ) : row.kind === 'result' ? (
            <li key={i} className="side-row verdict">
              <span className="side-who">Result</span>
              <span className="side-main">{row.text}</span>
            </li>
          ) : (
            <li key={i} className={`side-row ${row.human ? 'you' : 'bot'}`}>
              <span className="side-who">{row.human ? 'You' : 'Solver'}</span>
              <span className="side-main">
                <b>
                  {row.label}
                  {row.cost > 0 && <small> {row.cost}</small>}
                  {row.note && <small className="note-drew"> · {row.note}</small>}
                </b>
                {row.human && <MixCaption row={row} />}
              </span>
              {row.human && row.roll !== null && (
                <span className="side-grade">
                  <span className="roll">roll {row.roll}</span>
                  {row.mistake === 0 ? (
                    <span className="ok">The roll agrees</span>
                  ) : (
                    <span className="bad">
                      {Math.round(row.mistake ?? 0)}% off; the roll said {row.words[row.correct as Act]}
                    </span>
                  )}
                </span>
              )}
            </li>
          ),
        )}
      </ol>
      <p className="side-total">
        Mistakes this hand <b>{mistakes}%</b>
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
 * A card face, the bleed pip: the rank small in the top-left, one suit glyph
 * too big for the card cropped by its lower-right edge, in the suit's ink
 * (clubs green, diamonds blue, hearts oxblood). A card not yet dealt is a
 * dashed frame; face down is the chip's oxblood pinstripe.
 */
export function CardGlyph({ card, hidden, small }: { card?: CardIndex; hidden?: boolean; small?: boolean }) {
  const size = small ? 'pcard small' : 'pcard'
  if (hidden) return <span className={`${size} back`} aria-label="a card, face down" />
  if (card === undefined) return <span className={`${size} empty`} aria-label="no card yet" />
  return (
    <span className={`${size} face s${suitOf(card)}`} aria-label={sym(card)}>
      <b>{'23456'[rankOf(card)]}</b>
      <i aria-hidden="true">{SUIT_GLYPH[suitOf(card)]}</i>
    </span>
  )
}

/** A card as text inside a line of type, its suit in the suit's colour. */
const CardText = ({ card }: { card: CardIndex }) => (
  <span className={`ctext s${suitOf(card)}`}>
    {'23456'[rankOf(card)]}
    {SUIT_GLYPH[suitOf(card)]}
  </span>
)

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
 * The trainer's frame: a section on the page's paper. `PackBoundary` and the
 * loading state render the same frame around a sentence, so the trainer
 * never changes place.
 */
export function PlayWindow({ children }: { children: ReactNode }) {
  return (
    <section className="trainer" aria-label="Play a hand against the solver">
      {children}
    </section>
  )
}

/** The ledger: every settled hand of the sitting, newest first, as a hairline table. */
function Ledger({ settled, hands }: { settled: Hand[]; hands: number }) {
  return (
    <section className="ledger-block" aria-label="Settled hands">
      <p className="coach-k">Ledger · settled hands</p>
      {settled.length === 0 ? (
        <p className="playnote">No hand is settled yet. Each one lands here with your choices and the solver's mix at each.</p>
      ) : (
        <table className="hair handlog">
          <thead>
            <tr>
              <th>Hand</th>
              <th>Seat</th>
              <th>Your cards · board</th>
              <th>Your choices · the solver's mix</th>
              <th className="r">Mistakes</th>
              <th className="r">Result</th>
            </tr>
          </thead>
          <tbody>
            {settled.map((h, i) => {
              const you = h.humanSeat
              const hole = holeOf(h.deal, you, h.throws)
              const board = [board1(h.deal), ...(h.l2 !== null ? [board2(h.deal, h.throws as [Throw, Throw])] : [])]
              const mine = h.rows.filter((r): r is Extract<PlayRow, { kind: 'action' }> => r.kind === 'action' && r.human)
              const mistakes = mine.reduce((sum, r) => sum + Math.round(r.mistake ?? 0), 0)
              const result = h.result ?? 0
              return (
                <tr key={hands - i}>
                  <td className="n">{hands - i}</td>
                  <td className="n">P{you}</td>
                  <td className="n cards-cell">
                    {hole.map((c) => (
                      <CardText key={c} card={c} />
                    ))}
                    <span className="sep">·</span>
                    {board.map((c) => (
                      <CardText key={c} card={c} />
                    ))}
                  </td>
                  <td className="choices">
                    {mine.map((r, k) => (
                      <span key={k}>
                        {k > 0 && <span className="arrow"> → </span>}
                        <b>{r.label}</b>{' '}
                        <span className="mixline">
                          ({AGGRESSION.filter((a) => a in r.mix)
                            .map((a) => `${r.words[a]} ${pct(r.mix[a] ?? 0)}`)
                            .join(' · ')}
                          )
                        </span>
                      </span>
                    ))}
                  </td>
                  <td className="r n">{mistakes}%</td>
                  <td className={`r n ${result > 0 ? 'won' : result < 0 ? 'lost' : ''}`}>{result === 0 ? '0' : signed(result, Number.isInteger(result) ? 0 : 2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
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
      <PlayWindow>
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

  // A hand exists only once the pack loaded, so the manifest is here; its grade may still be null.
  const manifest = pack.manifest
  const grade = manifest?.grade ?? null
  const seatWorth = (seat: 0 | 1) => (grade ? `seat worth ${signed(seatValue(grade.valueP0, seat), 3)}` : 'seat value ungraded')

  // The turn line states the spot: the hand, where in it we are, and whose
  // decision it is; the line under it says what to do about it. `hands` is
  // banked in the same update that settles a hand, so once a hand is over it
  // is already counted: its number is `hands`, not the next.
  const n = over ? hands : hands + 1
  const result = hand.result ?? 0
  const stage = drawing ? 'the draw' : phase === 'round1' ? 'round 1' : over ? 'settled' : 'round 2'
  const headline = over
    ? result === 0
      ? 'A push.'
      : result > 0
        ? `You win ${fmt(result)}.`
        : `The solver wins ${fmt(-result)}.`
    : `${myTurn ? 'You' : 'The solver'} to act, ${stage}.`
  const subline = over
    ? 'Both holdings are up. The next hand swaps seats, and the deal pack moves on.'
    : !myTurn
      ? 'The solver reads its mix at this spot from the deal pack and never adapts.'
      : drawing
        ? 'Your three cards are shown low, mid, top. Throw one, or stand pat.'
        : `You are P${you}. Pick an action; the roll grades it against the strategy’s mix once you have.`

  const perHand = hands > 0 ? signed(chips / hands, 3) : '—'

  return (
    <PlayWindow>
      <div className="trainer-grid">
        <div className="trainer-main">
          <div className="turn">
            <p className="turn-k">
              Hand {n} · you are P{you} · {stage}
            </p>
            <p className="turn-say">{headline}</p>
            <p className="turn-sub">{subline}</p>
          </div>

          <div className="felt">
            <div className="lane villain">
              <span className="who">
                The solver <small>P{bot}</small>
              </span>
              <span className="cards">
                {botHole.map((c, i) => (reveal ? <CardGlyph key={c} card={c} /> : <CardGlyph key={i} hidden />))}
              </span>
              <span className="did">{lastActionOf(hand.rows, bot) ?? ''}</span>
              <Chips n={liveInFor[bot]} />
            </div>
            <div className="lane centre">
              <span className="who">Board</span>
              <span className="cards board">
                <CardGlyph card={b1} />
                <CardGlyph card={b2} />
              </span>
              <span className="did" />
              <span className="pot">
                <Chips n={pile} quiet /> Pot <b>{pot}</b>
              </span>
            </div>
            <div className="lane you">
              <span className="who">
                You <small>P{you}{drawing && myTurn ? ' · low, mid, top' : ''}</small>
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
              <button className="btn primary" onClick={next}>
                Deal the next hand
              </button>
            ) : myTurn ? (
              legal.map((a) => {
                const word = buttonWord(hand, a)
                return (
                  <button key={a} className="btn" onClick={() => act(a)}>
                    <i className={`akey ${bandOf(a, word)}`} aria-hidden />
                    {word}
                  </button>
                )
              })
            ) : (
              <span className="waiting">The solver is choosing…</span>
            )}
            {myTurn && hand.pendingRoll !== null && (
              <span className="rollcue">
                <span className="roll">roll {hand.pendingRoll}</span>
                The roll picks the strategy’s action for you.
              </span>
            )}
          </div>
        </div>

        <Coach rows={hand.rows} />
      </div>

      <section className="sitting" aria-label="This sitting">
        <p className="coach-k">This sitting</p>
        <dl className="stats">
          <div className="stat">
            <dt>Hands settled</dt>
            <dd>{hands}</dd>
          </div>
          <div className="stat">
            <dt>Your chips</dt>
            <dd>{chips >= 0 ? `+${fmt(chips)}` : `−${fmt(-chips)}`}</dd>
          </div>
          <div className="stat">
            <dt>Per hand</dt>
            <dd>{perHand}</dd>
          </div>
          {([0, 1] as const).map((seat) => {
            const played = session.seatHands[seat]
            return (
              <div className="stat" key={seat}>
                <dt>As P{seat}</dt>
                <dd>
                  {played === 0 ? '—' : signed(session.seatChips[seat] / played, 3)}
                  <small>
                    {played === 0 ? 'no hands' : `over ${played}`} · {seatWorth(seat)}
                  </small>
                </dd>
              </div>
            )
          })}
        </dl>
        <p className="playnote">
          Matching the bot in both seats averages zero{grade ? `; a perfect adversary makes at most ${grade.exploitability.toFixed(3)} a hand` : ''}.
          A hand swings by several chips, so short sessions are mostly noise.
        </p>
      </section>

      <Ledger settled={session.settled} hands={hands} />

      {manifest && (
        <p className="packline">
          {manifest.deals} deals · seed {manifest.seed} · {loadedDeals(manifest, pack.loaded)} loaded · digest{' '}
          {manifest.strategy.sha256.slice(0, 12)}
          {pack.error ? ` · a later chunk was refused (${pack.error}); play goes on with the hands already in` : ''}
        </p>
      )}
    </PlayWindow>
  )
}
