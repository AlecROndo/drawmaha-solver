/**
 * Mini-drawmaha's hand so far as a two-lane score: one lane per seat, one
 * column per street — board card 1, round 1, the draw, board card 2,
 * round 2 — with the board along the top and the pot along the bottom.
 * Every chip is a decision that was made; clicking one rewinds the hand to
 * just before it. The decision on screen is the yellow cell: the actor's
 * legal actions as buttons, each with the share of that seat's range that
 * takes it under the strategy; clicking one plays it. The board cells are
 * pickers: click one and the fifteen cards fan out beneath the score.
 *
 * The line is the public record only — what each seat did, how many cards
 * each threw, which board cards came — never anybody's cards; that is what
 * a range is conditioned on.
 */

import { type Bet, actorOf, betWord, isClosed, isFold, legalBets, replay } from '../mini/grammar'
import { MINI_DECK, type MiniCard } from '../mini/cards'
import { CardFace } from './bars'
import type { GameUI } from './game'

export interface MiniLine {
  board1: MiniCard
  /** round-1 letters so far: x check, c call, p pot, f fold */
  r1: string
  /** throw counts decided so far, seat 0 first */
  draws: (0 | 1)[]
  board2: MiniCard | null
  r2: string
}

/** Where the hand stands: whose decision, or which chance card is owed, or over. */
export type Decision =
  | { kind: 'bet'; round: 1 | 2; player: 0 | 1 }
  | { kind: 'draw'; player: 0 | 1 }
  | { kind: 'board2' }
  | { kind: 'over'; why: 'fold' | 'showdown' }

/** The decision the line is standing on, read off the grammar. */
export function decisionOf(line: MiniLine): Decision {
  const chips = replay(line.r2 ? [line.r1, line.r2] : [line.r1])
  if (isFold(line.r1)) return { kind: 'over', why: 'fold' }
  if (!isClosed(line.r1, chips.behind)) return { kind: 'bet', round: 1, player: actorOf(line.r1) }
  if (line.draws.length < 2) return { kind: 'draw', player: line.draws.length as 0 | 1 }
  if (line.board2 === null) return { kind: 'board2' }
  if (isFold(line.r2)) return { kind: 'over', why: 'fold' }
  // Both stacks empty after round 1 means round 2 never opens.
  if (chips.behind[0] === 0 && chips.behind[1] === 0) return { kind: 'over', why: 'showdown' }
  if (!isClosed(line.r2, chips.behind) || line.r2 === '') return { kind: 'bet', round: 2, player: actorOf(line.r2) }
  return { kind: 'over', why: 'showdown' }
}

export const linesOf = (line: MiniLine): string[] => (line.board2 === null ? [line.r1] : [line.r1, line.r2])

/** The options at a decision: a bet's letters, or the two throw counts. */
export function nextOptions(line: MiniLine): { key: string; word: string }[] {
  const d = decisionOf(line)
  if (d.kind === 'bet') {
    const lines = linesOf(line)
    const cur = d.round === 1 ? line.r1 : line.r2
    return legalBets(lines).map((b: Bet) => ({ key: b, word: betWord(b, cur) }))
  }
  if (d.kind === 'draw') return [{ key: '0', word: 'stand pat' }, { key: '1', word: 'throw one' }]
  return []
}

/**
 * The colour of an action's chip or button, by its key. Pink is the action
 * that changes the hand — a pot bet or raise ('p'), or throwing a card ('1');
 * a fold ('f') is the quiet ink; a check, call or stand pat is the yellow of
 * the hand going on as it was. One rule for the played chips and the to-act
 * buttons, so they agree.
 */
export const toneOf = (key: string): 'p' | 'd' | 'c' => (key === 'p' || key === '1' ? 'p' : key === 'f' ? 'd' : 'c')

/** The line after taking `key` at its current decision. */
export function advance(line: MiniLine, key: string): MiniLine {
  const d = decisionOf(line)
  if (d.kind === 'bet') {
    const bet = key as Bet
    const cur = d.round === 1 ? line.r1 : line.r2
    const sym = bet === 'c' ? (cur.endsWith('p') ? 'c' : 'x') : bet
    return d.round === 1 ? { ...line, r1: line.r1 + sym } : { ...line, r2: line.r2 + sym }
  }
  if (d.kind === 'draw') return { ...line, draws: [...line.draws, key === '1' ? 1 : 0] }
  throw new Error(`nothing to advance at a ${d.kind}`)
}

/** The cards not yet on the board, for the pickers. */
export const freeCards = (line: MiniLine, except: MiniCard | null): MiniCard[] =>
  MINI_DECK.filter((c) => c !== line.board1 && c !== line.board2 || c === except)

export function MiniScore({
  game,
  line,
  onLine,
  picking,
  onPick,
  shares,
}: {
  game: GameUI
  line: MiniLine
  onLine: (l: MiniLine) => void
  /** which board picker is open */
  picking: 1 | 2 | null
  onPick: (which: 1 | 2 | null) => void
  /** the reach-weighted share of the actor's range taking each next option, by key; null while loading */
  shares: Record<string, number> | null
}) {
  const d = decisionOf(line)
  const chips = replay(linesOf(line))
  const pot1 = replay([line.r1]).pot

  // Rewinds: to before letter k of a round, to a draw decision, to before board 2.
  const rewindR1 = (k: number) => onLine({ ...line, r1: line.r1.slice(0, k), draws: [], board2: null, r2: '' })
  const rewindDraw = (seat: 0 | 1) => onLine({ ...line, draws: line.draws.slice(0, seat), board2: null, r2: '' })
  const rewindR2 = (k: number) => onLine({ ...line, r2: line.r2.slice(0, k) })

  const chipsOf = (round: 1 | 2, seat: 0 | 1) => {
    const letters = round === 1 ? line.r1 : line.r2
    const items: React.ReactNode[] = []
    for (let k = 0; k < letters.length; k++) {
      if (actorOf(letters.slice(0, k)) !== seat) continue
      const sym = letters[k]
      const word = sym === 'x' ? 'check' : sym === 'c' ? 'call' : sym === 'f' ? 'fold' : letters.slice(0, k).endsWith('p') ? 'raise' : 'bet'
      items.push(
        <button key={k} type="button" className={`act ${toneOf(sym)}`} onClick={() => (round === 1 ? rewindR1(k) : rewindR2(k))} aria-label={`rewind to before seat ${seat}'s ${word}`}>
          {word}
        </button>,
      )
    }
    if (d.kind === 'bet' && d.round === round && d.player === seat) {
      items.push(<ToAct key="now" options={nextOptions(line)} shares={shares} onPlay={(k) => onLine(advance(line, k))} />)
    }
    return items.length ? items : <span className="none">·</span>
  }

  const drawChip = (seat: 0 | 1) => {
    const t = line.draws[seat]
    if (t !== undefined) {
      return (
        <button type="button" className="act d" onClick={() => rewindDraw(seat)} aria-label={`rewind to seat ${seat}'s draw`}>
          {t === 1 ? 'threw one' : 'stood pat'}
        </button>
      )
    }
    if (d.kind === 'draw' && d.player === seat) {
      return <ToAct options={nextOptions(line)} shares={shares} onPlay={(k) => onLine(advance(line, k))} />
    }
    return <span className="none">·</span>
  }

  const boardCell = (which: 1 | 2) => {
    const card = which === 1 ? line.board1 : line.board2
    const live = which === 1 || (line.draws.length === 2)
    if (!live) return <span className="none">—</span>
    return (
      <button type="button" className={`act boardpick ${picking === which ? 'you' : ''} ${card === null ? 'd' : ''}`} onClick={() => onPick(picking === which ? null : which)} aria-expanded={picking === which} aria-label={`choose board card ${which}`}>
        {card === null ? <span className="pickword">pick</span> : <CardFace game={game} card={card} />}
      </button>
    )
  }

  const streets = ['board 1', 'round 1', 'draw', 'board 2', 'round 2', 'showdown']
  return (
    <div className="miniscore">
      <div className="score" style={{ '--n': streets.length } as React.CSSProperties}>
        <div className="h" />
        {streets.map((h) => (
          <div key={h} className="h st">
            {h}
          </div>
        ))}

        <div className="lane">board</div>
        <div className="st board">{boardCell(1)}</div>
        <div className="st">
          <span className="none">—</span>
        </div>
        <div className="st">
          <span className="none">—</span>
        </div>
        <div className="st board">{boardCell(2)}</div>
        <div className="st">
          <span className="none">—</span>
        </div>
        <div className="st">
          <span className="none">{d.kind === 'over' ? (d.why === 'fold' ? 'folded' : 'showdown') : '—'}</span>
        </div>

        {([0, 1] as const).map((seat) => (
          <SeatLane key={seat} seat={seat} r1={chipsOf(1, seat)} draw={drawChip(seat)} r2={line.board2 === null ? <span className="none">·</span> : chipsOf(2, seat)} />
        ))}

        <div className="lane">pot</div>
        <div className="st pot">
          <b>2</b>
        </div>
        <div className="st pot">
          <b>{pot1}</b>
        </div>
        <div className="st pot">
          <b>{pot1}</b>
        </div>
        <div className="st pot">
          <b>{pot1}</b>
        </div>
        <div className="st pot">
          <b>{chips.pot}</b>
        </div>
        <div className="st pot">
          <span className="none">·</span>
        </div>
      </div>

      {picking !== null && (
        <div className="picker" role="listbox" aria-label={`board card ${picking}`}>
          {freeCards(line, picking === 1 ? line.board1 : line.board2).map((c) => {
            const chosen = (picking === 1 ? line.board1 : line.board2) === c
            return (
              <button
                key={c}
                type="button"
                role="option"
                aria-selected={chosen}
                className={`pickcard ${chosen ? 'on' : ''}`}
                onClick={() => {
                  // A new board card 1 restarts the hand; a new board card 2 restarts round 2.
                  onLine(picking === 1 ? { board1: c, r1: '', draws: [], board2: null, r2: '' } : { ...line, board2: c, r2: '' })
                  onPick(null)
                }}
              >
                <CardFace game={game} card={c} />
              </button>
            )
          })}
        </div>
      )}

      <p className="legend-line scorehint">
        {d.kind === 'over'
          ? d.why === 'fold'
            ? 'the hand ended on a fold · click a played chip to rewind to it'
            : 'showdown · click a played chip to rewind to it'
          : d.kind === 'board2'
            ? 'pick board card 2 to open round 2'
            : 'the yellow cell is the decision on screen: click an action to play it (the number is how much of that seat’s range takes it) · click a played chip to rewind · click a board card to change it'}
      </p>
    </div>
  )
}

/** The decision on screen: its legal actions as buttons, each with the share of the seat's range that takes it. */
function ToAct({ options, shares, onPlay }: { options: { key: string; word: string }[]; shares: Record<string, number> | null; onPlay: (key: string) => void }) {
  return (
    <span className="toact" aria-current="step">
      <span className="toact-lab">to act</span>
      {options.map((o) => (
        <button key={o.key} type="button" className={`act play ${toneOf(o.key)}`} onClick={() => onPlay(o.key)} aria-label={`play ${o.word}`}>
          {o.word}
          {shares && shares[o.key] !== undefined && <b>{Math.round(shares[o.key] * 100)}%</b>}
        </button>
      ))}
    </span>
  )
}

function SeatLane({ seat, r1, draw, r2 }: { seat: 0 | 1; r1: React.ReactNode; draw: React.ReactNode; r2: React.ReactNode }) {
  return (
    <>
      <div className="lane">seat {seat}</div>
      <div className="st">
        <span className="none">·</span>
      </div>
      <div className="st chips">{r1}</div>
      <div className="st">{draw}</div>
      <div className="st">
        <span className="none">·</span>
      </div>
      <div className="st chips">{r2}</div>
      <div className="st">
        <span className="none">·</span>
      </div>
    </>
  )
}
