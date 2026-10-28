/**
 * The hand so far as a strip, the backup layout (`index.ts` picks it with
 * `?score=strip`). One token per event, left to right in the order it
 * was played: the board card, each action tagged with its seat, each draw,
 * board card 2. The pot ticks along under every token, the street names sit
 * above the group they begin, and the decision on screen is the strip's live
 * head: a yellow-edged token holding the actor's options. There is no grid,
 * so there is no cell with nothing in it.
 */

import { type Step, byStreet, endWords, headLabel, headOf, rewindLabel, stepsOf, streetOf } from './steps'
import { BoardPick, Hint, Opts, Picker, type ScoreProps } from './parts'

export function StripScore(props: ScoreProps) {
  const { game, line, onLine, picking, onPick, shares } = props
  const steps = stepsOf(line)
  const head = headOf(line)
  const d = head.decision
  const groups = byStreet(steps)
  const headStreet = streetOf(d)
  // The head joins the group of its street, or opens that street as a new group.
  const last = groups[groups.length - 1]
  const headInLast = d.kind !== 'over' && d.kind !== 'board2' && last && last.street === headStreet

  const token = (s: Step, i: number) => {
    if (s.kind === 'board') {
      const owed = s.card === null
      return (
        <div key={i} className={`tok-wrap ${owed ? 'head' : ''}`} aria-current={owed ? 'step' : undefined}>
          <small className={`who ${owed ? 'lit' : ''}`}>{owed ? 'owed' : 'dealt'}</small>
          <BoardPick game={game} which={s.which} card={s.card} picking={picking} onPick={onPick} />
          <small className="potline">pot {s.pot}</small>
        </div>
      )
    }
    if (s.kind === 'end') {
      return (
        <div key={i} className="tok-wrap">
          <small className="who">{s.why === 'fold' ? 'over' : 'showdown'}</small>
          <span className="endtok">{s.why === 'showdown' ? 'showdown' : `seat ${s.winner} takes it`}</span>
          <small className="potline">pot {s.pot}</small>
        </div>
      )
    }
    return (
      <div key={i} className="tok-wrap">
        <small className="who">seat {s.seat}</small>
        <button type="button" className={`tok ${s.tone}`} onClick={() => onLine(s.back)} aria-label={rewindLabel(s)}>
          {s.word}
        </button>
        <small className="potline">pot {s.pot}</small>
      </div>
    )
  }

  const headTok = (
    <div className="tok-wrap head" aria-current="step">
      <small className="who lit">{headLabel(d)}</small>
      <Opts head={head} shares={shares} line={line} onLine={onLine} />
      <small className="potline">pot {head.pot}</small>
    </div>
  )

  return (
    <div className="sofar strip">
      <ol className="strip">
        {groups.map((g, gi) => (
          <li key={gi} className={`street ${g.street === 'end' ? 'over' : ''}`}>
            <span className="stname">{g.street === 'end' ? (g.steps[0].kind === 'end' ? endWords(g.steps[0]) : '') : g.street}</span>
            <div className="toks">
              {g.steps.map(token)}
              {headInLast && gi === groups.length - 1 && headTok}
            </div>
          </li>
        ))}
        {!headInLast && d.kind !== 'over' && d.kind !== 'board2' && (
          <li className="street">
            <span className="stname">{headStreet}</span>
            <div className="toks">{headTok}</div>
          </li>
        )}
      </ol>
      <Picker {...props} />
      <Hint d={d} />
    </div>
  )
}
