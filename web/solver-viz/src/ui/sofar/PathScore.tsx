/**
 * The hand so far as a path, the default layout, drawn down the tool's left
 * column: the board, then one numbered row per played step — the seat, what
 * it did, the pot after it — and at the foot the decision the line stands
 * on, its options branching beneath it, each with the share of the actor's
 * range that takes it as a number and a bar. Pressing a played step rewinds
 * to just before it; pressing an option plays it and it becomes the next
 * step; pressing a board card opens the fifteen to change it. The hand is a
 * path through the game tree, and this draws it as one. `StripScore` is the
 * backup layout; `index.ts` picks between them.
 */

import { endWords, headLabel, headOf, hueOf, pct, rewindLabel, stepsOf } from './steps'
import { advance } from '../MiniScore'
import { BoardPick, Hint, Picker, type ScoreProps } from './parts'

export function PathScore(props: ScoreProps) {
  const { game, line, onLine, picking, onPick, shares } = props
  const steps = stepsOf(line)
  const head = headOf(line)
  const d = head.decision
  const end = steps.find((s) => s.kind === 'end')
  const played = steps.filter((s) => s.kind === 'act' || s.kind === 'draw')

  return (
    <div className="sofar path">
      <div className="sf-board">
        <span className="k">Board</span>
        <BoardPick game={game} which={1} card={line.board1} picking={picking} onPick={onPick} />
        {line.board2 !== null && <BoardPick game={game} which={2} card={line.board2} picking={picking} onPick={onPick} />}
        <span className="sf-note">{line.board2 === null ? 'one card dealt, one to come' : 'both cards dealt'}</span>
      </div>
      <ol className="sf-steps" aria-label="the line so far">
        {played.map((s, i) => (
          <li key={i} className="sf-step">
            <span className="n">{i + 1}</span>
            <button type="button" className={`tok ${hueOf(s.tone, s.word)}`} onClick={() => onLine(s.back)} aria-label={rewindLabel(s)}>
              <span className="who">Seat {s.seat}</span> <span className="w">{s.word}</span>
            </button>
            <small className="potnum">pot {s.pot}</small>
          </li>
        ))}
        <li className="sf-step sf-head" aria-current="step">
          <span className="n">{d.kind === 'over' ? '' : played.length + 1}</span>
          <span className={`node ${d.kind === 'over' ? 'over' : 'lit'}`}>{d.kind === 'over' && end?.kind === 'end' ? endWords(end) : headLabel(d)}</span>
          {d.kind !== 'over' && <small className="potnum">pot {head.pot}</small>}
        </li>
      </ol>
      {d.kind !== 'over' && (
        <ul className="fork">
          {d.kind === 'board2' ? (
            <li className="branch">
              <BoardPick game={game} which={2} card={null} picking={picking} onPick={onPick} className="opt" />
              <span className="blab">board card 2</span>
            </li>
          ) : (
            head.options.map((o) => {
              const p = pct(shares, o.key)
              const share = shares && shares[o.key] !== undefined ? shares[o.key] : 0
              const hue = hueOf(o.tone, o.word)
              return (
                <li key={o.key} className="branch">
                  <button type="button" className={`opt ${hue}`} onClick={() => onLine(advance(line, o.key))} aria-label={`play ${o.word}`}>
                    <i className="sw" aria-hidden />
                    <span className="w">{o.word}</span>
                    {p && <b>{p}</b>}
                    <span className="go" aria-hidden>
                      play →
                    </span>
                  </button>
                  <span className="sharebar" aria-hidden>
                    <i className={hue} style={{ transform: `scaleX(${share})` }} />
                  </span>
                </li>
              )
            })
          )}
        </ul>
      )}
      <Picker {...props} />
      <Hint d={d} />
    </div>
  )
}
