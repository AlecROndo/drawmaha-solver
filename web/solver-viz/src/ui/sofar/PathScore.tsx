/**
 * The hand so far as a path, the default layout. The line is a breadcrumb
 * trail of compact crumbs joined by chevrons — 4♥ › s0 bet › s1 call › s0 pat › s1 threw one › 4♦ —
 * with the pot after each bet as a small number on the crumb, and at the
 * tip a fork: the decision's options branch off the end of the path as a
 * little tree, each branch carrying the share of the range that takes it as
 * a bar. Clicking back along the path rewinds; clicking a branch walks down
 * it and it becomes the next crumb. The hand is a path through the game
 * tree, and this draws it as one. `StripScore` is the backup layout; `index.ts`
 * picks between them.
 */

import { endWords, headLabel, headOf, pct, rewindLabel, stepsOf } from './steps'
import { advance } from '../MiniScore'
import { BoardPick, Hint, Picker, type ScoreProps } from './parts'

export function PathScore(props: ScoreProps) {
  const { game, line, onLine, picking, onPick, shares } = props
  const steps = stepsOf(line)
  const head = headOf(line)
  const d = head.decision
  const end = steps.find((s) => s.kind === 'end')

  return (
    <div className="sofar path">
      <div className="trail">
        <nav className="crumbline" aria-label="the line so far">
          {steps.map((s, i) => {
            // An owed board card is the fork's one branch, not a crumb yet.
            if (s.kind === 'board' && s.card === null) return null
            if (s.kind === 'board') {
              return (
                <span key={i} className="crumb-wrap">
                  <BoardPick game={game} which={s.which} card={s.card} picking={picking} onPick={onPick} className="crumb" />
                  <i className="chev" aria-hidden>
                    ›
                  </i>
                </span>
              )
            }
            if (s.kind === 'end') return null
            // The pot ticks onto a crumb only where it moved — a bet, a raise, a call — read off
            // the pots themselves, so a check or a draw never carries a stale number.
            const potChanged = i > 0 && s.pot !== steps[i - 1].pot
            return (
              <span key={i} className="crumb-wrap">
                <button type="button" className={`tok crumb ${s.tone}`} onClick={() => onLine(s.back)} aria-label={rewindLabel(s)}>
                  <span className="who">s{s.seat}</span>
                  <span className="w">{s.word}</span>
                  {potChanged && <small className="potnum">{s.pot}</small>}
                </button>
                <i className="chev" aria-hidden>
                  ›
                </i>
              </span>
            )
          })}
          <span className="tip">
            <span className={`node ${d.kind === 'over' ? 'over' : 'lit'}`} aria-current="step">
              <span className="w">{d.kind === 'over' && end?.kind === 'end' ? endWords(end) : headLabel(d)}</span>
              {d.kind !== 'over' && <small className="potnum">pot {head.pot}</small>}
            </span>
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
                    return (
                      <li key={o.key} className="branch">
                        <button type="button" className={`opt ${o.tone}`} onClick={() => onLine(advance(line, o.key))} aria-label={`play ${o.word}`}>
                          <span className="w">{o.word}</span>
                          {p && <b>{p}</b>}
                        </button>
                        <span className="sharebar" aria-hidden>
                          <i className={o.tone} style={{ transform: `scaleX(${share})` }} />
                        </span>
                      </li>
                    )
                  })
                )}
              </ul>
            )}
          </span>
        </nav>
      </div>
      <Picker {...props} />
      <Hint d={d} extra="the fork is the next decision" />
    </div>
  )
}
