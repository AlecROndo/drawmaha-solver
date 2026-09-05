import type { Engine } from '../sim/engine'
import { averageStrategy } from '../sim/ledger'
import { Chip } from './Chip'
import { fmt, fmtIter, fmtSigned, fmtVec } from './format'
import { FigureHead } from './site'

/**
 * The most recent round's arithmetic: a short label column on the left, the
 * worked line on the right. Full trace in step mode; a compact last-round
 * summary while running, because at 10k rounds a second the table would only
 * be a blur.
 */
export function TracePanel({ engine, running }: { engine: Engine; running: boolean }) {
  const round = engine.lastRound
  const mode = engine.config.mode
  const oppLabel = mode === 'vs-you' ? 'you played' : 'opponent played'
  const selfLabel = mode === 'vs-you' ? 'bot sampled' : 'sampled'

  return (
    <section className="sect g12" aria-label="Update trace">
      <div className="l4">
        <FigureHead n="Fig. 4" title="The arithmetic of one round">
          R += u − ⟨σ, u⟩ against the revealed action. The baseline is the strategy's expected
          utility, not the sampled action's. Pause and press <kbd>→</kbd> to walk one round.
        </FigureHead>
      </div>
      <div className="r8">
        {round === null ? (
          <p className="trace-empty">
            No rounds yet — press <kbd>Step ×1</kbd> to watch a single update happen.
          </p>
        ) : running ? (
          <p className="trace-compact">
            round {fmtIter(round.t)} · <Chip a={round.playerAction} /> vs{' '}
            <Chip a={round.oppAction} /> · ⟨σ, u⟩ = {fmtSigned(round.trace.expectedUtility)} · ΔR ={' '}
            {fmtVec(round.trace.increments, 3, true)} · avg ={' '}
            {fmtVec(averageStrategy(engine.ledgers[0]))}
          </p>
        ) : (
          <table className="trace">
            <tbody>
              <tr>
                <td className="lbl">round {fmtIter(round.t)}</td>
                <td>
                  {selfLabel} a = <Chip a={round.playerAction} />, {oppLabel} b ={' '}
                  <Chip a={round.oppAction} />
                </td>
              </tr>
              <tr>
                <td className="lbl">utility u = PAYOFF[:, b]</td>
                <td>
                  {fmtVec(round.trace.utility, 0, true)}{' '}
                  <span className="aside">
                    what rock, paper, scissors each score against b
                  </span>
                </td>
              </tr>
              <tr>
                <td className="lbl">acting strategy σ</td>
                <td>{fmtVec(round.trace.sigma)}</td>
              </tr>
              <tr>
                <td className="lbl">expected utility ⟨σ, u⟩</td>
                <td>{fmtSigned(round.trace.expectedUtility)}</td>
              </tr>
              <tr>
                <td className="lbl">regret increments u − ⟨σ, u⟩</td>
                <td>{fmtVec(round.trace.increments, 3, true)}</td>
              </tr>
              <tr>
                <td className="lbl">new cumulative R</td>
                <td>{fmtVec(engine.ledgers[0].R)}</td>
              </tr>
              <tr>
                <td className="lbl">next σ = R⁺ / ΣR⁺</td>
                <td>
                  R⁺ = {fmtVec(engine.ledgers[0].R.map((r) => Math.max(r, 0)))}, ΣR⁺ ={' '}
                  {fmt(engine.ledgers[0].R.reduce((s, r) => s + Math.max(r, 0), 0))} →{' '}
                  {fmtVec(round.nextSigma)}
                </td>
              </tr>
              <tr>
                <td className="lbl">average S / n</td>
                <td>
                  {fmtVec(averageStrategy(engine.ledgers[0]))} over n ={' '}
                  {fmtIter(engine.ledgers[0].n)}
                </td>
              </tr>
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}
