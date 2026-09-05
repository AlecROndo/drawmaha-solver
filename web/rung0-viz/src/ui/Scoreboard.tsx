import type { Engine } from '../sim/engine'
import { PAYOFF } from '../sim/game'
import { Chip } from './Chip'
import { fmtIter, fmtSigned } from './format'

/**
 * The scored modes (vs-fixed, vs-you) have a second question the self-play
 * page does not: how much is the learner actually winning? Cumulative chips,
 * EV per round, and the last ten rounds' outcomes.
 */
export function Scoreboard({ engine }: { engine: Engine }) {
  const vsYou = engine.config.mode === 'vs-you'
  const who = vsYou ? 'bot' : 'learner'
  const n = engine.iteration

  return (
    <section className="sect g12" aria-label="Scoreboard">
      <div className="l4">
        <h2 className="sec">The {who}'s chips</h2>
        <p className="serif dim" style={{ marginTop: 14, maxWidth: '32ch' }}>
          {vsYou
            ? 'The bot wins your chips by countering your habits: whatever you played too often becomes the action it regrets not beating.'
            : 'A best response to 50% rock earns +0.25 a round in theory. The ledger has to find it from regret alone.'}
        </p>
      </div>
      <div className="r8">
        <dl className="stat">
          <div>
            <dt>Rounds</dt>
            <dd>{fmtIter(n)}</dd>
          </div>
          <div>
            <dt>Cumulative chips</dt>
            <dd>{fmtSigned(engine.chips, 0)}</dd>
          </div>
          <div>
            <dt>Per round</dt>
            <dd>{n > 0 ? fmtSigned(engine.chips / n, 3) : '—'}</dd>
          </div>
        </dl>
        <div className="outcome-strip" aria-label="Recent rounds">
          {engine.recent.length === 0 ? (
            <span className="strip-empty">
              {vsYou ? 'Play a hand above to start the record.' : 'Step or run to start the record.'}
            </span>
          ) : (
            engine.recent.map((r) => {
              const p = PAYOFF[r.playerAction][r.oppAction]
              const res = p > 0 ? 'w' : p < 0 ? 'l' : 't'
              return (
                <div className="outcome" key={r.t}>
                  <div className="who">
                    <Chip a={r.playerAction} label="" /> vs <Chip a={r.oppAction} label="" />
                  </div>
                  <div className={`res ${res}`}>
                    {p > 0 ? 'W' : p < 0 ? 'L' : 'T'} {fmtSigned(p, 0)}
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>
    </section>
  )
}
