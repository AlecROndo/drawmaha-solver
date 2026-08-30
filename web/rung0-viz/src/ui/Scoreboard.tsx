import type { Engine } from '../sim/engine'
import { PAYOFF } from '../sim/game'
import { Chip } from './Chip'
import { fmtIter, fmtSigned } from './format'

/**
 * Scoreboard for the scored modes (vs-fixed, vs-you): cumulative chips,
 * EV/round, and the last ~10 rounds' outcomes.
 */
export function Scoreboard({ engine }: { engine: Engine }) {
  const vsYou = engine.config.mode === 'vs-you'
  const who = vsYou ? 'bot' : 'learner'
  const n = engine.iteration

  return (
    <section className="panel wide aside" aria-label="Scoreboard">
      <h2>Scoreboard — the {who}'s chips</h2>
      <p className="sub">
        {vsYou
          ? 'the bot wins your chips by countering your habits'
          : 'best response to 50% rock earns +0.25/round in theory'}
      </p>
      <div className="score-tiles">
        <div className="tile">
          <div className="k">rounds</div>
          <div className="v" style={{ fontVariantNumeric: 'tabular-nums' }}>
            {fmtIter(n)}
          </div>
        </div>
        <div className="tile">
          <div className="k">cumulative chips</div>
          <div className="v" style={{ fontVariantNumeric: 'tabular-nums' }}>
            {fmtSigned(engine.chips, 0)}
          </div>
        </div>
        <div className="tile">
          <div className="k">EV / round</div>
          <div className="v" style={{ fontVariantNumeric: 'tabular-nums' }}>
            {n > 0 ? fmtSigned(engine.chips / n, 3) : '—'}
          </div>
        </div>
      </div>
      {engine.recent.length > 0 && (
        <div className="outcome-strip" aria-label="Recent rounds">
          {engine.recent.map((r) => {
            const p = PAYOFF[r.playerAction][r.oppAction]
            const res = p > 0 ? 'w' : p < 0 ? 'l' : 't'
            return (
              <div className="outcome" key={r.t}>
                <div className="who">
                  <Chip a={r.playerAction} label="" /> vs <Chip a={r.oppAction} label="" />
                </div>
                <div className={`res ${res}`}>{p > 0 ? 'W' : p < 0 ? 'L' : 'T'} {fmtSigned(p, 0)}</div>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}
