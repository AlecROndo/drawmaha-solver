import type { Engine } from '../sim/engine'
import { ACTIONS, PAYOFF } from '../sim/game'
import { Chip } from './Chip'
import { fmtIter, fmtSigned } from './format'
import { Figure } from './site'

/**
 * The scored modes (vs-fixed, vs-you) have a second question self-play does
 * not: how much is the learner actually winning? Cumulative chips, chips per
 * round, and the last ten rounds as hairline rows, newest first.
 */
export function Scoreboard({ engine }: { engine: Engine }) {
  const vsYou = engine.config.mode === 'vs-you'
  const who = vsYou ? 'bot' : 'learner'
  const n = engine.iteration
  const rows = [...engine.recent].reverse()

  // The ceiling the learner is chasing: the best pure reply to the fixed
  // opponent's mix, worked out from the payoff table (+0.25, paper, at 50% rock).
  const dist = engine.config.fixedDist ?? [1 / 3, 1 / 3, 1 / 3]
  const replyEv = ACTIONS.map((_, a) => dist.reduce((s, q, b) => s + q * PAYOFF[a][b], 0))
  const best = replyEv.indexOf(Math.max(...replyEv))

  return (
    <Figure
      n="Figure 6"
      wide
      title={
        vsYou
          ? 'The bot wins by countering whatever you throw too often'
          : 'Regret alone finds the counter to a fixed opponent'
      }
      sub={
        vsYou ? (
          "The bot's chips against you, and your last ten rounds."
        ) : (
          <>
            The learner's chips against the fixed mix. The best reply to it, pure {ACTIONS[best]}, earns{' '}
            <b>{fmtSigned(replyEv[best], 3)}</b> a round.
          </>
        )
      }
      label="Scoreboard"
    >
      <dl className="stats">
        <div className="stat">
          <dt>Rounds</dt>
          <dd>{fmtIter(n)}</dd>
        </div>
        <div className="stat">
          <dt>The {who}'s chips</dt>
          <dd>{fmtSigned(engine.chips, 0)}</dd>
        </div>
        <div className="stat">
          <dt>Per round</dt>
          <dd>{n > 0 ? fmtSigned(engine.chips / n, 3) : '—'}</dd>
        </div>
      </dl>
      {rows.length === 0 ? (
        <p className="strip-empty">
          {vsYou ? 'Throw rock, paper or scissors above to start the record.' : 'Step or run to start the record.'}
        </p>
      ) : (
        <table className="hair recent" aria-label="The last ten rounds">
          <thead>
            <tr>
              <th>Round</th>
              <th>The {who}</th>
              <th>{vsYou ? 'You' : 'Opponent'}</th>
              <th>Result</th>
              <th className="r">Chips</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const p = PAYOFF[r.playerAction][r.oppAction]
              return (
                <tr key={r.t}>
                  <td className="n">{fmtIter(r.t)}</td>
                  <td>
                    <Chip a={r.playerAction} />
                  </td>
                  <td>
                    <Chip a={r.oppAction} />
                  </td>
                  <td className={p > 0 ? 'won' : p < 0 ? 'lost' : 'tied'}>
                    {p > 0 ? 'won' : p < 0 ? 'lost' : 'tied'}
                  </td>
                  <td className="r n">{fmtSigned(p, 0)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </Figure>
  )
}
