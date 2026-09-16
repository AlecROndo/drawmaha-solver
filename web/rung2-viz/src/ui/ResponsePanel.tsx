import { useState } from 'react'

import { Panel } from './site'
import type { Exploit } from './useExploit'

/**
 * The controls and the grade for an exploit run: what is locked, the button
 * that ships it to the local Python server, and — for a full-seat lock —
 * the exact numbers the response is graded against.
 */

/** One Leduc iteration walks 9,450 nodes (~50 ms), so budgets stay small. */
const ITERATION_CHOICES = [50, 200, 800]

const f5 = (x: number): string => x.toFixed(5)

export function ResponsePanel({ x }: { x: Exploit }) {
  const [iterations, setIterations] = useState(200)

  const guidance =
    x.lockedCount === 0
      ? 'Nothing locked yet. Lock a row with ◇ in a seat panel and drag it off Nash — or lock a whole seat below, the case with an exact answer to check against.'
      : x.fullSeat !== null
        ? `All 144 of P${x.fullSeat}’s spots are locked, so an exact best response exists. The re-solve will be graded against it.`
        : `${x.lockedCount} spot${x.lockedCount > 1 ? 's' : ''} locked. A partial lock still runs, but the rest of that seat keeps learning — a constrained equilibrium, not a best response, so there is no exact ceiling.`

  return (
    <Panel
      className="wide response"
      k="Fig. R · the response — the solver, given your frequencies"
      say={guidance}
      label="Run the exploiter"
    >
      <div className="response-controls">
        <label className="iters">
          iterations
          <select value={iterations} onChange={(e) => setIterations(Number(e.target.value))}>
            {ITERATION_CHOICES.map((n) => (
              <option key={n} value={n}>
                {n.toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <button
          className="btn solid"
          disabled={x.lockedCount === 0 || x.busy}
          onClick={() => void x.solve(iterations)}
        >
          {x.busy ? 'Solving…' : 'Run the solver'}
        </button>
        <button className="btn ghost" onClick={() => x.lockSeat(0)}>
          Lock all of P0
        </button>
        <button className="btn ghost" onClick={() => x.lockSeat(1)}>
          Lock all of P1
        </button>
        <button className="btn ghost" disabled={x.lockedCount === 0} onClick={x.clearLocks}>
          Clear
        </button>
      </div>

      {x.error && <p className="errnote">{x.error}</p>}

      {x.run && x.stale && (
        <p className="stale-note">
          The figures below are the previous run — they grade the locks as they were when you
          pressed Run, not the ones set now. Run again to bring them forward.
        </p>
      )}

      {x.run && x.run.exploiter !== null && (
        <dl className="stat">
          <div>
            <dt>equilibrium play earns</dt>
            <dd>{x.run.nashValue !== null ? f5(x.run.nashValue) : '—'}</dd>
          </div>
          <div>
            <dt>the exploiter takes</dt>
            <dd>{x.run.ceiling !== null ? f5(x.run.ceiling) : '—'}</dd>
          </div>
          <div>
            <dt>the band between</dt>
            <dd>
              {x.run.ceiling !== null && x.run.nashValue !== null
                ? f5(x.run.ceiling - x.run.nashValue)
                : '—'}
            </dd>
          </div>
          <div>
            <dt>CFR still short of it by</dt>
            <dd>{x.run.gap !== null ? x.run.gap.toFixed(6) : '—'}</dd>
          </div>
        </dl>
      )}

      {x.run && x.run.exploiter === null && (
        <p className="response-note">
          A constrained equilibrium, not a best response: both seats kept learning around your
          locks. The bars above show where every strategy settled — full colour now, the muted
          lane where Nash stood.
        </p>
      )}

      {!x.run && !x.error && x.lockedCount > 0 && (
        <p className="response-note">
          Ready. The run POSTs your locked rows to the local Python solver and re-solves with
          them held still — expect roughly a second per twenty iterations.
        </p>
      )}
    </Panel>
  )
}
