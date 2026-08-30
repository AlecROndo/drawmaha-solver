import type { Action } from './sim/game'
import { ACTIONS } from './sim/game'
import { ACTION_COLORS } from './ui/format'
import { Controls } from './ui/Controls'
import { ConvergenceChart } from './ui/ConvergenceChart'
import { ExploitChart } from './ui/ExploitChart'
import { LedgerPanel } from './ui/LedgerPanel'
import { Scoreboard } from './ui/Scoreboard'
import { SimplexPlot } from './ui/SimplexPlot'
import { TracePanel } from './ui/TracePanel'
import { useSimulation } from './ui/useSimulation'

const ACTION_GLYPHS = ['✊', '✋', '✌️']

export default function App() {
  const sim = useSimulation()
  const scored = sim.mode !== 'self-play'

  return (
    <>
      <header className="app-header">
        <h1>Regret matching learns rock-paper-scissors</h1>
        <p>
          The current strategy chases yesterday's regrets and cycles forever; its running average is
          what converges to Nash. Watch both happen live.
        </p>
      </header>

      <Controls sim={sim} />

      {sim.mode === 'vs-you' && (
        <div className="play-buttons" aria-label="Play a hand">
          {ACTIONS.map((name, i) => (
            <button key={name} onClick={() => sim.playUserAction(i as Action)}>
              <span aria-hidden>{ACTION_GLYPHS[i]}</span>
              <span className="chip">
                <span className="dot" style={{ background: ACTION_COLORS[i] }} />
                {name}
              </span>
            </button>
          ))}
        </div>
      )}

      <div className="panels">
        <LedgerPanel engine={sim.engine} />
        <TracePanel engine={sim.engine} running={sim.running} />
        <ConvergenceChart engine={sim.engine} />
        <ExploitChart engine={sim.engine} />
        <SimplexPlot engine={sim.engine} version={sim.version} />
        {scored ? (
          <Scoreboard engine={sim.engine} />
        ) : (
          <section className="panel" aria-label="Reading guide">
            <h2>What to look for</h2>
            <p className="sub">three behaviors, one algorithm</p>
            <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--ink-2)' }}>
              <li>
                <strong style={{ color: 'var(--ink)' }}>The current strategy never settles.</strong>{' '}
                Regret chases the last winner in a permanent orbit — the gray trail on the simplex.
              </li>
              <li>
                <strong style={{ color: 'var(--ink)' }}>The average is the product.</strong> S/n
                spirals into (⅓, ⅓, ⅓); its exploitability falls like 1/√T.
              </li>
              <li>
                <strong style={{ color: 'var(--ink)' }}>Step mode shows the arithmetic.</strong>{' '}
                Press <kbd>→</kbd> and follow one round through the update trace.
              </li>
            </ul>
          </section>
        )}
      </div>

      <p className="kbd-hint">
        <kbd>space</kbd> run / pause · <kbd>→</kbd> step ×1 · same seed → bit-identical run
      </p>
    </>
  )
}
