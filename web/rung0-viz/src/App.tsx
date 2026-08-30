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
        <p className="eyebrow">drawmaha solver · rung 0 · regret matching</p>
        <h1>Regret matching learns rock-paper-scissors</h1>
        <p className="dek">
          A live experiment in one algorithm. The current strategy chases yesterday's regrets and
          cycles forever; its running average is what converges to Nash — watch both happen below.
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
        <SimplexPlot engine={sim.engine} version={sim.version} />
        <TracePanel engine={sim.engine} running={sim.running} />
        <ConvergenceChart engine={sim.engine} />
        <ExploitChart engine={sim.engine} />
        {scored ? (
          <Scoreboard engine={sim.engine} />
        ) : (
          <section className="panel wide aside" aria-label="Reading guide">
            <h2>What to look for</h2>
            <p className="sub">three behaviors, one algorithm</p>
            <ul className="guide-list">
              <li>
                <strong>The current strategy never settles.</strong> Regret chases the last winner
                in a permanent orbit — the gray trail in Fig. 2.
              </li>
              <li>
                <strong>The average is the product.</strong> S/n spirals into (⅓, ⅓, ⅓); its
                exploitability falls like 1/√T (Fig. 5).
              </li>
              <li>
                <strong>Step mode shows the arithmetic.</strong> Pause, then press <kbd>→</kbd> and
                follow one round through Fig. 3.
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
