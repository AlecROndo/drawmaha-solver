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
import { Footer, Nav } from './ui/site'
import { useSimulation } from './ui/useSimulation'

const ACTION_GLYPHS = ['✊', '✋', '✌️']

export default function App() {
  const sim = useSimulation()
  const scored = sim.mode !== 'self-play'

  return (
    <>
      <div className="wrap">
        <Nav here="/rung0" />

        <header className="hero">
          <h1>Regret matching learns rock-paper-scissors</h1>
          <p className="dek">
            A live experiment in one algorithm. The current strategy chases yesterday's regrets
            and cycles forever; its running average is what converges to Nash. Both happen below.
          </p>
        </header>

        <div className="toolbar">
          <Controls sim={sim} />
          {sim.mode === 'vs-you' && (
            <div className="play-buttons" aria-label="Play a hand">
              {ACTIONS.map((name, i) => (
                <button className="btn ghost" key={name} onClick={() => sim.playUserAction(i as Action)}>
                  <span aria-hidden>{ACTION_GLYPHS[i]}</span>
                  <span className="chip">
                    <span className="dot" style={{ background: ACTION_COLORS[i] }} />
                    {name}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* The one full-bleed moment: the ledger breaks the container, because
          it is the thing the page is about. */}
      <LedgerPanel engine={sim.engine} />

      <div className="wrap">
        <section className="sect g12">
          <SimplexPlot engine={sim.engine} version={sim.version} />
          <ExploitChart engine={sim.engine} />
        </section>

        <TracePanel engine={sim.engine} running={sim.running} />

        <ConvergenceChart engine={sim.engine} />

        {scored ? (
          <Scoreboard engine={sim.engine} />
        ) : (
          <section className="sect g12" aria-label="Reading guide">
            <div className="l4">
              <h2 className="sec">What to look for</h2>
              <p className="serif dim" style={{ marginTop: 14, maxWidth: '30ch' }}>
                Three behaviours, one algorithm.
              </p>
            </div>
            <ul className="rows r8">
              <li>
                <span>
                  <b>The current strategy never settles.</b>{' '}
                  <span className="serif dim">
                    Regret chases the last winner in a permanent orbit.
                  </span>
                </span>
                <span className="m">Fig. 2</span>
              </li>
              <li>
                <span>
                  <b>The average is the product.</b>{' '}
                  <span className="serif dim">
                    S/n spirals into (⅓, ⅓, ⅓); its exploitability falls like 1/√T.
                  </span>
                </span>
                <span className="m">Fig. 3</span>
              </li>
              <li>
                <span>
                  <b>Step mode shows the arithmetic.</b>{' '}
                  <span className="serif dim">
                    Pause, then press → and follow a single update, line by line.
                  </span>
                </span>
                <span className="m">Fig. 4</span>
              </li>
            </ul>
          </section>
        )}

        <p className="kbd-hint">
          <kbd>space</kbd> run / pause · <kbd>→</kbd> step ×1 · same seed, bit-identical run
        </p>
      </div>

      <Footer />
    </>
  )
}
