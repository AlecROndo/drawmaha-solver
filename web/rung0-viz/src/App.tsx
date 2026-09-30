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
import { Footer, Hero, TopBar } from './ui/site'
import { useSimulation } from './ui/useSimulation'

const ACTION_GLYPHS = ['✊', '✋', '✌️']

export default function App() {
  const sim = useSimulation()
  const scored = sim.mode !== 'self-play'

  return (
    <>
      <TopBar here={0} />

      <main className="wrap">
        <Hero
          eyebrow="Rung 0 of 4 · complete · regret matching"
          done
          title={
            <>
              Regret matching learns <em>rock-paper-scissors.</em>
            </>
          }
          icon="chip"
          iconLabel="A poker chip lying on the felt with a second leaning on it, drawn in ASCII"
        >
          <p className="lede">
            One algorithm, running live. The current strategy chases yesterday's regrets and cycles
            forever; <b>its running average is the thing that converges to Nash.</b> Both happen
            below, and you can stop it mid-round to read the arithmetic.
          </p>
        </Hero>

        <Controls sim={sim} />

        {sim.mode === 'vs-you' && (
          <div className="play-buttons" aria-label="Play a hand">
            {ACTIONS.map((name, i) => (
              <button className="btn" key={name} onClick={() => sim.playUserAction(i as Action)}>
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
          <ExploitChart engine={sim.engine} />
          <TracePanel engine={sim.engine} running={sim.running} />
          <ConvergenceChart engine={sim.engine} />
          {scored && <Scoreboard engine={sim.engine} />}
        </div>

        {!scored && (
          <section className="stop" aria-label="Reading guide">
            <div className="stop-head">
              <div>
                <p className="kicker">What to look for</p>
                <h2>Three behaviours, one algorithm.</h2>
              </div>
              <p>
                The figures above are one run of one update rule. Each row here names a thing the
                rule does that a table of its numbers could not show, and the figure it shows in.
              </p>
            </div>
            <table className="rows">
              <tbody>
                <tr>
                  <td className="g">The current strategy never settles</td>
                  <td className="p">Regret chases the last winner, in a permanent orbit.</td>
                  <td className="s">Fig. 2</td>
                </tr>
                <tr>
                  <td className="g">The average is the product</td>
                  <td className="p">
                    S/n spirals into (⅓, ⅓, ⅓) and its exploitability falls like 1/√T.
                  </td>
                  <td className="s">Fig. 3</td>
                </tr>
                <tr>
                  <td className="g">Step mode shows the arithmetic</td>
                  <td className="p">Pause, then press → and follow a single update, line by line.</td>
                  <td className="s">Fig. 4</td>
                </tr>
              </tbody>
            </table>

            <p className="kbd-hint">
              <kbd>space</kbd> run / pause · <kbd>→</kbd> step ×1 · same seed, bit-identical run
            </p>
          </section>
        )}
      </main>

      <Footer />
    </>
  )
}
