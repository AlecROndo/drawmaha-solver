import { Controls, ModeSwitch } from './ui/Controls'
import { ConvergenceChart } from './ui/ConvergenceChart'
import { ExploitChart } from './ui/ExploitChart'
import { LedgerPanel } from './ui/LedgerPanel'
import { Scoreboard } from './ui/Scoreboard'
import { SimplexPlot } from './ui/SimplexPlot'
import { TracePanel } from './ui/TracePanel'
import { Footer, PageHead, TopBar } from './ui/site'
import { useSimulation } from './ui/useSimulation'

export default function App() {
  const sim = useSimulation()
  const scored = sim.mode !== 'self-play'

  return (
    <>
      <TopBar here="benchmarks" />

      <main className="wrap">
        <PageHead
          eyebrow="Benchmarks · rock-paper-scissors"
          title="Regret matching, graded against the uniform Nash."
          lede={
            <>
              Two regret matchers play each other in your browser. The current strategy chases last round's regret and
              never settles; its running average converges to (⅓, ⅓, ⅓) and is exploitable for{' '}
              <b>0.0009 chips per round after 100,000 rounds</b>.
            </>
          }
          controls={<ModeSwitch sim={sim} />}
        >
          <p className="note">
            The first of four games graded against a known answer. <a href="/benchmarks">All four benchmarks</a>
          </p>
        </PageHead>

        <Controls sim={sim} />

        <div className="panels">
          <LedgerPanel engine={sim.engine} />
          <SimplexPlot engine={sim.engine} version={sim.version} />
          <ExploitChart engine={sim.engine} />
          <TracePanel engine={sim.engine} running={sim.running} />
          <ConvergenceChart engine={sim.engine} />
          {scored && <Scoreboard engine={sim.engine} />}
        </div>

        {!scored && (
          <section className="stop" aria-label="What the figures show">
            <div className="stop-head">
              <div>
                <p className="kicker">What the figures show</p>
                <h2>The current strategy cycles; the average converges.</h2>
              </div>
              <p>
                Every figure above is one run of one update rule. Each row names a behaviour of the rule and the figure
                that shows it.
              </p>
            </div>
            <table className="rows">
              <tbody>
                <tr>
                  <td className="g">The current strategy never settles</td>
                  <td className="p">Regret chases the last winner, so σ orbits the centre of the triangle.</td>
                  <td className="s">
                    <a href="#fig-2">Figure 2</a>
                  </td>
                </tr>
                <tr>
                  <td className="g">The average is the answer</td>
                  <td className="p">S/n spirals into (⅓, ⅓, ⅓), and its exploitability falls like 1/√T.</td>
                  <td className="s">
                    <a href="#fig-3">Figure 3</a>
                  </td>
                </tr>
                <tr>
                  <td className="g">Step mode writes out the arithmetic</td>
                  <td className="p">Pause and press → to follow one update, line by line.</td>
                  <td className="s">
                    <a href="#fig-4">Figure 4</a>
                  </td>
                </tr>
              </tbody>
            </table>

            <p className="kbd-hint">
              <kbd>Space</kbd> runs and pauses · <kbd>→</kbd> plays one round · the same seed replays the same run, bit
              for bit
            </p>
          </section>
        )}
      </main>

      <Footer />
    </>
  )
}
