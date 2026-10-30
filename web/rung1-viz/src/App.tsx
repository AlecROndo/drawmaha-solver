import { useEffect, useState } from 'react'
import { AnswerSheet } from './ui/AnswerSheet'
import { ExploitChart } from './ui/ExploitChart'
import { ExploitTab } from './ui/ExploitTab'
import { GameTree, TreeLegend } from './ui/GameTree'
import { PlayPanel } from './ui/PlayPanel'
import { Figure, Footer, PageHead, TopBar } from './ui/site'
import { SOLVE, useSolve } from './ui/useSolve'

const TABS = [
  { id: 'solve', label: 'Solve', blurb: 'CFR finds the equilibrium' },
  { id: 'exploit', label: 'Exploit', blurb: 'CFR beats a strategy you lock' },
] as const

type TabId = (typeof TABS)[number]['id']

const isTabId = (value: string): value is TabId => TABS.some((t) => t.id === value)

const LAST = SOLVE.iterations.length - 1

/** A chip count with a true minus sign. */
const num = (v: number, digits: number) => (v < 0 ? `−${Math.abs(v).toFixed(digits)}` : v.toFixed(digits))

/**
 * The tab named in the URL hash, so a view can be linked to and reloaded into.
 * Anything else — `#play`, which the site's menu can point at — falls back to
 * the solve tab, because that is the view the target lives on.
 */
const tabFromHash = (): TabId => {
  const hash = typeof location === 'object' ? location.hash.replace('#', '') : ''
  return isTabId(hash) ? hash : 'solve'
}

function SolveTab() {
  const { index, last, playing, toggle, scrub } = useSolve()
  const t = SOLVE.iterations[index]
  const alphaNow = SOLVE.bet['J'][index]
  const kingNow = SOLVE.bet['K'][index]

  return (
    <>
      <div className="panels">
        <Figure
          n="Figure 1"
          wide
          title="Twelve bars walk onto the closed-form equilibrium"
          sub="Four decision nodes × three cards: the twelve information sets, each bar CFR's average strategy at the playhead."
          caption={`Each tick is the closed form at the α = ${SOLVE.alpha.toFixed(3)} this run found. Edge widths are how often each action is taken.`}
          label="Game tree and strategy"
        >
          <div className="transport" role="group" aria-label="Playback">
            <button className="btn" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}>
              {playing ? 'Pause' : index >= last ? 'Replay' : 'Play'}
            </button>
            <input
              type="range"
              min={0}
              max={last}
              value={index}
              onChange={(e) => scrub(Number(e.target.value))}
              aria-label="Iteration"
            />
            <span className="readout">
              iteration <b>{t.toLocaleString()}</b>
            </span>
          </div>
          <GameTree index={index} bet={SOLVE.bet} hairline={SOLVE.closedForm} />
          <TreeLegend hairlineIs="closed-form equilibrium" />
        </Figure>

        <ExploitChart index={index} onScrub={scrub} />
        <AnswerSheet index={index} />
        <PlayPanel />
      </div>

      <section className="stop" aria-label="Reading guide">
        <div className="stop-head">
          <div>
            <p className="eyebrow">What to look for</p>
            <h2>The run shows three things its strategy table cannot.</h2>
          </div>
          <p>
            The figures above are one solve of one small game. Each row names a thing the run does that a table of its
            final strategy would hide, and the figure it shows in.
          </p>
        </div>
        <table className="rows">
          <tbody>
            <tr>
              <td className="g">The bluff is discovered, not supplied</td>
              <td className="p">
                The jack's opening bar and the king's move together: now {alphaNow.toFixed(3)} and{' '}
                {kingNow.toFixed(3)}, a ratio of {alphaNow > 0.005 ? (kingNow / alphaNow).toFixed(2) : '—'}. It settles
                at 3, and α itself is free anywhere in [0, ⅓].
              </td>
              <td className="s">Figure 1</td>
            </tr>
            <tr>
              <td className="g">Two answers, one run</td>
              <td className="p">
                The average falls down the log axes; the current strategy, the thing the algorithm is actually playing,
                stays between 0.1 and 0.4 and ends worse than it was at ten iterations. Only the average carries a guarantee, so the
                average is what CFR reports.
              </td>
              <td className="s">Figure 2</td>
            </tr>
            <tr>
              <td className="g">The ticks are the known answer</td>
              <td className="p">
                Kuhn is solved in closed form, so each bar has a tick where it belongs. This solve landed at α ={' '}
                {SOLVE.alpha.toFixed(3)} with a game value of {num(SOLVE.gameValue, 5)} against the exact −1/18 ={' '}
                {num(SOLVE.gameValueExact, 5)}.
              </td>
              <td className="s">Figures 1, 3</td>
            </tr>
          </tbody>
        </table>
      </section>
    </>
  )
}

export default function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash)

  // A link to /rung1#play asks for the play panel. That is not a tab id, so
  // the page opens on the solve tab and then has to take the visitor to the
  // panel they asked for, rather than dropping them at the top.
  useEffect(() => {
    if (location.hash !== '#play') return
    document.getElementById('play')?.scrollIntoView({ block: 'center' })
  }, [])

  const show = (id: TabId) => {
    setTab(id)
    // Replace rather than push: the tabs are two views of one page, and
    // stacking them in history would make Back mean "previous tab" instead of
    // "the page I came from".
    history.replaceState(null, '', `#${id}`)
  }

  return (
    <>
      <TopBar here="benchmarks" />

      <main className="wrap">
        <PageHead
          eyebrow="Benchmarks · Kuhn poker"
          title={`Vanilla CFR solves Kuhn poker to ${SOLVE.exploitabilityAverage[LAST].toFixed(5)} chips a hand.`}
          lede={
            <>
              Three cards, one bet, twelve information sets, and an equilibrium known in closed form. After 100,000
              iterations the average strategy is exploitable for{' '}
              <b>{SOLVE.exploitabilityAverage[LAST].toFixed(5)} chips a hand</b>, and the game value is{' '}
              <b>{num(SOLVE.gameValue, 5)}</b> against the exact −1/18. Nobody tells the solver to bluff; it finds that a
              jack bluffs a third as often as a king bets.
            </>
          }
        >
          <a className="back" href="https://drawmaha.app/benchmarks">
            All benchmarks →
          </a>
        </PageHead>

        <div className="views" role="tablist" aria-label="Views">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              role="tab"
              aria-selected={tab === entry.id}
              className={tab === entry.id ? 'view current' : 'view'}
              onClick={() => show(entry.id)}
            >
              <b>{entry.label}</b>
              <span>{entry.blurb}</span>
            </button>
          ))}
        </div>

        {tab === 'solve' ? <SolveTab /> : <ExploitTab />}

        <p className="foot">
          {tab === 'solve'
            ? '100,000 vanilla CFR iterations, solved by '
            : 'Exploit runs are live: the locked spots are posted to a local Python process running '}
          <code>src/drawmaha_solver/kuhn/</code>
          {tab === 'solve'
            ? ' and exported to JSON. The page renders the solver’s own numbers rather than re-implementing it. Vanilla CFR enumerates the whole tree and never samples, so the run is deterministic: no seed, the same figures every time.'
            : ', the same walk with your spots held still. No CFR is re-implemented in TypeScript; the browser draws numbers Python computed, and the ceiling they are graded against comes from a module that has never imported the solver.'}
        </p>
      </main>

      <Footer />
    </>
  )
}
