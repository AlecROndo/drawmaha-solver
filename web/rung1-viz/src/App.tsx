import { useEffect, useState } from 'react'
import { ExploitChart } from './ui/ExploitChart'
import { ExploitTab } from './ui/ExploitTab'
import { GameTree, TreeLegend } from './ui/GameTree'
import { PlayPanel } from './ui/PlayPanel'
import { IdentityRail, LadderLine, Panel, Squiggle } from './ui/site'
import { SOLVE, useSolve } from './ui/useSolve'

const TABS = [
  { id: 'solve', label: 'Solve', blurb: 'watch CFR find the equilibrium' },
  { id: 'exploit', label: 'Exploit', blurb: 'lock a strategy, watch CFR beat it' },
] as const

type TabId = (typeof TABS)[number]['id']

const isTabId = (value: string): value is TabId => TABS.some((t) => t.id === value)

/**
 * The tab named in the URL hash, so a view can be linked to and reloaded into.
 * Anything else — `#play`, which the identity rail's button points at — falls
 * back to the solve tab, because that is the view the target lives on.
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
        <Panel
          n="01"
          wide
          k="Fig. 1 · the game"
          title="The game, with the strategy drawn on it"
          say="Four decision nodes × three possible cards = the twelve information sets. Each bar is what CFR's average strategy does there; each tick is the closed form."
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
        </Panel>

        <ExploitChart index={index} />
        <PlayPanel />
      </div>

      <section aria-label="Reading guide">
        <p className="stop">Stop 03 · what to look for</p>
        <h2 className="big">Three things a table cannot show.</h2>
        <table className="rows">
          <tbody>
            <tr>
              <td className="g">The bluff is discovered, not supplied</td>
              <td className="p">
                The jack's opening bar and the king's move together: now {alphaNow.toFixed(3)} and{' '}
                {kingNow.toFixed(3)}, a ratio of{' '}
                {alphaNow > 0.005 ? (kingNow / alphaNow).toFixed(2) : '—'}. It settles at 3, and α
                itself is free anywhere in [0, ⅓].
              </td>
              <td className="s">Fig. 1</td>
            </tr>
            <tr>
              <td className="g">Two answers, one run</td>
              <td className="p">
                The average marches down the diagonal; the current strategy — the thing the
                algorithm is actually playing — stays near 0.2 and is briefly worse late than
                early. Only the average has a guarantee, which is why the rung-4 dashboard must
                query the average policy and never the final iterate.
              </td>
              <td className="s">Fig. 2</td>
            </tr>
            <tr>
              <td className="g">The ticks are the known answer</td>
              <td className="p">
                Kuhn is solved in closed form, so each bar has a tick where it belongs. This solve
                landed at α = {SOLVE.alpha.toFixed(3)} with a game value of{' '}
                {SOLVE.gameValue.toFixed(5)} against the exact −1/18 ={' '}
                {SOLVE.gameValueExact.toFixed(5)}.
              </td>
              <td className="s">Fig. 1</td>
            </tr>
          </tbody>
        </table>
      </section>
    </>
  )
}

export default function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash)

  // The identity rail offers "Play the solver" at /rung1#play. That is not a
  // tab id, so the page opens on the solve tab and then has to take the
  // visitor to the panel they asked for, rather than dropping them at the top.
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
      <LadderLine here={1} />

      <div className="shell">
        <IdentityRail now="Rung 1, complete" next="Rung 2 · Leduc poker" />

        <main>
          <section>
            <p className="stop">Stop 02 · rung 1</p>
            <h2 className="big">CFR solves Kuhn poker, and discovers how often to bluff.</h2>
            <Squiggle />
            <p className="lede">
              Three cards, one bet, twelve information sets: the smallest poker with hidden
              information, and one of the few with a known exact answer. Nobody tells the solver to
              bluff. It works out that a jack should bluff one third as often as a king value-bets.
            </p>

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
                : 'Exploit runs are live: the locked spots are POSTed to a local Python process running '}
              src/drawmaha_solver/kuhn/
              {tab === 'solve'
                ? ' and exported to JSON — this page renders the solver’s own numbers rather than re-implementing it. Vanilla CFR enumerates the whole tree and never samples, so the run is deterministic: no seed, same figures every time.'
                : ' — the same walk, with your spots held still. No CFR is re-implemented in TypeScript; the browser only draws numbers Python computed, and the ceiling it is graded against comes from a module that has never imported the solver.'}
            </p>
          </section>
        </main>
      </div>
    </>
  )
}
