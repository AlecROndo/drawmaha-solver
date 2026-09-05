import { useEffect, useState } from 'react'
import { ExploitChart } from './ui/ExploitChart'
import { ExploitTab } from './ui/ExploitTab'
import { GameTree, TreeLegend } from './ui/GameTree'
import { PlayPanel } from './ui/PlayPanel'
import { Footer, Nav, FigureHead } from './ui/site'
import { SOLVE, useSolve } from './ui/useSolve'

const TABS = [
  { id: 'solve', label: 'Solve', blurb: 'watch CFR find the equilibrium' },
  { id: 'exploit', label: 'Exploit', blurb: 'lock a strategy, watch CFR beat it' },
] as const

type TabId = (typeof TABS)[number]['id']

const isTabId = (value: string): value is TabId => TABS.some((t) => t.id === value)

/**
 * The tab named in the URL hash, so a view can be linked to and reloaded into.
 * Anything else — `#play`, which the site's nav button points at — falls back
 * to the solve tab, because that is the view the target lives on.
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
      {/* The one full-bleed moment: the tree breaks the container, because the
          whole page is an argument about what is drawn on it. */}
      <section className="plate" aria-label="Game tree and strategy">
        <div className="plate-head">
          <FigureHead n="Fig. 1" title="The game, with the strategy drawn on it">
            Four decision nodes × three possible cards = the twelve information sets. Each bar is
            what CFR's average strategy does there; each tick is the closed form.
          </FigureHead>
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
        </div>
        <GameTree index={index} bet={SOLVE.bet} hairline={SOLVE.closedForm} />
        <TreeLegend hairlineIs="closed-form equilibrium" />
      </section>

      <div className="wrap">
        <section className="sect g12">
          <ExploitChart index={index} />
          <PlayPanel />
        </section>

        <section className="sect g12" aria-label="Reading guide">
          <div className="l4">
            <h2 className="sec">What to look for</h2>
            <p className="serif dim" style={{ marginTop: 14, maxWidth: '30ch' }}>
              Three things this page can show that a table cannot.
            </p>
          </div>
          <ul className="rows r8">
            <li>
              <span>
                <b>The bluff is discovered, not supplied.</b>{' '}
                <span className="serif dim">
                  The jack's opening bar and the king's move together: now{' '}
                  <span className="mono">{alphaNow.toFixed(3)}</span> and{' '}
                  <span className="mono">{kingNow.toFixed(3)}</span>, a ratio of{' '}
                  <span className="mono">
                    {alphaNow > 0.005 ? (kingNow / alphaNow).toFixed(2) : '—'}
                  </span>
                  . It settles at 3, and α itself is free anywhere in [0, ⅓].
                </span>
              </span>
              <span className="m">Fig. 1</span>
            </li>
            <li>
              <span>
                <b>Two answers, one run.</b>{' '}
                <span className="serif dim">
                  The average marches down the diagonal; the current strategy — the thing the
                  algorithm is actually playing — stays near 0.2 and is briefly worse late than
                  early. Only the average has a guarantee, which is why the rung-4 dashboard must
                  query the average policy and never the final iterate.
                </span>
              </span>
              <span className="m">Fig. 2</span>
            </li>
            <li>
              <span>
                <b>The ticks are the known answer.</b>{' '}
                <span className="serif dim">
                  Kuhn is solved in closed form, so each bar has a tick where it belongs. This
                  solve landed at α = <span className="mono">{SOLVE.alpha.toFixed(3)}</span> with a
                  game value of <span className="mono">{SOLVE.gameValue.toFixed(5)}</span> against
                  the exact −1/18 ={' '}
                  <span className="mono">{SOLVE.gameValueExact.toFixed(5)}</span>.
                </span>
              </span>
              <span className="m">Fig. 1</span>
            </li>
          </ul>
        </section>
      </div>
    </>
  )
}

export default function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash)

  // The site's nav offers "Play against CFR" at /rung1#play. That is not a tab
  // id, so the page opens on the solve tab and then has to take the visitor to
  // the panel they asked for, rather than dropping them at the top.
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
      <div className="wrap">
        <Nav here="/rung1" />

        <header className="hero">
          <h1>CFR solves Kuhn poker, and discovers how often to bluff</h1>
          <p className="dek">
            Three cards, one bet, twelve information sets: the smallest poker with hidden
            information, and one of the few with a known exact answer. Nobody tells the solver to
            bluff. It works out that a jack should bluff one third as often as a king value-bets.
          </p>
        </header>

        <div className="tabs" role="tablist" aria-label="Views">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              role="tab"
              aria-selected={tab === entry.id}
              className={tab === entry.id ? 'tab current' : 'tab'}
              onClick={() => show(entry.id)}
            >
              <b>{entry.label}</b>
              <span>{entry.blurb}</span>
            </button>
          ))}
        </div>
      </div>

      {tab === 'solve' ? <SolveTab /> : <ExploitTab />}

      <div className="wrap">
        <p className="foot">
          {tab === 'solve'
            ? '100,000 vanilla CFR iterations, solved by '
            : 'Exploit runs are live: the locked spots are POSTed to a local Python process running '}
          <span className="mono">src/drawmaha_solver/kuhn/</span>
          {tab === 'solve'
            ? ' and exported to JSON — this page renders the solver’s own numbers rather than re-implementing it. Vanilla CFR enumerates the whole tree and never samples, so the run is deterministic: no seed, same figures every time.'
            : ' — the same walk, with your spots held still. No CFR is re-implemented in TypeScript; the browser only draws numbers Python computed, and the ceiling it is graded against comes from a module that has never imported the solver.'}
        </p>
      </div>

      <Footer />
    </>
  )
}
