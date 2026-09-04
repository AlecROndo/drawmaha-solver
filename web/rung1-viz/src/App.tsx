import { useState } from 'react'
import { ExploitChart } from './ui/ExploitChart'
import { ExploitTab } from './ui/ExploitTab'
import { GameTree } from './ui/GameTree'
import { PlayPanel } from './ui/PlayPanel'
import { SOLVE, useSolve } from './ui/useSolve'

const TABS = [
  { id: 'solve', label: 'Solve', blurb: 'watch CFR find the equilibrium' },
  { id: 'exploit', label: 'Exploit', blurb: 'lock a strategy, watch CFR beat it' },
] as const

type TabId = (typeof TABS)[number]['id']

const isTabId = (value: string): value is TabId => TABS.some((t) => t.id === value)

/** The tab named in the URL hash, so a view can be linked to and reloaded into. */
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
      <div className="transport" role="group" aria-label="Playback">
        <button onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}>
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

      <div className="panels">
        <GameTree
          index={index}
          bet={SOLVE.bet}
          hairline={SOLVE.closedForm}
          heading="Fig. 1 · The game, with the strategy drawn on it"
          sub="four decision nodes × three possible cards = the twelve information sets · bar is P(bet), which reads as P(call) at the two nodes facing a bet · hairline is the closed form"
        />
        <ExploitChart index={index} />
        <PlayPanel />

        <section className="panel wide aside" aria-label="Reading guide">
          <h2>What to look for</h2>
          <p className="sub">three things this page can show that a table cannot</p>
          <ul className="guide-list">
            <li>
              <strong>The bluff is discovered, not supplied.</strong> In Fig. 1 the jack's
              opening bar and the king's move together: right now{' '}
              <span className="mono">{alphaNow.toFixed(3)}</span> and{' '}
              <span className="mono">{kingNow.toFixed(3)}</span>, a ratio of{' '}
              <span className="mono">
                {alphaNow > 0.005 ? (kingNow / alphaNow).toFixed(2) : '—'}
              </span>
              . It settles at 3, and α itself is free anywhere in [0, ⅓].
            </li>
            <li>
              <strong>Two answers, one run.</strong> Fig. 2 draws both. The average marches down
              the diagonal; the current strategy — the thing the algorithm is actually playing —
              stays around 0.2 and is briefly worse late than it was early. Only the average has
              a convergence guarantee, which is why the rung-4 dashboard must query the average
              policy and never the final iterate.
            </li>
            <li>
              <strong>The hairlines are the known answer.</strong> Kuhn is solved in closed form,
              so each bar in Fig. 1 has a tick where it belongs. This solve landed at α ={' '}
              <span className="mono">{SOLVE.alpha.toFixed(3)}</span> with a game value of{' '}
              <span className="mono">{SOLVE.gameValue.toFixed(5)}</span> against the exact{' '}
              <span className="mono">−1/18 = {SOLVE.gameValueExact.toFixed(5)}</span>.
            </li>
          </ul>
        </section>
      </div>
    </>
  )
}

export default function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash)

  const show = (id: TabId) => {
    setTab(id)
    // Replace rather than push: the tabs are two views of one page, and
    // stacking them in history would make Back mean "previous tab" instead of
    // "the page I came from".
    history.replaceState(null, '', `#${id}`)
  }

  return (
    <>
      <header className="app-header">
        <p className="eyebrow">
          <a href="/">drawmaha solver</a> · rung 1 · counterfactual regret minimization
        </p>
        <h1>CFR solves Kuhn poker, and discovers how often to bluff</h1>
        <p className="dek">
          Three cards, one bet, twelve information sets — the smallest poker with hidden
          information, and one of the few with a known exact answer. Nobody tells the solver to
          bluff. Press play and watch it work out that a jack should bluff exactly one third as
          often as a king value-bets — then lock a strategy of your own and watch the same
          machinery take it apart.
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
            {entry.label}
            <span className="tab-blurb">{entry.blurb}</span>
          </button>
        ))}
      </div>

      {tab === 'solve' ? <SolveTab /> : <ExploitTab />}

      <p className="foot">
        {tab === 'solve'
          ? '100,000 vanilla CFR iterations, solved by '
          : 'Exploit runs are live: the locked spots are POSTed to a local Python process running '}
        <span className="mono">src/drawmaha_solver/kuhn/</span>
        {tab === 'solve'
          ? ' and exported to JSON — this page renders the solver’s own numbers rather than re-implementing it. Vanilla CFR enumerates the whole tree and never samples, so the run is deterministic: no seed, same figures every time.'
          : ' — the same walk, with your spots held still. No CFR is re-implemented in TypeScript; the browser only draws numbers Python computed, and the ceiling it is graded against comes from a module that has never imported the solver.'}
      </p>
    </>
  )
}
