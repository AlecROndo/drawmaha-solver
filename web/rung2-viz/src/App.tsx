import { useEffect, useState } from 'react'

import { freshSession, type Session } from './play'
import { SOLVE } from './solve'
import { advance, jump, type Step, type Walk } from './timeline'
import { NextActions, Timeline } from './ui/Timeline'
import { PlayPanel } from './ui/PlayPanel'
import { ResponsePanel } from './ui/ResponsePanel'
import { SeatPanel } from './ui/SeatPanel'
import { useExploit, type Exploit } from './ui/useExploit'
import { Figure, Footer, PageHead, TopBar } from './ui/site'

const TABS = [
  { id: 'walk', label: 'Walk the line', blurb: 'One hand, node by node, both seats as ranges' },
  { id: 'play', label: 'Play the solver', blurb: 'Heads-up against the committed average' },
] as const

type TabId = (typeof TABS)[number]['id']

/** A signed value with a true minus sign, the way the page prints every game value. */
const fmt = (v: number): string => `${v < 0 ? '\u2212' : ''}${Math.abs(v).toFixed(4)}`

/** A gap between two values, unsigned. */
const fmt4 = (v: number): string => Math.abs(v).toFixed(4)

const isTabId = (value: string): value is TabId => TABS.some((t) => t.id === value)

/** The tab named in the URL hash, so a view can be linked to and reloaded into. */
const tabFromHash = (): TabId => {
  const hash = location.hash.replace('#', '')
  return isTabId(hash) ? hash : 'walk'
}

/**
 * The line the walk opens on: check, check, the jack turns, check — deep
 * enough that the timeline, the board station, and both seats' panels all
 * have something to say before the visitor touches anything.
 */
const OPENING: Step[] = [
  { k: 'a', v: 'c' },
  { k: 'a', v: 'c' },
  { k: 'b', v: 'J' },
  { k: 'a', v: 'c' },
]

function WalkTab({ walk, setWalk, x }: { walk: Walk; setWalk: (w: Walk) => void; x: Exploit }) {
  return (
    <>
      <Figure
        n="Figure 1"
        className="timeline"
        title="A hand of Leduc is a line of decisions, one station each"
        sub="Press a station to rewind to it; press the board card to change it. The track is solid as far as the hand has gone."
        caption={
          <>
            Rewinding keeps the stations ahead. Choosing a different action at a station replaces them, and the strip
            says so before you press.
          </>
        }
        label="The action timeline"
      >
        <Timeline walk={walk} onJump={(n) => setWalk(jump(walk, n))} />
        <NextActions walk={walk} onStep={(step) => setWalk(advance(walk, step))} />
      </Figure>

      <div className="panels">
        <SeatPanel seat={0} walk={walk} x={x} />
        <SeatPanel seat={1} walk={walk} x={x} />
        <ResponsePanel x={x} />
      </div>

      <p className="foot">
        The percentages on the round-1 buttons are how often the average strategy reaches each branch, so the 288
        spots are weighted by how often play arrives at them. Round-2 buttons carry none: a reach conditioned on the
        board is not computed. Exploit runs are live: the locked rows are POSTed to a local Python process running
        src/drawmaha_solver/leduc/, the same walk with your rows held still. No CFR runs in TypeScript; the browser
        draws numbers Python computed.
      </p>
    </>
  )
}

export default function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash)
  // Both tabs' state lives here so flipping tabs loses nothing: the walk
  // keeps its explored line (and the exploit locks and last run with it),
  // the play tab its bankroll and hand in progress.
  const [walk, setWalk] = useState<Walk>({ path: OPENING, cur: OPENING.length })
  const x = useExploit()
  const [play, setPlay] = useState<Session>(() => freshSession(SOLVE.strategy))

  // The top bar's "Play the solver" button points at #play from every page.
  // On this page that is a hash change with no reload, so listen.
  useEffect(() => {
    const onHash = () => setTab(tabFromHash())
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
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
          eyebrow="Benchmarks · Leduc poker"
          title="Vanilla CFR on Leduc poker grades 0.011 chips per hand."
          lede={
            <>
              Six cards, two betting rounds, one board card, 288 information sets. After{' '}
              {SOLVE.iterations.toLocaleString()} iterations the average strategy is exploitable by{' '}
              <b>{SOLVE.exploitabilityAverage.toFixed(3)} chips/hand</b>, and its game value of{' '}
              {fmt(SOLVE.gameValue)} sits {fmt4(SOLVE.gameValue - SOLVE.gameValueExact)} from OpenSpiel’s exact{' '}
              {fmt(SOLVE.gameValueExact)}.
            </>
          }
          controls={
            <dl className="stats head-stats">
              <div className="stat">
                <dt>Exploitability</dt>
                <dd>
                  {SOLVE.exploitabilityAverage.toFixed(3)} <small>chips/hand</small>
                </dd>
              </div>
              <div className="stat">
                <dt>Game value, seat 0</dt>
                <dd>
                  {fmt(SOLVE.gameValue)} <small>vs {fmt(SOLVE.gameValueExact)}</small>
                </dd>
              </div>
            </dl>
          }
        >
          <p className="note">
            Every number on this page is the Python solver’s output, read from the committed{' '}
            {SOLVE.iterations.toLocaleString()}-iteration solve. The exact value is OpenSpiel’s linear-program
            answer, frozen in the repository as a fixture.
          </p>
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

        {tab === 'walk' ? (
          <WalkTab walk={walk} setWalk={setWalk} x={x} />
        ) : (
          <PlayPanel session={play} setSession={setPlay} />
        )}
      </main>

      <Footer />
    </>
  )
}
