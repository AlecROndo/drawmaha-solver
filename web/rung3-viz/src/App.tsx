import { useEffect, useState } from 'react'

import { useSession } from './deals'
import { FINAL_ITERATION, FROZEN, SIZES, WORKERS, perHundred, series } from './research'
import { PlayPanel } from './ui/PlayPanel'
import { ResearchTab } from './ui/ResearchTab'
import { Footer, Hero, TopBar } from './ui/site'

const TABS = [
  { id: 'research', label: 'The research', blurb: 'the race, the rules, what the answer does' },
  { id: 'play', label: 'Play the solver', blurb: 'heads-up against the frozen strategy, with a coach' },
] as const

type TabId = (typeof TABS)[number]['id']

const isTabId = (value: string): value is TabId => TABS.some((t) => t.id === value)

/** The tab named in the URL hash, so a view can be linked to and reloaded into. */
const tabFromHash = (): TabId => {
  const hash = location.hash.replace('#', '')
  return isTabId(hash) ? hash : 'research'
}

export default function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash)
  // The session lives here so flipping tabs loses nothing: a hand in
  // progress and the bankroll survive a look at the research.
  const [session, setSession, pack] = useSession()

  // The top bar's "Play the solver" button points at #play. On this page that
  // is a hash change with no reload, so listen.
  useEffect(() => {
    const onHash = () => setTab(tabFromHash())
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])

  const show = (id: TabId) => {
    setTab(id)
    // Replace rather than push: the tabs are two views of one page.
    history.replaceState(null, '', `#${id}`)
  }

  const lcfr = perHundred(series('lcfr').at(-1)!.exploitability)
  const cfrplus = perHundred(series('cfrplus').at(-1)!.exploitability)

  return (
    <>
      <TopBar here={3} />

      <main className="wrap">
        <Hero
          eyebrow="Rung 3 of 4 · complete · sampled CFR with a draw"
          done
          title={
            <>
              Keeping negative regret wins, <em>by three to four times.</em>
            </>
          }
          icon="deck"
          iconLabel="The deck, one card lifted through the gate, drawn in ASCII"
        >
          <p className="lede">
            Mini-drawmaha is the first game on this ladder with no referee and no whole-tree walk:{' '}
            {SIZES[2].infosets.toLocaleString()} information sets, a draw whose chance node depends on the action taken,
            and a pot that pays twice. Four regret rules — vanilla, LCFR, CFR+, DCFR — raced to{' '}
            {((FINAL_ITERATION * WORKERS) / 1e6).toFixed(0)} million sampled hands per seat under lockstep MCCFR, each
            graded by the exact best-response walk. <b>The two that keep negative regret won by 3–4×</b>, and LCFR’s
            average is frozen as the rung’s answer. Read the race and what the answer does, or sit down against it with
            a coach that shows you its mix at every spot you face.
          </p>
          <p className="note">
            LCFR at {((FINAL_ITERATION * WORKERS) / 1e6).toFixed(0)}M hands per seat is exploitable for{' '}
            <b>{FROZEN.exploitability.toFixed(4)} chips/hand</b> ({lcfr.toFixed(1)} per hundred; CFR+ {cfrplus.toFixed(1)}).
            P0 is worth {FROZEN.value_p0.toFixed(4)} a hand in self-play. Seed 0, one run per rule.
          </p>
        </Hero>

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

        {tab === 'research' ? (
          <ResearchTab />
        ) : (
          <>
            <div className="block">
              <PlayPanel session={session} setSession={setSession} pack={pack} />
            </div>
            <p className="foot">
              The bot never looks a strategy up in the browser. Python dealt {pack.manifest?.deals ?? 600} hands as fixed
              deck orders and recorded the frozen strategy’s mix at every decision either player can reach in each —
              about 480 nodes a deal — plus the showdown of every post-draw world (uv run minidraw-deals). The page
              follows the path it is on and reads the vector. The pot-limit betting grammar is the one thing transcribed
              to TypeScript, and minidraw.test.ts pins it against the same thirteen round-1 lines the Python suite pins.
              Your rolls grade you against that mix; the solver’s cards and mix stay face down until the hand ends.
            </p>
          </>
        )}
      </main>

      <Footer />
    </>
  )
}
