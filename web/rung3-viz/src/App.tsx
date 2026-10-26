/**
 * `/rung3`: three tabs over mini-drawmaha, in the order a visitor meets
 * them. The research (the race and what the answer does); Learn — the range
 * interface over the strategy at every spot, the same view `/solver` shows,
 * imported from that app rather than copied (`ui/LearnTab.tsx`); and the
 * trainer (play the frozen strategy with a coach). The tab, and the Learn
 * tab's view and strategy, live in the URL hash (`learn.ts`) so any of them
 * can be linked to, and the Rungs ladder in the top bar links all three.
 */

import { useEffect, useLayoutEffect, useState } from 'react'

// The range interface's stylesheet rides along with its components. Its one
// page-wide rule hangs off `body.wide`, set below only while Learn shows.
import '../../solver-viz/src/index.css'

import { useSession } from './deals'
import { type PageView, type TabId, formatHash, parseHash } from './learn'
import { FINAL_ITERATION, FROZEN, SIZES, WORKERS, perHundred, series } from './research'
import { LearnTab } from './ui/LearnTab'
import { PackBoundary } from './ui/PackBoundary'
import { PlayPanel } from './ui/PlayPanel'
import { ResearchTab } from './ui/ResearchTab'
import { Footer, Hero, TopBar } from './ui/site'

const TABS: readonly { id: TabId; label: string; blurb: string }[] = [
  { id: 'research', label: 'The research', blurb: 'the race, the rules, what the answer does' },
  { id: 'learn', label: 'Learn', blurb: 'every spot, every hand it could hold — tabular or Deep CFR' },
  { id: 'play', label: 'Play the solver', blurb: 'heads-up against the frozen strategy, with a coach' },
]

export default function App() {
  const [page, setPage] = useState<PageView>(() => parseHash(location.hash))
  const { tab } = page
  // The session lives here so flipping tabs loses nothing: a hand in
  // progress and the bankroll survive a look at the research.
  const [session, setSession, pack] = useSession()

  // The top bar's "Play the solver" button points at #play. On this page that
  // is a hash change with no reload, so listen.
  useEffect(() => {
    const onHash = () => setPage(parseHash(location.hash))
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])

  // Only the Learn tab wants the whole browser (solver-viz's index.css,
  // `body.wide`). A layout effect, so the page is wide before it paints:
  // opening `#learn/...` directly lays the grid out once, not at 1312 px and
  // then again. The cleanup narrows it when the tab changes or the page goes.
  useLayoutEffect(() => {
    if (tab !== 'learn') return
    document.body.classList.add('wide')
    return () => document.body.classList.remove('wide')
  }, [tab])

  const show = (next: PageView) => {
    setPage(next)
    // Replace rather than push: the tabs are views of one page.
    history.replaceState(null, '', formatHash(next))
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
          icon="cigar"
          iconLabel="A lit cigar resting on a chip, ash at its tip and smoke rising, drawn in ASCII"
        >
          <p className="lede">
            Mini-drawmaha is the first game on this ladder with no referee and no whole-tree walk:{' '}
            {SIZES[2].infosets.toLocaleString()} information sets, a draw whose chance node depends on the action taken,
            and a pot that pays twice. Four regret rules — vanilla, LCFR, CFR+, DCFR — raced to{' '}
            {((FINAL_ITERATION * WORKERS) / 1e6).toFixed(0)} million sampled hands per seat under lockstep MCCFR, each
            graded by the exact best-response walk. <b>The two that keep negative regret won by 3–4×</b>, and LCFR’s
            average is frozen as the rung’s answer. Read the race and what the answer does, sit down against it with a
            coach that shows you its mix at every spot you face, or open any spot and see what the strategy does with
            every hand it could hold there — the frozen table, or rung 4’s Deep CFR beside it.
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
              onClick={() => show({ ...page, tab: entry.id })}
            >
              <b>{entry.label}</b>
              <span>{entry.blurb}</span>
            </button>
          ))}
        </div>

        {tab === 'research' ? (
          <ResearchTab />
        ) : tab === 'learn' ? (
          <LearnTab
            view={page.view}
            strategy={page.strategy}
            onView={(view) => show({ ...page, view })}
            onStrategy={(strategy) => show({ ...page, strategy })}
          />
        ) : (
          <>
            <div className="block">
              <PackBoundary>
                <PlayPanel session={session} setSession={setSession} pack={pack} />
              </PackBoundary>
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
