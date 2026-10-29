/**
 * `/rung3`: a research publication over mini-drawmaha, with two tools
 * beside it. One masthead for the page (the finding, the date, the author,
 * a summary), then three views: the research (the race, the clock, and the
 * net, as chapters with captioned figures); Learn — the range interface over
 * the strategy at every spot, the same view `/solver` shows, imported from
 * that app rather than copied (`ui/LearnTab.tsx`); and the trainer (play
 * the frozen strategy with a coach). The tab, and the Learn tab's view and
 * strategy, live in the URL hash (`learn.ts`) so any of them can be linked
 * to, and the Rungs ladder in the top bar links all three.
 */

import { useEffect, useLayoutEffect, useState } from 'react'

// The range interface's stylesheet rides along with its components. Its one
// page-wide rule hangs off `body.wide`, set below only while Learn shows.
import '../../solver-viz/src/index.css'

import { useSession } from './deals'
import { BEST, KEY_NUMBERS, lcfrAt, markOf } from './deep'
import { type PageView, type TabId, formatHash, parseHash } from './learn'
import { FINAL_ITERATION, SIZES, WORKERS, perHundred, series } from './research'
import { LearnTab } from './ui/LearnTab'
import { Masthead } from './ui/Masthead'
import { PackBoundary } from './ui/PackBoundary'
import { PlayPanel } from './ui/PlayPanel'
import { ResearchTab } from './ui/ResearchTab'
import { fraction } from './ui/scales'
import { Footer, TopBar } from './ui/site'

const TABS: readonly { id: TabId; label: string; blurb: string }[] = [
  { id: 'research', label: 'The research', blurb: 'the race, the answer, and the net, in nine chapters' },
  { id: 'learn', label: 'Learn', blurb: 'every spot, every hand it could hold — tabular or Deep CFR' },
  { id: 'play', label: 'Play the solver', blurb: 'heads-up against the frozen strategy, with a coach' },
]

export default function App() {
  const [page, setPage] = useState<PageView>(() => parseHash(location.hash))
  const { tab } = page
  // The session lives here so flipping tabs loses nothing: a hand in
  // progress and the bankroll survive a look at the research.
  const [session, setSession, pack] = useSession()

  // The top bar's "Play the solver" button points at #play, and the contents
  // rail points at chapters. On this page those are hash changes with no
  // reload, so listen; an unknown segment keeps the research tab.
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
  const hands = (FINAL_ITERATION * WORKERS) / 1e6

  return (
    <>
      <TopBar here="research" />

      <main className="wrap publication">
        <Masthead
          eyebrow="Research · rungs 3 and 4 · mini-drawmaha"
          title={
            <>
              Keeping negative regret wins by four times, <em>and a network is an eighth of the table at a million hands.</em>
            </>
          }
          date="October 27, 2026"
          byline="Alec Ondo"
          tags={['CFR', 'MCCFR', 'Deep CFR', 'exploitability']}
        >
          <p className="summary">
            Mini-drawmaha is the first game on this ladder with no referee and no whole-tree walk:{' '}
            {SIZES[2].infosets.toLocaleString()} information sets, a draw whose chance node depends on the action taken,
            and a pot that pays twice. Four regret rules raced to {hands.toFixed(0)} million sampled hands per seat under
            lockstep MCCFR, each graded by the exact best-response walk, and{' '}
            <b>the two that keep negative regret won by three to four times</b> ({lcfr.toFixed(1)} against{' '}
            {cfrplus.toFixed(1)} chips per hundred hands). Deep CFR then trained on the same game, the same walk and the
            same hands: at one million hands per seat the best network is exploitable for{' '}
            {perHundred(markOf(BEST, 100).exact.exploitability).toFixed(1)} against the table's{' '}
            {perHundred(lcfrAt(1_000_000)).toFixed(1)}, <b>about {fraction(KEY_NUMBERS.priceAt1M)}</b> ({KEY_NUMBERS.priceAt1M.toFixed(2)} of the table's), and its
            curve has flattened by five million hands where the table's keeps falling. One seed throughout: a gap under
            about ten percent is suggestive, not settled.
          </p>
        </Masthead>

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
