/**
 * `/research`, `/learn` and `/play` (and `/rung3`, where the app is built): a
 * research publication over mini-drawmaha with two tools beside it. Each tab
 * opens on its own head — the publication's title block and results strip,
 * or a page head for Learn and Play — with the three views under it: the
 * research (nine chapters, seventeen figures); Learn, the range interface
 * over the strategy at every spot, the same view `/solver` shows, imported
 * from that app rather than copied (`ui/LearnTab.tsx`); and Play, the
 * trainer against the frozen strategy with a coach. Vercel rewrites the
 * three product paths to this page, so the tab is read from the path as well
 * as the hash (`learn.ts`), and the Learn tab's view and strategy live in the
 * hash so any of them can be linked to.
 */

import { useEffect, useLayoutEffect, useState } from 'react'

// The range interface's stylesheet rides along with its components. Its one
// page-wide rule hangs off `body.wide`, set below only while Learn shows.
import '../../solver-viz/src/index.css'

import { useSession } from './deals'
import { BEST, POT_ARMS, lcfrAt, markOf } from './deep'
import { type PageView, type TabId, addressOf, parseLocation } from './learn'
import { FINAL_ITERATION, SIZES, WORKERS, perHundred, series } from './research'
import { LearnTab } from './ui/LearnTab'
import { Masthead, type Result } from './ui/Masthead'
import { PackBoundary } from './ui/PackBoundary'
import { PlayPanel, packWords } from './ui/PlayPanel'
import { ResearchTab } from './ui/ResearchTab'
import { Footer, PageHead, TopBar } from './ui/site'

const TABS: readonly { id: TabId; label: string; blurb: string; title: string }[] = [
  { id: 'research', label: 'Research', blurb: 'Nine chapters, seventeen figures', title: 'Mini-drawmaha, solved' },
  { id: 'learn', label: 'Learn', blurb: 'Every spot, every hand it could hold', title: 'Learn' },
  { id: 'play', label: 'Play', blurb: 'Heads-up against the solver, with a coach', title: 'Play the solver' },
]

const REPO = 'https://github.com/AlecROndo/drawmaha-solver'

const here = () => parseLocation(location.pathname, location.hash)

export default function App() {
  const [page, setPage] = useState<PageView>(here)
  const { tab } = page
  // The session lives here so flipping tabs loses nothing: a hand in
  // progress and the bankroll survive a look at the research.
  const [session, setSession, pack] = useSession()

  // The research tab's callouts link to #play and #learn/..., and the
  // contents rail to chapters. On this page those are hash changes with no
  // reload, so listen; a hash that names no tab keeps the path's.
  useEffect(() => {
    const onMove = () => setPage(here())
    addEventListener('hashchange', onMove)
    addEventListener('popstate', onMove)
    return () => {
      removeEventListener('hashchange', onMove)
      removeEventListener('popstate', onMove)
    }
  }, [])

  // Only the Learn tab wants the whole browser (solver-viz's index.css,
  // `body.wide`). A layout effect, so the page is wide before it paints:
  // opening `/learn` directly lays the grid out once, not at 1160 px and
  // then again. The cleanup narrows it when the tab changes or the page goes.
  useLayoutEffect(() => {
    if (tab !== 'learn') return
    document.body.classList.add('wide')
    return () => document.body.classList.remove('wide')
  }, [tab])

  useEffect(() => {
    document.title = `${TABS.find((t) => t.id === tab)!.title} · Drawmaha Solver`
  }, [tab])

  const show = (next: PageView) => {
    setPage(next)
    // Replace rather than push: the tabs are views of one page.
    history.replaceState(null, '', addressOf(location.pathname, next))
  }

  const hands = FINAL_ITERATION * WORKERS
  const lcfr = perHundred(series('lcfr').at(-1)!.exploitability)
  const cfrplus = perHundred(series('cfrplus').at(-1)!.exploitability)
  const sqrtPot = POT_ARMS.find((a) => a.potPower === 0.5 && a.lr === 0.0005)!
  const sqrtPotAt2M = perHundred(sqrtPot.exact[200])
  const best1M = perHundred(markOf(BEST, 100).exact.exploitability)
  const table1M = perHundred(lcfrAt(1_000_000))

  const results: Result[] = [
    {
      value: lcfr.toFixed(1),
      label: <>chips per hundred hands: tabular LCFR at {hands / 1e6} million hands per seat, graded by the exact best response</>,
      href: '#ch-03',
      chapter: '03',
    },
    {
      value: `${Math.round(cfrplus / lcfr)}×`,
      label: <>LCFR's margin over CFR+ ({cfrplus.toFixed(1)}) at the same mark, both raced on one seed</>,
      href: '#ch-03',
      chapter: '03',
    },
    {
      value: sqrtPotAt2M.toFixed(1),
      label: (
        <>
          chips per hundred hands: Deep CFR's exact average with ÷ √pot targets at 2 million hands per seat, where LCFR
          grades {perHundred(lcfrAt(2_000_000)).toFixed(1)}
        </>
      ),
      href: '#ch-06',
      chapter: '06',
    },
  ]

  const views = (
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
  )

  return (
    <>
      <TopBar here={tab === 'play' ? 'play' : 'research'} />

      <main className={`wrap rung3 on-${tab}`}>
        {tab === 'research' ? (
          <Masthead
            eyebrow="Research · Mini-drawmaha · 27 October 2026"
            title="Mini-drawmaha, solved: keeping negative regret wins by four times, and a network beats the table at equal hands"
            dek={
              <>
                A draw game of {(SIZES[2].infosets / 1e6).toFixed(1)} million information sets, with a pot that pays twice,
                is solved by sampled CFR and graded by an exact best response. The two regret rules that keep negative
                regret beat CFR+ and DCFR by three to four times. Deep CFR on the same game and the same hands is
                exploitable for {best1M.toFixed(1)} chips per hundred hands at one million hands per seat, where the table
                is at {table1M.toFixed(1)}.
              </>
            }
            meta={[
              <b key="a">Alec Ondo</b>,
              <span key="c">9 chapters · 17 figures · one seed throughout</span>,
              <a key="d" href={`${REPO}/tree/main/figures`}>
                Figure data
              </a>,
              <a key="g" href={REPO}>
                Code
              </a>,
              <a key="s" href={`${REPO}/blob/main/paper/drawmaha-ml-survey.pdf`}>
                ML survey (PDF)
              </a>,
            ]}
            results={results}
          />
        ) : tab === 'learn' ? (
          <PageHead
            eyebrow="Learn · Mini-drawmaha"
            title="Every spot, every hand it could hold"
            lede="Pick a spot and read the strategy's mix for every hand in range. Three strategies sit behind one switch: the tabular LCFR answer, Deep CFR's policy net Π and Deep CFR's exact average."
          />
        ) : (
          <PageHead
            eyebrow="Play · Mini-drawmaha"
            title="Play the solver"
            lede={`Hands are dealt from a ${pack.manifest?.deals ?? 600}-deal pack. The solver plays its frozen strategy exactly and shows you its mix after every choice you make.`}
          >
            <p className="note">{packWords(pack.manifest)}</p>
          </PageHead>
        )}

        {views}

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
            <PackBoundary>
              <PlayPanel session={session} setSession={setSession} pack={pack} />
            </PackBoundary>
            <p className="foot">
              The bot never looks a strategy up in the browser. Python dealt {pack.manifest?.deals ?? 600} hands as fixed
              deck orders and recorded the frozen strategy’s mix at every decision either player can reach in each, about
              480 nodes a deal, plus the showdown of every post-draw world (uv run minidraw-deals). The page follows the
              path it is on and reads the vector. The pot-limit betting grammar is the one thing transcribed to
              TypeScript, and minidraw.test.ts pins it against the same thirteen round-1 lines the Python suite pins. Your
              rolls grade you against that mix; the solver’s cards and mix stay face down until the hand ends.
            </p>
          </>
        )}
      </main>

      <Footer />
    </>
  )
}
