/**
 * The solver's range interface: one page, two games. Full Drawmaha runs on
 * a sampled range and an illustrative policy (`ui/FullRange.tsx`, built
 * ahead of the rung-4 network); mini-drawmaha runs on the rung-3 frozen
 * LCFR strategy, every number the solver's (`ui/MiniRange.tsx`). The game
 * switch sits under the hero; each game has the same two tabs, your range
 * and range vs range, and the tab lives in the URL hash so a view can be
 * linked to.
 */

import { useEffect, useState } from 'react'

import { FullRange } from './ui/FullRange'
import { MiniRange } from './ui/MiniRange'
import { Footer, Hero, TopBar } from './ui/site'

const GAMES = [
  { id: 'mini', label: 'Mini-drawmaha · solved', blurb: 'rung 3’s frozen LCFR strategy, every number the solver’s' },
  { id: 'full', label: 'Full Drawmaha · illustrative', blurb: 'the interface ahead of rung 4’s network, over a sampled range' },
] as const
type GameId = (typeof GAMES)[number]['id']

const TABS = [
  { id: 'range', label: 'Your range', blurb: 'the grid, the ribbon, the ledger, the deck, the draw' },
  { id: 'versus', label: 'Range vs range', blurb: 'their range beside yours, and the hand as ranges' },
] as const
export type TabId = (typeof TABS)[number]['id']

const isGameId = (v: string): v is GameId => GAMES.some((g) => g.id === v)
const isTabId = (v: string): v is TabId => TABS.some((t) => t.id === v)

/** `#mini/range`: the game and the tab, so a view can be linked to and reloaded into. */
function fromHash(): { game: GameId; tab: TabId } {
  const [g, t] = location.hash.replace('#', '').split('/')
  return { game: isGameId(g) ? g : 'mini', tab: isTabId(t) ? t : 'range' }
}

export default function App() {
  const [{ game, tab }, setView] = useState(fromHash)
  useEffect(() => {
    const onHash = () => setView(fromHash())
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])
  const show = (next: { game: GameId; tab: TabId }) => {
    setView(next)
    // Replace rather than push: the tabs are views of one page.
    history.replaceState(null, '', `#${next.game}/${next.tab}`)
  }

  return (
    <>
      <TopBar here={4} />
      <main className="wrap">
        <Hero
          eyebrow="Rung 4 of 4 · in progress · the interface"
          title={
            <>
              Everything you could have here, <em>and what to do with it.</em>
            </>
          }
          icon="cards"
          iconLabel="Five playing cards fanned, drawn in ASCII"
        >
          <p className="lede">
            Pick a spot. The page lays out every hand you could be holding there, sorted by the two halves of the pot:
            what your cards make on their own (the <b>inner</b> hand) and what two of them make with the board (the{' '}
            <b>outer</b> hand). The grid is that map; the ribbon beside it sorts the same hands by strength; the ledger
            opens any region down to a single hand; the deck lets you build yours card by card; the draw board takes over
            when the decision is what to throw.
          </p>
          <p className="note">
            <b>Mini-drawmaha is solved</b>: its range at every spot is the reach-weighted set under rung 3’s frozen LCFR
            strategy, every percentage is that strategy’s, and the only rounding is the export’s 1/250 step.{' '}
            <b>Full Drawmaha is not yet</b>: its composition is
            real (sampled and classified exactly) but its action mix is an illustrative stand-in for the rung-4 network,
            and the page says so wherever it is drawn.
          </p>
        </Hero>

        <div className="views games" role="tablist" aria-label="Games">
          {GAMES.map((g) => (
            <button key={g.id} role="tab" aria-selected={game === g.id} className={game === g.id ? 'view current' : 'view'} onClick={() => show({ game: g.id, tab })}>
              <b>{g.label}</b>
              <span>{g.blurb}</span>
            </button>
          ))}
        </div>
        <div className="views" role="tablist" aria-label="Views">
          {TABS.map((entry) => (
            <button key={entry.id} role="tab" aria-selected={tab === entry.id} className={tab === entry.id ? 'view current' : 'view'} onClick={() => show({ game, tab: entry.id })}>
              <b>{entry.label}</b>
              <span>{entry.blurb}</span>
            </button>
          ))}
        </div>

        {game === 'full' ? <FullRange tab={tab} /> : <MiniRange tab={tab} />}
      </main>
      <Footer />
    </>
  )
}
