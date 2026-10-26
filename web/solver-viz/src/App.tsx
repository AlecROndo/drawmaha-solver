/**
 * The solver's range interface: one page, two games. Full Drawmaha runs on
 * a sampled range and an illustrative policy (`ui/FullRange.tsx`, built
 * ahead of the rung-4 network); mini-drawmaha runs on a solved strategy,
 * every number the solver's (`ui/MiniRange.tsx`) — rung 3's tabular LCFR
 * average or either of rung 4's Deep CFR answers, behind a switch on that
 * view. The game switch sits under the hero; each game has the same two
 * tabs, your range and range vs range; and the game, the tab and the
 * strategy live in the URL hash (`ui/hash.ts`) so a view can be linked to.
 */

import { useEffect, useState } from 'react'

import { FullRange } from './ui/FullRange'
import { type View, formatHash, parseHash } from './ui/hash'
import { MiniRange } from './ui/MiniRange'
import { Footer, Hero, TopBar } from './ui/site'

export type { TabId } from './ui/hash'

const GAMES = [
  { id: 'mini', label: 'Mini-drawmaha · solved', blurb: 'rung 3’s tabular LCFR or rung 4’s Deep CFR, every number the solver’s' },
  { id: 'full', label: 'Full Drawmaha · illustrative', blurb: 'the interface ahead of rung 4’s network, over a sampled range' },
] as const

const TABS = [
  { id: 'range', label: 'Your range', blurb: 'the grid, the ribbon, the ledger, the deck, the draw' },
  { id: 'versus', label: 'Range vs range', blurb: 'their range beside yours, and the hand as ranges' },
] as const

export default function App() {
  const [view, setView] = useState<View>(() => parseHash(location.hash))
  useEffect(() => {
    const onHash = () => setView(parseHash(location.hash))
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])
  const show = (next: View) => {
    setView(next)
    // Replace rather than push: the tabs and the switch are views of one page.
    history.replaceState(null, '', formatHash(next))
  }
  const { game, tab, strategy } = view

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
            <b>Mini-drawmaha is solved</b>: its range at every spot is the reach-weighted set under a frozen strategy —
            rung 3’s tabular LCFR, or rung 4’s Deep CFR as the policy net Π or as its exact average, chosen by a switch on
            the view — every percentage is that strategy’s, and the only rounding is the export’s 1/250 step.{' '}
            <b>Full Drawmaha is not yet</b>: its composition is real (sampled and classified exactly) but its action mix
            is an illustrative stand-in for the rung-4 network, and the page says so wherever it is drawn.
          </p>
        </Hero>

        <div className="views games" role="tablist" aria-label="Games">
          {GAMES.map((g) => (
            <button key={g.id} role="tab" aria-selected={game === g.id} className={game === g.id ? 'view current' : 'view'} onClick={() => show({ ...view, game: g.id })}>
              <b>{g.label}</b>
              <span>{g.blurb}</span>
            </button>
          ))}
        </div>
        <div className="views" role="tablist" aria-label="Views">
          {TABS.map((entry) => (
            <button key={entry.id} role="tab" aria-selected={tab === entry.id} className={tab === entry.id ? 'view current' : 'view'} onClick={() => show({ ...view, tab: entry.id })}>
              <b>{entry.label}</b>
              <span>{entry.blurb}</span>
            </button>
          ))}
        </div>

        {game === 'full' ? <FullRange tab={tab} /> : <MiniRange tab={tab} strategy={strategy} onStrategy={(s) => show({ ...view, strategy: s })} />}
      </main>
      <Footer />
    </>
  )
}
