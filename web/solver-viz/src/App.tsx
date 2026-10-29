/**
 * The range explorer: one page, two games. Full Drawmaha runs on a sampled
 * range and an illustrative policy (`ui/FullRange.tsx`, built ahead of the
 * network that will replace the policy); mini-drawmaha runs on a solved
 * strategy, every number the solver's (`ui/MiniRange.tsx`) — the tabular
 * LCFR average or either of the Deep CFR answers. The page head carries the
 * two page-wide choices, the game (a segmented control) and the strategy (a
 * select, the same state as the radio list in the tool's right column);
 * under it the two views, your range and range vs range. The game, the view
 * and the strategy live in the URL hash (`ui/hash.ts`) so a view can be
 * linked to.
 */

import { useEffect, useLayoutEffect, useState } from 'react'

import { STRATEGIES, Strategies, describeStrategy } from './mini/strategies'
import { FullRange } from './ui/FullRange'
import { type GameId, type View, formatHash, parseHash } from './ui/hash'
import { MiniRange } from './ui/MiniRange'
import { Footer, PageHead, TopBar } from './ui/site'

export type { TabId } from './ui/hash'

const GAMES: readonly { id: GameId; label: string }[] = [
  { id: 'mini', label: 'Mini-drawmaha · solved' },
  { id: 'full', label: 'Full Drawmaha · illustrative' },
]

const TABS = [
  { id: 'range', label: 'Your range' },
  { id: 'versus', label: 'Range vs range' },
] as const

/** The strategy select's grades, read from each export's manifest: one small fetch per strategy, kept. */
function useGrades(): Partial<Record<string, string>> {
  const [grades, setGrades] = useState<Partial<Record<string, string>>>({})
  useEffect(() => {
    const set = new Strategies(import.meta.env.BASE_URL)
    let live = true
    for (const s of STRATEGIES) {
      set
        .manifest(s.id)
        .then((m) => {
          const g = describeStrategy(m).grade
          if (live && g) setGrades((have) => ({ ...have, [s.id]: g }))
        })
        // A manifest that fails only costs its option the grade; choosing it reports the error in the tool.
        .catch(() => undefined)
    }
    return () => {
      live = false
    }
  }, [])
  return grades
}

export default function App() {
  const [view, setView] = useState<View>(() => parseHash(location.hash))
  useEffect(() => {
    const onHash = () => setView(parseHash(location.hash))
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])
  // The tool wants more than the site's 1160 px column (index.css, `body.wide`).
  // A layout effect, so the page is wide before its first paint rather than
  // laid out narrow and then again.
  useLayoutEffect(() => {
    document.body.classList.add('wide')
    return () => document.body.classList.remove('wide')
  }, [])
  const show = (next: View) => {
    setView(next)
    // Replace rather than push: the tabs and the switches are views of one page.
    history.replaceState(null, '', formatHash(next))
  }
  const { game, tab, strategy } = view
  const grades = useGrades()

  const controls = (
    <>
      <div className="rx-ctl">
        <span className="rx-ctl-k">Game</span>
        <div className="seg" role="group" aria-label="Game">
          {GAMES.map((g) => (
            <button key={g.id} type="button" aria-pressed={game === g.id} onClick={() => show({ ...view, game: g.id })}>
              {g.label}
            </button>
          ))}
        </div>
      </div>
      <label className="rx-ctl">
        <span className="rx-ctl-k">Strategy</span>
        {game === 'mini' ? (
          <select className="rx-select" value={strategy} onChange={(e) => show({ ...view, strategy: e.target.value as View['strategy'] })}>
            {STRATEGIES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
                {grades[s.id] ? ` · exploitability ${grades[s.id]}` : ''}
              </option>
            ))}
          </select>
        ) : (
          <select className="rx-select" value="illustrative" disabled>
            <option value="illustrative">Illustrative policy · not a solver’s</option>
          </select>
        )}
      </label>
    </>
  )

  return (
    <>
      <TopBar here="solver" />
      <main className="wrap">
        <PageHead
          title="Range explorer"
          lede={
            game === 'mini'
              ? 'The solver’s range at any spot of mini-drawmaha, as the frozen strategy plays it.'
              : 'Every holding at a flop of Full Drawmaha, sampled exactly; the action mix is illustrative.'
          }
          controls={controls}
        />

        <div className="views rx-views" role="tablist" aria-label="Views">
          {TABS.map((entry) => (
            <button key={entry.id} type="button" role="tab" aria-selected={tab === entry.id} className={tab === entry.id ? 'view current' : 'view'} onClick={() => show({ ...view, tab: entry.id })}>
              <b>{entry.label}</b>
            </button>
          ))}
        </div>

        {game === 'full' ? <FullRange tab={tab} /> : <MiniRange tab={tab} strategy={strategy} onStrategy={(s) => show({ ...view, strategy: s })} />}
      </main>
      <Footer />
    </>
  )
}
