import { useState } from 'react'

import { SOLVE } from './solve'
import { advance, jump, type Step, type Walk } from './timeline'
import { NextActions, Timeline } from './ui/Timeline'
import { SeatPanel } from './ui/SeatPanel'
import { IdentityRail, LadderLine, Panel, Squiggle } from './ui/site'

/**
 * The line the page opens on: check, check, the jack turns, check — deep
 * enough that the timeline, the board station, and both seats' panels all
 * have something to say before the visitor touches anything.
 */
const OPENING: Step[] = [
  { k: 'a', v: 'c' },
  { k: 'a', v: 'c' },
  { k: 'b', v: 'J' },
  { k: 'a', v: 'c' },
]

export default function App() {
  const [walk, setWalk] = useState<Walk>({ path: OPENING, cur: OPENING.length })

  return (
    <>
      <LadderLine here={2} />
      <div className="shell">
        <IdentityRail now="Rung 2 · Leduc poker" next="Rung 3 · mini-Drawmaha" />
        <main>
          <section>
            <p className="stop">Rung 2 · Leduc poker · 288 information sets</p>
            <h2 className="big">The action timeline, drawn as the ladder.</h2>
            <Squiggle />
            <p className="lede">
              The site draws its validation ladder as a line with a station per rung, filled as
              far as the climb has got. A hand of Leduc is the same shape — a line of decisions —
              so this page reuses the motif: one station per action, the board card is a station
              too, and the rail is solid as far as the hand has been played.{' '}
              <strong>Click any station to rewind to it; click the board card to change it.</strong>{' '}
              Rewinding never deletes the stations ahead; only choosing a different action forks
              the line, and the page warns before it does. You never see a card — each seat is its whole range, which is the
              only thing either player could actually condition on.
            </p>
            <p className="note script">
              {SOLVE.iterations.toLocaleString()}-iteration solve — every number on this page is
              the Python solver’s output: exploitability {SOLVE.exploitabilityAverage.toFixed(4)}{' '}
              chips/hand, game value {SOLVE.gameValue.toFixed(4)} against the exact{' '}
              {SOLVE.gameValueExact.toFixed(4)}.
            </p>
          </section>

          <section>
            <Panel
              className="timeline wide"
              k="Fig. 1 · the hand so far — click a station to rewind"
              label="The action timeline"
            >
              <Timeline walk={walk} onJump={(n) => setWalk(jump(walk, n))} />
              <NextActions walk={walk} onStep={(step) => setWalk(advance(walk, step))} />
            </Panel>
          </section>

          <section className="panels">
            <SeatPanel seat={0} walk={walk} />
            <SeatPanel seat={1} walk={walk} />
          </section>

          <p className="foot">
            The percentages on the round-1 next-action buttons are how often play actually
            reaches each branch under the solved strategy, so those spots stop being equally
            weighted; round-2 buttons carry none — a board-conditioned reach is future work,
            alongside the exploiter.
            Editing a range and watching the solver punish it needs a Leduc exploiter — that is
            the next PR, the rung-1 exploit tab’s equivalent. Everything shown here is real
            solver output.
          </p>
        </main>
      </div>
    </>
  )
}
