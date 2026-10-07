/**
 * The solver's range interface: one window that answers "here is the spot,
 * here is everything I could be holding, what is done with each region, and
 * with mine". The hand so far is a two-lane score above the window; inside,
 * the spot header's controls open the secondary views (the three panes from
 * the mix bar, the ledger from an action, the deck from the cards button),
 * the two-half grid sits beside the strength ribbon and the equity plane,
 * and the draw board takes the grid's place when the cursor is on the draw.
 * A second tab puts the other seat's range beside yours with the filmstrip
 * at its foot.
 *
 * Everything on the page is an aggregate over one sampled range: N holdings
 * drawn from the cards unseen on the flop and classified twice. The
 * composition is real to sampling error; the action mix is an illustrative
 * function of each hand's two equities, because rung 4 is not trained yet.
 * The page says so in its status line and nowhere pretends otherwise.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'

import { density, deckStats, drawTable, examples, gridView, percentileOf, ribbon, type RibbonKey } from './engine/aggregate'
import { parseBoard, type Card } from './engine/cards'
import { equityAgainst } from './engine/equity'
import { EMPTY_FILTER, maskFor, type Filter } from './engine/filter'
import { equitiesOfHand, mixOfHand, policyFor, throwCounts, throwOfHand, type Node } from './engine/policy'
import { exactRange, sampleRange } from './engine/sample'
import { Deck } from './ui/Deck'
import { DrawBoard } from './ui/DrawBoard'
import { Grid, type Pin } from './ui/Grid'
import { Ledger } from './ui/Ledger'
import { Panes } from './ui/Panes'
import { Plane } from './ui/Plane'
import { Ribbon } from './ui/Ribbon'
import { OneLine, Score, type Cursor, type HandLine } from './ui/Score'
import { Sheet } from './ui/Sheet'
import { SpotHead, type Opened, type SortKey } from './ui/SpotHead'
import { Versus } from './ui/Versus'
import { count } from './ui/bars'
import { Footer, Hero, Panel, TopBar } from './ui/site'

/** Holdings sampled per seat. Enough that a 0.2% cell holds ~100 hands; cheap enough to resample on a board change. */
const N = 50_000
const RIBBON_BINS = 120
const PLANE_BINS = 18
const DEFAULT_BOARD = 'Ks 9s 4d'

const TABS = [
  { id: 'range', label: 'Your range', blurb: 'the grid, the ribbon, the ledger, the deck, the draw' },
  { id: 'versus', label: 'Range vs range', blurb: 'their range beside yours, and the hand as ranges' },
] as const
type TabId = (typeof TABS)[number]['id']
const isTabId = (v: string): v is TabId => TABS.some((t) => t.id === v)
const tabFromHash = (): TabId => {
  const hash = location.hash.replace('#', '')
  return isTabId(hash) ? hash : 'range'
}

export default function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash)
  useEffect(() => {
    const onHash = () => setTab(tabFromHash())
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])
  const show = (id: TabId) => {
    setTab(id)
    history.replaceState(null, '', `#${id}`)
  }

  // The hand: its board, what seat 0 did, and which of your two decisions is on screen.
  const [line, setLine] = useState<HandLine>({ board: parseBoard(DEFAULT_BOARD), villainAct: 'pot', cursor: 'you' })
  const [boardText, setBoardText] = useState(DEFAULT_BOARD)
  const [boardError, setBoardError] = useState<string | null>(null)
  const onBoard = useCallback((text: string) => {
    setBoardText(text)
    try {
      const board = parseBoard(text)
      setBoardError(null)
      setLine((l) => ({ ...l, board }))
      setFilter(EMPTY_FILTER)
      setPin(null)
    } catch (e) {
      // The range on screen stays the last good one; the error is printed under the field.
      setBoardError(e instanceof Error ? e.message : String(e))
    }
  }, [])
  const node: Node = line.villainAct === 'pot' ? 'facing' : 'open'

  // The two seats' ranges: your sample and theirs, from the same board.
  const sample = useMemo(() => sampleRange(line.board, N, 1), [line.board])
  const other = useMemo(() => sampleRange(line.board, N, 2), [line.board])
  const pol = useMemo(() => policyFor(sample, node), [sample, node])
  const theirOpen = useMemo(() => policyFor(other, 'open'), [other])
  const throws = useMemo(() => throwCounts(sample), [sample])
  const theirThrows = useMemo(() => throwCounts(other), [other])

  // The filter: whole rows, whole columns, and the cards placed from the deck.
  // Held cards are not a mask over the sample — a sample cannot narrow to a
  // hand — but an exact enumeration of every holding that contains them,
  // with equities still measured against the whole sample. `range` is the
  // set every view below is drawn over; `sample` stays the reference.
  const [filter, setFilter] = useState<Filter>(EMPTY_FILTER)
  const held = filter.held
  const range = useMemo(() => (held.length ? exactRange(line.board, held, sample) : sample), [line.board, held, sample])
  const rangePol = useMemo(() => (range === sample ? policyFor(sample, node) : policyFor(range, node)), [range, sample, node])
  const rangeThrows = useMemo(() => (range === sample ? throwCounts(sample) : throwCounts(range)), [range, sample])
  const rowColFilter = useMemo<Filter>(() => ({ rows: filter.rows, cols: filter.cols, held: [] }), [filter.rows, filter.cols])
  const mask = useMemo(() => maskFor(range, rowColFilter), [range, rowColFilter])
  const view = useMemo(() => gridView(range, rangePol, mask), [range, rangePol, mask])
  const whole = useMemo(() => gridView(sample, pol, null), [sample, pol])
  const theirs = useMemo(() => gridView(other, theirOpen, null), [other, theirOpen])

  const [pin, setPin] = useState<Pin | null>(null)
  const [opened, setOpened] = useState<Opened>(null)
  const [sort, setSort] = useState<SortKey>(null)
  const [rkey, setRkey] = useState<RibbonKey>('total')

  // With five cards placed the range is one hand, and a one-slice ribbon says
  // nothing: the ribbon and the plane then show the whole range with the hand
  // marked on it, which is the question they exist to answer.
  const full = held.length === 5
  const ribbonBins = useMemo(
    () => (full ? ribbon(sample, pol, null, rkey, RIBBON_BINS) : ribbon(range, rangePol, mask, rkey, RIBBON_BINS)),
    [full, sample, pol, range, rangePol, mask, rkey],
  )
  const plane = useMemo(() => (full ? density(sample, pol, null, PLANE_BINS) : density(range, rangePol, mask, PLANE_BINS)), [full, sample, pol, range, rangePol, mask])
  const deck = useMemo(() => deckStats(range, rangePol, mask), [range, rangePol, mask])
  const draws = useMemo(() => drawTable(range, rangeThrows, mask), [range, rangeThrows, mask])

  // The held hand, when five cards are placed: its own direct answer, not an aggregate.
  const handMix = useMemo(() => (full ? mixOfHand(held, line.board, sample, node) : null), [full, held, line.board, sample, node])
  const handThrow = useMemo(() => (full ? throwOfHand(held, line.board, sample) : null), [full, held, line.board, sample])
  const handPct = useMemo(() => (full ? percentileOf(sample, pol, rkey, held, line.board) : null), [full, sample, pol, rkey, held, line.board])
  const handPoint = useMemo(() => {
    if (!full) return null
    const eq = equitiesOfHand(held, line.board, sample)
    return { inner: eq.eqI, outer: eq.eqO }
  }, [full, held, line.board, sample])

  const toggleHeld = (c: Card) =>
    setFilter((f) => ({ ...f, held: f.held.includes(c) ? f.held.filter((x) => x !== c) : f.held.length < 5 ? [...f.held, c] : f.held }))

  const examplesFor = useCallback((row: number, col: number) => examples(range, rangePol, mask, row, col, 5), [range, rangePol, mask])

  const equityOf = useCallback(
    (row: number, col: number) => {
      const cellMask = new Uint8Array(sample.n)
      for (let i = 0; i < sample.n; i++) cellMask[i] = sample.row[i] === row && sample.col[i] === col && (mask === null || mask[i] === 1) ? 1 : 0
      const weights = line.villainAct === 'pot' ? theirOpen.p : theirOpen.c
      return equityAgainst(sample, cellMask, other, weights)
    },
    [sample, other, mask, theirOpen, line.villainAct],
  )
  const throwMix = (table: { share: number; counts: number[] }[]) => {
    const total = table.reduce((s, r) => s + r.share, 0) || 1
    return [0, 1, 2, 3, 4, 5].map((k) => table.reduce((s, r) => s + r.share * r.counts[k], 0) / total)
  }

  const onCursor = (cursor: Cursor) => {
    setLine((l) => ({ ...l, cursor }))
    setOpened(null)
    setSort(null)
  }

  return (
    <>
      <TopBar here={4} />
      <main className="wrap">
        <Hero
          eyebrow="Rung 4 of 4 · in progress · the interface, ahead of the net"
          title={
            <>
              Everything you could have here, <em>and what to do with it.</em>
            </>
          }
          icon="cards"
          iconLabel="Five playing cards fanned, drawn in ASCII"
        >
          <p className="lede">
            Pick a flop. The page deals every five-card holding you could be dealt around it and sorts them by the two
            halves of the pot: what the five cards make on their own (the <b>inner</b> hand) and what two of them make
            with the board (the <b>outer</b>, Omaha hand). The grid is that map; the ribbon beside it sorts the same
            holdings by strength; the ledger opens any region down to a single hand; the deck lets you build yours card
            by card; the draw board takes over when the decision is how many to throw.
          </p>
          <p className="note">
            The composition is real: {count(N)} holdings sampled from the 49 unseen cards and classified exactly. The{' '}
            <b>action mix is illustrative</b> — rung 4’s network is not trained yet, so a smooth function of each hand’s
            two showdown-now equities stands in for the policy. Shapes and counts are honest; the percentages are not a
            solver’s.
          </p>
        </Hero>

        <div className="views" role="tablist" aria-label="Views">
          {TABS.map((entry) => (
            <button key={entry.id} role="tab" aria-selected={tab === entry.id} className={tab === entry.id ? 'view current' : 'view'} onClick={() => show(entry.id)}>
              <b>{entry.label}</b>
              <span>{entry.blurb}</span>
            </button>
          ))}
        </div>

        <div className="block">
          <Panel n="01" k="the hand so far — click a cell to move to that decision" wide>
            <Score line={line} onToggleVillain={() => setLine((l) => ({ ...l, villainAct: l.villainAct === 'pot' ? 'check' : 'pot' }))} onCursor={onCursor} />
          </Panel>
        </div>

        {tab === 'range' ? (
          <div className="block">
            <section className="panel wide rangewin" aria-label="your range at this spot">
              <div className="titlebar">
                <span className="dots">
                  <i />
                  <i />
                  <i />
                </span>
                <span className="k">02 · your range</span>
                <OneLine line={line} />
                <span className="no">02</span>
              </div>
              <div className="pane">
                <SpotHead
                  node={node}
                  cursor={line.cursor}
                  all={whole.all}
                  left={view.all}
                  n={N}
                  exact={range === sample ? null : range.n}
                  held={held}
                  boardText={boardText}
                  boardError={boardError}
                  onBoard={onBoard}
                  opened={opened}
                  onOpen={setOpened}
                  sort={sort}
                  onSort={(k) => {
                    setSort(k)
                    if (k && !pin) setPin({ row: -1, col: -1 })
                  }}
                  onUnhold={toggleHeld}
                />
                {opened === 'panes' && (
                  <Sheet title="one pane per action" onClose={() => setOpened(null)}>
                    <Panes view={view} node={node} />
                  </Sheet>
                )}
                {opened === 'deck' && (
                  <Sheet title="the deck · tap cards to hold them" onClose={() => setOpened(null)}>
                    <Deck board={line.board} held={held} potLift={deck.potLift} onToggle={toggleHeld} />
                  </Sheet>
                )}
                {handMix && line.cursor === 'you' && (
                  <div className="handanswer">
                    <span className="lbl">your hand · {node === 'facing' ? 'fold / call / pot' : 'check / pot'}</span>
                    <span className="mixbar" style={{ height: 10 }}>
                      <i className="f" style={{ width: `${handMix.f * 100}%` }} />
                      <i className="c" style={{ width: `${handMix.c * 100}%` }} />
                      <i className="p" style={{ width: `${handMix.p * 100}%` }} />
                    </span>
                    <span className="nums">
                      {node === 'facing' && <>fold <b>{Math.round(handMix.f * 100)}%</b> · </>}
                      {node === 'facing' ? 'call' : 'check'} <b>{Math.round(handMix.c * 100)}%</b> · pot <b>{Math.round(handMix.p * 100)}%</b> · equity{' '}
                      <b>{Math.round(handMix.eq * 100)}%</b>
                    </span>
                  </div>
                )}
                <div className="body">
                  <div className="main">
                    {line.cursor === 'you' ? (
                      <>
                        <Grid
                          view={view}
                          node={node}
                          rows={filter.rows}
                          cols={filter.cols}
                          pin={pin && pin.row >= 0 ? pin : null}
                          onRows={(rows) => setFilter((f) => ({ ...f, rows }))}
                          onCols={(cols) => setFilter((f) => ({ ...f, cols }))}
                          onPin={setPin}
                        />
                        {(pin || sort) && (
                          <Ledger view={view} node={node} sort={sort} pin={pin && pin.row >= 0 ? pin : null} onPin={setPin} examplesFor={examplesFor} />
                        )}
                      </>
                    ) : (
                      <DrawBoard table={draws} hand={handThrow && full ? { cards: held, ...handThrow } : null} />
                    )}
                  </div>
                  {line.cursor === 'you' && (
                    <aside className="side">
                      <Ribbon bins={ribbonBins} rkey={rkey} onKey={setRkey} percentile={handPct} all={full ? whole.all : view.all} />
                      <Plane cells={plane} bins={PLANE_BINS} hand={handPoint} />
                    </aside>
                  )}
                </div>
              </div>
              <div className="statusbar">
                <span>
                  {count(N)} holdings sampled · composition exact to sampling · <b>policy illustrative — rung 4 not trained</b>
                </span>
                <span>
                  {range !== sample
                    ? `${count(range.n)} holdings contain the ${held.length === 1 ? 'card' : 'cards'} you placed — enumerated exactly, not sampled`
                    : mask
                      ? `${count(mask.reduce((s, v) => s + v, 0))} of ${count(N)} sampled hands in the filter`
                      : 'nothing filtered'}
                </span>
              </div>
            </section>
          </div>
        ) : (
          <div className="block">
            <Panel n="02" k="range vs range — their range after what they did, beside yours" wide>
              <Versus mine={whole} theirs={theirs} node={node} villainAct={line.villainAct} equityOf={equityOf} throwsMine={throwMix(drawTable(sample, throws, null))} throwsTheirs={throwMix(drawTable(other, theirThrows, null))} />
            </Panel>
          </div>
        )}

        <p className="foot">
          Every number on this page is computed in the browser from {count(N)} sampled holdings per seat on the flop you
          typed: each is scored as the five cards alone and as the best two with the board, and its two equities are its
          percentile in the sample. The action mix is not a solver’s. It is the same illustrative rule the design studies
          used, so the interface could be built and used before rung 4’s network exists; when it does, the policy module
          is the one file that changes.
        </p>
      </main>
      <Footer />
    </>
  )
}
