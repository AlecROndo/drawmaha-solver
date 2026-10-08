/**
 * Mini-drawmaha's range window, over the rung-3 frozen LCFR strategy: the
 * two-lane score with the hand so far and the next actions, then the window
 * — the spot header whose controls open the secondary views, the two-half
 * grid beside the ribbon and the plane, the ledger under the grid, the draw
 * board in the grid's place at a draw decision — and, on the second tab,
 * the other seat's range beside yours with exact showdown equities.
 *
 * Every number is the solver's, read from its export and exact to that
 * export's 1/250 rounding (`mini/range.ts`). The actor's range at a decision
 * is the reach-weighted set of its private states: every physical hole (and
 * discard) it could hold, weighted by the product of its own action
 * probabilities along the line. The other seat's range is the
 * same for the other seat. Equities are showdown-now between the two
 * ranges with card removal: at round 2 both halves, exactly; with one board
 * card only the inner half can be scored, so the outer equity, the plane
 * and the scoop are absent and the page says so.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'

import type { TabId } from '../App'
import { type Cell, type GridView, type RibbonKey, deckStats, density, drawTable, examples, gridView, keyValue, percentileOf, ribbon } from '../engine/aggregate'
import type { RangeTable } from '../engine/table'
import { type MiniCard, MINI_SUIT_GLYPH, miniGlyph, parseMini } from '../mini/cards'
import { drawOrder } from '../mini/canonical'
import { MiniData } from '../mini/data'
import type { MiniSpot } from '../mini/points'
import { type MiniRange as Range, equities, reachRange } from '../mini/range'
import { MINI_INNER_ROWS, MINI_R1_COLS, MINI_R2_COLS } from '../mini/regions'
import { CardFace, MixBar, actionWord, count, pct, pct0 } from './bars'
import { Deck } from './Deck'
import { DrawBoard } from './DrawBoard'
import type { GameUI } from './game'
import { Grid, type Pin } from './Grid'
import { Ledger } from './Ledger'
import { type MiniLine, MiniScore, decisionOf, linesOf, nextOptions } from './MiniScore'
import { Panes } from './Panes'
import { Plane } from './Plane'
import { Ribbon } from './Ribbon'
import { Sheet } from './Sheet'
import { Panel } from './site'
import { Thumb } from './Versus'
import type { Opened, SortKey } from './SpotHead'

const RIBBON_BINS = 60
const PLANE_BINS = 12
const THROW_OPTIONS = ['stand pat', 'throw low', 'throw mid', 'throw top']

/** The 15-card game as the shared views see it: 6 rows, 5 or 7 columns, three to a hand. */
function miniGame(boardCards: 1 | 2): GameUI {
  return {
    rows: MINI_INNER_ROWS,
    cols: boardCards === 1 ? MINI_R1_COLS : MINI_R2_COLS,
    handSize: 3,
    deck: {
      ranks: ['6', '5', '4', '3', '2'],
      suits: MINI_SUIT_GLYPH.map((glyph, s) => ({ glyph, red: s > 0 })),
      // rank index 0 is the six = engine rank 4; suits c d h = 0..2
      cardAt: (rankIndex, suitIndex) => (4 - rankIndex) * 3 + suitIndex,
    },
    label: (c) => miniGlyph(c),
    isRed: (c) => c % 3 > 0,
  }
}

/** The actor's range and the other seat's, with the equities between them, at one spot. */
interface Ranges {
  spot: MiniSpot
  mine: Range
  theirs: Range
  eq: ReturnType<typeof equities>
}

/** The range as the shared views' table: the mix split into fold / check-call / pot, the equities as computed. */
function tableOf(r: Ranges, boardCards: 1 | 2): RangeTable {
  const { mine, eq } = r
  const n = mine.n
  const w = mine.mix.length / Math.max(1, n)
  const col = (sym: string) => mine.point.actions.indexOf(sym)
  const pick = (sym: string) => {
    const j = col(sym)
    const out = new Float32Array(n)
    if (j >= 0) for (let i = 0; i < n; i++) out[i] = mine.mix[i * w + j]
    return out
  }
  const cards = new Uint8Array(n * 3)
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) cards[i * 3 + k] = mine.cards[i * 4 + k]
  const total = new Float32Array(n)
  for (let i = 0; i < n; i++) total[i] = eq.outer ? (eq.inner[i] + eq.outer[i]) / 2 : eq.inner[i]
  return {
    n,
    rows: MINI_INNER_ROWS.length,
    cols: boardCards === 1 ? MINI_R1_COLS.length : MINI_R2_COLS.length,
    handSize: 3,
    deckSize: 15,
    cards,
    weight: mine.weight,
    row: mine.row,
    col: mine.col,
    f: pick('f'),
    c: pick('c'),
    p: pick('p'),
    eq: total,
    eqI: eq.inner,
    eqO: eq.outer,
  }
}

/** The spot the line stands on, or null when the hand is over or a chance card is owed. */
function spotOf(line: MiniLine): MiniSpot | null {
  const d = decisionOf(line)
  if (d.kind !== 'bet' && d.kind !== 'draw') return null
  const board = line.board2 === null ? [line.board1] : [line.board1, line.board2]
  const draws = d.kind === 'draw' ? line.draws.slice(0, d.player) : line.draws
  return { board, lines: linesOf(line), draws, player: d.player }
}

const spotKey = (s: MiniSpot | null): string => (s === null ? '' : JSON.stringify(s))

export function MiniRange({ tab }: { tab: TabId }) {
  const [data, setData] = useState<MiniData | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    MiniData.load(`${import.meta.env.BASE_URL}mini/`)
      .then((d) => live && setData(d))
      .catch((e: unknown) => live && setLoadError(e instanceof Error ? e.message : String(e)))
    return () => {
      live = false
    }
  }, [])

  const [line, setLine] = useState<MiniLine>({ board1: parseMini('4h'), r1: '', draws: [], board2: null, r2: '' })
  const [picking, setPicking] = useState<1 | 2 | null>(null)
  const [rows, setRows] = useState<ReadonlySet<number> | null>(null)
  const [cols, setCols] = useState<ReadonlySet<number> | null>(null)
  const [held, setHeld] = useState<MiniCard[]>([])
  const [pin, setPin] = useState<Pin | null>(null)
  const [opened, setOpened] = useState<Opened>(null)
  const [sort, setSort] = useState<SortKey>(null)
  const [rkey, setRkey] = useState<RibbonKey>('total')
  const [hover, setHover] = useState<Pin | null>(null)

  const decision = decisionOf(line)
  const spot = useMemo(() => spotOf(line), [line])
  const boardCards: 1 | 2 = line.board2 === null ? 1 : 2
  const game = useMemo(() => miniGame(boardCards), [boardCards])

  // A new line resets what was pinned and held; a held card that is now on the board is gone.
  const onLine = (next: MiniLine) => {
    setLine(next)
    setPin(null)
    setSort(null)
    setOpened(null)
    setRows(null)
    setCols(null)
    setHeld((h) => h.filter((c) => c !== next.board1 && c !== next.board2))
  }

  // The ranges at the spot, computed off the main render: the exact enumeration is a few thousand states.
  const [ranges, setRanges] = useState<Ranges | null>(null)
  const [computing, setComputing] = useState(false)
  const key = spotKey(spot)
  useEffect(() => {
    if (!data || !spot) return
    let live = true
    setComputing(true)
    ;(async () => {
      const mine = await reachRange(data, spot, spot.player, [])
      const theirs = await reachRange(data, spot, (1 - spot.player) as 0 | 1, held)
      const eq = equities(mine, theirs)
      if (live) {
        setRanges({ spot, mine, theirs, eq })
        setComputing(false)
      }
    })().catch((e: unknown) => {
      if (live) {
        setLoadError(e instanceof Error ? e.message : String(e))
        setComputing(false)
      }
    })
    return () => {
      live = false
    }
    // `key` stands in for `spot`, which is a fresh object every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, key, held])

  const current = ranges && spotKey(ranges.spot) === key ? ranges : null
  const table = useMemo(() => (current ? tableOf(current, current.spot.board.length as 1 | 2) : null), [current])

  // The mask: whole rows, whole columns, and the states holding every held card.
  const mask = useMemo(() => {
    if (!current) return null
    if (rows === null && cols === null && held.length === 0) return null
    const m = new Uint8Array(current.mine.n)
    for (let i = 0; i < current.mine.n; i++) {
      if (rows !== null && !rows.has(current.mine.row[i])) continue
      if (cols !== null && !cols.has(current.mine.col[i])) continue
      let ok = true
      for (const h of held) {
        if (current.mine.cards[i * 4] !== h && current.mine.cards[i * 4 + 1] !== h && current.mine.cards[i * 4 + 2] !== h) ok = false
      }
      if (ok) m[i] = 1
    }
    return m
  }, [current, rows, cols, held])

  const view = useMemo(() => (table ? gridView(table, mask) : null), [table, mask])
  const whole = useMemo(() => (table ? gridView(table, null) : null), [table])
  const theirsView = useMemo(() => (current ? gridView(theirTable(current), null) : null), [current])
  const ribbonBins = useMemo(() => (table ? ribbon(table, mask, rkey === 'outer' || rkey === 'scoop' ? (table.eqO ? rkey : 'inner') : rkey, RIBBON_BINS) : []), [table, mask, rkey])
  const plane = useMemo(() => (table && table.eqO ? density(table, mask, PLANE_BINS) : null), [table, mask])
  const deck = useMemo(() => (table ? deckStats(table, mask) : null), [table, mask])
  const isDraw = decision.kind === 'draw'
  const draws = useMemo(() => {
    if (!table || !current || !isDraw) return null
    return drawTable(table, current.mine.mix, 4, mask)
  }, [table, current, isDraw, mask])
  const examplesFor = useCallback((row: number, col: number) => (table ? examples(table, mask, row, col, 6) : []), [table, mask])

  // The held hand's own answers, when all three cards are placed.
  const full = held.length === 3
  const handRow = useMemo(() => {
    if (!full || !current) return -1
    for (let i = 0; i < current.mine.n; i++) if (mask === null || mask[i]) return i
    return -1
  }, [full, current, mask])
  const handPct = useMemo(() => (table && handRow >= 0 ? percentileOf(table, rkey === 'outer' || rkey === 'scoop' ? (table.eqO ? rkey : 'inner') : rkey, keyValue(table, rkey === 'outer' || rkey === 'scoop' ? (table.eqO ? rkey : 'inner') : rkey, handRow)) : null), [table, handRow, rkey])
  const handPoint = useMemo(() => (table && table.eqO && handRow >= 0 ? { inner: table.eqI[handRow], outer: table.eqO[handRow] } : null), [table, handRow])
  const handThrow = useMemo(() => {
    if (!full || !current || !isDraw || !view) return null
    // The three cards in the canonical draw order are what 'throw low / mid / top' point at.
    const order = drawOrder({ hole: held, thrown: [], board: [line.board1] })
    const w = current.mine.mix.length / current.mine.n
    const counts = [0, 0, 0, 0]
    let wsum = 0
    for (let i = 0; i < current.mine.n; i++) {
      if (mask !== null && !mask[i]) continue
      const wi = current.mine.weight[i]
      wsum += wi
      for (let k = 0; k < 4; k++) counts[k] += wi * current.mine.mix[i * w + k]
    }
    if (wsum === 0) return null
    const perCard = held.map((c) => counts[1 + order.indexOf(c)] / wsum)
    return { cards: held, counts: counts.map((v) => v / wsum), perCard }
  }, [full, current, isDraw, view, held, line.board1, mask])

  // The next strip's percentages: the reach-weighted share of the actor's range taking each option.
  const shares = useMemo(() => {
    if (!current) return null
    const out: Record<string, number> = {}
    const w = current.mine.mix.length / current.mine.n
    const sum = (j: number) => {
      let s = 0
      for (let i = 0; i < current.mine.n; i++) s += current.mine.weight[i] * current.mine.mix[i * w + j]
      return s
    }
    if (current.mine.point.draw) {
      const pat = sum(current.mine.point.actions.indexOf('n'))
      out['0'] = pat
      out['1'] = 1 - pat
    } else {
      for (const o of nextOptions(line)) {
        const j = current.mine.point.actions.indexOf(o.key)
        if (j >= 0) out[o.key] = sum(j)
      }
    }
    return out
  }, [current, line])

  const node = current && current.mine.point.actions.includes('f') ? 'facing' : 'open'
  const toggleHeld = (c: MiniCard) => setHeld((h) => (h.includes(c) ? h.filter((x) => x !== c) : h.length < 3 ? [...h, c] : h))

  const equityOf = useCallback(
    (row: number, col: number) => {
      if (!current) return null
      let w = 0
      let inner = 0
      let outer = 0
      let scoop = 0
      for (let i = 0; i < current.mine.n; i++) {
        if (current.mine.row[i] !== row || current.mine.col[i] !== col) continue
        const wi = current.mine.weight[i]
        w += wi
        inner += wi * current.eq.inner[i]
        if (current.eq.outer) outer += wi * current.eq.outer[i]
        if (current.eq.scoop) scoop += wi * current.eq.scoop[i]
      }
      if (w === 0) return null
      return { inner: inner / w, outer: current.eq.outer ? outer / w : null, scoop: current.eq.scoop ? scoop / w : null }
    },
    [current],
  )

  const strategy = data?.index.strategy
  const title =
    decision.kind === 'bet' ? `Seat ${decision.player} to act, round ${decision.round}` : decision.kind === 'draw' ? `Seat ${decision.player} to draw` : decision.kind === 'board2' ? 'Board card 2 is owed' : decision.why === 'fold' ? 'The hand ended on a fold' : 'Showdown'
  const sub = current ? `${count(current.mine.n)} private states · ${current.spot.board.length === 1 ? 'one board card: the inner half can be scored now, the outer half cannot' : 'both board cards out: both halves scored exactly'}` : loadError ? '' : computing ? 'computing the exact range…' : data ? 'click a chip to rewind' : 'loading the strategy’s export…'

  return (
    <>
      <div className="block">
        <Panel n="01" k="the hand so far — click a chip to rewind, a board card to change it" wide>
          <MiniScore game={game} line={line} onLine={onLine} picking={picking} onPick={setPicking} shares={shares} />
        </Panel>
      </div>

      {tab === 'range' ? (
        <div className="block">
          <section className="panel wide rangewin" aria-label="the actor's range at this spot">
            <div className="titlebar">
              <span className="dots">
                <i />
                <i />
                <i />
              </span>
              <span className="k">02 · the range to act</span>
              <span className="no">02</span>
            </div>
            <div className="pane">
              <div className="spothead mini">
                <div className="who">
                  {title}
                  <small>{sub}</small>
                  {loadError && <em className="err">{loadError}</em>}
                </div>
                <div className="mixblock">
                  {current && whole && !isDraw && (
                    <>
                      <button type="button" className={`hbtn mixbtn ${opened === 'panes' ? 'on' : ''}`} onClick={() => setOpened(opened === 'panes' ? null : 'panes')} aria-expanded={opened === 'panes'} aria-label="the whole range's mix; opens one map per action">
                        <MixBar mix={whole.all} h={10} />
                      </button>
                      <div className="acts">
                        {(node === 'facing' ? (['f', 'c', 'p'] as const) : (['c', 'p'] as const)).map((k) => (
                          <button key={k} type="button" className={`hbtn act-${k} ${sort === k ? 'on' : ''}`} onClick={() => setSort(sort === k ? null : k)} aria-pressed={sort === k}>
                            <i />
                            {actionWord(k, node)} <b>{pct(whole.all[k])}</b>
                          </button>
                        ))}
                        <span className="hint">{sort ? `ledger sorted by ${actionWord(sort, node)}` : 'click an action to open the ledger'}</span>
                      </div>
                    </>
                  )}
                  {current && isDraw && shares && (
                    <span className="mixkey">
                      <span className="c">
                        <i />
                        stand pat <b>{pct0(shares['0'] ?? 0)}</b>
                      </span>
                      <span className="p">
                        <i />
                        throw one <b>{pct0(shares['1'] ?? 0)}</b>
                      </span>
                      <span className="dimmer">which card: the board below, by the hand's canonical order</span>
                    </span>
                  )}
                </div>
                <div className="holdblock">
                  <button type="button" className={`hbtn cardsbtn ${opened === 'deck' ? 'on' : ''}`} onClick={() => setOpened(opened === 'deck' ? null : 'deck')} aria-expanded={opened === 'deck'}>
                    cards
                  </button>
                  <span className="holding" aria-label="the cards you have placed">
                    {Array.from({ length: 3 }, (_, i) =>
                      held[i] !== undefined ? (
                        <button key={held[i]} type="button" className="hbtn heldcard" onClick={() => toggleHeld(held[i])} aria-label="remove the card">
                          <CardFace game={game} card={held[i]} />
                        </button>
                      ) : (
                        <CardFace key={`g${i}`} game={game} ghost />
                      ),
                    )}
                  </span>
                  <span className="count">
                    {view && current ? (
                      <>
                        <b>{pct(view.all.share)}</b> of the range
                        <br />
                        {mask ? `${count(view.all.n)} of ${count(current.mine.n)} states in the filter` : `${count(current.mine.n)} states · exact`}
                      </>
                    ) : (
                      <>
                        <b>—</b>
                      </>
                    )}
                  </span>
                </div>
              </div>

              {opened === 'panes' && view && (
                <Sheet title="one pane per action" onClose={() => setOpened(null)}>
                  <Panes game={game} view={view} node={node} />
                </Sheet>
              )}
              {opened === 'deck' && deck && (
                <Sheet title="the deck · tap cards to hold them" onClose={() => setOpened(null)}>
                  <Deck game={game} board={line.board2 === null ? [line.board1] : [line.board1, line.board2]} held={held} potLift={deck.potLift} onToggle={toggleHeld} />
                </Sheet>
              )}
              {full && view && current && view.all.n > 0 && view.all.share === 0 && (
                <div className="handanswer">
                  <span className="lbl">your hand</span>
                  <span className="nums">
                    the strategy never brings this hand here: at an earlier decision it took another action with probability one, so its reach is zero
                  </span>
                </div>
              )}
              {full && view && current && !isDraw && view.all.n > 0 && view.all.share > 0 && (
                <div className="handanswer">
                  <span className="lbl">your hand · {node === 'facing' ? 'fold / call / pot' : 'check / pot'}</span>
                  <MixBar mix={view.all} h={10} />
                  <span className="nums">
                    {node === 'facing' && <>fold <b>{pct0(view.all.f)}</b> · </>}
                    {actionWord('c', node)} <b>{pct0(view.all.c)}</b> · pot <b>{pct0(view.all.p)}</b>
                    {handRow >= 0 && table && <> · inner equity <b>{pct0(table.eqI[handRow])}</b>{table.eqO && <> · outer <b>{pct0(table.eqO[handRow])}</b></>}</>}
                  </span>
                </div>
              )}

              {view && table && current ? (
                <div className="body">
                  <div className="main">
                    {isDraw ? (
                      <DrawBoard game={game} options={THROW_OPTIONS} table={draws ?? []} hand={handThrow} fullHint="place three cards (the cards button) to see which of them is thrown" />
                    ) : (
                      <>
                        <Grid game={game} view={view} node={node} rows={rows} cols={cols} pin={pin && pin.row >= 0 ? pin : null} onRows={setRows} onCols={setCols} onPin={setPin} />
                        {(pin || sort) && <Ledger game={game} view={view} node={node} sort={sort} pin={pin && pin.row >= 0 ? pin : null} onPin={setPin} examplesFor={examplesFor} />}
                      </>
                    )}
                  </div>
                  {!isDraw && (
                    <aside className="side">
                      <Ribbon bins={ribbonBins} rkey={table.eqO ? rkey : rkey === 'outer' || rkey === 'scoop' ? 'inner' : rkey} onKey={setRkey} percentile={handPct} all={view.all} />
                      {plane ? <Plane cells={plane} bins={PLANE_BINS} hand={handPoint} /> : <p className="legend-line">the equity plane needs both board cards: with one out, only the inner half can be scored</p>}
                    </aside>
                  )}
                </div>
              ) : (
                <p className="legend-line" style={{ marginTop: 18 }}>
                  {spot === null ? 'no decision stands here — rewind with a chip, or pick board card 2' : computing ? 'computing…' : loadError ?? 'loading…'}
                </p>
              )}
            </div>
            <div className="statusbar">
              <span>
                {strategy ? (
                  <>
                    rung-3 <b>{strategy.rule.toUpperCase()}</b> · {count(strategy.iteration)} iterations × {strategy.workers} workers · every number is the frozen strategy’s
                  </>
                ) : (
                  'loading the strategy’s export…'
                )}
              </span>
              <span>{current ? `${count(current.mine.n)} states enumerated exactly · reach-weighted` : ''}</span>
            </div>
          </section>
        </div>
      ) : (
        <div className="block">
          <Panel n="02" k="range vs range — the other seat's range beside the actor's, with exact showdown equities" wide>
            {whole && theirsView && current ? (
              <div className="versus">
                <div className="vs">
                  <div className="side">
                    <div className="lbl">
                      <span>seat {1 - current.spot.player} · their range here</span>
                      <span className="dimmer">{count(current.theirs.n)} states</span>
                    </div>
                    <Thumb game={game} view={theirsView} weight={(c) => c.share} plain />
                    <p className="legend-line">ink = how much of their range is here, given what they have done so far{held.length ? ' and the cards you hold' : ''}</p>
                  </div>
                  <div className="mid">
                    <div className="vsn">vs</div>
                    <div className="eq">
                      <span className="dimmer">{hover ? `${game.rows[hover.row].label} × ${game.cols[hover.col].label}` : 'hover a cell on the actor’s side'}</span>
                      {(() => {
                        const e = hover ? equityOf(hover.row, hover.col) : null
                        return (
                          <>
                            <b>{e ? pct0(e.outer === null ? e.inner : (e.inner + e.outer) / 2) : '—'}</b>
                            <span className="dimmer">{current.spot.board.length === 2 ? 'showdown equity vs their range' : 'inner-half equity vs their range'}</span>
                            <div className="two">
                              <span>
                                inner <b>{e ? pct0(e.inner) : '—'}</b>
                              </span>
                              <span>
                                outer <b>{e && e.outer !== null ? pct0(e.outer) : '—'}</b>
                              </span>
                            </div>
                            <div className="two">
                              <span>
                                scoop <b>{e && e.scoop !== null ? pct0(e.scoop) : '—'}</b>
                              </span>
                              <span />
                            </div>
                          </>
                        )
                      })()}
                    </div>
                    <span className="dimmer small">{current.spot.board.length === 2 ? 'exact: both halves scored, pairs with a shared card skipped' : 'one board card out: the outer half is not yet scorable'}</span>
                  </div>
                  <div className="side">
                    <div className="lbl">
                      <span>seat {current.spot.player} · to act · whole range</span>
                      <span className="dimmer">colour = its mix</span>
                    </div>
                    <Thumb game={game} view={whole} weight={(c) => c.share} hover={hover} onHover={setHover} />
                    <p className="legend-line">ink = share of the range · colour = what the region mostly does now</p>
                  </div>
                </div>
              </div>
            ) : (
              <p className="legend-line">{spot === null ? 'no decision stands here — rewind with a chip, or pick board card 2' : 'computing…'}</p>
            )}
          </Panel>
        </div>
      )}

      <p className="foot">
        The strategy is rung 3’s frozen LCFR average, exported once per public decision point (141 of them) as the mix of
        every private key — 6.2 million rows. The browser rebuilds the ranges exactly: it enumerates every hole (and
        discard) the actor could hold given the board, relabels suits jointly the way the solver’s key does, reads each
        state’s mix, and weights the state by the product of the actor’s own action probabilities along the line. Nothing
        here is sampled or illustrative; the one rounding is the export’s, where each probability is a byte in 1/250ths,
        so an action the strategy takes under 0.2% of the time reads as never.
      </p>
    </>
  )
}

/** The other seat's range as a table with no mix: only its weights, regions and equities matter here. */
function theirTable(r: Ranges): RangeTable {
  const t = r.theirs
  const cards = new Uint8Array(t.n * 3)
  for (let i = 0; i < t.n; i++) for (let k = 0; k < 3; k++) cards[i * 3 + k] = t.cards[i * 4 + k]
  const zero = new Float32Array(t.n)
  return {
    n: t.n,
    rows: MINI_INNER_ROWS.length,
    cols: r.spot.board.length === 1 ? MINI_R1_COLS.length : MINI_R2_COLS.length,
    handSize: 3,
    deckSize: 15,
    cards,
    weight: t.weight,
    row: t.row,
    col: t.col,
    f: zero,
    c: zero,
    p: zero,
    eq: zero,
    eqI: zero,
    eqO: null,
  }
}

export type { Cell, GridView }
