/**
 * The grid's two filters, in the tool's left column: keep whole inner rows,
 * keep whole outer columns. Each is a list of the axis's regions with the
 * share of the whole range each holds, so the filter is also the axis's
 * marginal. More than one region can be kept, so each is a checkbox; "all"
 * clears the filter, and keeping every region is the same as keeping none
 * (the filter goes back to null rather than holding a set that excludes
 * nothing). The shares are the unfiltered range's, so they do not move as
 * you narrow.
 */

import type { GridView } from '../engine/aggregate'
import { pct } from './bars'
import type { GameUI } from './game'
import type { Region } from '../engine/regions'
import { Section } from './Tool'

type Keep = ReadonlySet<number> | null

/** `set` with `i` toggled; null when it would keep nothing or everything. */
export function toggled(set: Keep, i: number, size: number): Keep {
  const next = new Set(set ?? [])
  if (next.has(i)) next.delete(i)
  else next.add(i)
  return next.size === 0 || next.size === size ? null : next
}

function Axis({ title, regions, shares, keep, onKeep }: { title: string; regions: readonly Region[]; shares: number[]; keep: Keep; onKeep: (k: Keep) => void }) {
  return (
    <Section title={title}>
      <div className="radios rx-filter">
        <label>
          <input type="checkbox" checked={keep === null} onChange={() => onKeep(null)} />
          <span className="nm">all</span>
          <span className="v">100%</span>
        </label>
        {regions.map((r, i) => (
          <label key={r.label} title={r.long}>
            <input type="checkbox" checked={keep?.has(i) ?? false} onChange={() => onKeep(toggled(keep, i, regions.length))} />
            <span className="nm">{r.label}</span>
            <span className="v">{pct(shares[i])}</span>
          </label>
        ))}
      </div>
    </Section>
  )
}

export function Filters({ game, whole, rows, cols, onRows, onCols }: { game: GameUI; whole: GridView; rows: Keep; cols: Keep; onRows: (k: Keep) => void; onCols: (k: Keep) => void }) {
  const inHand = game.handSize === 3 ? 'three' : 'five'
  return (
    <>
      <Axis title={`Inner · the ${inHand} in hand`} regions={game.rows} shares={whole.rows.map((r) => r.share)} keep={rows} onKeep={onRows} />
      <Axis title="Outer · two with the board" regions={game.cols} shares={whole.cols.map((c) => c.share)} keep={cols} onKeep={onCols} />
    </>
  )
}
