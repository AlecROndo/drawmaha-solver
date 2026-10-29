/**
 * The pieces the tool's three columns are built from, shared by both games:
 * a titled section on a hairline, the "this spot" figures as hairline rows,
 * the selected region read out in full, and the slots of the hand being
 * built from the deck. Layout only; every number is passed in.
 */

import type { Cell } from '../engine/aggregate'
import type { Node } from '../engine/policy'
import { CardFace, actionWord, pct, pct0 } from './bars'
import type { GameUI } from './game'

/** A column's section: a small uppercase heading on a hairline, an optional aside at its right, then the body. */
export function Section({ title, aside, className, children }: { title: string; aside?: React.ReactNode; className?: string; children: React.ReactNode }) {
  return (
    <section className={['rx-sec', className ?? ''].filter(Boolean).join(' ')}>
      <h2 className="rx-h">
        <span>{title}</span>
        {aside && <span className="rx-h-aside">{aside}</span>}
      </h2>
      {children}
    </section>
  )
}

/** Name / value rows on hairlines. */
export function Stats({ rows }: { rows: readonly (readonly [React.ReactNode, React.ReactNode])[] }) {
  return (
    <dl className="rx-stats">
      {rows.map(([k, v], i) => (
        <div key={i}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

/**
 * The region the grid has pinned, read out: its name, its share, how many
 * private states (or sampled holdings) it holds, and its mix action by
 * action. `noun` names what `cell.n` counts. The hands in it are one level
 * down in the ledger, which the pin opens at the same region.
 */
export function Selected({
  name,
  cell,
  node,
  noun,
  onClear,
}: {
  name: React.ReactNode
  cell: Cell
  node: Node
  noun: string
  onClear: () => void
}) {
  const keys = node === 'facing' ? (['f', 'c', 'p'] as const) : (['c', 'p'] as const)
  return (
    <Section
      title="Selected region"
      className="rx-selected"
      aside={
        <button type="button" className="hbtn rx-clear" onClick={onClear}>
          clear ×
        </button>
      }
    >
      <p className="rx-sel-name">{name}</p>
      <Stats
        rows={[
          ['Share of the range', pct(cell.share)],
          [noun, cell.n.toLocaleString('en-US')],
          ...keys.map(
            (k) =>
              [
                <span key={k} className={`rx-sw ${k}`}>
                  <i />
                  {actionWord(k, node)}
                </span>,
                pct0(cell[k]),
              ] as const,
          ),
        ]}
      />
    </Section>
  )
}

/** The hand being built: one slot per card, a placed card removable by a press. */
export function HeldSlots({ game, held, onUnhold }: { game: GameUI; held: readonly number[]; onUnhold: (c: number) => void }) {
  return (
    <span className="holding" aria-label="the cards you have placed">
      {Array.from({ length: game.handSize }, (_, i) =>
        held[i] !== undefined ? (
          <button key={held[i]} type="button" className="hbtn heldcard" onClick={() => onUnhold(held[i])} aria-label="remove the card">
            <CardFace game={game} card={held[i]} />
          </button>
        ) : (
          <CardFace key={`g${i}`} game={game} ghost />
        ),
      )}
    </span>
  )
}
