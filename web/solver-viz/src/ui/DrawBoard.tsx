/**
 * The draw board: at the draw the action is one of thirty-two subsets of
 * your five cards, and no fold/call/pot colouring survives that. So the
 * draw gets its own board — rows are the inner classes, columns are how
 * many you throw, a cell is how often — and under it the held hand with a
 * throw probability under each card, which is the only honest per-hand
 * answer. Replaces the grid when the cursor is on the draw.
 */

import type { Card } from '../engine/cards'
import { INNER_ROWS } from '../engine/regions'
import { CardFace, byRankDesc, pct, pct0 } from './bars'

export function DrawBoard({
  table,
  hand,
}: {
  table: { share: number; counts: number[] }[]
  /** the held hand's throw answer, when five cards are placed */
  hand: { cards: readonly Card[]; counts: number[]; perCard: number[] } | null
}) {
  const overall = [0, 1, 2, 3, 4, 5].map((k) => table.reduce((s, r) => s + r.share * r.counts[k], 0))
  const total = table.reduce((s, r) => s + r.share, 0) || 1
  return (
    <div className="drawblock">
      <div className="drawb">
        <div className="h left">inner class · share</div>
        {[0, 1, 2, 3, 4, 5].map((k) => (
          <div key={k} className="h">
            {k === 0 ? 'stand pat' : `throw ${k}`}
          </div>
        ))}
        <div className="h">mode</div>
        {INNER_ROWS.map((r, i) => {
          const row = table[i]
          if (!row || row.share === 0) return null
          const top = row.counts.indexOf(Math.max(...row.counts))
          return (
            <DrawRow key={r.label} label={r.label} share={row.share} counts={row.counts} top={top} />
          )
        })}
        <div className="rh all">
          <span>whole range</span>
          <small>{pct(total)}</small>
        </div>
        {overall.map((v, k) => (
          <div key={k} className="cell" style={{ '--o': (v / total) * 0.95 } as React.CSSProperties}>
            <i />
            <span>{v / total > 0.004 ? pct0(v / total) : '·'}</span>
          </div>
        ))}
        <div className="mode" />
      </div>
      {hand ? (
        <div className="throwhand">
          <div className="lbl">your hand · how often each card is thrown</div>
          <div className="slots">
            {byRankDesc(hand.cards).map((c) => {
              const idx = hand.cards.indexOf(c)
              const p = hand.perCard[idx]
              return (
                <div key={c} className={`slot ${p < 0.5 ? 'keep' : ''}`}>
                  <CardFace card={c} size="lg" />
                  <span className="t">
                    <i style={{ height: `${p * 100}%` }} />
                  </span>
                  <span>{pct0(p)}</span>
                </div>
              )
            })}
            <div className="say">
              {hand.counts.map((v, k) => (v > 0.004 ? <span key={k}>{k === 0 ? 'stand pat' : `throw ${k}`} <b>{pct0(v)}</b></span> : null))}
            </div>
          </div>
        </div>
      ) : (
        <p className="legend-line">place five cards (the cards button) to see which of them are thrown</p>
      )}
    </div>
  )
}

function DrawRow({ label, share, counts, top }: { label: string; share: number; counts: number[]; top: number }) {
  return (
    <>
      <div className="rh">
        <span>{label}</span>
        <small>{pct(share)}</small>
      </div>
      {counts.map((v, k) => (
        <div key={k} className={`cell ${k === top ? 'top' : ''}`} style={{ '--o': v * 0.95 } as React.CSSProperties}>
          <i />
          <span>{v > 0.004 ? pct0(v) : '·'}</span>
        </div>
      ))}
      <div className="mode">{top === 0 ? 'pat' : `throw ${top}`}</div>
    </>
  )
}
