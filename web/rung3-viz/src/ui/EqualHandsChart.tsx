import { useState } from 'react'

import { BEST, lcfrAt } from '../deep'
import { perHundred } from '../research'
import { fmtHands, hoursAt, logScale } from './scales'

/**
 * Two clocks on one ruler. The ruler is hands per seat on a log axis, with
 * a notch at each of the best arm's six graded marks. Above it reads Deep
 * CFR's clock — the iteration t, and under it the hours of training that
 * iteration had cost — and below it the table's clock, the lockstep
 * iteration 1,000 t that had seen the same hands. Every notch lines up
 * through all three rows because hands per seat is the axis both are read
 * on; that alignment, not any number, is what the figure says. The bottom
 * row is the table's own grade at each notch, the denominator the price in
 * the next figure divides by, as a dot whose height is the grade on a log
 * scale, falling left to right. Hover or arrow a notch and the readout
 * under the figure states that column's row in words.
 */

const W = 760
const H = 252
const M = { l: 248, r: 64 }
const ROW = { deep: 36, hours: 52, ruler: 108, hands: 128, table: 166, gradeTop: 196, gradeBottom: 240 }

const x = logScale(1e5, 5e6, M.l, W - M.r)
const yGrade = logScale(20, 400, ROW.gradeBottom, ROW.gradeTop)

export function EqualHandsChart() {
  const marks = BEST.marks
  const [at, setAt] = useState(marks.length - 1)
  const mark = marks[at]

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setAt((i) => Math.max(0, i - 1))
    if (e.key === 'ArrowRight') setAt((i) => Math.min(marks.length - 1, i + 1))
  }

  const rowLabel = (y: number, text: string, sub?: string) => (
    <>
      <text x={M.l - 36} y={y} textAnchor="end" fontSize="10" letterSpacing="0.1em" fill="var(--panel-dim)">
        {text.toUpperCase()}
      </text>
      {sub && (
        <text x={M.l - 36} y={y + 13} textAnchor="end" fontSize="10" fill="var(--iv-45)">
          {sub}
        </text>
      )}
    </>
  )

  return (
    <>
      <svg
        className="chart equal"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Deep CFR's iteration and the table's lockstep iteration on one axis of hands per seat, with the table's grade at each mark"
        tabIndex={0}
        onKeyDown={onKey}
      >
        {rowLabel(ROW.deep + 4, 'Deep CFR', 'iteration, hours trained')}
        {rowLabel(ROW.ruler + 4, 'hands per seat', 'the shared axis')}
        {rowLabel(ROW.table + 4, 'the table', 'lockstep iteration')}
        {rowLabel(ROW.gradeTop + 14, 'LCFR grades', 'chips per 100 hands')}

        {/* the ruler */}
        <line x1={M.l} x2={W - M.r} y1={ROW.ruler} y2={ROW.ruler} stroke="var(--panel-dim)" />

        {marks.map((m, i) => {
          const cx = x(m.hands)
          const on = i === at
          const ink = on ? 'var(--panel-mark)' : 'var(--iv-80)'
          const grade = perHundred(lcfrAt(m.hands))
          return (
            <g key={m.t}>
              {/* the column: one hairline through every row, because the notch is the same notch */}
              <line x1={cx} x2={cx} y1={ROW.deep - 14} y2={ROW.gradeBottom} stroke={on ? 'var(--iv-30)' : 'var(--panel-hair)'} />
              {/* Deep CFR's clock */}
              <text x={cx} y={ROW.deep + 4} textAnchor="middle" fontSize="13" fontWeight={on ? 600 : 500} fill={ink}>
                {m.t}
              </text>
              <text x={cx} y={ROW.hours + 4} textAnchor="middle" fontSize="10" fill="var(--iv-45)">
                {hoursAt(m.wallClockS)}
              </text>
              {/* the notch and its hands */}
              <line x1={cx} x2={cx} y1={ROW.ruler - 7} y2={ROW.ruler + 7} stroke={ink} strokeWidth={on ? 2 : 1.25} />
              <text x={cx} y={ROW.hands + 4} textAnchor="middle" fontSize="11" fill={ink}>
                {fmtHands(m.hands)}
              </text>
              {/* the table's clock */}
              <text x={cx} y={ROW.table + 4} textAnchor="middle" fontSize="12" fontWeight={on ? 600 : 400} fill={ink}>
                {(m.t * 1000).toLocaleString()}
              </text>
              {/* the table's grade: a stem from the row's floor to a dot at the grade */}
              <line x1={cx} x2={cx} y1={ROW.gradeBottom} y2={yGrade(grade)} stroke="var(--table-ink)" strokeWidth={2} />
              <circle cx={cx} cy={yGrade(grade)} r={on ? 5 : 4} fill="var(--table-ink)" stroke="var(--field)" strokeWidth={2} />
              <text x={cx + 9} y={yGrade(grade) + 4} fontSize="11" fill={ink}>
                {grade.toFixed(1)}
              </text>
            </g>
          )
        })}
        <line x1={M.l} x2={W - M.r} y1={ROW.gradeBottom} y2={ROW.gradeBottom} stroke="var(--panel-hair)" />

        {marks.map((m, i) => {
          const cx = x(m.hands)
          const left = i === 0 ? M.l - 30 : (x(marks[i - 1].hands) + cx) / 2
          const right = i === marks.length - 1 ? W : (x(marks[i + 1].hands) + cx) / 2
          return <rect key={m.t} x={left} y={0} width={right - left} height={H} fill="transparent" onMouseEnter={() => setAt(i)} onFocus={() => setAt(i)} />
        })}
      </svg>
      <dl className="stat">
        <div>
          <dt>hands per seat</dt>
          <dd>{mark.hands.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Deep CFR iteration</dt>
          <dd>{mark.t}</dd>
        </div>
        <div>
          <dt>lockstep iteration</dt>
          <dd>{(mark.t * 1000).toLocaleString()}</dd>
        </div>
        <div>
          <dt style={{ color: 'var(--table-ink)' }}>LCFR grades</dt>
          <dd>{perHundred(lcfrAt(mark.hands)).toFixed(1)}</dd>
        </div>
        <div>
          <dt>trained for</dt>
          <dd>{hoursAt(mark.wallClockS)}</dd>
        </div>
      </dl>
    </>
  )
}
