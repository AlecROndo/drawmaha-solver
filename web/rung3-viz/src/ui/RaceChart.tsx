import { useState } from 'react'

import {
  CHECKPOINTS,
  KEEPS_NEGATIVE,
  RULES,
  RULE_NAME,
  UNIFORM_RANDOM,
  WORKERS,
  perHundred,
  series,
  type Rule,
} from '../research'
import { RULE_INK } from '../inks'
import { Panel } from './site'

/**
 * The race: exploitability against hands per seat, both axes log, one line
 * per regret rule. The two rules that keep negative regret (vanilla, LCFR)
 * pull away from the two that throw it away (CFR+, DCFR), which lie on top
 * of each other. Uniform random play is the dashed reference at the top,
 * and the dotted guide is 1/√N through vanilla's first point. Hover or arrow
 * a checkpoint to read the four numbers at that column.
 */

const W = 560
const H = 300
const M = { l: 54, r: 112, t: 26, b: 34 }

const X_MIN = 1e5
const X_MAX = 2e7
const logx = (hands: number) =>
  M.l + (Math.log10(hands / X_MIN) / Math.log10(X_MAX / X_MIN)) * (W - M.l - M.r)

/** Chips per hundred hands, log: 1 at the floor, 1000 at the ceiling. */
const Y_TOP = 1000
const Y_BOTTOM = 1
const logy = (perHundredHands: number) => {
  const clamped = Math.min(Math.max(perHundredHands, Y_BOTTOM), Y_TOP)
  return M.t + (Math.log10(Y_TOP / clamped) / Math.log10(Y_TOP / Y_BOTTOM)) * (H - M.t - M.b)
}

const X_TICKS = [1e5, 1e6, 1e7, 2e7]
const Y_TICKS = [1, 10, 100, 1000]

const hands = (n: number) => (n >= 1e6 ? `${n / 1e6}M` : `${n / 1e3}k`)

const path = (rule: Rule) =>
  series(rule)
    .map((p, i) => `${i ? 'L' : 'M'} ${logx(p.hands)} ${logy(perHundred(p.exploitability))}`)
    .join(' ')

export function RaceChart() {
  const [at, setAt] = useState(CHECKPOINTS.length - 1)
  const iteration = CHECKPOINTS[at]
  const column = RULES.map((rule) => ({ rule, point: series(rule)[at] }))
  const xAt = logx(iteration * WORKERS)

  // the 1/√N guide, anchored at vanilla's first point
  const first = series('vanilla')[0]
  const guide = (handsAt: number) => perHundred(first.exploitability) * Math.sqrt(first.hands / handsAt)

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setAt((i) => Math.max(0, i - 1))
    if (e.key === 'ArrowRight') setAt((i) => Math.min(CHECKPOINTS.length - 1, i + 1))
  }

  return (
    <Panel
      n="01"
      wide
      k="Fig. 1 · the race — hover or arrow a checkpoint"
      title="Keeping negative regret wins, by three to four times."
      say={
        <>
          Chips per hundred hands a perfect adversary wins against each rule’s linear average, by the exact
          best-response walk, against sampled hands per seat ({WORKERS} per iteration). Both axes log.{' '}
          <b>Seed 0, one run per rule.</b>
        </>
      }
      label="Exploitability of the four regret rules over the race"
    >
      <div className="legend">
        {RULES.map((rule) => (
          <span key={rule} className="item">
            <span className={`swatch ${rule}`} style={{ background: RULE_INK[rule] }} />
            {RULE_NAME[rule]}
            <small>{KEEPS_NEGATIVE[rule] ? ' · keeps negative regret' : ' · discards it'}</small>
          </span>
        ))}
      </div>
      <svg
        className="chart race"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Exploitability against hands per seat, four regret rules"
        tabIndex={0}
        onKeyDown={onKey}
      >
        {Y_TICKS.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={logy(v)} y2={logy(v)} stroke="var(--panel-hair)" />
            <text x={M.l - 8} y={logy(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--panel-dim)">
              {v}
            </text>
          </g>
        ))}
        {X_TICKS.map((v) => (
          <g key={v}>
            <line x1={logx(v)} x2={logx(v)} y1={M.t} y2={H - M.b} stroke="var(--panel-hair)" />
            <text x={logx(v)} y={H - 12} textAnchor="middle" fontSize="11" fill="var(--panel-dim)">
              {hands(v)}
            </text>
          </g>
        ))}
        <text x={W - M.r + 8} y={H - 12} fontSize="10" fill="var(--panel-dim)">
          hands / seat
        </text>
        <text x={4} y={M.t - 12} fontSize="10" fill="var(--panel-dim)">
          chips / 100 hands
        </text>

        {/* uniform random: the scale */}
        <line
          x1={M.l}
          x2={W - M.r}
          y1={logy(perHundred(UNIFORM_RANDOM.exploitability))}
          y2={logy(perHundred(UNIFORM_RANDOM.exploitability))}
          stroke="var(--panel-dim)"
          strokeDasharray="4 4"
        />
        <text
          x={W - M.r + 8}
          y={logy(perHundred(UNIFORM_RANDOM.exploitability)) + 4}
          fontSize="10.5"
          fill="var(--panel-dim)"
        >
          random {perHundred(UNIFORM_RANDOM.exploitability).toFixed(0)}
        </text>

        {/* the 1/√N guide */}
        <line
          x1={logx(first.hands)}
          y1={logy(guide(first.hands))}
          x2={logx(X_MAX)}
          y2={logy(guide(X_MAX))}
          stroke="var(--panel-dim)"
          strokeDasharray="1.5 4"
          strokeOpacity="0.8"
        />
        <text x={logx(2.5e6) + 6} y={logy(guide(2.5e6)) - 7} fontSize="10.5" fill="var(--panel-dim)">
          1/√N
        </text>

        {RULES.map((rule) => (
          <path
            key={rule}
            d={path(rule)}
            fill="none"
            stroke={RULE_INK[rule]}
            strokeWidth={rule === 'lcfr' ? 2.2 : 1.6}
            strokeDasharray={rule === 'dcfr' ? '6 4' : undefined}
          />
        ))}

        {/* the selected checkpoint: a playhead and a dot per rule */}
        <line x1={xAt} x2={xAt} y1={M.t} y2={H - M.b} stroke="var(--panel-mark)" strokeWidth={1} />
        {column.map(({ rule, point }) => (
          <circle
            key={rule}
            cx={xAt}
            cy={logy(perHundred(point.exploitability))}
            r={rule === 'lcfr' ? 4 : 3}
            fill={RULE_INK[rule]}
          />
        ))}
        {/* direct labels at the right edge for the two clusters */}
        <text x={logx(X_MAX) + 8} y={logy(perHundred(series('lcfr').at(-1)!.exploitability)) + 4} fontSize="11" fill={RULE_INK.lcfr}>
          LCFR {perHundred(series('lcfr').at(-1)!.exploitability).toFixed(1)}
        </text>
        <text x={logx(X_MAX) + 8} y={logy(perHundred(series('vanilla').at(-1)!.exploitability)) - 6} fontSize="11" fill={RULE_INK.vanilla}>
          vanilla {perHundred(series('vanilla').at(-1)!.exploitability).toFixed(1)}
        </text>
        <text x={logx(X_MAX) + 8} y={logy(perHundred(series('cfrplus').at(-1)!.exploitability)) + 4} fontSize="11" fill={RULE_INK.cfrplus}>
          CFR+ · DCFR {perHundred(series('cfrplus').at(-1)!.exploitability).toFixed(0)}
        </text>

        {/* hover targets: one column per checkpoint */}
        {CHECKPOINTS.map((it, i) => {
          const x = logx(it * WORKERS)
          const left = i === 0 ? M.l : (logx(CHECKPOINTS[i - 1] * WORKERS) + x) / 2
          const right = i === CHECKPOINTS.length - 1 ? W - M.r : (logx(CHECKPOINTS[i + 1] * WORKERS) + x) / 2
          return (
            <rect
              key={it}
              x={left}
              y={M.t}
              width={right - left}
              height={H - M.t - M.b}
              fill="transparent"
              onMouseEnter={() => setAt(i)}
              onFocus={() => setAt(i)}
            />
          )
        })}
      </svg>
      <dl className="stat">
        <div>
          <dt>hands per seat</dt>
          <dd>{(iteration * WORKERS).toLocaleString()}</dd>
        </div>
        {column.map(({ rule, point }) => (
          <div key={rule}>
            <dt style={{ color: RULE_INK[rule] }}>{RULE_NAME[rule]}</dt>
            <dd>{perHundred(point.exploitability).toFixed(1)}</dd>
          </div>
        ))}
        <div>
          <dt>P0 value (LCFR)</dt>
          <dd>{column[1].point.value_p0.toFixed(4)}</dd>
        </div>
      </dl>
    </Panel>
  )
}
