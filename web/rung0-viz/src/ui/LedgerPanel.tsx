import { useState } from 'react'
import type { Engine } from '../sim/engine'
import { ACTIONS, exploitability } from '../sim/game'
import { averageStrategy, strategy } from '../sim/ledger'
import { ACTION_COLORS, fmt, fmtIter } from './format'
import { Panel } from './site'

interface BarsProps {
  values: readonly number[]
  /** 'signed' renders a zero baseline with negative bars muted below it. */
  kind: 'signed' | 'share'
  thirdTick?: boolean
  digits?: number
}

/** Plot height, and the strip above it kept clear for the value text. */
const H = 150
const LABEL_ROOM = 22

function BarValue({ top, children }: { top: number; children: React.ReactNode }) {
  return (
    <span
      className="bar-value"
      style={{ position: 'absolute', top, left: 0, right: 0, textAlign: 'center', zIndex: 1 }}
    >
      {children}
    </span>
  )
}

function Bars({ values, kind, thirdTick, digits = 3 }: BarsProps) {
  if (kind === 'share') {
    // A share of 1 is a full-height bar, so the plot stops short of the top and
    // the number always has its own strip to sit in.
    const plot = H - LABEL_ROOM
    const top = (v: number) => LABEL_ROOM + plot * (1 - v)
    return (
      <div className="bars" style={{ height: H }}>
        {thirdTick && (
          <div className="third-line" style={{ top: top(1 / 3) }}>
            <span className="third-label">⅓</span>
          </div>
        )}
        <div className="zero-line" style={{ top: H - 1 }} />
        {values.map((v, i) => (
          <div className="bar-slot" key={i}>
            <div className="bar-track">
              <BarValue top={top(v) - 18}>{fmt(v, digits)}</BarValue>
              <div
                className="bar-fill"
                style={{ top: top(v), height: plot * v, background: ACTION_COLORS[i] }}
              />
            </div>
          </div>
        ))}
      </div>
    )
  }
  // signed bars around a centered zero baseline
  const maxAbs = Math.max(1, ...values.map((v) => Math.abs(v)))
  const zeroY = H / 2
  return (
    <div className="bars" style={{ height: H }}>
      <div className="zero-line" style={{ top: zeroY }} />
      {values.map((v, i) => {
        const h = (Math.abs(v) / maxAbs) * (zeroY - LABEL_ROOM)
        const pos = v >= 0
        return (
          <div className="bar-slot" key={i}>
            <div className="bar-track">
              <BarValue top={pos ? zeroY - h - 18 : zeroY + h + 4}>{fmt(v, digits)}</BarValue>
              <div
                className={`bar-fill${pos ? '' : ' neg'}`}
                style={{
                  top: pos ? zeroY - h : zeroY + 1,
                  height: Math.max(h, 1),
                  background: pos ? ACTION_COLORS[i] : 'var(--panel-dim)',
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

function BarNames() {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-around', padding: '0 4px' }}>
      {ACTIONS.map((name, i) => (
        <span className="bar-name" key={name}>
          <span className="dot" style={{ background: ACTION_COLORS[i] }} />
          {name}
        </span>
      ))}
    </div>
  )
}

function Stage({ label, values, kind, thirdTick, digits }: BarsProps & { label: string }) {
  return (
    <div className="ledger-stage">
      <span className="k">{label}</span>
      <Bars values={values} kind={kind} thirdTick={thirdTick} digits={digits} />
      <BarNames />
    </div>
  )
}

/**
 * The full pipeline of one brain — cumulative regrets R → current strategy σ →
 * average strategy S/n. It takes the widest panel on the page because it is
 * the thing the page is about; every other figure is commentary on it.
 */
export function LedgerPanel({ engine }: { engine: Engine }) {
  const [player, setPlayer] = useState(0)
  const selfPlay = engine.config.mode === 'self-play'
  const p = selfPlay ? player : 0
  const L = engine.ledgers[p]
  const avg = averageStrategy(L)

  return (
    <Panel
      n="01"
      wide
      k="Fig. 1 · the ledger"
      title={`The ledger after round ${fmtIter(engine.iteration)}`}
      say="R accumulates what each action would have earned over what the strategy expected. Positive regret becomes the next strategy; the average of every strategy played is the product."
      label="Regret-matching ledger"
    >
      <div className="ledger-head">
        {selfPlay && (
          <div className="mode-tabs" role="tablist" aria-label="Displayed player">
            {[0, 1].map((i) => (
              <button key={i} role="tab" aria-selected={player === i} onClick={() => setPlayer(i)}>
                player {i}
              </button>
            ))}
          </div>
        )}
        <span className="grow" />
        <span className="readout">
          exploitability of the average <b>{fmt(exploitability(avg), 4)}</b>
        </span>
      </div>

      <div className="ledger-flow">
        <Stage label="Cumulative regret R" values={L.R} kind="signed" digits={1} />
        <div className="ledger-arrow" aria-hidden>
          →
        </div>
        <Stage label="Current σ = R⁺ / ΣR⁺" values={strategy(L)} kind="share" digits={2} />
        <div className="ledger-arrow" aria-hidden>
          →
        </div>
        <Stage label="Average S / n" values={avg} kind="share" thirdTick />
      </div>

      <p className="say">
        Negative regret is clipped by the floor and never drives play — σ is built from
        R⁺ = max(R, 0) only, so any bar pushed below zero renders greyed.
      </p>
    </Panel>
  )
}
