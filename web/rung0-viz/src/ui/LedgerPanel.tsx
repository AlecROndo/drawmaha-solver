import { useState } from 'react'
import type { Engine } from '../sim/engine'
import { ACTIONS, exploitability } from '../sim/game'
import { averageStrategy, strategy } from '../sim/ledger'
import { AVG, CUR, INK, fmt, fmtIter } from './format'
import { Figure } from './site'

interface BarsProps {
  values: readonly number[]
  /** 'signed' renders a zero baseline with negative bars muted below it. */
  kind: 'signed' | 'share'
  /** the colour of the estimate the stage shows: ink for R, grey for σ, the accent for S/n */
  color: string
  thirdTick?: boolean
  digits?: number
}

/** Plot height, and the strip above it kept clear for the value text. */
const H = 150
const LABEL_ROOM = 22

function BarValue({ top, children }: { top: number; children: React.ReactNode }) {
  return (
    <span className="bar-value" style={{ top }}>
      {children}
    </span>
  )
}

function Bars({ values, kind, color, thirdTick, digits = 3 }: BarsProps) {
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
              <div className="bar-fill" style={{ top: top(v), height: plot * v, background: color }} />
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
                  background: pos ? color : undefined,
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
    <div className="bar-names">
      {ACTIONS.map((name) => (
        <span className="bar-name" key={name}>
          {name}
        </span>
      ))}
    </div>
  )
}

function Stage({ label, ...bars }: BarsProps & { label: React.ReactNode }) {
  return (
    <div className="ledger-stage">
      <span className="stage-k">{label}</span>
      <Bars {...bars} />
      <BarNames />
    </div>
  )
}

/**
 * The full pipeline of one learner — cumulative regrets R → current strategy
 * σ → average strategy S/n. It takes the widest figure on the page because it
 * is the thing the page is about; every other figure is commentary on it.
 */
export function LedgerPanel({ engine }: { engine: Engine }) {
  const [player, setPlayer] = useState(0)
  const selfPlay = engine.config.mode === 'self-play'
  const p = selfPlay ? player : 0
  const L = engine.ledgers[p]
  const avg = averageStrategy(L)

  return (
    <Figure
      n="Figure 1"
      wide
      title="Positive regret sets the next throw; the average of every throw is the answer"
      sub={
        <>
          The ledger of {selfPlay ? `player ${p}` : 'the learner'} after round <b>{fmtIter(engine.iteration)}</b>.
          Exploitability of the average: <b>{fmt(exploitability(avg), 4)}</b> chips per round.
        </>
      }
      caption="σ is built from R⁺ = max(R, 0) only, so a regret below zero never drives play; its bar is drawn hollow."
      label="Regret-matching ledger"
    >
      {selfPlay && (
        <div className="ledger-head">
          <div className="seg mode-tabs" role="group" aria-label="Player shown">
            {[0, 1].map((i) => (
              <button key={i} type="button" aria-pressed={player === i} onClick={() => setPlayer(i)}>
                Player {i}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="ledger-flow">
        <Stage label="Cumulative regret R" values={L.R} kind="signed" color={INK} digits={1} />
        <div className="ledger-arrow" aria-hidden>
          →
        </div>
        <Stage
          label={
            <>
              <span className="sw" style={{ background: CUR }} />
              Current σ = R⁺ / ΣR⁺
            </>
          }
          values={strategy(L)}
          kind="share"
          color={CUR}
          digits={2}
        />
        <div className="ledger-arrow" aria-hidden>
          →
        </div>
        <Stage
          label={
            <>
              <span className="sw" style={{ background: AVG }} />
              Average S / n
            </>
          }
          values={avg}
          kind="share"
          color={AVG}
          thirdTick
        />
      </div>
    </Figure>
  )
}
