import { useState } from 'react'
import type { Engine } from '../sim/engine'
import { ACTIONS } from '../sim/game'
import { averageStrategy, strategy } from '../sim/ledger'
import { ACTION_COLORS, fmt } from './format'

interface BarsProps {
  values: readonly number[]
  /** 'signed' renders a zero baseline with negative bars muted below it. */
  kind: 'signed' | 'share'
  thirdTick?: boolean
  digits?: number
}

function Bars({ values, kind, thirdTick, digits = 3 }: BarsProps) {
  const H = 132
  if (kind === 'share') {
    const zeroY = H
    return (
      <div className="bars" style={{ height: H }}>
        {thirdTick && (
          <div className="third-line" style={{ top: H * (1 - 1 / 3) }}>
            <span className="third-label">⅓</span>
          </div>
        )}
        <div className="zero-line" style={{ top: zeroY - 1 }} />
        {values.map((v, i) => (
          <div className="bar-slot" key={i}>
            <div className="bar-track">
              <span
                className="bar-value"
                style={{
                  position: 'absolute',
                  top: Math.min(H * (1 - v), H - 6) - 18,
                  left: 0,
                  right: 0,
                  textAlign: 'center',
                }}
              >
                {fmt(v, digits)}
              </span>
              <div
                className="bar-fill"
                style={{ top: H * (1 - v), height: H * v, background: ACTION_COLORS[i] }}
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
        const h = (Math.abs(v) / maxAbs) * (H / 2 - 20)
        const pos = v >= 0
        return (
          <div className="bar-slot" key={i}>
            <div className="bar-track">
              <span
                className="bar-value"
                style={{
                  position: 'absolute',
                  top: pos ? zeroY - h - 18 : zeroY + h + 3,
                  left: 0,
                  right: 0,
                  textAlign: 'center',
                }}
              >
                {fmt(v, digits)}
              </span>
              <div
                className={`bar-fill${pos ? '' : ' neg'}`}
                style={{
                  top: pos ? zeroY - h : zeroY + 1,
                  height: Math.max(h, 1),
                  background: pos ? ACTION_COLORS[i] : 'var(--muted)',
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
    <div style={{ display: 'flex', justifyContent: 'space-around', padding: '2px 4px 0' }}>
      {ACTIONS.map((name, i) => (
        <span className="bar-name" key={name}>
          <span className="dot" style={{ width: 8, height: 8, borderRadius: '50%', background: ACTION_COLORS[i] }} />
          {name}
        </span>
      ))}
    </div>
  )
}

/**
 * Hero panel: the full pipeline of one brain — cumulative regrets R →
 * current strategy σ → average strategy S/n — animated on every update.
 */
export function LedgerPanel({ engine }: { engine: Engine }) {
  const [player, setPlayer] = useState(0)
  const selfPlay = engine.config.mode === 'self-play'
  const p = selfPlay ? player : 0
  const L = engine.ledgers[p]
  const hasNegative = L.R.some((r) => r < 0)

  return (
    <section className="panel hero" aria-label="Regret-matching ledger">
      {selfPlay && (
        <div className="player-toggle" role="tablist" aria-label="Displayed player">
          {[0, 1].map((i) => (
            <button key={i} role="tab" aria-selected={player === i} onClick={() => setPlayer(i)}>
              player {i}
            </button>
          ))}
        </div>
      )}
      <h2>The ledger — one brain, updated every round</h2>
      <p className="sub">
        Positive regret ("I wish I had played that more") drives play; the average of everything
        played so far is what converges to Nash.
      </p>
      <div className="ledger-flow">
        <div className="ledger-stage">
          <h3>Cumulative regret R</h3>
          <p className="stage-sub">signed; only the part above zero matters</p>
          <Bars values={L.R} kind="signed" digits={1} />
          <BarNames />
        </div>
        <div className="ledger-arrow" aria-hidden>
          →
        </div>
        <div className="ledger-stage">
          <h3>Current strategy σ = R⁺ / ΣR⁺</h3>
          <p className="stage-sub">cycles forever, never settles</p>
          <Bars values={strategy(L)} kind="share" />
          <BarNames />
        </div>
        <div className="ledger-arrow" aria-hidden>
          →
        </div>
        <div className="ledger-stage">
          <h3>Average strategy S / n</h3>
          <p className="stage-sub">converges to (⅓, ⅓, ⅓)</p>
          <Bars values={averageStrategy(L)} kind="share" thirdTick />
          <BarNames />
        </div>
      </div>
      {hasNegative && (
        <p className="floor-note">
          Grayed bars below zero are clipped by the floor: negative regret never drives play — σ is
          built from R⁺ = max(R, 0) only.
        </p>
      )}
    </section>
  )
}
