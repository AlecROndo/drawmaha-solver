import { useEffect, useState } from 'react'
import type { Mode } from '../sim/engine'
import type { Action, Vec3 } from '../sim/game'
import { ACTIONS } from '../sim/game'
import { fmtSpeed } from './format'
import type { Simulation } from './useSimulation'

const MODES: { key: Mode; label: string }[] = [
  { key: 'self-play', label: 'Self-play' },
  { key: 'vs-fixed', label: 'Vs a fixed mix' },
  { key: 'vs-you', label: 'Vs you' },
]

const SPEED_MIN = 10
const SPEED_MAX = 50_000

function sliderToSpeed(v: number): number {
  // log scale 10 → 50k over slider 0..100
  const s = SPEED_MIN * (SPEED_MAX / SPEED_MIN) ** (v / 100)
  return Math.round(s)
}

function speedToSlider(s: number): number {
  return (Math.log(s / SPEED_MIN) / Math.log(SPEED_MAX / SPEED_MIN)) * 100
}

export function Controls({ sim }: { sim: Simulation }) {
  const auto = sim.mode !== 'vs-you'
  const [seedText, setSeedText] = useState(String(sim.seed))

  useEffect(() => setSeedText(String(sim.seed)), [sim.seed])

  const commitSeed = () => {
    const s = Number.parseInt(seedText, 10)
    if (Number.isFinite(s) && s !== sim.seed) sim.setSeed(s)
    else setSeedText(String(sim.seed))
  }

  // space = run/pause, → = step ×1 (never while typing in a field)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
      if (!auto) return
      if (e.code === 'Space') {
        e.preventDefault()
        sim.setRunning(!sim.running)
      } else if (e.code === 'ArrowRight') {
        e.preventDefault()
        sim.stepN(1)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [auto, sim])

  // Raw slider weights; the opponent distribution is their normalization, so
  // dragging one slider never fights the user by rescaling itself.
  const [weights, setWeights] = useState<Vec3>([50, 25, 25])

  const setWeight = (i: number, w: number) => {
    const next = [...weights] as Vec3
    next[i] = w
    setWeights(next)
    const sum = next[0] + next[1] + next[2]
    if (sum > 0) sim.setFixedDist([next[0] / sum, next[1] / sum, next[2] / sum] as Vec3)
  }

  return (
    <div className="console" role="group" aria-label="Run controls">
      <div className="console-row">
        {auto ? (
          <>
            <button
              type="button"
              className="btn primary run"
              onClick={() => sim.setRunning(!sim.running)}
            >
              {sim.running ? 'Pause' : 'Run'}
            </button>
            <button type="button" className="btn" onClick={() => sim.stepN(1)}>
              Step ×1
            </button>
            <button type="button" className="btn" onClick={() => sim.stepN(100)}>
              ×100
            </button>
            <button type="button" className="btn" onClick={() => sim.stepN(10_000)}>
              ×10k
            </button>
          </>
        ) : (
          <>
            <span className="field-k">Throw</span>
            {ACTIONS.map((name, i) => (
              <button type="button" className="btn" key={name} onClick={() => sim.playUserAction(i as Action)}>
                {name[0].toUpperCase() + name.slice(1)}
              </button>
            ))}
          </>
        )}

        <span className="grow" />

        {auto && (
          <label className="field">
            <span className="field-k">Speed</span>
            <input
              type="range"
              min={0}
              max={100}
              value={speedToSlider(sim.speed)}
              onChange={(e) => sim.setSpeed(sliderToSpeed(Number(e.target.value)))}
            />
            <span className="readout">{fmtSpeed(sim.speed)} rounds / s</span>
          </label>
        )}

        <label className="field">
          <span className="field-k">Seed</span>
          <input
            type="text"
            inputMode="numeric"
            value={seedText}
            onChange={(e) => setSeedText(e.target.value)}
            onBlur={commitSeed}
            onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
            aria-label="RNG seed"
          />
        </label>

        <button type="button" className="btn quiet" onClick={sim.reset}>
          Reset
        </button>
      </div>

      {/* No round counter here: Figure 1's sub-title states the round. */}

      {sim.mode === 'vs-fixed' && (
        <div className="console-row dist" aria-label="Opponent distribution">
          <span className="field-k">The opponent throws</span>
          {ACTIONS.map((name, i) => (
            <label className="field" key={name}>
              {name}
              <input
                type="range"
                min={0}
                max={100}
                value={weights[i]}
                onChange={(e) => setWeight(i, Number(e.target.value))}
              />
              <span className="pct">{Math.round(sim.fixedDist[i] * 100)}%</span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Three readings of one page — self-play, a fixed opponent, you — as the
 * theme's segmented control in the page head. Changing it rebuilds the run.
 */
export function ModeSwitch({ sim }: { sim: Simulation }) {
  return (
    <div className="seg mode-tabs" role="group" aria-label="Opponent">
      {MODES.map((m) => (
        <button key={m.key} type="button" aria-pressed={sim.mode === m.key} onClick={() => sim.setMode(m.key)}>
          {m.label}
        </button>
      ))}
    </div>
  )
}
