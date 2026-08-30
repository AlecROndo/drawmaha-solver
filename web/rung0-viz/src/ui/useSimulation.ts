import { useCallback, useEffect, useRef, useState } from 'react'
import { Engine, type Mode } from '../sim/engine'
import type { Action, Vec3 } from '../sim/game'

export interface Simulation {
  engine: Engine
  mode: Mode
  seed: number
  fixedDist: Vec3
  running: boolean
  speed: number
  /** Bumps every time engine state changed — panels re-read the engine when it does. */
  version: number
  setMode: (m: Mode) => void
  setSeed: (s: number) => void
  setFixedDist: (d: Vec3) => void
  setRunning: (r: boolean) => void
  setSpeed: (s: number) => void
  stepN: (n: number) => void
  playUserAction: (a: Action) => void
  reset: () => void
}

const DEFAULT_DIST: Vec3 = [0.5, 0.25, 0.25]

export function useSimulation(): Simulation {
  const [mode, setModeState] = useState<Mode>('self-play')
  const [seed, setSeedState] = useState(42)
  const [fixedDist, setFixedDistState] = useState<Vec3>(DEFAULT_DIST)
  const [running, setRunning] = useState(false)
  const [speed, setSpeed] = useState(1000)
  const [version, setVersion] = useState(0)

  const engineRef = useRef<Engine | null>(null)
  if (engineRef.current === null) {
    engineRef.current = new Engine({ mode, seed, fixedDist })
  }

  const bump = useCallback(() => setVersion((v) => v + 1), [])

  const rebuild = useCallback(
    (m: Mode, s: number, d: Vec3) => {
      engineRef.current = new Engine({ mode: m, seed: s, fixedDist: d })
      setRunning(false)
      bump()
    },
    [bump],
  )

  const setMode = useCallback(
    (m: Mode) => {
      setModeState(m)
      rebuild(m, seed, fixedDist)
    },
    [rebuild, seed, fixedDist],
  )

  const setSeed = useCallback(
    (s: number) => {
      setSeedState(s)
      rebuild(mode, s, fixedDist)
    },
    [rebuild, mode, fixedDist],
  )

  const setFixedDist = useCallback(
    (d: Vec3) => {
      setFixedDistState(d)
      // Applied live: the learner re-adapts to the new opponent mid-run.
      engineRef.current!.config.fixedDist = d
      bump()
    },
    [bump],
  )

  const reset = useCallback(() => rebuild(mode, seed, fixedDist), [rebuild, mode, seed, fixedDist])

  const stepN = useCallback(
    (n: number) => {
      engineRef.current!.step(n)
      bump()
    },
    [bump],
  )

  const playUserAction = useCallback(
    (a: Action) => {
      engineRef.current!.playUserAction(a)
      bump()
    },
    [bump],
  )

  // Run loop: batch iterations inside requestAnimationFrame; React renders
  // once per frame from a snapshot, never per iteration.
  useEffect(() => {
    if (!running) return
    let raf = 0
    let last = performance.now()
    let carry = 0
    const loop = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.25)
      last = now
      carry += speed * dt
      const n = Math.floor(carry)
      if (n > 0) {
        carry -= n
        engineRef.current!.step(n)
        bump()
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [running, speed, bump])

  return {
    engine: engineRef.current,
    mode,
    seed,
    fixedDist,
    running,
    speed,
    version,
    setMode,
    setSeed,
    setFixedDist,
    setRunning,
    setSpeed,
    stepN,
    playUserAction,
    reset,
  }
}
