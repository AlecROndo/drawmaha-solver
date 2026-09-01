import { useCallback, useEffect, useState } from 'react'
import solveData from '../data/solve.json'

/**
 * Playback over a solve exported by the Python solver.
 *
 * The trajectory is 55 log-spaced checkpoints of one 100,000-iteration run, so
 * "play" scrubs a recording rather than solving live. That is the point: every
 * number on this page is one `kuhn-analysis` printed and the Python suite
 * pins, not a re-implementation's.
 */

export interface Solve {
  iterations: number[]
  exploitabilityAverage: number[]
  exploitabilityCurrent: number[]
  /** infoset label ("K", "Jpb", ...) to P(bet) at each checkpoint */
  bet: Record<string, number[]>
  /** the closed form, at the alpha this run happened to find */
  closedForm: Record<string, number>
  alpha: number
  gameValue: number
  gameValueExact: number
}

export const SOLVE = solveData as Solve

const FRAME_MS = 110

/** True when the visitor has asked the OS for less animation. */
const prefersReducedMotion = (): boolean =>
  typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches

export function useSolve() {
  const last = SOLVE.iterations.length - 1

  // Lazy initialisers, read once at mount. Someone who asked for reduced
  // motion should still get the finding, so they land on the solved state
  // rather than an animation they opted out of.
  const [index, setIndex] = useState(() => (prefersReducedMotion() ? last : 0))
  const [playing, setPlaying] = useState(() => !prefersReducedMotion())

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      setIndex((i) => {
        if (i >= last) {
          setPlaying(false)
          return last
        }
        return i + 1
      })
    }, FRAME_MS)
    return () => clearInterval(id)
  }, [playing, last])

  const toggle = useCallback(() => {
    // Pressing play at the end replays instead of doing nothing.
    setIndex((i) => (i >= last ? 0 : i))
    setPlaying((p) => !p)
  }, [last])

  const scrub = useCallback((i: number) => {
    setPlaying(false)
    setIndex(i)
  }, [])

  return { index, last, playing, toggle, scrub }
}
