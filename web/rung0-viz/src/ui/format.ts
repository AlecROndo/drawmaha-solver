import { ACTIONS, type Action } from '../sim/game'

export const ACTION_COLORS = ['var(--rock)', 'var(--paper)', 'var(--scissors)'] as const

export function actionName(a: Action): string {
  return ACTIONS[a]
}

/** Fixed-decimals number, e.g. 0.333. */
export function fmt(x: number, digits = 3): string {
  return x.toFixed(digits)
}

/** Signed fixed-decimals number, e.g. +1.000 / −0.500. */
export function fmtSigned(x: number, digits = 3): string {
  const s = x.toFixed(digits)
  return x >= 0 ? `+${s}` : s
}

/** Vector as (a, b, c). */
export function fmtVec(v: readonly number[], digits = 3, signed = false): string {
  const f = signed ? (x: number) => fmtSigned(x, digits) : (x: number) => fmt(x, digits)
  return `(${v.map(f).join(', ')})`
}

/** Iteration count with thin-space grouping, e.g. 52 400. */
export function fmtIter(t: number): string {
  return t.toLocaleString('en-US').replace(/,/g, ' ')
}

/** Speed like 10, 250, 1.2k, 50k. */
export function fmtSpeed(s: number): string {
  if (s >= 1000) {
    const k = s / 1000
    return `${k >= 10 ? Math.round(k) : k.toFixed(1)}k`
  }
  return String(Math.round(s))
}
