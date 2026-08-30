import type { HistoryPoint } from '../sim/engine'

export interface LogXScale {
  x: (t: number) => number
  decades: number[]
  maxT: number
}

/** Log-10 x scale over [1, maxT], at least two decades so early runs aren't degenerate. */
export function makeLogX(maxT: number, x0: number, x1: number): LogXScale {
  const top = Math.max(maxT, 100)
  const logMax = Math.log10(top)
  const decades: number[] = []
  for (let d = 0; d <= Math.ceil(logMax); d++) decades.push(10 ** d)
  return {
    x: (t: number) => x0 + (Math.log10(Math.max(t, 1)) / logMax) * (x1 - x0),
    decades: decades.filter((d) => d <= top),
    maxT: top,
  }
}

export function decadeLabel(d: number): string {
  if (d >= 1_000_000) return `${d / 1_000_000}M`
  if (d >= 1000) return `${d / 1000}k`
  return String(d)
}

/**
 * Spread direct-label y positions so none overlap: keeps each label as close
 * to its line as the minimum gap allows, clamped to [lo, hi].
 */
export function spreadLabels(ys: number[], minGap: number, lo: number, hi: number): number[] {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y)
  const placed: number[] = []
  for (let k = 0; k < order.length; k++) {
    placed.push(k === 0 ? Math.max(order[k].y, lo) : Math.max(order[k].y, placed[k - 1] + minGap))
  }
  for (let k = order.length - 1; k >= 0; k--) {
    const cap = k === order.length - 1 ? hi : placed[k + 1] - minGap
    if (placed[k] > cap) placed[k] = cap
  }
  const out = new Array<number>(ys.length)
  order.forEach(({ i }, k) => {
    out[i] = placed[k]
  })
  return out
}

/** Nearest history point to a target t (history is sorted by t). */
export function nearestPoint(history: HistoryPoint[], t: number): HistoryPoint | null {
  if (history.length === 0) return null
  let lo = 0
  let hi = history.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (history[mid].t < t) lo = mid + 1
    else hi = mid
  }
  if (lo > 0 && Math.abs(history[lo - 1].t - t) < Math.abs(history[lo].t - t)) lo -= 1
  return history[lo]
}
