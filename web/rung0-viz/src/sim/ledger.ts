import type { Vec3 } from './game'

/** One learner's state: cumulative regret R, cumulative strategy sum S, round count n. */
export interface Ledger {
  R: Vec3
  S: Vec3
  n: number
}

/** What one update did — the numbers the step-mode trace panel displays. */
export interface UpdateTrace {
  sigma: Vec3
  utility: Vec3
  expectedUtility: number
  increments: Vec3
}

export function newLedger(): Ledger {
  return { R: [0, 0, 0], S: [0, 0, 0], n: 0 }
}

/** σ = R⁺ / ΣR⁺ if any regret is positive, else uniform. */
export function strategy(L: Ledger): Vec3 {
  const p0 = Math.max(L.R[0], 0)
  const p1 = Math.max(L.R[1], 0)
  const p2 = Math.max(L.R[2], 0)
  const sum = p0 + p1 + p2
  if (sum > 0) return [p0 / sum, p1 / sum, p2 / sum]
  return [1 / 3, 1 / 3, 1 / 3]
}

/** Average strategy S/n — the object that converges to Nash. Uniform before any round. */
export function averageStrategy(L: Ledger): Vec3 {
  if (L.n === 0) return [1 / 3, 1 / 3, 1 / 3]
  return [L.S[0] / L.n, L.S[1] / L.n, L.S[2] / L.n]
}

/**
 * One regret-matching update against the revealed utility vector u.
 * Baseline is the strategy's EXPECTED utility ⟨σ,u⟩ (the counterfactual-regret
 * form CFR uses), not the sampled action's utility u[a].
 */
export function update(L: Ledger, u: Vec3): UpdateTrace {
  const sigma = strategy(L)
  const expectedUtility = sigma[0] * u[0] + sigma[1] * u[1] + sigma[2] * u[2]
  const increments: Vec3 = [
    u[0] - expectedUtility,
    u[1] - expectedUtility,
    u[2] - expectedUtility,
  ]
  L.R[0] += increments[0]
  L.R[1] += increments[1]
  L.R[2] += increments[2]
  L.S[0] += sigma[0]
  L.S[1] += sigma[1]
  L.S[2] += sigma[2]
  L.n += 1
  return { sigma, utility: u, expectedUtility, increments }
}
