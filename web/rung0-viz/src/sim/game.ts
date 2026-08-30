export const ROCK = 0
export const PAPER = 1
export const SCISSORS = 2
export const ACTIONS = ['rock', 'paper', 'scissors'] as const
export type Action = 0 | 1 | 2
export type Vec3 = [number, number, number]

/** PAYOFF[a][b] = row player's chips when playing a against b. Zero-sum, antisymmetric. */
export const PAYOFF: readonly (readonly number[])[] = [
  [0, -1, 1],
  [1, 0, -1],
  [-1, 1, 0],
]

/** Column PAYOFF[:, b] — what every own action would have scored against the revealed b. */
export function utilityVs(b: Action): Vec3 {
  return [PAYOFF[0][b], PAYOFF[1][b], PAYOFF[2][b]]
}

/**
 * exploit(σ) = max over b of Σₐ σ[a]·PAYOFF[b][a] — what a best-responding
 * adversary who knows σ earns per round. Exactly 0 at Nash; pure rock → 1.
 */
export function exploitability(sigma: readonly number[]): number {
  let best = -Infinity
  for (let b = 0; b < 3; b++) {
    let ev = 0
    for (let a = 0; a < 3; a++) ev += sigma[a] * PAYOFF[b][a]
    if (ev > best) best = ev
  }
  return best
}
