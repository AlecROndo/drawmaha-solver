/**
 * Range against range: how a masked part of one seat's sample fares against
 * a weighted range of the other seat's, on each half of the pot.
 *
 * Terms: a *holding* is a sampled hand of five hole cards; its *inner* hand
 * is the holding as a poker hand, its *outer* hand the best two-hole-plus-
 * three-board Omaha hand (`classify.ts`); Drawmaha pays half the pot to each.
 * A *region* is a part of the grid, given here as a 0/1 mask (`filter.ts`).
 * The other seat's range is its own sample on the same board (a second
 * seed) with a weight per hand — the probability the *illustrative policy*
 * (`policy.ts`) bets it, say. Equity here is showdown-now: the weighted share
 * of the other range a hand's score beats, ties at half. Card removal between
 * the two seats is ignored on purpose: both samples are drawn from the same
 * 49 unseen cards, so two "opposing" hands may share a card. The error is
 * small and the page says so.
 */

import type { Policy } from './policy'
import { type RangeSample, lowerBound } from './sample'

/** The other seat's betting range: one weight per hand, its probability of potting (a copy). */
export function weightsForBet(pol: Policy): Float32Array {
  return pol.p.slice()
}

/** A weighted CDF over the distinct scores of one half of a range. */
interface WeightedScores {
  scores: Float64Array
  /** Weight strictly below `scores[k]`. */
  below: Float64Array
  /** Weight at or below `scores[k]`. */
  upTo: Float64Array
  total: number
}

function weightedScores(score: Uint32Array, weights: Float32Array): WeightedScores {
  const n = score.length
  const packed = new Float64Array(n)
  for (let i = 0; i < n; i++) packed[i] = score[i] * n + i
  packed.sort()
  const scores: number[] = []
  const below: number[] = []
  const upTo: number[] = []
  let running = 0
  let k = 0
  while (k < n) {
    const sc = Math.floor(packed[k] / n)
    scores.push(sc)
    below.push(running)
    while (k < n && Math.floor(packed[k] / n) === sc) {
      running += weights[packed[k] - sc * n]
      k++
    }
    upTo.push(running)
  }
  return { scores: Float64Array.from(scores), below: Float64Array.from(below), upTo: Float64Array.from(upTo), total: running }
}

/** The weighted share of a range `score` beats, ties at half. */
function beats(w: WeightedScores, score: number): number {
  const k = lowerBound(w.scores, score)
  if (k < w.scores.length && w.scores[k] === score) return (w.below[k] + 0.5 * (w.upTo[k] - w.below[k])) / w.total
  return (k < w.scores.length ? w.below[k] : w.total) / w.total
}

const cdfCache = new WeakMap<Float32Array, { other: RangeSample; inner: WeightedScores; outer: WeightedScores }>()

/**
 * For the hands of `s` in `mask` (null = every hand), the mean probability of
 * beating a hand drawn from `other` with probability ∝ `weights` (one weight
 * per hand of `other`, not all zero), on the inner and outer halves; `total`
 * is their average; `scoop` is the mean over hands of P(win inner) ×
 * P(win outer) and `scooped` the mean of P(lose inner) × P(lose outer), both
 * treating the two halves as independent draws. Card removal between the
 * seats is ignored (see the module note). Throws when the mask holds no hand.
 */
export function equityAgainst(
  s: RangeSample, mask: Uint8Array | null, other: RangeSample, weights: Float32Array,
): { inner: number; outer: number; total: number; scoop: number; scooped: number } {
  if (mask !== null && mask.length !== s.n) throw new Error(`mask has ${mask.length} entries for a sample of ${s.n}`)
  if (weights.length !== other.n) throw new Error(`weights has ${weights.length} entries for the other range of ${other.n}`)

  let cdf = cdfCache.get(weights)
  if (cdf === undefined || cdf.other !== other) {
    cdf = { other, inner: weightedScores(other.innerScore, weights), outer: weightedScores(other.outerScore, weights) }
    if (!(cdf.inner.total > 0)) throw new Error('the other range has no weight: every weight is zero')
    cdfCache.set(weights, cdf)
  }

  let m = 0
  let inner = 0
  let outer = 0
  let scoop = 0
  let scooped = 0
  for (let i = 0; i < s.n; i++) {
    if (mask !== null && mask[i] === 0) continue
    const pI = beats(cdf.inner, s.innerScore[i])
    const pO = beats(cdf.outer, s.outerScore[i])
    m++
    inner += pI
    outer += pO
    scoop += pI * pO
    scooped += (1 - pI) * (1 - pO)
  }
  if (m === 0) throw new Error('the mask holds no hand to take equity over')
  inner /= m
  outer /= m
  return { inner, outer, total: (inner + outer) / 2, scoop: scoop / m, scooped: scooped / m }
}
