/**
 * The ILLUSTRATIVE policy: a smooth action mix and a throw-count rule written
 * on each hand's two showdown-now equities. Rung 4 is not trained; nothing
 * here is a solver output, and every page that draws these numbers says so.
 *
 * Terms: a *holding* is five hole cards; its *inner* equity is the share of
 * the sampled range its five-card hand beats, its *outer* equity the share
 * its best two-hole-plus-three-board Omaha hand beats (`sample.ts`). The
 * policy first lifts each equity by a bonus for the half's draw flags
 * (four-flush, four-straight; flush draw, straight draw), blends the two into
 * `eq`, and reads the pot, fold and call probabilities off logistic curves of
 * `eq` and of the *scoop proxy* `pI × pO` — the same curves as the Python
 * study's `policy()`, so the two agree. The proxy is not a scoop probability:
 * `pI` and `pO` are percentiles against a uniform sample, not win chances
 * against a betting range, and the two halves are built from the same five
 * cards, so they are positively correlated and the product understates how
 * often a strong-both-ways hand scoops. It is a feature the curves read, and
 * is named a proxy wherever it is drawn. A *node* is where the hand acts:
 * `'facing'` a pot bet (fold / call / pot) or `'open'` first to act (check /
 * pot, fold = 0). The throw rule is a fixed distribution over how many cards
 * to throw at the draw, keyed on the inner class and the draw flags.
 */

import type { Card } from './cards'
import { rankOf, suitOf } from './cards'
import { bestOuterTwo, classifyInner, classifyOuter, straightOuts } from './classify'
import { popcount, rankMaskOf } from './evaluate'
import { type RangeSample, assertSameBoard, beatFraction, sortedScores } from './sample'

/** Facing a pot bet: fold / call / pot. First to act: check / pot, fold = 0. */
export type Node = 'facing' | 'open'

export interface Policy {
  f: Float32Array
  c: Float32Array
  p: Float32Array
  /** The blended, draw-lifted equity the mix is read from. */
  eq: Float32Array
}

const sig = (x: number): number => 1 / (1 + Math.exp(-x))

interface Flags {
  fourFlush: boolean
  fourStraight: number
  flushDraw: boolean
  straightDraw: number
}

/**
 * The mix for one hand from its two equities and draw flags. Draws lift the
 * equity of their half (a four-flush most, a gutshot least) because the
 * showdown-now percentile undercounts a hand that will usually improve; the
 * pot curve turns on at eq ≈ 0.74 and the scoop curve at the proxy
 * pI × pO ≈ 0.55 (independence assumed, see the module note); big combined
 * outer draws semi-bluff in proportion to how weak the inner is; the fold
 * curve turns on below eq ≈ 0.47 and only over what is not potted, so the
 * three always sum to one. The `'open'` node reuses the facing-node pot
 * curve as its bet curve: opening and raising are different decisions and a
 * trained policy would put the threshold elsewhere, but the illustration
 * has one curve and says so here.
 */
function mixFrom(pI: number, pO: number, flags: Flags, node: Node): { f: number; c: number; p: number; eq: number } {
  const pI2 = Math.min(1, pI + (flags.fourFlush ? 0.14 : 0) + (flags.fourStraight === 2 ? 0.06 : flags.fourStraight === 1 ? 0.03 : 0))
  const pO2 = Math.min(
    1,
    pO + (flags.flushDraw ? 0.16 : 0) + (flags.straightDraw === 3 ? 0.1 : flags.straightDraw === 2 ? 0.07 : flags.straightDraw === 1 ? 0.03 : 0),
  )
  const eq = 0.5 * (pI2 + pO2)
  const scoop = pI2 * pO2
  let p = 0.92 * sig(16 * (eq - 0.74)) + 0.55 * sig(18 * (scoop - 0.55))
  if (flags.flushDraw && flags.straightDraw > 0) p += 0.22 * (1 - pI2)
  p = Math.min(0.98, p)
  if (node === 'open') return { f: 0, c: 1 - p, p, eq }
  const f = 0.97 * sig(14 * (0.47 - eq)) * (1 - p)
  return { f, c: Math.max(0, 1 - p - f), p, eq }
}

function flagsAt(s: RangeSample, i: number): Flags {
  return {
    fourFlush: s.innerFourFlush[i] === 1,
    fourStraight: s.innerFourStraight[i],
    flushDraw: s.outerFlushDraw[i] === 1,
    straightDraw: s.outerStraightDraw[i],
  }
}

/** The illustrative mix of every hand in the sample at `node`. */
export function policyFor(s: RangeSample, node: Node): Policy {
  const f = new Float32Array(s.n)
  const c = new Float32Array(s.n)
  const p = new Float32Array(s.n)
  const eq = new Float32Array(s.n)
  for (let i = 0; i < s.n; i++) {
    const m = mixFrom(s.innerEquity[i], s.outerEquity[i], flagsAt(s, i), node)
    f[i] = m.f
    c[i] = m.c
    p[i] = m.p
    eq[i] = m.eq
  }
  return { f, c, p, eq }
}

/**
 * The two equities of a hand placed against the sample: the share of each
 * sorted score array it beats, ties at half. A hand in the sample gets back
 * exactly its own `innerEquity` / `outerEquity`.
 */
export function equitiesOfHand(hole: readonly Card[], board: readonly Card[], s: RangeSample): { eqI: number; eqO: number; flags: Flags; innerCat: number } {
  assertSameBoard(s, board)
  const inner = classifyInner(hole)
  const outer = classifyOuter(hole, board)
  const sorted = sortedScores(s)
  return {
    eqI: beatFraction(sorted.inner, inner.score),
    eqO: beatFraction(sorted.outer, outer.score),
    flags: { fourFlush: inner.fourFlush, fourStraight: inner.fourStraight, flushDraw: outer.flushDraw, straightDraw: outer.straightDraw },
    innerCat: inner.cat,
  }
}

/**
 * The illustrative mix of one hand, whether or not it was sampled: the hand
 * is classified directly on `board` (which must be the sample's board) and
 * placed against the sample's scores.
 */
export function mixOfHand(hole: readonly Card[], board: readonly Card[], s: RangeSample, node: Node): { f: number; c: number; p: number; eq: number } {
  const { eqI, eqO, flags } = equitiesOfHand(hole, board, s)
  return mixFrom(eqI, eqO, flags, node)
}

/**
 * P(throw k), k = 0..5, for one hand. Made hands (a straight or better)
 * stand pat; two pair and trips share one rule, standing pat 45 % and
 * breaking for one 55 % — a simplification for trips, which a draw-poker
 * player would usually break for two, kept so the inner class alone decides
 * the count; a four-flush or open four-straight throws its odd card; a pair
 * throws three (or two, keeping a kicker); nothing keeps a flush draw's two
 * or throws three or four.
 */
function throwRule(innerCat: number, flags: Flags): number[] {
  if (innerCat >= 4) return [1, 0, 0, 0, 0, 0]
  if (innerCat >= 2) return [0.45, 0.55, 0, 0, 0, 0]
  if (flags.fourFlush || flags.fourStraight === 2) return [0, 1, 0, 0, 0, 0]
  if (innerCat === 1) return [0, 0, 0.4, 0.6, 0, 0]
  if (flags.flushDraw) return [0, 0, 1, 0, 0, 0]
  return [0, 0, 0, 0.5, 0.5, 0]
}

/** n × 6: the throw-count distribution of every hand, hand i at [6i, 6i + 6). */
export function throwCounts(s: RangeSample): Float32Array {
  const out = new Float32Array(s.n * 6)
  for (let i = 0; i < s.n; i++) {
    const dist = throwRule(s.innerCat[i], flagsAt(s, i))
    for (let k = 0; k < 6; k++) out[i * 6 + k] = dist[k]
  }
  return out
}

/**
 * The throw-count distribution of one hand and a throw probability per hole
 * card, in the order given: for each count k the rule names the exact k
 * cards thrown, and perCard[i] = Σₖ P(k) · [card i ∈ set(k)], so Σ perCard is
 * the expected count. The sets, illustrative like everything here: a
 * four-flush throws its off-suit card; an open four-straight the card
 * outside the four straight ranks (a second card of a straight rank counts
 * as outside; the lower if two qualify); two pair and trips the lowest card
 * outside the groups; a pair its three kickers at k = 3 and its two lowest
 * kickers at k = 2 (ties by suit); nothing throws its lowest cards, never
 * one of the outer flush-draw suit when it has that draw and k ≤ 3, and
 * keeping the two cards of the best outer hand when k ≤ 3; a made hand
 * throws nothing. When the preferred cards run out before k, the set is
 * filled with the lowest-ranked cards not yet in it.
 */
export function throwOfHand(hole: readonly Card[], board: readonly Card[], s: RangeSample): { counts: number[]; perCard: number[] } {
  const { flags, innerCat } = equitiesOfHand(hole, board, s)
  const counts = throwRule(innerCat, flags)
  const perCard = hole.map(() => 0)
  for (let k = 1; k < 6; k++) {
    if (counts[k] === 0) continue
    for (const i of throwSet(k, hole, board, innerCat, flags)) perCard[i] += counts[k]
  }
  return { counts, perCard }
}

/** The indices of the k cards thrown at count k (see `throwOfHand`). */
function throwSet(k: number, hole: readonly Card[], board: readonly Card[], innerCat: number, flags: Flags): number[] {
  const preferred = throwOrder(k, hole, board, innerCat, flags)
  const set = preferred.slice(0, k)
  if (set.length < k) {
    const rest = byLowest(hole, hole.map((_, i) => i).filter((i) => !set.includes(i)))
    set.push(...rest.slice(0, k - set.length))
  }
  return set
}

/** Indices sorted lowest card first: by rank, then by suit index. */
function byLowest(hole: readonly Card[], indices: number[]): number[] {
  return [...indices].sort((a, b) => hole[a] - hole[b])
}

/**
 * The cards the rule would rather throw, most willingly first; mirrors the
 * branch order of `throwRule`.
 */
function throwOrder(k: number, hole: readonly Card[], board: readonly Card[], innerCat: number, flags: Flags): number[] {
  const rankCount = new Uint8Array(13)
  const suitCount = new Uint8Array(4)
  for (const c of hole) {
    rankCount[rankOf(c)]++
    suitCount[suitOf(c)]++
  }
  const all = hole.map((_, i) => i)
  const singles = all.filter((i) => rankCount[rankOf(hole[i])] === 1)
  if (innerCat >= 2) return byLowest(hole, singles)
  if (flags.fourFlush) return byLowest(hole, all.filter((i) => suitCount[suitOf(hole[i])] !== 4))
  if (flags.fourStraight === 2) return byLowest(hole, outsideStraight(hole))
  if (innerCat === 1) return byLowest(hole, singles)

  const boardSuit = new Uint8Array(4)
  for (const c of board) boardSuit[suitOf(c)]++
  let pool = all
  if (flags.flushDraw && k <= 3) {
    pool = pool.filter((i) => !(boardSuit[suitOf(hole[i])] === 2 && suitCount[suitOf(hole[i])] >= 2))
  }
  if (k > 3) return byLowest(hole, pool)
  const best = bestOuterTwo(hole, board)
  const isBest = (i: number) => i === best.a || i === best.b
  return [...byLowest(hole, pool.filter((i) => !isBest(i))), ...byLowest(hole, pool.filter(isBest))]
}

/**
 * The cards outside a four-straight: those whose rank takes no part in any
 * four ranks with outs, plus every card past the first of a straight rank.
 */
function outsideStraight(hole: readonly Card[]): number[] {
  const mask = rankMaskOf(hole)
  let straightRanks = 0
  if (popcount(mask) === 4) straightRanks = mask
  else for (let r = 0; r < 13; r++) if (mask & (1 << r) && straightOuts(mask & ~(1 << r)) !== 0) straightRanks |= mask & ~(1 << r)
  let used = 0
  const outside: number[] = []
  hole.forEach((c, i) => {
    const bit = 1 << rankOf(c)
    if ((straightRanks & bit) !== 0 && (used & bit) === 0) used |= bit
    else outside.push(i)
  })
  return outside
}
