/**
 * The two readings of a holding on a board, each with its draw flags.
 *
 * Terms: a *holding* is five hole cards; the *inner* hand is the holding as
 * a five-card poker hand on its own, flagged for a four-flush (four of one
 * suit, no made flush) and a four-straight (four distinct ranks that one more
 * rank would complete: gutshot when one rank does, open when two or more).
 * The *outer* hand is the best Omaha hand — exactly two hole cards and three
 * board cards — flagged for a flush draw (two board cards and at least two
 * hole cards of one suit, no made flush) and a straight draw (two hole ranks
 * and two board ranks that one, two, or three-or-more ranks complete: gut,
 * open, wrap). The flags feed the *regions* of the grid and the *illustrative
 * policy's* equity bonuses. The draw rules are written for the flop; on a
 * longer board the same counting is applied verbatim.
 */

import { type Card, suitOf } from './cards'
import { categoryOf, popcount, rankMaskOf, score5 } from './evaluate'

export interface Inner {
  cat: number
  score: number
  fourFlush: boolean
  /** 0 none, 1 gutshot, 2 open. */
  fourStraight: 0 | 1 | 2
}

export interface Outer {
  cat: number
  score: number
  flushDraw: boolean
  /** 0 none, 1 gut, 2 open, 3 wrap. */
  straightDraw: 0 | 1 | 2 | 3
}

const STRAIGHT_WINDOWS: readonly number[] = [...Array.from({ length: 9 }, (_, lo) => 0b11111 << lo), (1 << 12) | 0b1111]

/**
 * The ranks (as a mask) that would complete a straight for a four-rank mask:
 * every straight window containing all four contributes its missing rank.
 */
export function straightOuts(four: number): number {
  let outs = 0
  for (const w of STRAIGHT_WINDOWS) if ((four & w) === four) outs |= w & ~four
  return outs
}

/**
 * The outs of a holding's four-straights: every four distinct ranks of the
 * five, as `straightOuts` sees them, unioned.
 */
export function innerStraightOuts(rankMask: number): number {
  const distinct = popcount(rankMask)
  if (distinct === 4) return straightOuts(rankMask)
  if (distinct !== 5) return 0
  let outs = 0
  for (let r = 0; r < 13; r++) if (rankMask & (1 << r)) outs |= straightOuts(rankMask & ~(1 << r))
  return outs
}

const suitCounts = new Uint8Array(4)

/** Read five hole cards as a poker hand, with the four-flush and four-straight flags. */
export function classifyInner(hole: readonly Card[]): Inner {
  if (hole.length !== 5) throw new Error(`a holding is five cards, got ${hole.length}`)
  const score = score5(hole)
  const cat = categoryOf(score)
  suitCounts.fill(0)
  let maxSuit = 0
  for (const c of hole) maxSuit = Math.max(maxSuit, ++suitCounts[suitOf(c)])
  const fourFlush = cat < 5 && maxSuit === 4
  // A made straight (or better) is not drawing to one.
  const outs = cat < 4 ? popcount(innerStraightOuts(rankMaskOf(hole))) : 0
  const fourStraight = outs >= 2 ? 2 : outs === 1 ? 1 : 0
  return { cat, score, fourFlush, fourStraight }
}

const five: Card[] = [0, 0, 0, 0, 0]

/**
 * The best Omaha selection: the highest `score5` over every two hole cards
 * and three board cards, with the indices into `hole` of the two chosen.
 * Among equal scores the first pair in index order wins.
 */
export function bestOuterTwo(hole: readonly Card[], board: readonly Card[]): { score: number; a: number; b: number } {
  if (hole.length < 2) throw new Error(`the outer hand needs at least two hole cards, got ${hole.length}`)
  if (board.length < 3 || board.length > 5) throw new Error(`a board is three to five cards, got ${board.length}`)
  let best = -1
  let bestA = 0
  let bestB = 1
  for (let a = 0; a < hole.length; a++) {
    for (let b = a + 1; b < hole.length; b++) {
      five[0] = hole[a]
      five[1] = hole[b]
      for (let i = 0; i < board.length; i++) {
        for (let j = i + 1; j < board.length; j++) {
          for (let k = j + 1; k < board.length; k++) {
            five[2] = board[i]
            five[3] = board[j]
            five[4] = board[k]
            const s = score5(five)
            if (s > best) {
              best = s
              bestA = a
              bestB = b
            }
          }
        }
      }
    }
  }
  return { score: best, a: bestA, b: bestB }
}

/**
 * The outs of the outer straight draws: every two distinct hole ranks with
 * every two distinct board ranks that make four distinct ranks, unioned.
 */
export function outerStraightOuts(holeRanks: number, boardRanks: number): number {
  let outs = 0
  for (let h1 = 0; h1 < 13; h1++) {
    if (!(holeRanks & (1 << h1))) continue
    for (let h2 = h1 + 1; h2 < 13; h2++) {
      if (!(holeRanks & (1 << h2))) continue
      for (let b1 = 0; b1 < 13; b1++) {
        if (!(boardRanks & (1 << b1))) continue
        for (let b2 = b1 + 1; b2 < 13; b2++) {
          if (!(boardRanks & (1 << b2))) continue
          const four = (1 << h1) | (1 << h2) | (1 << b1) | (1 << b2)
          if (popcount(four) === 4) outs |= straightOuts(four)
        }
      }
    }
  }
  return outs
}

const holeSuits = new Uint8Array(4)
const boardSuits = new Uint8Array(4)

/**
 * The best Omaha hand of `hole` (at least two cards) on `board` (three to
 * five cards), with the flush-draw and straight-draw flags.
 */
export function classifyOuter(hole: readonly Card[], board: readonly Card[]): Outer {
  const { score } = bestOuterTwo(hole, board)
  const cat = categoryOf(score)
  holeSuits.fill(0)
  boardSuits.fill(0)
  for (const c of hole) holeSuits[suitOf(c)]++
  for (const c of board) boardSuits[suitOf(c)]++
  let flushDraw = false
  for (let s = 0; s < 4; s++) if (boardSuits[s] === 2 && holeSuits[s] >= 2) flushDraw = true
  // A made flush (or better) is not drawing to one; a made straight is not drawing to one.
  if (cat >= 5) flushDraw = false
  const outs = cat < 4 ? popcount(outerStraightOuts(rankMaskOf(hole), rankMaskOf(board))) : 0
  const straightDraw = outs >= 3 ? 3 : outs === 2 ? 2 : outs === 1 ? 1 : 0
  return { cat, score, flushDraw, straightDraw }
}
