/**
 * Enough of Leduc to play one hand in the browser: the deck, the showdown
 * ladder, the settlement, sampling the committed solve's mixed strategy, and
 * the loop that drives a hand forward between the human's clicks. The payout
 * half is a transcript of `src/drawmaha_solver/leduc/game.py`, pinned by
 * `play.test.ts` against the same numbers the referee fixture pins — the ±3
 * fold-to-a-re-raise included. The solver itself stays in Python. Everything
 * random takes an injectable uniform draw so tests replay exact hands.
 */

import { ANTE, BETS, keyFor, legal, word, type Act, type Rank, type Strategy } from './leduc'
import { boardPending, lineOver } from './timeline'

export interface Card {
  rank: Rank
  /** two indistinguishable copies per rank; the suit only names the card */
  suit: 0 | 1
}

export const DECK: Card[] = (['J', 'Q', 'K'] as Rank[]).flatMap((rank) =>
  ([0, 1] as const).map((suit) => ({ rank, suit })),
)

const RANK_VALUE: Record<Rank, number> = { J: 0, Q: 1, K: 2 }
const PAIR_STRENGTH = 3

/** Pairing the board beats every unpaired card; otherwise rank order. */
export const strength = (privateRank: Rank, board: Rank): number =>
  privateRank === board ? PAIR_STRENGTH : RANK_VALUE[privateRank]

/** Chips (P0, P1) put in during one round; a folder stops where they stood. */
function contributions(line: string, size: number): [number, number] {
  const paid: [number, number] = [0, 0]
  let raises = 0
  for (let i = 0; i < line.length; i++) {
    if (line[i] === 'f') break
    if (line[i] === 'r') raises += 1
    paid[i % 2] = raises * size
  }
  return paid
}

/**
 * Chips to P0 at a finished hand (P1's are the negation).
 *
 * A fold pays the winner everything the folder had put in, ante included; a
 * showdown pays the loser's (equal) stake to the stronger hand, and equal
 * unpaired ranks push.
 */
export function settle(l1: string, board: Rank | null, l2: string, r0: Rank, r1: Rank): number {
  const one = contributions(l1, BETS[0])
  const two = board === null ? [0, 0] : contributions(l2, BETS[1])
  const paid: [number, number] = [ANTE + one[0] + two[0], ANTE + one[1] + two[1]]

  if (l1.includes('f') || l2.includes('f')) {
    // Seats alternate within each round, so the folder's seat is the fold's
    // index parity in its own round's line.
    const folder = (l1.includes('f') ? l1.indexOf('f') : l2.indexOf('f')) % 2
    return folder === 0 ? -paid[0] : paid[1]
  }

  if (board === null) throw new Error('a hand without a board can only end in a fold')
  const s0 = strength(r0, board)
  const s1 = strength(r1, board)
  if (s0 === s1) return 0
  return s0 > s1 ? paid[1] : -paid[0]
}

/**
 * One action drawn from the solve's mixed strategy at `key`, using the
 * supplied uniform draw — injected so tests replay exact hands. A spot the
 * export somehow lacks falls back to uniform over the legal actions.
 */
export function sampleAction(
  strategy: Record<string, Partial<Record<Act, number>>>,
  key: string,
  legalActs: Act[],
  uniform: () => number = Math.random,
): Act {
  const row = strategy[key]
  const u = uniform()
  if (!row) return legalActs[Math.min(Math.floor(u * legalActs.length), legalActs.length - 1)]
  let cumulative = 0
  for (const act of legalActs) {
    cumulative += row[act] ?? 0
    if (u < cumulative) return act
  }
  return legalActs[legalActs.length - 1]
}

/** A uniformly shuffled copy of the deck. */
export function shuffled(uniform: () => number = Math.random): Card[] {
  const cards = [...DECK]
  for (let i = cards.length - 1; i > 0; i--) {
    const j = Math.floor(uniform() * (i + 1))
    ;[cards[i], cards[j]] = [cards[j], cards[i]]
  }
  return cards
}

/** One hand in progress against the solve, transcript included. */
export interface Hand {
  cards: [Card, Card]
  boardCard: Card
  humanSeat: 0 | 1
  l1: string
  board: Rank | null
  l2: string
  lines: string[]
  /** chips to the human, set exactly once when the hand ends */
  result: number | null
}

const node = (h: Hand) => ({ l1: h.l1, board: h.board, l2: h.l2 })
export const roundLine = (h: Hand): string => (h.board === null ? h.l1 : h.l2)
const actorOf = (h: Hand): 0 | 1 => (roundLine(h).length % 2) as 0 | 1

const appendAct = (h: Hand, act: Act): void => {
  if (h.board === null) h.l1 += act
  else h.l2 += act
}

/**
 * Run the hand forward — the board turning, the bot acting — until it is the
 * human's move or the hand is over, then settle. Mutates its own copy.
 */
export function drive(prev: Hand, strategy: Strategy, uniform: () => number = Math.random): Hand {
  const h: Hand = { ...prev, lines: [...prev.lines] }
  for (;;) {
    if (lineOver(node(h))) break
    if (boardPending(node(h))) {
      h.board = h.boardCard.rank
      h.lines.push(`the board turns ${h.board}`)
      continue
    }
    if (actorOf(h) === h.humanSeat) return h
    const bot = h.cards[1 - h.humanSeat].rank
    const line = roundLine(h)
    const act = sampleAction(strategy, keyFor(bot, h.board, h.l1, h.l2), legal(line), uniform)
    h.lines.push(`the solver ${word(act, line)}s`)
    appendAct(h, act)
  }
  const toP0 = settle(h.l1, h.board, h.l2, h.cards[0].rank, h.cards[1].rank)
  h.result = h.humanSeat === 0 ? toP0 : -toP0
  const you = h.cards[h.humanSeat].rank
  const bot = h.cards[1 - h.humanSeat].rank
  const verdict =
    h.result === 0 ? 'a push' : h.result > 0 ? `you win ${h.result}` : `the solver wins ${-h.result}`
  h.lines.push(`you held ${you} · the solver held ${bot} — ${verdict}`)
  return h
}

/** A fresh hand off a shuffled deck, driven to the human's first decision. */
export function deal(
  humanSeat: 0 | 1,
  strategy: Strategy,
  uniform: () => number = Math.random,
): Hand {
  const deck = shuffled(uniform)
  return drive(
    {
      cards: [deck[0], deck[1]],
      boardCard: deck[2],
      humanSeat,
      l1: '',
      board: null,
      l2: '',
      lines: [],
      result: null,
    },
    strategy,
    uniform,
  )
}

/** A sitting against the solver: the live hand plus the running bankroll. */
export interface Session {
  hand: Hand
  chips: number
  hands: number
}

/** A fresh sit-down: nothing banked, you in seat 0, hand one already dealt. */
export const freshSession = (strategy: Strategy, uniform: () => number = Math.random): Session => ({
  hand: deal(0, strategy, uniform),
  chips: 0,
  hands: 0,
})

/**
 * The human plays `act`: log it, extend the line, drive to the next stop, and
 * bank the result in the same transition it appears — so there is nothing to
 * double-count if the update is replayed. A finished hand is left untouched.
 */
export function playAct(
  s: Session,
  act: Act,
  strategy: Strategy,
  uniform: () => number = Math.random,
): Session {
  if (s.hand.result !== null) return s
  const played: Hand = { ...s.hand, lines: [...s.hand.lines, `you ${word(act, roundLine(s.hand))}`] }
  appendAct(played, act)
  const done = drive(played, strategy, uniform)
  return done.result === null
    ? { ...s, hand: done }
    : { hand: done, chips: s.chips + done.result, hands: s.hands + 1 }
}

/** Deal the next hand, seats swapped; the bankroll carries. */
export const nextHand = (
  s: Session,
  strategy: Strategy,
  uniform: () => number = Math.random,
): Session => ({
  ...s,
  hand: deal((1 - s.hand.humanSeat) as 0 | 1, strategy, uniform),
})
