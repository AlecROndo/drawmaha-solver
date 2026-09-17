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

/**
 * The roll: a mixed strategy made playable by hand. Aggression owns the LOW
 * rolls — bet/raise first, then check/call, then fold — so "bet 50%" means
 * every roll under 50 is a bet. `prescribed` names the action whose segment
 * holds the roll; `mistakeOf` grades a deviation by how many points deep the
 * roll sat inside the action you should have taken — checking on a 40 when
 * the bet region runs to 50 is a 10% mistake.
 */

/** Aggression descending: the order the roll's segments stack in. */
export const AGGRESSION: Act[] = ['r', 'c', 'f']

/** The [start, end) roll segment (0-100) an action owns under `mix`. */
function segmentOf(mix: Partial<Record<Act, number>>, act: Act): [number, number] {
  let start = 0
  for (const a of AGGRESSION) {
    if (a === act) break
    start += (mix[a] ?? 0) * 100
  }
  return [start, start + (mix[act] ?? 0) * 100]
}

/** The action the strategy prescribes for this roll. */
export function prescribed(mix: Partial<Record<Act, number>>, roll: number): Act {
  let start = 0
  const present = AGGRESSION.filter((a) => a in mix)
  for (const a of present) {
    start += (mix[a] ?? 0) * 100
    if (roll < start) return a
  }
  // An empty mix is a spot the export lacks; check/call is always legal.
  return present[present.length - 1] ?? 'c'
}

/** Percentage points between the roll and the played action's segment. */
export function mistakeOf(mix: Partial<Record<Act, number>>, roll: number, act: Act): number {
  const [start, end] = segmentOf(mix, act)
  if (roll < start) return start - roll
  if (roll >= end) return roll - end
  return 0
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

/** One node of the transcript: an action with its true mix, or the board. */
export type PlayRow =
  | {
      kind: 'action'
      seat: 0 | 1
      human: boolean
      act: Act
      /** the round line the actor faced — what names `c` a call vs a check */
      line: string
      /** the poker word for `act` in its context */
      label: string
      /** the solve's whole mix at the actor's spot */
      mix: Partial<Record<Act, number>>
      /** the pre-drawn roll — human decisions only */
      roll: number | null
      /** what that roll prescribed — human decisions only */
      correct: Act | null
      /** points of deviation; 0 is a followed prescription */
      mistake: number | null
    }
  | { kind: 'board'; rank: Rank }

/** One hand in progress against the solve, transcript included. */
export interface Hand {
  cards: [Card, Card]
  boardCard: Card
  humanSeat: 0 | 1
  l1: string
  board: Rank | null
  l2: string
  lines: string[]
  rows: PlayRow[]
  /** the roll for the live human decision, drawn when the turn arrives */
  pendingRoll: number | null
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
  const h: Hand = { ...prev, lines: [...prev.lines], rows: [...prev.rows] }
  for (;;) {
    if (lineOver(node(h))) break
    if (boardPending(node(h))) {
      h.board = h.boardCard.rank
      h.lines.push(`the board turns ${h.board}`)
      h.rows.push({ kind: 'board', rank: h.board })
      continue
    }
    if (actorOf(h) === h.humanSeat) {
      // The turn arrives: the roll is drawn now, once, so the sidebar can
      // show it BEFORE the click — it is the instruction, not the grade.
      // floor, not round: uniform over 0-99, the integers of the [0,100)
      // segment space — rounding would give 0 and 100 half a width each.
      if (h.pendingRoll === null) h.pendingRoll = Math.floor(uniform() * 100)
      return h
    }
    const bot = h.cards[1 - h.humanSeat].rank
    const line = roundLine(h)
    const key = keyFor(bot, h.board, h.l1, h.l2)
    const act = sampleAction(strategy, key, legal(line), uniform)
    h.lines.push(`the solver ${word(act, line)}s`)
    h.rows.push({
      kind: 'action',
      seat: (1 - h.humanSeat) as 0 | 1,
      human: false,
      act,
      line,
      label: word(act, line),
      mix: strategy[key] ?? {},
      roll: null,
      correct: null,
      mistake: null,
    })
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
      rows: [],
      pendingRoll: null,
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
  const h = s.hand
  const line = roundLine(h)
  const key = keyFor(h.cards[h.humanSeat].rank, h.board, h.l1, h.l2)
  const mix = strategy[key] ?? {}
  const roll = h.pendingRoll
  const played: Hand = {
    ...h,
    lines: [...h.lines, `you ${word(act, line)}`],
    rows: [
      ...h.rows,
      {
        kind: 'action',
        seat: h.humanSeat,
        human: true,
        act,
        line,
        label: word(act, line),
        mix,
        roll,
        correct: roll === null ? null : prescribed(mix, roll),
        mistake: roll === null ? null : mistakeOf(mix, roll, act),
      },
    ],
    pendingRoll: null,
  }
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
