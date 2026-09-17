/**
 * Leduc's rules for the browser: betting grammar, pot accounting, infoset
 * keys, and reach weights. A transcript of the rules half of
 * `src/drawmaha_solver/leduc/game.py` — CFR is deliberately NOT ported; the
 * strategy this page renders comes from `data/solve.json`, exported and
 * tested on the Python side, so there is no second copy of the solver to
 * drift. What lives here is arithmetic over published probabilities.
 */

export type Act = 'f' | 'c' | 'r'
export type Rank = 'J' | 'Q' | 'K'

/** Infoset key → that rank's mixed strategy over its legal actions. */
export type Strategy = Record<string, Partial<Record<Act, number>>>

export const RANKS: Rank[] = ['J', 'Q', 'K']
export const ANTE = 1
export const BETS: [number, number] = [2, 4]

const raises = (line: string): number => (line.match(/r/g) ?? []).length

/** The actions open to whoever acts on `line`: two raises cap a round. */
export function legal(line: string): Act[] {
  const n = raises(line)
  if (n === 0) return ['c', 'r']
  if (n < 2) return ['f', 'c', 'r']
  return ['f', 'c']
}

/** Whether a round's betting is finished. */
export function closed(line: string): boolean {
  if (!line) return false
  if (line.endsWith('f')) return true
  return raises(line) === 0 ? line.length >= 2 : line.endsWith('c') && line.length >= 2
}

/** What the symbol is called here: facing a bet, `c` is a call, `r` a raise. */
export const word = (act: Act, line: string): string =>
  ({
    f: 'fold',
    c: line.endsWith('r') ? 'call' : 'check',
    r: line.endsWith('r') ? 'raise' : 'bet',
  })[act]

/** The Python solver's `str(InfoSet)` label, e.g. "K:" or "J:cc|Q:r". */
export const keyFor = (rank: Rank, board: Rank | null, l1: string, l2: string): string =>
  board === null ? `${rank}:${l1}` : `${rank}:${l1}|${board}:${l2}`

/**
 * Chips in the middle after a given prefix of the hand.
 *
 * Tracks what each seat has actually put in per round: a raise moves the
 * level and the raiser up to it, a call brings the caller level, a check
 * moves nothing. An uncalled bet is one seat's chips — the whole reason this
 * cannot be counted as raises × size × 2.
 */
/** Chips each seat has put in during one round — the bets standing in front
 * of a seat before the round closes and they slide to the middle. */
export function roundInFor(line: string, size: number): [number, number] {
  const inFor: [number, number] = [0, 0]
  let level = 0
  for (let i = 0; i < line.length; i++) {
    const seat = i % 2
    const act = line[i]
    if (act === 'r') {
      level += size
      inFor[seat] = level
    } else if (act === 'c') {
      inFor[seat] = Math.max(inFor[seat], level)
    } else {
      break // a fold ends the round; nothing more goes in
    }
  }
  return inFor
}

export function potAfter(l1: string, board: Rank | null, l2: string): number {
  const one = roundInFor(l1, BETS[0])
  let pot = 2 * ANTE + one[0] + one[1]
  if (board !== null) {
    const two = roundInFor(l2, BETS[1])
    pot += two[0] + two[1]
  }
  return pot
}

/** The point in a hand the page is looking at, as the two lines and a board. */
interface Node {
  l1: string
  board: Rank | null
  l2: string
}

/**
 * How many ways the deal produces (P0 rank, P1 rank, this board): two cards
 * per rank, drawn without replacement — so a board rank a player also holds
 * has one card left, and a rank held by both players and the board has none.
 */
function dealCount(r0: Rank, r1: Rank, board: Rank | null): number {
  let count = 2 * (r1 === r0 ? 1 : 2)
  if (board !== null) count *= 2 - Number(board === r0) - Number(board === r1)
  return count
}

/** P(this line's actions | the two seats hold r0, r1), both rounds. */
function lineProb(r0: Rank, r1: Rank, s: Node, strategy: Strategy): number {
  let p = 1
  for (let i = 0; i < s.l1.length; i++) {
    const rank = i % 2 === 0 ? r0 : r1
    p *= strategy[keyFor(rank, null, s.l1.slice(0, i), '')]?.[s.l1[i] as Act] ?? 0
  }
  if (s.board !== null) {
    for (let i = 0; i < s.l2.length; i++) {
      const rank = i % 2 === 0 ? r0 : r1
      p *= strategy[keyFor(rank, s.board, s.l1, s.l2.slice(0, i))]?.[s.l2[i] as Act] ?? 0
    }
  }
  return p
}

/**
 * How often play reaches a round-1 line, before the board.
 *
 * A forward pass over the exported strategy, weighted by the deal. The flat
 * per-node mean would treat the seat behind a bet as a uniform deck, but the
 * bettor's own card has left it — rung 1's `actionFrequency` documents the
 * same correction. Reading the solve is not redoing it.
 */
export function reach(l1: string, strategy: Strategy): number {
  const s: Node = { l1, board: null, l2: '' }
  let total = 0
  for (const r0 of RANKS) {
    for (const r1 of RANKS) {
      total += (dealCount(r0, r1, null) / 30) * lineProb(r0, r1, s, strategy)
    }
  }
  return total
}

/**
 * What `seat` likely holds here: P(rank | the board and every action so far),
 * as [J, Q, K] summing to 1.
 *
 * This is the bar that keeps a red strategy row honest — a rank can bet 100%
 * of the time and still be almost never held, because the deal, the board,
 * and the seat's own earlier actions have already filtered it. Bayes over the
 * deal: deal weight × the probability both seats' strategies produce this
 * exact line. Unreachable nodes fall back to the deal alone — the strategy
 * factor is gone, but a board card has still left the deck — rather than
 * dividing by zero.
 */
export function rankWeights(seat: 0 | 1, s: Node, strategy: Strategy): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0]
  const deal: [number, number, number] = [0, 0, 0]
  RANKS.forEach((r0, i0) => {
    RANKS.forEach((r1, i1) => {
      const count = dealCount(r0, r1, s.board)
      if (count === 0) return
      deal[seat === 0 ? i0 : i1] += count
      out[seat === 0 ? i0 : i1] += count * lineProb(r0, r1, s, strategy)
    })
  })
  const total = out[0] + out[1] + out[2]
  if (total === 0) {
    const dealTotal = deal[0] + deal[1] + deal[2]
    return [deal[0] / dealTotal, deal[1] / dealTotal, deal[2] / dealTotal]
  }
  return [out[0] / total, out[1] / total, out[2] / total]
}
