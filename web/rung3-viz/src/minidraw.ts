/**
 * Enough of mini-drawmaha to play one hand in the browser from a deal pack:
 * the 15-card deck and its symbols, the pot-limit betting grammar (what may
 * be done, what ends the hand, what closes a round, what a bet costs), the
 * throw codes, and the settlement. A transcript of the rules half of
 * `src/drawmaha_solver/minidrawmaha/game.py`, pinned by `minidraw.test.ts`
 * against the same thirteen round-1 lines and chip ladder the Python suite
 * pins.
 *
 * What is deliberately NOT here: the infoset key (a joint canonical suit
 * relabelling) and hand ranking. Both stay in Python; the deal pack carries
 * the strategy's mix at every reachable node and the showdown of every
 * post-draw world, so the browser only follows a path and reads.
 */

// ---------------------------------------------------------------- cards

/** A card is its index in the deck: rank × 3 + suit, ranks 2–6, suits c d h. */
export type CardIndex = number

export const RANKS = '23456'
export const SUITS = 'cdh'
export const DECK_SIZE = RANKS.length * SUITS.length

export const rankOf = (c: CardIndex): number => Math.floor(c / SUITS.length)
export const suitOf = (c: CardIndex): number => c % SUITS.length
/** `4h`, the way the CLI and the Python suite spell a card. */
export const sym = (c: CardIndex): string => RANKS[rankOf(c)] + SUITS[suitOf(c)]

/** The suit's glyph for a face: ♣ ♦ ♥. */
export const SUIT_GLYPH = ['♣', '♦', '♥'] as const

// -------------------------------------------------------------- betting

export const ANTE = 1
/** 25 behind the ante: a bet, a raise and an all-in, nothing deeper. */
export const STACK = 26

/** A betting action: fold, check/call, pot. The same three the ledgers index. */
export type Bet = 'f' | 'c' | 'p'
/** A throw at the draw: stand pat, or the low/mid/top card in canonical order. */
export type Throw = 'n' | 'l' | 'm' | 't'
export type Act = Bet | Throw

export const BETS: Bet[] = ['f', 'c', 'p']
export const THROWS: Throw[] = ['n', 'l', 'm', 't']

export const isThrow = (a: Act): a is Throw => a === 'n' || a === 'l' || a === 'm' || a === 't'

/**
 * A betting line in the symbol notation the Python side prints: `x` check,
 * `c` call, `p` bet or raise, `f` fold. `x` and `c` are one action doing two
 * jobs, so a replay reads either as check/call.
 */
export type Line = string

export interface Chips {
  pot: number
  /** chips each seat has put in across the hand, ante included */
  committed: [number, number]
  /** chips each seat still has */
  behind: [number, number]
  /** chips each seat has put in during the round in progress */
  inRound: [number, number]
}

/** Chips a pot-sized bet or raise puts in: the call plus the pot after it, capped by the stack. */
const potSize = (pot: number, owed: number, behind: number): number =>
  owed + Math.min(pot + owed, behind - owed)

/**
 * Replay the betting and report where the chips are. `lines` holds one line
 * per round begun; the last is the round in progress. A folder stops paying
 * where they stood.
 */
export function replay(lines: Line[]): Chips {
  let pot = 2 * ANTE
  const committed: [number, number] = [ANTE, ANTE]
  const behind: [number, number] = [STACK - ANTE, STACK - ANTE]
  let inRound: [number, number] = [0, 0]
  for (const line of lines) {
    inRound = [0, 0]
    for (let turn = 0; turn < line.length; turn++) {
      const player = turn % 2
      const other = 1 - player
      const owed = inRound[other] - inRound[player]
      const letter = line[turn]
      if (letter === 'f') break
      const put = letter === 'p' ? potSize(pot, owed, behind[player]) : Math.min(owed, behind[player])
      inRound[player] += put
      committed[player] += put
      behind[player] -= put
      pot += put
    }
  }
  return { pot, committed, behind, inRound }
}

export const isFold = (line: Line): boolean => line.endsWith('f')

/** True once the stakes are level and nobody is left to act, or both stacks are empty. */
export const isClosed = (line: Line, behind: [number, number]): boolean =>
  (behind[0] === 0 && behind[1] === 0) || (line.length >= 2 && /[xc]$/.test(line))

/** Whose turn in a betting round: P0 opens every round. */
export const actorOf = (line: Line): 0 | 1 => (line.length % 2) as 0 | 1

/** What the player to act may do, in ledger order: fold, check/call, pot. */
export function legalBets(lines: Line[]): Bet[] {
  const chips = replay(lines)
  const line = lines[lines.length - 1]
  const player = actorOf(line)
  const owed = chips.inRound[1 - player] - chips.inRound[player]
  const behind = chips.behind[player]
  if (owed === 0) return behind > 0 ? ['c', 'p'] : ['c']
  return behind > owed ? ['f', 'c', 'p'] : ['f', 'c']
}

/** What a betting action costs the actor here. */
export function costOf(lines: Line[], bet: Bet): number {
  if (bet === 'f') return 0
  const chips = replay(lines)
  const line = lines[lines.length - 1]
  const player = actorOf(line)
  const owed = chips.inRound[1 - player] - chips.inRound[player]
  if (bet === 'c') return Math.min(owed, chips.behind[player])
  return potSize(chips.pot, owed, chips.behind[player])
}

const facingBet = (line: Line): boolean => line.endsWith('p')

/** The letter a betting action writes into the line here. */
export const betSymbol = (bet: Bet, line: Line): string =>
  bet === 'c' ? (facingBet(line) ? 'c' : 'x') : bet

/** The poker word for a betting action in its context. */
export function betWord(bet: Bet, line: Line): string {
  if (bet === 'f') return 'fold'
  if (bet === 'c') return facingBet(line) ? 'call' : 'check'
  return facingBet(line) ? 'raise' : 'bet'
}

/** The word for a throw, naming the card when one is given. */
export function throwWord(t: Throw, card?: CardIndex): string {
  if (t === 'n') return 'stand pat'
  return card === undefined ? `throw ${{ l: 'low', m: 'mid', t: 'top' }[t]}` : `throw ${sym(card)}`
}

/** Which of the actor's three canonical positions a throw discards. */
export const throwIndex = (t: Throw): number => ({ n: -1, l: 0, m: 1, t: 2 })[t]

// ------------------------------------------------------------ settlement

/** Who took one half of the pot: P0, P1, or a chop. */
export type Side = 0 | 1 | 2
export const CHOP: Side = 2

/** Each seat's share of a half the pot decided by `side`. */
const halfShares = (side: Side): [number, number] =>
  side === CHOP ? [0.25, 0.25] : side === 0 ? [0.5, 0] : [0, 0.5]

/**
 * Net chips to P0 at a finished hand (P1's are the negation). A fold hands
 * the folder's own stake to the other seat; a showdown pays each half of the
 * pot on its own side and nets off what each seat put in.
 */
export function settle(lines: Line[], inner: Side | null, outer: Side | null): number {
  const chips = replay(lines)
  const last = lines[lines.length - 1]
  if (isFold(last)) {
    const folder = ((last.length - 1) % 2) as 0 | 1
    return folder === 0 ? -chips.committed[0] : chips.committed[1]
  }
  if (inner === null || outer === null) throw new Error('a showdown needs both halves decided')
  const share = halfShares(inner)[0] + halfShares(outer)[0]
  return share * chips.pot - chips.committed[0]
}
