/**
 * One hand of mini-drawmaha against the frozen strategy, driven from a deal
 * pack. A deal is a fixed deck order plus the strategy's mix at every decision
 * either player can reach in it (see `deal_pack.py`); the hand here only
 * follows its path — round 1, the two throws, board card 2, round 2 — and
 * reads the vector at each node. Everything random takes an injectable
 * uniform draw so tests replay exact hands.
 *
 * The coach is rung 2's: when your turn arrives a roll in 0–99 is drawn and
 * shown BEFORE you click; the strategy's mix partitions the hundred rolls
 * into segments, aggressive first, and the segment your roll fell in is the
 * action the strategy would have taken this time. Following it is ✓;
 * deviating is graded by how deep the roll sat inside the segment you left.
 */

import {
  BETS,
  CHOP,
  DECK_SIZE,
  THROWS,
  actorOf,
  betSymbol,
  betWord,
  costOf,
  isClosed,
  isFold,
  isThrow,
  legalBets,
  replay,
  roundTree,
  settle,
  sym,
  throwIndex,
  throwWord,
  type Act,
  type Bet,
  type CardIndex,
  type Line,
  type Side,
  type Throw,
} from './minidraw'

// ------------------------------------------------------------- the pack

export interface Showdown {
  holes: [CardIndex[], CardIndex[]]
  board: [CardIndex, CardIndex]
  inner: Side
  outer: Side
  /** [inner, outer] category names per seat */
  cats: [[string, string], [string, string]]
}

/** One pre-dealt hand, as `deal_pack.py` writes it. Vectors are per-mille. */
export interface Deal {
  deck: CardIndex[]
  order: [CardIndex[], CardIndex[]]
  r1: Record<string, number[]>
  d0: Record<string, number[]>
  d1: Record<string, Record<string, number[]>>
  r2: Record<string, Record<string, Record<string, number[]>>>
  show: Record<string, Showdown>
}

export interface Manifest {
  seed: number
  deals: number
  chunk: number
  chunks: string[]
  strategy: { sha256: string; rule: string; column: string; iteration: number; workers: number; seed: number }
}

/** Where the stub starts: two holes and board card 1 come off the top. */
const STUB = 7

export const count = (t: Throw | null): number => (t !== null && t !== 'n' ? 1 : 0)

/** The card the deck hands `seat` for its throw, or null if it stood pat (or has not drawn). */
export function replacementOf(deal: Deal, seat: 0 | 1, throws: [Throw | null, Throw | null]): CardIndex | null {
  if (!count(throws[seat])) return null
  return deal.deck[STUB + (seat === 1 ? count(throws[0]) : 0)]
}

const ascending = (cards: CardIndex[]): CardIndex[] => [...cards].sort((a, b) => a - b)

/** What `seat` holds now, given the throws so far. */
export function holeOf(deal: Deal, seat: 0 | 1, throws: [Throw | null, Throw | null]): CardIndex[] {
  const dealt = deal.deck.slice(3 * seat, 3 * seat + 3)
  const t = throws[seat]
  if (t === null || t === 'n') return ascending(dealt)
  const thrown = deal.order[seat][throwIndex(t)]
  const got = replacementOf(deal, seat, throws)
  return ascending([...dealt.filter((c) => c !== thrown), got as CardIndex])
}

export const board1 = (deal: Deal): CardIndex => deal.deck[6]
export const board2 = (deal: Deal, throws: [Throw, Throw]): CardIndex =>
  deal.deck[STUB + count(throws[0]) + count(throws[1])]
export const comboOf = (throws: [Throw, Throw]): string => throws[0] + throws[1]

/** The sixteen post-draw worlds, P0's throw then P1's. */
export const COMBOS: string[] = THROWS.flatMap((a) => THROWS.map((b) => a + b))

// ------------------------------------------------------ the pack's shape

/** A per-mille vector: `width` non-negative integers that partition 1000. */
const isVector = (v: unknown, width: number): v is number[] =>
  Array.isArray(v) &&
  v.length === width &&
  v.every((x) => Number.isInteger(x) && x >= 0) &&
  v.reduce((sum: number, x: number) => sum + x, 0) === 1000

const sameCards = (a: CardIndex[] | undefined, b: CardIndex[]): boolean =>
  Array.isArray(a) && ascending(a).join() === ascending(b).join()

// The grammar is the same for every deal, so it is grown once: round 1, and
// round 2 after each of the seven lines that reach the draw.
const ROUND1 = roundTree([])
const ROUND2 = new Map(ROUND1.closed.map((line) => [line, roundTree([line])]))

/**
 * Why `deal` cannot be played, or null when it is whole. Every node the
 * betting grammar can reach must carry a vector of the right width, every
 * post-draw world a showdown on the cards the deck actually deals. This is
 * the pack checked against the grammar the browser plays by — a node the
 * Python export missed, or a stale chunk, is refused here and never reaches
 * the table as a silently uniform bot.
 */
export function validateDeal(deal: Deal): string | null {
  const deck = deal.deck
  const whole = Array.from({ length: DECK_SIZE }, (_, i) => i)
  if (!sameCards(deck, whole)) return 'the deck is not a permutation of the fifteen cards'
  for (const seat of [0, 1] as const) {
    if (!sameCards(deal.order?.[seat], deck.slice(3 * seat, 3 * seat + 3))) return `P${seat}'s draw order is not their three cards`
  }
  for (const node of ROUND1.nodes) {
    if (!isVector(deal.r1?.[node], legalBets([node]).length)) return `round 1 at '${node}' is missing or malformed`
  }
  for (const line of ROUND1.closed) {
    if (!isVector(deal.d0?.[line], THROWS.length)) return `P0's draw after ${line} is missing or malformed`
    for (const drew of ['0', '1']) {
      if (!isVector(deal.d1?.[line]?.[drew], THROWS.length)) return `P1's draw after ${line} (P0 drew ${drew}) is missing or malformed`
    }
    const round2 = ROUND2.get(line)!
    for (const combo of COMBOS) {
      for (const node of round2.nodes) {
        if (!isVector(deal.r2?.[line]?.[combo]?.[node], legalBets([line, node]).length)) {
          return `round 2 after ${line}/${combo} at '${node}' is missing or malformed`
        }
      }
    }
  }
  for (const combo of COMBOS) {
    const throws = [combo[0], combo[1]] as [Throw, Throw]
    const show = deal.show?.[combo]
    if (!show) return `the showdown of ${combo} is missing`
    for (const seat of [0, 1] as const) {
      if (!sameCards(show.holes?.[seat], holeOf(deal, seat, throws))) return `the showdown of ${combo} holds cards P${seat} does not`
    }
    if ((show.board ?? []).join() !== [board1(deal), board2(deal, throws)].join()) return `the showdown of ${combo} is on the wrong board`
    if (![0, 1, CHOP].includes(show.inner) || ![0, 1, CHOP].includes(show.outer)) return `the showdown of ${combo} has a half undecided`
  }
  return null
}

// ------------------------------------------------------------- the mix

export type Mix = Partial<Record<Act, number>>

/**
 * A per-mille vector over `legal`, as probabilities. A vector that is missing
 * or the wrong width is a broken pack, and the point of the pack is that the
 * bot plays the frozen strategy exactly — so it throws, naming `where`,
 * rather than quietly playing uniform in the strategy's name. `validateDeal`
 * runs on every chunk as it loads, so this is the backstop, not the check.
 */
export function toMix(legal: Act[], vector: number[] | undefined, where: string): Mix {
  if (!vector || vector.length !== legal.length) throw new Error(`the deal pack has no vector at ${where}`)
  const mix: Mix = {}
  for (let i = 0; i < legal.length; i++) mix[legal[i]] = vector[i] / 1000
  return mix
}

/**
 * The roll's segment order: aggression owns the LOW rolls. Bet/raise first,
 * then check/call, then fold; at the draw, standing pat first, then the three
 * throws in canonical order.
 */
export const AGGRESSION: Act[] = ['p', 'c', 'f', 'n', 'l', 'm', 't']

const present = (mix: Mix): Act[] => AGGRESSION.filter((a) => a in mix)

/** The [start, end) roll segment (0–100) an action owns under `mix`. */
function segmentOf(mix: Mix, act: Act): [number, number] {
  let start = 0
  for (const a of present(mix)) {
    if (a === act) break
    start += (mix[a] ?? 0) * 100
  }
  return [start, start + (mix[act] ?? 0) * 100]
}

/** The action the strategy prescribes for this roll. */
export function prescribed(mix: Mix, roll: number): Act {
  let start = 0
  const acts = present(mix)
  for (const a of acts) {
    start += (mix[a] ?? 0) * 100
    if (roll < start) return a
  }
  return acts[acts.length - 1] ?? 'c'
}

/** Percentage points between the roll and the played action's segment. */
export function mistakeOf(mix: Mix, roll: number, act: Act): number {
  const [start, end] = segmentOf(mix, act)
  if (roll < start) return start - roll
  if (roll >= end) return roll - end
  return 0
}

/** One action drawn from `mix` over `legal`, with the supplied uniform draw. */
export function sampleAction(mix: Mix, legal: Act[], uniform: () => number = Math.random): Act {
  const u = uniform()
  let cumulative = 0
  for (const act of legal) {
    cumulative += mix[act] ?? 0
    if (u < cumulative) return act
  }
  return legal[legal.length - 1]
}

// ------------------------------------------------------------- the hand

export type Phase = 'round1' | 'draw' | 'board2' | 'round2' | 'over'

/** One node of the transcript: an action with its true mix, or a board card. */
export type PlayRow =
  | {
      kind: 'action'
      seat: 0 | 1
      human: boolean
      act: Act
      /** the poker word, or the table's view of a draw */
      label: string
      /** what the human learnt that the table did not: `drew 5c` */
      note: string | null
      /** chips the action put in */
      cost: number
      /** the strategy's whole mix at the actor's spot */
      mix: Mix
      /** the legal actions at that spot, in ledger order */
      legal: Act[]
      /** each legal action's word in this context, for the mix caption */
      words: Partial<Record<Act, string>>
      roll: number | null
      correct: Act | null
      mistake: number | null
    }
  | { kind: 'board'; n: 1 | 2; card: CardIndex }
  | { kind: 'result'; text: string }

export interface Hand {
  deal: Deal
  humanSeat: 0 | 1
  l1: Line
  throws: [Throw | null, Throw | null]
  /** null until board card 2 lands */
  l2: Line | null
  rows: PlayRow[]
  pendingRoll: number | null
  /** chips to the human, set exactly once when the hand ends */
  result: number | null
}

export const linesOf = (h: Hand): Line[] => (h.l2 === null ? [h.l1] : [h.l1, h.l2])

export function phaseOf(h: Hand): Phase {
  if (isFold(h.l1)) return 'over'
  if (!isClosed(h.l1, replay([h.l1]).behind)) return 'round1'
  if (h.throws[0] === null || h.throws[1] === null) return 'draw'
  if (h.l2 === null) return 'board2'
  const chips = replay([h.l1, h.l2])
  if (isFold(h.l2) || isClosed(h.l2, chips.behind)) return 'over'
  return 'round2'
}

/** Whose decision it is; only meaningful in round1, draw and round2. */
export function actorAt(h: Hand): 0 | 1 {
  const phase = phaseOf(h)
  if (phase === 'round1') return actorOf(h.l1)
  if (phase === 'draw') return h.throws[0] === null ? 0 : 1
  return actorOf(h.l2 ?? '')
}

/** The legal actions and the strategy's mix at the node the hand is at. */
export function spotOf(h: Hand): { legal: Act[]; mix: Mix } {
  const phase = phaseOf(h)
  if (phase === 'round1') {
    const legal = legalBets([h.l1])
    return { legal, mix: toMix(legal, h.deal.r1[h.l1], `round 1 '${h.l1}'`) }
  }
  if (phase === 'draw') {
    const seat = actorAt(h)
    const drew = String(count(h.throws[0]))
    const vector = seat === 0 ? h.deal.d0[h.l1] : h.deal.d1[h.l1]?.[drew]
    return { legal: THROWS, mix: toMix(THROWS, vector, `P${seat}'s draw after ${h.l1}${seat === 1 ? ` (P0 drew ${drew})` : ''}`) }
  }
  const legal = legalBets([h.l1, h.l2 ?? ''])
  const combo = comboOf(h.throws as [Throw, Throw])
  return { legal, mix: toMix(legal, h.deal.r2[h.l1]?.[combo]?.[h.l2 ?? ''], `round 2 after ${h.l1}/${combo} at '${h.l2 ?? ''}'`) }
}

/** The physical card a throw discards for the player to act, or null for a pat. */
export function thrownCard(h: Hand, t: Throw): CardIndex | null {
  if (t === 'n') return null
  return h.deal.order[actorAt(h)][throwIndex(t)]
}

/** The word for `act` at the hand's current node, from the actor's side of the table. */
export function wordFor(h: Hand, act: Act, mine: boolean): string {
  if (isThrow(act)) {
    if (mine) return throwWord(act, thrownCard(h, act) ?? undefined)
    return act === 'n' ? 'stands pat' : 'draws one'
  }
  const line = phaseOf(h) === 'round1' ? h.l1 : (h.l2 ?? '')
  return betWord(act as Bet, line)
}

/** Apply one action at the node the hand is at. Pure: returns the next hand. */
function applyAct(h: Hand, act: Act): Hand {
  const phase = phaseOf(h)
  if (phase === 'round1') return { ...h, l1: h.l1 + betSymbol(act as Bet, h.l1) }
  if (phase === 'draw') {
    const throws: [Throw | null, Throw | null] = [...h.throws]
    throws[actorAt(h)] = act as Throw
    return { ...h, throws }
  }
  return { ...h, l2: (h.l2 ?? '') + betSymbol(act as Bet, h.l2 ?? '') }
}

const row = (h: Hand, act: Act, mine: boolean, roll: number | null): PlayRow => {
  const { legal, mix } = spotOf(h)
  const seat = actorAt(h)
  let note: string | null = null
  if (isThrow(act) && act !== 'n' && mine) {
    const after = applyAct(h, act)
    const got = replacementOf(after.deal, seat, after.throws)
    note = got === null ? null : `drew ${sym(got)}`
  }
  return {
    kind: 'action',
    seat,
    human: mine,
    act,
    label: wordFor(h, act, mine),
    note,
    cost: isThrow(act) ? 0 : costOf(linesOf(h), act as Bet),
    mix,
    legal,
    words: Object.fromEntries(legal.map((a) => [a, wordFor(h, a, mine)])),
    roll,
    correct: roll === null ? null : prescribed(mix, roll),
    mistake: roll === null ? null : mistakeOf(mix, roll, act),
  }
}

const cardWords = (cards: CardIndex[]): string => cards.map(sym).join(' ')

/** Who took a half, from the human's side, with both categories. */
function halfWords(name: string, side: Side, humanSeat: 0 | 1, cats: [string, string]): string {
  const who = side === CHOP ? 'chop' : side === humanSeat ? 'you' : 'the solver'
  return `${name}: ${who} (${cats[0]} v ${cats[1]})`
}

/** The last line of a finished hand: the reveal and the verdict. */
export function verdictOf(h: Hand): string {
  const result = h.result ?? 0
  const bot = (1 - h.humanSeat) as 0 | 1
  const verdict =
    result === 0 ? 'a push' : result > 0 ? `you win ${fmt(result)}` : `the solver wins ${fmt(-result)}`
  const throws = h.throws
  const shown = `you held ${cardWords(holeOf(h.deal, h.humanSeat, throws))} · the solver held ${cardWords(holeOf(h.deal, bot, throws))}`
  if (h.l2 === null || isFold(h.l2)) return `${shown} — ${verdict}`
  const show = h.deal.show[comboOf(throws as [Throw, Throw])]
  const inner = halfWords('inner', show.inner, h.humanSeat, [show.cats[h.humanSeat][0], show.cats[bot][0]])
  const outer = halfWords('outer', show.outer, h.humanSeat, [show.cats[h.humanSeat][1], show.cats[bot][1]])
  return `${shown} · ${inner} · ${outer} — ${verdict}`
}

/** A chip amount: whole when it is, else to the quarter. */
export const fmt = (n: number): string => (Number.isInteger(n) ? String(n) : n.toFixed(2).replace(/0$/, ''))

/**
 * Run the hand forward — the board landing, the solver acting — until it is
 * the human's move or the hand is over, then settle. Pure: returns a new hand.
 */
export function drive(prev: Hand, uniform: () => number = Math.random): Hand {
  let h: Hand = { ...prev, rows: [...prev.rows] }
  for (;;) {
    const phase = phaseOf(h)
    if (phase === 'over') break
    if (phase === 'board2') {
      const card = board2(h.deal, h.throws as [Throw, Throw])
      h.rows.push({ kind: 'board', n: 2, card })
      h = { ...h, l2: '' }
      continue
    }
    if (actorAt(h) === h.humanSeat) {
      // The turn arrives: the roll is drawn now, once, so the sidebar shows
      // it BEFORE the click — it is the instruction, not the grade. floor,
      // not round: uniform over the integers 0–99.
      if (h.pendingRoll === null) h = { ...h, pendingRoll: Math.floor(uniform() * 100) }
      return h
    }
    const { legal, mix } = spotOf(h)
    const act = sampleAction(mix, legal, uniform)
    h.rows.push(row(h, act, false, null))
    h = applyAct(h, act)
  }
  const lines = linesOf(h)
  const show =
    h.l2 !== null && !isFold(h.l2) ? h.deal.show[comboOf(h.throws as [Throw, Throw])] : null
  const toP0 = settle(lines, show?.inner ?? null, show?.outer ?? null)
  h = { ...h, result: h.humanSeat === 0 ? toP0 : -toP0 }
  h.rows.push({ kind: 'result', text: verdictOf(h) })
  return h
}

/** A fresh hand off `deal`, driven to the human's first decision. */
export function dealHand(deal: Deal, humanSeat: 0 | 1, uniform: () => number = Math.random): Hand {
  return drive(
    {
      deal,
      humanSeat,
      l1: '',
      throws: [null, null],
      l2: null,
      rows: [{ kind: 'board', n: 1, card: board1(deal) }],
      pendingRoll: null,
      result: null,
    },
    uniform,
  )
}

// ---------------------------------------------------------- the session

/** A sitting against the solver: the live hand, the bankroll, and the deals still to come. */
export interface Session {
  hand: Hand | null
  chips: number
  hands: number
  seatChips: [number, number]
  seatHands: [number, number]
  /** deals not yet played this sitting, in the order they will be dealt */
  pending: Deal[]
  /** deals already played, reshuffled back in when the pending run out */
  used: Deal[]
}

/** A uniformly shuffled copy. */
export function shuffled<T>(items: T[], uniform: () => number = Math.random): T[] {
  const out = [...items]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(uniform() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

/** A fresh sit-down: nothing banked, you in seat 0, hand one dealt if any deal has arrived. */
export function freshSession(deals: Deal[], uniform: () => number = Math.random): Session {
  const s: Session = {
    hand: null,
    chips: 0,
    hands: 0,
    seatChips: [0, 0],
    seatHands: [0, 0],
    pending: shuffled(deals, uniform),
    used: [],
  }
  return s.pending.length ? nextHand(s, uniform) : s
}

/** More deals arrive (a later chunk of the pack): shuffled into the queue. */
export const addDeals = (s: Session, deals: Deal[], uniform: () => number = Math.random): Session => ({
  ...s,
  pending: [...s.pending, ...shuffled(deals, uniform)],
})

/** True when the queue is short enough that the next chunk should be fetched. */
export const runningLow = (s: Session, floor = 10): boolean => s.pending.length < floor

/**
 * The human plays `act`: log it, extend the hand, drive to the next stop, and
 * bank the result in the same transition it appears — so a replayed update
 * cannot double-count. A finished or absent hand is left untouched.
 */
export function playAct(s: Session, act: Act, uniform: () => number = Math.random): Session {
  const h = s.hand
  if (h === null || h.result !== null) return s
  const played: Hand = { ...applyAct(h, act), rows: [...h.rows, row(h, act, true, h.pendingRoll)], pendingRoll: null }
  const done = drive(played, uniform)
  if (done.result === null) return { ...s, hand: done }
  const seat = done.humanSeat
  const seatChips: [number, number] = [...s.seatChips]
  const seatHands: [number, number] = [...s.seatHands]
  seatChips[seat] += done.result
  seatHands[seat] += 1
  return { ...s, hand: done, chips: s.chips + done.result, hands: s.hands + 1, seatChips, seatHands }
}

/** Deal the next hand, seats swapped; the bankroll carries. */
export function nextHand(s: Session, uniform: () => number = Math.random): Session {
  let pending = s.pending
  let used = s.hand ? [...s.used, s.hand.deal] : s.used
  if (pending.length === 0) {
    if (used.length === 0) return s
    pending = shuffled(used, uniform)
    used = []
  }
  const [deal, ...rest] = pending
  const seat = s.hand ? ((1 - s.hand.humanSeat) as 0 | 1) : 0
  return { ...s, hand: dealHand(deal, seat, uniform), pending: rest, used }
}

/** The legal betting actions, for a caller that wants only those. */
export const legalBetsOf = (h: Hand): Bet[] => legalBets(linesOf(h))

export { BETS }
