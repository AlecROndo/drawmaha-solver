/**
 * A seat's range at a spot — every private state it could be in, weighted
 * by how likely its own strategy was to bring that state here — and the
 * showdown equities of one range against another. Both are exact GIVEN THE
 * EXPORT: the enumeration misses no state and the equities are pairwise
 * over every legal pair, but every probability is read from the export's
 * byte rows, where the exporter rounded it to a multiple of 1/`scale`
 * (1/250, `range_export.quantise`). A reach is a product of up to five such
 * entries, so its rounding compounds, and an action the strategy takes less
 * than 0.2% of the time is a zero byte — a state reached only through such
 * an action has zero reach here, which is the export's resolution, not the
 * strategy's behaviour.
 *
 * Terms. A *state* is one physical hand a seat could hold at the spot: three
 * hole cards and, once it has drawn, the card it *threw* (or no card, 255).
 * Its *key* is the state canonicalised with the board (`canonical.ts`); the
 * strategy's *mix* at the spot is the key's row of the point's chunk
 * (`data.ts`). A state's *reach* is the product, over the seat's OWN earlier
 * decisions in the spot, of the probability its strategy gave the action it
 * actually took — read at each earlier public point with the key the state
 * had THEN. The deal and the deck's draws are equally likely for every state
 * of one shape, so those chance factors are the same constant for all of
 * them and are not applied; `weight` is reach normalised to sum 1 over the
 * range. The seat may be the one to act (`rangeAt`) or the other one
 * (`reachRange` with the other player): the other seat's reach is just the
 * same product over ITS decisions so far, and with none taken yet its range
 * is uniform.
 *
 * The one subtlety is a *thrower's* state. After the draw the seat holds
 * (hole, thrown) but not the memory of which hole card arrived from the deck,
 * so the state is the merge of THREE histories — the *pre-hole* was `thrown`
 * plus any *kept pair* of the three hole cards, and the third card is the
 * replacement. Each pre-hole has its own round-1 reach and its own
 * probability of throwing that card (the throw column is the card's position
 * in the pre-hole's draw order), while the replacement's chance is the same
 * in all three, so
 *
 *   reach(hole, thrown) = Σ over kept pairs of [ Π round-1 own-action probs(pre-hole)
 *                                                 × P(throw `thrown` | pre-hole) ]
 *                         × Π round-2 own-action probs(hole, thrown)
 *
 * A stand-pat seat's state has one history: reach = Π round-1 probs(hole)
 * × P(stand pat | hole) × Π round-2 probs.
 *
 * Equities are pairwise: my state against each of the other seat's states
 * whose cards are disjoint from mine, weighted by the other range's weights
 * (Bayes over a uniform deal: card removal restricts, their reach weights —
 * and since each seat's reach is a product over its OWN decisions only, the
 * two seats' weights are independent and pairwise disjointness is the whole
 * constraint). Inner and outer equities count a tie as half; a *scoop* is
 * winning BOTH halves outright on the same pair, so a chop on either half is
 * not one.
 */

import { MINI_DECK, miniLabel, type MiniCard } from './cards'
import { canonicalPicture, drawOrder } from './canonical'
import type { MiniData, PointMeta, Shape } from './data'
import { actorOf } from './grammar'
import { describeSpot, pointFor, type MiniSpot } from './points'
import { innerCategory, innerScore, outerCategory, outerScore } from './ranking'
import { r1Col, r2Col } from './regions'

/** The discard slot of a state that has thrown nothing. */
export const NO_DISCARD = 255

export interface MiniRange {
  spot: MiniSpot
  /** whose range this is; `point` and `mix` describe the ACTOR at the spot, not necessarily this seat */
  player: 0 | 1
  point: PointMeta
  n: number
  /** n × 4: hole (3, ascending) then the thrown card or `NO_DISCARD` */
  cards: Uint8Array
  /**
   * reach-weighted, sums to 1 over the range. Float32 because the shared
   * `RangeTable` reads it; the narrowing costs ~6e-8 relative per entry, far
   * under the export's 1/250 rounding, and cannot underflow: a nonzero reach
   * is at least 250^-5 (five byte entries) over a total of at most 3n, so
   * the smallest nonzero weight is ~1e-16 against float32's 1e-38 floor.
   * `equities` accumulates these in doubles.
   */
  weight: Float32Array
  /** inner row 0..5 = `innerCategory − 1` */
  row: Uint8Array
  /** `r1Col` on one board card, `r2Col` on two */
  col: Uint8Array
  /** the actor's own range: n × point.actions.length probabilities, columns = `point.actions`; the other seat's: empty */
  mix: Float32Array
  innerScore: Int32Array
  /** −1 on a one-card board */
  outerScore: Int32Array
}

/** One of the seat's earlier decisions: where it was taken, and its chunk. */
interface Stage {
  point: PointMeta
  chunk: Uint8Array
}

/** A betting stage also knows which column the seat took. */
interface BetStage extends Stage {
  col: number
}

/** The ledger symbol a line letter stands for: `x` and `c` are both check/call. */
function betSymbolOf(letter: string): string {
  if (letter === 'x' || letter === 'c') return 'c'
  if (letter === 'p' || letter === 'f') return letter
  throw new Error(`${JSON.stringify(letter)} is not a betting symbol (x c p f)`)
}

function columnOf(point: PointMeta, symbol: string): number {
  const col = point.actions.indexOf(symbol)
  if (col < 0) throw new Error(`action ${symbol} is not legal at public point ${point.id} (${point.actions.join(' ')})`)
  return col
}

/** The positions in `line` where `player` acted. */
function ownTurns(line: string, player: 0 | 1): number[] {
  const turns: number[] = []
  for (let k = 0; k < line.length; k++) if (actorOf(line.slice(0, k)) === player) turns.push(k)
  return turns
}

/** A sorted triple as one integer, for memo tables keyed by pre-hole. */
const tripleCode = (a: MiniCard, b: MiniCard, c: MiniCard): number => (a * 15 + b) * 15 + c

/** Three cards in ascending order. */
function sorted3(a: MiniCard, b: MiniCard, c: MiniCard): [MiniCard, MiniCard, MiniCard] {
  const t: [MiniCard, MiniCard, MiniCard] = [a, b, c]
  if (t[0] > t[1]) [t[0], t[1]] = [t[1], t[0]]
  if (t[1] > t[2]) [t[1], t[2]] = [t[2], t[1]]
  if (t[0] > t[1]) [t[0], t[1]] = [t[1], t[0]]
  return t
}

/**
 * `player`'s range at `spot`, exact under the exported strategy (module
 * comment: exact to the export's 1/`scale` rounding). The spot must
 * be a decision (of either seat). Enumerates every physical state of
 * `player` not holding a card in `exclude` — its shape follows the seat's
 * own draw count at that moment: three hole cards before it has drawn or
 * when it stood pat, hole plus thrown card when it threw — and weights each
 * by its reach (module comment). When `player` is the seat to act, `mix` is
 * its strategy at the spot; otherwise `mix` is empty and `point` still names
 * the actor's point. Async only because the chunks are fetched lazily; every
 * point the spot needs is loaded first and the arithmetic then runs
 * synchronously. Throws when the spot is not a decision, when a state's key
 * is missing from its shape, when a chunk is malformed, or when no state has
 * positive reach.
 */
export async function reachRange(data: MiniData, spot: MiniSpot, player: 0 | 1, exclude: readonly MiniCard[]): Promise<MiniRange> {
  const point = pointFor(data.index, spot)
  const isActor = player === spot.player
  const board = spot.board
  const b1 = board[0]
  const line1 = spot.lines[0]
  const drawn = spot.draws.length > player
  const thrower = drawn && spot.draws[player] === 1
  if (isActor && point.discards !== (thrower ? 1 : 0)) {
    throw new Error(`point ${point.id} says the actor threw ${point.discards}, but the spot's draws ${spot.draws.join(',')} say ${thrower ? 1 : 0}`)
  }
  const shape2: Shape = thrower ? 'b2d1' : 'b2d0'

  // Where the seat's own earlier decisions were taken, in the order taken.
  const r1Points = ownTurns(line1, player).map((k) => ({
    point: pointFor(data.index, { board: [b1], lines: [line1.slice(0, k)], draws: [], player }),
    symbol: betSymbolOf(line1[k]),
  }))
  const drawPoint = drawn
    ? pointFor(data.index, { board: [b1], lines: [line1], draws: player === 1 ? [spot.draws[0]] : [], player })
    : null
  const r2Points =
    spot.lines.length === 2
      ? ownTurns(spot.lines[1], player).map((k) => ({
          point: pointFor(data.index, { board, lines: [line1, spot.lines[1].slice(0, k)], draws: spot.draws, player }),
          symbol: betSymbolOf(spot.lines[1][k]),
        }))
      : []
  for (const { point: p } of r1Points) if (p.shape !== 'b1d0') throw new Error(`round-1 point ${p.id} has shape ${p.shape}, want b1d0`)
  if (drawPoint !== null && drawPoint.shape !== 'b1d0') throw new Error(`draw point ${drawPoint.id} has shape ${drawPoint.shape}, want b1d0`)
  for (const { point: p } of r2Points) if (p.shape !== shape2) throw new Error(`round-2 point ${p.id} has shape ${p.shape}, but P${player}'s states are ${shape2}`)
  if (isActor && board.length === 2 && point.shape !== shape2) throw new Error(`point ${point.id} has shape ${point.shape}, but the actor's states are ${shape2}`)
  if (isActor && board.length === 1 && point.shape !== 'b1d0') throw new Error(`point ${point.id} has shape ${point.shape} on a one-card board`)

  const needed = [...(isActor ? [point] : []), ...r1Points.map((s) => s.point), ...(drawPoint ? [drawPoint] : []), ...r2Points.map((s) => s.point)]
  const chunks = new Map(await Promise.all(needed.map(async (p) => [p.id, await data.point(p.id)] as const)))
  const chunkOf = (p: PointMeta): Uint8Array => {
    const c = chunks.get(p.id)
    if (c === undefined) throw new Error(`chunk for point ${p.id} was not loaded`)
    return c
  }
  const scale = data.index.scale
  const r1Stages: BetStage[] = r1Points.map((s) => ({ point: s.point, chunk: chunkOf(s.point), col: columnOf(s.point, s.symbol) }))
  const drawStage: Stage | null = drawPoint ? { point: drawPoint, chunk: chunkOf(drawPoint) } : null
  const r2Stages: BetStage[] = r2Points.map((s) => ({ point: s.point, chunk: chunkOf(s.point), col: columnOf(s.point, s.symbol) }))
  const here = isActor ? chunkOf(point) : null
  const width = isActor ? point.actions.length : 0

  const probAt = (stage: Stage, row: number, col: number): number => stage.chunk[row * stage.point.actions.length + col] / scale

  // Round-1 facts of a pre-hole, memoised: the same pre-hole is asked about
  // once as a round-1 state but up to eleven times as a thrower's history.
  const row1Memo = new Map<number, number>()
  const row1Of = (pre: readonly MiniCard[]): number => {
    const code = tripleCode(pre[0], pre[1], pre[2])
    let row = row1Memo.get(code)
    if (row === undefined) {
      row = data.keyIndex('b1d0', canonicalPicture({ hole: [...pre], thrown: [], board: [b1] }).key)
      row1Memo.set(code, row)
    }
    return row
  }
  const reach1Memo = new Map<number, number>()
  const reach1Of = (pre: readonly MiniCard[]): number => {
    const code = tripleCode(pre[0], pre[1], pre[2])
    let r = reach1Memo.get(code)
    if (r === undefined) {
      const row = row1Of(pre)
      r = 1
      for (const s of r1Stages) r *= probAt(s, row, s.col)
      reach1Memo.set(code, r)
    }
    return r
  }
  /** P(the seat throws `thrown` from `pre`), or stands pat when `thrown` is NO_DISCARD. */
  const throwProb = (pre: readonly MiniCard[], thrown: MiniCard): number => {
    if (drawStage === null) throw new Error(`P${player} has not drawn at ${describeSpot(spot)}`)
    let symbol: string
    if (thrown === NO_DISCARD) symbol = 'n'
    else {
      const pos = drawOrder({ hole: [...pre], thrown: [], board: [b1] }).indexOf(thrown)
      if (pos < 0) throw new Error(`${miniLabel(thrown)} is not in the pre-hole ${pre.map(miniLabel).join(' ')}`)
      symbol = 'lmt'[pos]
    }
    return probAt(drawStage, row1Of(pre), columnOf(drawStage.point, symbol))
  }

  const banned = new Set<MiniCard>([...board, ...exclude])
  const unseen = MINI_DECK.filter((c) => !banned.has(c))
  const u = unseen.length
  const holes = (u * (u - 1) * (u - 2)) / 6
  const n = thrower ? holes * (u - 3) : holes
  if (n === 0) throw new Error(`every state of P${player} at ${describeSpot(spot)} is excluded by ${exclude.map(miniLabel).join(' ')}`)

  const cards = new Uint8Array(n * 4)
  const weight = new Float32Array(n)
  const row = new Uint8Array(n)
  const col = new Uint8Array(n)
  const mix = new Float32Array(n * width)
  const inner = new Int32Array(n)
  const outer = new Int32Array(n)
  const reach = new Float64Array(n)
  // The two-card key is only built when something reads it: the actor's mix
  // at round 2, or a round-2 stage of either seat.
  const needKey2 = board.length === 2 && (isActor || r2Stages.length > 0)
  let total = 0
  let i = 0

  const place = (hole: readonly [MiniCard, MiniCard, MiniCard], thrown: MiniCard): void => {
    let r: number
    if (!drawn) {
      r = reach1Of(hole)
    } else if (!thrower) {
      r = reach1Of(hole) * throwProb(hole, NO_DISCARD)
    } else {
      // Three histories, one per kept pair; the replacement is the odd card out.
      const preA = sorted3(hole[0], hole[1], thrown)
      const preB = sorted3(hole[0], hole[2], thrown)
      const preC = sorted3(hole[1], hole[2], thrown)
      r = reach1Of(preA) * throwProb(preA, thrown) + reach1Of(preB) * throwProb(preB, thrown) + reach1Of(preC) * throwProb(preC, thrown)
    }
    let keyRow2 = -1
    if (needKey2) {
      keyRow2 = data.keyIndex(shape2, canonicalPicture({ hole: [...hole], thrown: thrown === NO_DISCARD ? [] : [thrown], board: [...board] }).key)
    }
    for (const s of r2Stages) r *= probAt(s, keyRow2, s.col)
    reach[i] = r
    total += r
    cards[i * 4] = hole[0]
    cards[i * 4 + 1] = hole[1]
    cards[i * 4 + 2] = hole[2]
    cards[i * 4 + 3] = thrown
    if (here !== null) {
      const mixRow = board.length === 2 ? keyRow2 : row1Of(hole)
      for (let a = 0; a < width; a++) mix[i * width + a] = here[mixRow * width + a] / scale
    }
    const sI = innerScore(hole)
    inner[i] = sI
    row[i] = innerCategory(sI) - 1
    if (board.length === 2) {
      const sO = outerScore(hole, board)
      outer[i] = sO
      col[i] = r2Col(outerCategory(sO))
    } else {
      outer[i] = -1
      col[i] = r1Col(hole, b1)
    }
    i++
  }

  for (let a = 0; a < u; a++) {
    for (let b = a + 1; b < u; b++) {
      for (let c = b + 1; c < u; c++) {
        const hole: [MiniCard, MiniCard, MiniCard] = [unseen[a], unseen[b], unseen[c]]
        if (!thrower) place(hole, NO_DISCARD)
        else for (let d = 0; d < u; d++) if (d !== a && d !== b && d !== c) place(hole, unseen[d])
      }
    }
  }
  if (i !== n) throw new Error(`enumerated ${i} states of P${player} at ${describeSpot(spot)}, expected ${n}`)
  if (total === 0) throw new Error(`every state of P${player} at ${describeSpot(spot)} has zero reach under the strategy`)
  for (let k = 0; k < n; k++) weight[k] = reach[k] / total

  return { spot, player, point, n, cards, weight, row, col, mix, innerScore: inner, outerScore: outer }
}

/** The actor's range at `spot`: `reachRange` for the seat to act, with its mix. */
export const rangeAt = (data: MiniData, spot: MiniSpot, exclude: readonly MiniCard[]): Promise<MiniRange> =>
  reachRange(data, spot, spot.player, exclude)

/** One bit per card a state accounts for: its hole and, if any, its thrown card. */
function cardMasks(range: MiniRange): Uint16Array {
  const masks = new Uint16Array(range.n)
  for (let i = 0; i < range.n; i++) {
    let m = (1 << range.cards[i * 4]) | (1 << range.cards[i * 4 + 1]) | (1 << range.cards[i * 4 + 2])
    const thrown = range.cards[i * 4 + 3]
    if (thrown !== NO_DISCARD) m |= 1 << thrown
    masks[i] = m
  }
  return masks
}

/**
 * For each of my states, the probability of winning each half against
 * `theirs` — over their states whose cards are disjoint from mine, weighted
 * by their `weight`, ties at half — and of scooping (both halves won
 * outright on one pair). `outer` and `scoop` are null on a one-card board.
 * A state of mine that no state of theirs can sit beside (their whole
 * weight lies on states sharing a card with it) has no opponent to measure
 * against and reads NaN. Both ranges must be over the same board. Two
 * thrower ranges are 2,860 × 2,860 pairs; the loop allocates nothing.
 */
export function equities(mine: MiniRange, theirs: MiniRange): { inner: Float32Array; outer: Float32Array | null; scoop: Float32Array | null } {
  const sameBoard = mine.spot.board.length === theirs.spot.board.length && mine.spot.board.every((c, k) => c === theirs.spot.board[k])
  if (!sameBoard) {
    throw new Error(`equities need one board: ${mine.spot.board.map(miniLabel).join(' ')} vs ${theirs.spot.board.map(miniLabel).join(' ')}`)
  }
  const withOuter = mine.spot.board.length === 2
  const maskM = cardMasks(mine)
  const maskT = cardMasks(theirs)
  const inner = new Float32Array(mine.n)
  const outer = withOuter ? new Float32Array(mine.n) : null
  const scoop = withOuter ? new Float32Array(mine.n) : null
  const wT = theirs.weight
  const iT = theirs.innerScore
  const oT = theirs.outerScore
  for (let i = 0; i < mine.n; i++) {
    const mi = maskM[i]
    const si = mine.innerScore[i]
    const oi = mine.outerScore[i]
    let w = 0
    let win = 0
    let tie = 0
    let oWin = 0
    let oTie = 0
    let both = 0
    for (let j = 0; j < theirs.n; j++) {
      if ((mi & maskT[j]) !== 0) continue
      const wj = wT[j]
      if (wj === 0) continue
      w += wj
      const sj = iT[j]
      const innerWin = si > sj
      if (innerWin) win += wj
      else if (si === sj) tie += wj
      if (withOuter) {
        const oj = oT[j]
        if (oi > oj) {
          oWin += wj
          if (innerWin) both += wj
        } else if (oi === oj) oTie += wj
      }
    }
    inner[i] = w > 0 ? (win + tie / 2) / w : NaN
    if (outer !== null && scoop !== null) {
      outer[i] = w > 0 ? (oWin + oTie / 2) / w : NaN
      scoop[i] = w > 0 ? both / w : NaN
    }
  }
  return { inner, outer, scoop }
}
