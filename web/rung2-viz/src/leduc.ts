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
export function potAfter(l1: string, board: Rank | null, l2: string): number {
  const round = (line: string, size: number): number => {
    const inFor = [0, 0]
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
    return inFor[0] + inFor[1]
  }
  let pot = 2 * ANTE + round(l1, BETS[0])
  if (board !== null) pot += round(l2, BETS[1])
  return pot
}

/**
 * How often play reaches a round-1 line, before the board.
 *
 * A forward pass over the exported strategy, weighted by the deal: six cards,
 * two per rank, so an ordered rank pair weighs 2/30 when the ranks match and
 * 4/30 when they differ. The flat per-node mean would treat the seat behind a
 * bet as a uniform deck, but the bettor's own card has left it — rung 1's
 * `actionFrequency` documents the same correction. Reading the solve is not
 * redoing it.
 */
export function reach(l1: string, strategy: Strategy): number {
  let total = 0
  for (const r0 of RANKS) {
    for (const r1 of RANKS) {
      const dealWeight = r0 === r1 ? 2 / 30 : 4 / 30
      let lineWeight = 1
      for (let i = 0; i < l1.length; i++) {
        const rank = i % 2 === 0 ? r0 : r1
        const row = strategy[keyFor(rank, null, l1.slice(0, i), '')]
        lineWeight *= row?.[l1[i] as Act] ?? 0
      }
      total += dealWeight * lineWeight
    }
  }
  return total
}
