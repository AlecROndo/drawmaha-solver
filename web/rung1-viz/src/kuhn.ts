/**
 * Kuhn poker's rules and tree layout, for the browser.
 *
 * A transcript of the rules half of `src/drawmaha_solver/kuhn/game.py`. CFR is
 * deliberately NOT ported: the solve this page renders is exported from the
 * Python solver to `data/solve.json`, so there is no second copy of the
 * reach-weighted walk to drift. What lives here is the payoff table the play
 * panel needs — thirty lines with no hidden state, pinned by `kuhn.test.ts`
 * against the same outcomes the Python suite pins.
 */

export type Card = 0 | 1 | 2
export type Act = 'p' | 'b'

export const CARDS: Card[] = [0, 1, 2]
export const CARD_SYMBOL = ['J', 'Q', 'K'] as const
export const CARD_NAME = ['jack', 'queen', 'king'] as const

/** The six deals, (P0 card, P1 card), each equally likely. */
export const DEALS: [Card, Card][] = [
  [0, 1],
  [0, 2],
  [1, 0],
  [1, 2],
  [2, 0],
  [2, 1],
]

/** The four spots where somebody acts, in tree order. */
export const DECISIONS = ['', 'p', 'b', 'pb'] as const
export type Decision = (typeof DECISIONS)[number]

const TERMINALS = ['pp', 'pbp', 'pbb', 'bp', 'bb']

/** Whose turn it is: P0 acts on even-length histories. */
export const actor = (history: string): 0 | 1 => (history.length % 2) as 0 | 1

export const isTerminal = (history: string): boolean => TERMINALS.includes(history)

/** What `act` is called here: a pass is a check or a fold, a bet is a bet or a call. */
export function actionLabel(act: Act, history: string): string {
  const facingBet = history.endsWith('b')
  if (act === 'p') return facingBet ? 'fold' : 'check'
  return facingBet ? 'call' : 'bet'
}

/**
 * Chips to (P0, P1) at a finished hand.
 *
 * A pass with a bet outstanding is a fold, worth the ante. Otherwise it is a
 * showdown, worth the ante alone if the hand was checked down and the ante
 * plus the bet if a bet was called.
 */
export function returns(history: string, cards: [Card, Card]): [number, number] {
  const contested = history.includes('b')
  let winner: 0 | 1
  let stake: number
  if (contested && history.endsWith('p')) {
    const folder = (history.length - 1) % 2
    winner = (1 - folder) as 0 | 1
    stake = 1
  } else {
    winner = cards[0] > cards[1] ? 0 : 1
    stake = contested ? 2 : 1
  }
  return winner === 0 ? [stake, -stake] : [-stake, stake]
}

/** The solver's infoset label: own card plus public history, e.g. "Kpb". */
export const infosetKey = (card: Card, history: string): string =>
  CARD_SYMBOL[card] + history

/** P(bet) at one infoset — the exported solve, read at whichever checkpoint. */
export type BetProbability = (card: Card, history: string) => number

/**
 * How often `act` is taken at `history`, across every hand that gets there.
 *
 * Not the flat mean of the node's three infosets: reaching `p` means the opener
 * checked, and they check a queen always but a king barely a third of the time,
 * so the hands sitting at that node are not the deck. Each card is weighted by
 * how often the acting player actually arrives holding it — at alpha ≈ 0.22 the
 * player facing a check holds {J 0.32, Q 0.26, K 0.42}, not thirds.
 *
 * This is a forward pass over the exported strategy: it multiplies published
 * probabilities along one line, and never touches regrets or counterfactual
 * values. Reading the solve is not redoing it.
 */
export function actionFrequency(
  history: Decision,
  act: Act,
  betProbability: BetProbability,
): number {
  const seat = actor(history)
  let reach = 0
  let bets = 0
  for (const cards of DEALS) {
    // Chance deals the six pairs uniformly, so the 1/6 cancels in the ratio.
    let weight = 1
    for (let i = 0; i < history.length; i++) {
      const prefix = history.slice(0, i)
      const p = betProbability(cards[actor(prefix)], prefix)
      weight *= history[i] === 'b' ? p : 1 - p
    }
    reach += weight
    bets += weight * betProbability(cards[seat], history)
  }
  // A node no hand reaches has no frequency to report; fall back to the flat
  // mean so the edge still draws rather than vanishing on a divide by zero.
  const p =
    reach > 0
      ? bets / reach
      : CARDS.reduce<number>((sum, card) => sum + betProbability(card, history), 0) / CARDS.length
  return act === 'b' ? p : 1 - p
}

// ---------------------------------------------------------------------------
// Tree layout
// ---------------------------------------------------------------------------

export interface DecisionNode {
  key: Decision
  cx: number
  cy: number
  caption: string
}

export interface TerminalNode {
  key: string
  cx: number
  cy: number
  text: string
}

export interface Edge {
  from: Decision
  act: Act
  to: string
}

/**
 * The tree runs left to right, one column per action taken, because that is
 * the order a hand happens in and it lays the whole game on one line inside
 * the page's full-bleed plate. Every coordinate below is a centre.
 */
export const TREE_W = 1010
export const TREE_H = 400

export const NODE_W = 214
export const NODE_H = 96

export const TERMINAL_W = 176
export const TERMINAL_H = 34

/** Node centres, placed by hand so no edge crosses another. */
export const DECISION_NODES: DecisionNode[] = [
  { key: '', cx: 137, cy: 198, caption: 'P0 opens' },
  { key: 'p', cx: 407, cy: 78, caption: 'P1 after a check' },
  { key: 'b', cx: 407, cy: 318, caption: 'P1 facing a bet' },
  { key: 'pb', cx: 677, cy: 78, caption: 'P0 facing a bet, after checking' },
]

export const TERMINAL_NODES: TerminalNode[] = [
  { key: 'pp', cx: 658, cy: 195, text: 'showdown for 1' },
  { key: 'bp', cx: 658, cy: 287, text: 'P1 folds · P0 +1' },
  { key: 'bb', cx: 658, cy: 353, text: 'showdown for 2' },
  { key: 'pbp', cx: 928, cy: 57, text: 'P0 folds · P1 +1' },
  { key: 'pbb', cx: 928, cy: 123, text: 'showdown for 2' },
]

export const EDGES: Edge[] = [
  { from: '', act: 'p', to: 'p' },
  { from: '', act: 'b', to: 'b' },
  { from: 'p', act: 'p', to: 'pp' },
  { from: 'p', act: 'b', to: 'pb' },
  { from: 'b', act: 'p', to: 'bp' },
  { from: 'b', act: 'b', to: 'bb' },
  { from: 'pb', act: 'p', to: 'pbp' },
  { from: 'pb', act: 'b', to: 'pbb' },
]
