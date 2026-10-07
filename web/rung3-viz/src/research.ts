/**
 * The research view's numbers, read from the two files `minidraw-analysis`
 * writes and nothing else: `figures/rung3/grades.json` (every graded
 * checkpoint of the four-rule race, by the exact best-response walk, plus a
 * uniform-random reference) and `figures/rung3/answer_sheet.json` (the frozen
 * strategy's grades and what it does in round 1 and at the draw, by inner
 * category). Both are copied into `src/data/` and nothing here is typed in.
 */

import gradesData from './data/grades.json'
import sheetData from './data/answer_sheet.json'

export type Rule = 'vanilla' | 'lcfr' | 'cfrplus' | 'dcfr'
export type Column = 'linear' | 'uniform' | 'quadratic'

export const RULES: Rule[] = ['vanilla', 'lcfr', 'cfrplus', 'dcfr']
export const RULE_NAME: Record<Rule, string> = { vanilla: 'vanilla', lcfr: 'LCFR', cfrplus: 'CFR+', dcfr: 'DCFR' }

/** Which rules keep negative regret; the race's finding splits on this. */
export const KEEPS_NEGATIVE: Record<Rule, boolean> = { vanilla: true, lcfr: true, cfrplus: false, dcfr: false }

export interface Grade {
  br0: number
  br1: number
  exploitability: number
  value_p0: number
  seed?: number
  workers?: number
}

const GRADES = gradesData as Record<string, Grade>

export interface Point extends Grade {
  rule: Rule
  column: Column
  iteration: number
  /** sampled hands per seat: iterations × workers */
  hands: number
}

/** `lcfr/002000000/linear` → its parts, or null for the reference row. */
function parseKey(key: string): { rule: Rule; iteration: number; column: Column } | null {
  const m = /^(vanilla|lcfr|cfrplus|dcfr)\/(\d+)\/(linear|uniform|quadratic)$/.exec(key)
  if (!m) return null
  return { rule: m[1] as Rule, iteration: Number(m[2]), column: m[3] as Column }
}

export const POINTS: Point[] = Object.entries(GRADES)
  .flatMap(([key, grade]) => {
    const parsed = parseKey(key)
    if (!parsed) return []
    return [{ ...grade, ...parsed, hands: parsed.iteration * (grade.workers ?? 1) }]
  })
  .sort((a, b) => a.iteration - b.iteration)

/** W, the hands per seat one lockstep iteration samples. */
export const WORKERS = POINTS[0]?.workers ?? 1

/** One rule's trajectory under one averaging column, ascending in iterations. */
export const series = (rule: Rule, column: Column = 'linear'): Point[] =>
  POINTS.filter((p) => p.rule === rule && p.column === column)

/** Every iteration the race graded at, ascending. */
export const CHECKPOINTS: number[] = [...new Set(POINTS.map((p) => p.iteration))].sort((a, b) => a - b)
export const FINAL_ITERATION = CHECKPOINTS[CHECKPOINTS.length - 1]

/** Every (rule, column) graded at the end of the run, for the averaging figure. */
export const FINAL_COLUMNS: Point[] = POINTS.filter((p) => p.iteration === FINAL_ITERATION)

/** What uniform random play is exploitable for: the scale the race is read against. */
export const UNIFORM_RANDOM: Grade = GRADES.uniform_random

/** Chips per hand → chips per hundred hands, the unit the README speaks in. */
export const perHundred = (x: number): number => x * 100

// ------------------------------------------------------- the answer sheet

export type Category = 'high_card' | 'pair' | 'straight' | 'flush' | 'straight_flush' | 'trips'
/** Inner categories in this deck's rarity order, commonest first (hands.py). */
export const CATEGORIES: Category[] = ['high_card', 'pair', 'straight', 'flush', 'straight_flush', 'trips']
export const CATEGORY_NAME: Record<Category, string> = {
  high_card: 'high card',
  pair: 'pair',
  straight: 'straight',
  flush: 'flush',
  straight_flush: 'straight flush',
  trips: 'trips',
}

type Spot = Record<Category, number[]>

interface Sheet {
  race_final_linear: Record<string, { hands_per_seat: number; exploitability: number }>
  frozen_strategy: Grade & { rule: string; column: string; iteration: number; workers: number; seed: number; sha256: string }
  round_one: { p0_open: Spot; p1_checked_to: Spot; p1_facing_bet: Spot }
  draw: { p0: Spot; p1_after_pat: Spot; p1_after_draw: Spot }
  columns: { round_one: { p0_open: string[]; p1_checked_to: string[]; p1_facing_bet: string[] }; draw: string[] }
}

export const SHEET = sheetData as unknown as Sheet
export const FROZEN = SHEET.frozen_strategy

export interface SpotFigure {
  key: string
  title: string
  columns: string[]
  rows: { category: Category; mix: number[] }[]
}

const spotFigure = (key: string, title: string, spot: Spot, columns: string[]): SpotFigure => ({
  key,
  title,
  columns,
  rows: CATEGORIES.map((category) => ({ category, mix: spot[category] })),
})

/** Round 1's three spots, each a stacked bar per category. */
export const ROUND_ONE: SpotFigure[] = [
  spotFigure('p0_open', 'P0 opens', SHEET.round_one.p0_open, SHEET.columns.round_one.p0_open),
  spotFigure('p1_checked_to', 'P1, checked to', SHEET.round_one.p1_checked_to, SHEET.columns.round_one.p1_checked_to),
  spotFigure('p1_facing_bet', 'P1, facing a bet', SHEET.round_one.p1_facing_bet, SHEET.columns.round_one.p1_facing_bet),
]

/** The draw's three spots. */
export const DRAW: SpotFigure[] = [
  spotFigure('p0', 'P0 draws', SHEET.draw.p0, SHEET.columns.draw),
  spotFigure('p1_after_pat', 'P1, after P0 stood pat', SHEET.draw.p1_after_pat, SHEET.columns.draw),
  spotFigure('p1_after_draw', 'P1, after P0 drew', SHEET.draw.p1_after_draw, SHEET.columns.draw),
]

/** The share of a draw spot's mix that throws a card at all. */
export const throwRate = (mix: number[]): number => 1 - mix[0]

// ------------------------------------------------------- the game's size

/** Information sets per rung, the ladder the homepage draws; mini-drawmaha's is the packed table's row count. */
export const SIZES = [
  { rung: 1, game: 'Kuhn', infosets: 12 },
  { rung: 2, game: 'Leduc', infosets: 288 },
  { rung: 3, game: 'mini-drawmaha', infosets: 6_220_050 },
] as const

/** Decision nodes in mini-drawmaha's full tree — the number that rules a full walk out. */
export const DECISION_NODES = 85.1e9
/** Distinct private deals, the root chance event the sampler draws from. */
export const DEALS = 100_100
/** Probabilities in the frozen strategy, and its size compressed. */
export const FROZEN_FLOATS = 14_253_840
export const FROZEN_MB = 18
/** Rows the LCFR run never reached, which play uniformly. */
export const UNREACHED_ROWS = 8_036
/** Median iterations between visits to a row at the end of the run. */
export const MEDIAN_REVISIT = 37_000
/** Halvings that take a float64 regret to exactly zero. */
export const HALVINGS_TO_ZERO = 1_075
