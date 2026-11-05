/**
 * The strategies `/solver` can show for mini-drawmaha — one export each,
 * written by `minidraw-ranges --release <name>` into its own directory under
 * `public/` — and the words the page says about whichever one is loaded.
 *
 * Three strategies of one game. Rung 3's *tabular LCFR* average is the
 * yardstick: 2,000,000 lockstep iterations over 10 workers, 20M hands per
 * seat. Rung 4's two *Deep CFR* objects are frozen at iteration 500 of the
 * 4 × 256 arm (5M hands per seat): *Π*, the policy net, a fresh net fit on
 * the policy memory and the only average a game too big for tables can keep;
 * and the *exact average* of the iterates, which mini-drawmaha is small
 * enough to hold beside it. The three exports share their key tables and
 * their point walk byte for byte — only the per-point mix chunks differ — so
 * switching strategy under a spot keeps the spot's states and changes their
 * weights and mixes, which is what makes the switch honest. Every export
 * holds a row for every state, reached or not, and the three differ in what
 * an unvisited row holds: the two averages show a flat mix there (an empty
 * row averages to uniform), while Π shows the net's generalisation from the
 * rows it did see — a real mix, and the reason a net is interesting. So
 * under the switch an unvisited row is never missing, and it is where the
 * two kinds of answer differ most.
 *
 * Every export's `index.json` names its strategy (rule, column, iteration,
 * release and digest) and carries its grade — the exact best response's
 * exploitability, copied from the committed grades file at export time and
 * held to it by `tests/minidrawmaha/test_range_export.py` — so the status
 * line is read from the one file the page loads, never typed here.
 */

import { MiniData, type Fetcher, type MiniIndex } from './data'

/**
 * The two blocks every manifest of a shipped strategy opens with — `/solver`'s
 * `index.json` and `/rung3`'s deal-pack manifest alike, written by one Python
 * routine — so both pages describe a strategy in the same words.
 */
export type StrategyManifest = Pick<MiniIndex, 'strategy' | 'grade'>

export type StrategyId = 'tabular' | 'deep-pi' | 'deep-exact'

export interface StrategyChoice {
  id: StrategyId
  /** the export directory under the app's base, with its trailing slash */
  dir: string
  label: string
  blurb: string
}

export const STRATEGIES: readonly StrategyChoice[] = [
  { id: 'tabular', dir: 'mini/', label: 'Tabular LCFR', blurb: '20M hands per seat' },
  { id: 'deep-pi', dir: 'mini-deep-pi/', label: 'Deep CFR · Π', blurb: 'the policy net · 5M hands' },
  { id: 'deep-exact', dir: 'mini-deep-exact/', label: 'Deep CFR · exact', blurb: 'the exact average · 5M hands' },
]

export const DEFAULT_STRATEGY: StrategyId = 'tabular'

export const isStrategyId = (v: string): v is StrategyId => STRATEGIES.some((s) => s.id === v)

/** The choice an id names; throws on an id the list lacks, which a type-checked caller cannot produce. */
export function strategyChoice(id: StrategyId): StrategyChoice {
  const found = STRATEGIES.find((s) => s.id === id)
  if (found === undefined) throw new Error(`${id} is not a strategy (have ${STRATEGIES.map((s) => s.id).join(', ')})`)
  return found
}

// ------------------------------------------------------------- the words

export interface StrategyWords {
  /** who solved it and what object this is: "tabular LCFR", "Deep CFR · Π, the policy net" */
  name: string
  /** how far it was trained: iterations, workers, hands per seat */
  detail: string
  /** the graded exploitability to three decimals, or null when the manifest carries no grade */
  grade: string | null
}

const COLUMN_WORDS: Record<string, string> = {
  pi: 'Π, the policy net',
  exact: 'the exact average',
  current: 'the current iterate',
}

/** 20,000,000 → "20M"; anything not a whole million is written out. */
function hands(n: number): string {
  return n % 1_000_000 === 0 ? `${n / 1_000_000}M` : n.toLocaleString('en-US')
}

/**
 * The status line's words for a loaded manifest. A tabular rule is named by
 * its rule in capitals and its lockstep size; the deep rule by the
 * object its column names. The grade is quoted to three decimals, the way
 * the run log and the README quote every grade. The Play tab on `/rung3`
 * reads its deal pack's manifest through this too, so the bot there is named
 * exactly as the Learn tab names the same strategy.
 */
export function describeStrategy(index: StrategyManifest): StrategyWords {
  const { strategy, grade } = index
  const seen = grade ? ` · ${hands(grade.handsPerSeat)} hands per seat` : ''
  if (strategy.rule === 'deep') {
    return {
      name: `Deep CFR · ${COLUMN_WORDS[strategy.column] ?? strategy.column}`,
      detail: `iteration ${strategy.iteration.toLocaleString('en-US')}${seen}`,
      grade: grade ? grade.exploitability.toFixed(3) : null,
    }
  }
  return {
    name: `tabular ${strategy.rule.toUpperCase()}`,
    detail: `${strategy.iteration.toLocaleString('en-US')} iterations × ${strategy.workers} workers${seen}`,
    grade: grade ? grade.exploitability.toFixed(3) : null,
  }
}

// --------------------------------------------------------------- loading

/**
 * The exports, loaded on demand and kept: a manifest per strategy for the
 * switch's labels (one small fetch each), and the full `MiniData` for
 * whichever strategies have been shown. A failed load is not kept, so the
 * next request retries it.
 */
export class Strategies {
  private readonly base: string
  private readonly fetcher: Fetcher | undefined
  private readonly manifests = new Map<StrategyId, Promise<MiniIndex>>()
  private readonly loaded = new Map<StrategyId, Promise<MiniData>>()

  /** `base` is the app's base URL with its trailing slash (`/solver/`); the exports sit under it. */
  constructor(base: string, fetcher?: Fetcher) {
    this.base = base
    this.fetcher = fetcher
  }

  private dir(id: StrategyId): string {
    return `${this.base}${strategyChoice(id).dir}`
  }

  private kept<T>(cache: Map<StrategyId, Promise<T>>, id: StrategyId, start: () => Promise<T>): Promise<T> {
    const cached = cache.get(id)
    if (cached !== undefined) return cached
    const pending = start()
    pending.catch(() => cache.delete(id))
    cache.set(id, pending)
    return pending
  }

  manifest(id: StrategyId): Promise<MiniIndex> {
    return this.kept(this.manifests, id, () => MiniData.manifest(this.dir(id), this.fetcher))
  }

  data(id: StrategyId): Promise<MiniData> {
    return this.kept(this.loaded, id, () => MiniData.load(this.dir(id), this.fetcher))
  }
}
