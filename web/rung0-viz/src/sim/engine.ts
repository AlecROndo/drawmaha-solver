import { mulberry32, sampleIndex } from './rng'
import { PAYOFF, exploitability, utilityVs, type Action, type Vec3 } from './game'
import {
  averageStrategy,
  newLedger,
  strategy,
  update,
  type Ledger,
  type UpdateTrace,
} from './ledger'

export type Mode = 'self-play' | 'vs-fixed' | 'vs-you'

export interface EngineConfig {
  mode: Mode
  seed: number
  /** Opponent distribution for vs-fixed mode. */
  fixedDist?: Vec3
}

/** One decimated sample of the run, enough to draw every chart and trail. */
export interface HistoryPoint {
  t: number
  avg: Vec3
  cur: Vec3
  exploit: number
  /** Second learner, self-play only. */
  avg1?: Vec3
  cur1?: Vec3
  /** Learner chips per round so far, scored modes only. */
  ev?: number
}

/** Everything the update-trace panel and scoreboard need about the last round. */
export interface RoundRecord {
  t: number
  playerAction: Action
  oppAction: Action
  payoff: number
  trace: UpdateTrace
  nextSigma: Vec3
}

/**
 * Decimation stride: keep every round to 1k, every 10th to 10k, every 100th
 * to 100k, and so on — memory stays bounded and log-x charts stay faithful.
 */
export function historyStride(t: number): number {
  if (t <= 1000) return 1
  return 10 ** (Math.ceil(Math.log10(t)) - 3)
}

export class Engine {
  readonly config: EngineConfig
  readonly ledgers: Ledger[]
  readonly history: HistoryPoint[] = []
  /** Last ~10 rounds, newest last — the scoreboard strip. */
  readonly recent: RoundRecord[] = []
  iteration = 0
  /** Learner's cumulative chips (scored modes). */
  chips = 0
  lastRound: RoundRecord | null = null
  private readonly rng: () => number

  constructor(config: EngineConfig) {
    this.config = config
    this.rng = mulberry32(config.seed)
    this.ledgers = config.mode === 'self-play' ? [newLedger(), newLedger()] : [newLedger()]
  }

  /** Run `count` automatic rounds (self-play / vs-fixed). No-op in vs-you mode. */
  step(count: number): void {
    if (this.config.mode === 'vs-you') return
    for (let i = 0; i < count; i++) {
      if (this.config.mode === 'self-play') this.stepSelfPlay()
      else this.stepVsFixed()
    }
  }

  private stepSelfPlay(): void {
    const [L0, L1] = this.ledgers
    const a0 = sampleIndex(strategy(L0), this.rng) as Action
    const a1 = sampleIndex(strategy(L1), this.rng) as Action
    const trace = update(L0, utilityVs(a1))
    update(L1, utilityVs(a0))
    this.finishRound(a0, a1, PAYOFF[a0][a1], trace, false)
  }

  private stepVsFixed(): void {
    const L0 = this.ledgers[0]
    const dist = this.config.fixedDist ?? [0.5, 0.25, 0.25]
    const a = sampleIndex(strategy(L0), this.rng) as Action
    const b = sampleIndex(dist, this.rng) as Action
    const trace = update(L0, utilityVs(b))
    this.finishRound(a, b, PAYOFF[a][b], trace, true)
  }

  /**
   * vs-you: the bot samples from its current strategy simultaneously with the
   * user's click, both are revealed, and the bot's ledger updates on the
   * user's action. Returns the round for immediate display.
   */
  playUserAction(userAction: Action): PlayResult {
    const L0 = this.ledgers[0]
    const botAction = sampleIndex(strategy(L0), this.rng) as Action
    const trace = update(L0, utilityVs(userAction))
    const payoff = PAYOFF[botAction][userAction]
    this.finishRound(botAction, userAction, payoff, trace, true)
    return { botAction, userAction, payoff }
  }

  private finishRound(
    playerAction: Action,
    oppAction: Action,
    payoff: number,
    trace: UpdateTrace,
    scored: boolean,
  ): void {
    this.iteration += 1
    if (scored) this.chips += payoff
    this.lastRound = {
      t: this.iteration,
      playerAction,
      oppAction,
      payoff,
      trace,
      nextSigma: strategy(this.ledgers[0]),
    }
    this.recent.push(this.lastRound)
    if (this.recent.length > 10) this.recent.shift()
    if (this.iteration % historyStride(this.iteration) === 0) this.record(scored)
  }

  private record(scored: boolean): void {
    const L0 = this.ledgers[0]
    const point: HistoryPoint = {
      t: this.iteration,
      avg: averageStrategy(L0),
      cur: strategy(L0),
      exploit: exploitability(averageStrategy(L0)),
    }
    if (this.config.mode === 'self-play') {
      point.avg1 = averageStrategy(this.ledgers[1])
      point.cur1 = strategy(this.ledgers[1])
    }
    if (scored) point.ev = this.chips / this.iteration
    this.history.push(point)
  }
}

/** What playUserAction reveals to the vs-you UI. */
export interface PlayResult {
  botAction: Action
  userAction: Action
  /** Bot's chips this round (user's chips are the negation). */
  payoff: number
}
