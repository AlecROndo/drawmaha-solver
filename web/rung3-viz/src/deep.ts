/**
 * Rung 4's numbers, read from copies of the committed grade files under
 * `src/data/rung4/` and nothing else: every arm of Deep CFR trained on this
 * game, graded three ways at each mark by the exact best-response walk
 * (`figures/rung4/*.json`, keyed `deep/<t>/<exact|pi|current>`), the
 * net-size sweep's verdict (`sweep.json`), the regret-rule screens and the
 * sign-flip measurement behind them, and the policy-net loss sweep
 * (`pi_sweep/`). Every file is in chips per hand; the page multiplies by
 * 100 in one place (`perHundred`, in `research.ts`). The copies are held
 * byte for byte to the originals by `tests/test_rung3_page_data.py`.
 *
 * Each file is checked as it is read, not cast: a regenerated file with a
 * half-graded mark, a missing field or marks that disagree on their config
 * fails here with the arm, the mark and the field in the message, instead of
 * as an `undefined` in a chart.
 */

import { series } from './research'

import grades64 from './data/rung4/grades.json'
import sweepData from './data/rung4/sweep.json'
import s128x3t200 from './data/rung4/sweep_128x3_t200.json'
import s128x3s8000 from './data/rung4/sweep_128x3-s8000.json'
import s128x4 from './data/rung4/sweep_128x4.json'
import s256x4 from './data/rung4/sweep_256x4.json'
import s256x4s8000 from './data/rung4/sweep_256x4-s8000.json'
import s256x4s16000 from './data/rung4/sweep_256x4-s16000-lr5e-4.json'
import ruleDcfr from './data/rung4/rule_dcfr-256x4.json'
import ruleCfrplus from './data/rung4/rule_cfrplus-256x4.json'
import signFlipsData from './data/rung4/rule_sign_flips.json'
import piResults from './data/rung4/pi_sweep_results.json'
import piReach from './data/rung4/pi_sweep_reach_weighted.json'

// ------------------------------------------------------------ the grade files

/** One graded object: the exact average, Π or the current iterate at one mark. */
export interface Graded {
  br0: number
  br1: number
  exploitability: number
  value_p0: number
}

interface Config {
  width: number
  depth: number
  steps: number
  lr: number
  workers: number
  walks: number
  rule?: string
  device?: string
}

interface DeepGrade extends Graded {
  hands_per_seat: number
  wall_clock_s: number
  config: Config
}

type GradeFile = Record<string, DeepGrade>

export type Objective = 'exact' | 'pi' | 'current'
export const OBJECTIVES: Objective[] = ['exact', 'pi', 'current']
export const OBJECTIVE_NAME: Record<Objective, string> = {
  exact: 'exact average',
  pi: 'Π, the policy net',
  current: 'current iterate',
}

/** One grid mark of one arm: three objects, each graded, and the hands they had seen. */
export interface Mark {
  t: number
  hands: number
  wallClockS: number
  exact: Graded
  pi: Graded
  current: Graded
}

export type Rule = 'linear' | 'dcfr' | 'cfr+'

/** One trained arm: where it ran, its net and refit, and its graded marks in order. */
export interface Arm {
  id: string
  /** depth × width, the way the run log spells it */
  net: string
  width: number
  depth: number
  steps: number
  lr: number
  workers: number
  rule: Rule
  where: 'Mac' | 'Modal'
  marks: Mark[]
}

const KEY = /^deep\/(\d+)\/(exact|pi|current)$/

const GRADE_FIELDS = ['br0', 'br1', 'exploitability', 'value_p0', 'hands_per_seat', 'wall_clock_s'] as const
const CONFIG_FIELDS = ['width', 'depth', 'steps', 'lr', 'workers'] as const
/** The settings every mark of one arm must share: the arm is one run, so one config. */
const ARM_FIELDS = [...CONFIG_FIELDS, 'rule'] as const

const isNumber = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x)

/** One grade file as the trainer writes it, every `deep/<t>/<objective>` entry checked field by field. */
function gradeFile(id: string, raw: unknown): GradeFile {
  if (typeof raw !== 'object' || raw === null) throw new Error(`${id}: the grade file is not an object`)
  const file: GradeFile = {}
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!KEY.test(key)) throw new Error(`${id}: ${key} is not a deep/<t>/<objective> key`)
    const grade = value as Partial<DeepGrade> | null
    for (const field of GRADE_FIELDS) {
      if (!isNumber(grade?.[field])) throw new Error(`${id}: ${key} has no numeric ${field}`)
    }
    const config = grade?.config as Partial<Config> | undefined
    for (const field of CONFIG_FIELDS) {
      if (!isNumber(config?.[field])) throw new Error(`${id}: ${key} has no numeric config.${field}`)
    }
    file[key] = grade as DeepGrade
  }
  if (Object.keys(file).length === 0) throw new Error(`${id}: the grade file has no marks`)
  return file
}

/** The marks of one arm in order, each with all three objects graded at the same hands. */
function marksOf(id: string, file: GradeFile): Mark[] {
  const byT = new Map<number, Partial<Record<Objective, DeepGrade>>>()
  for (const [key, grade] of Object.entries(file)) {
    const m = KEY.exec(key)!
    const t = Number(m[1])
    byT.set(t, { ...(byT.get(t) ?? {}), [m[2] as Objective]: grade })
  }
  return [...byT.entries()]
    .sort(([a], [b]) => a - b)
    .map(([t, objects]) => {
      const missing = OBJECTIVES.filter((o) => !objects[o])
      if (missing.length) throw new Error(`${id}: mark ${t} is graded without ${missing.join(' and ')}`)
      const { exact, pi, current } = objects as Record<Objective, DeepGrade>
      for (const o of ['pi', 'current'] as const) {
        if (objects[o]!.hands_per_seat !== exact.hands_per_seat) {
          throw new Error(`${id}: mark ${t} grades ${o} at ${objects[o]!.hands_per_seat} hands and exact at ${exact.hands_per_seat}`)
        }
      }
      return { t, hands: exact.hands_per_seat, wallClockS: exact.wall_clock_s, exact, pi, current }
    })
}

/** One arm from its grade file, read raw from the JSON import and checked here. */
export function armOf(id: string, raw: unknown, where: Arm['where']): Arm {
  const file = gradeFile(id, raw)
  const marks = marksOf(id, file)
  const configs = Object.values(file).map((g) => g.config)
  const first = configs[0]
  for (const field of ARM_FIELDS) {
    const odd = configs.find((c) => c[field] !== first[field])
    if (odd) throw new Error(`${id}: the marks disagree on ${field} (${String(first[field])} and ${String(odd[field])})`)
  }
  return {
    id,
    net: `${first.depth} × ${first.width}`,
    width: first.width,
    depth: first.depth,
    steps: first.steps,
    lr: first.lr,
    workers: first.workers,
    rule: (first.rule as Rule | undefined) ?? 'linear',
    where,
    marks,
  }
}

/**
 * The Modal size sweep is one file of three runs, each graded once, at the
 * mark the file itself records (`iterations`, with `hands_per_seat` beside
 * it); the key the trainer would have written is rebuilt from that mark.
 */
function sweepArm(id: string, size: string): Arm {
  const sweep = sweepData as unknown as { iterations: unknown; hands_per_seat: unknown; sizes: Record<string, Record<Objective, DeepGrade>> }
  if (!Number.isInteger(sweep.iterations) || !isNumber(sweep.hands_per_seat)) throw new Error(`${id}: the sweep file records no mark`)
  const objects = sweep.sizes?.[size]
  if (!objects) throw new Error(`${id}: the sweep has no size ${size}`)
  const file = Object.fromEntries(
    OBJECTIVES.map((o) => {
      if (objects[o]?.hands_per_seat !== sweep.hands_per_seat) {
        throw new Error(`${id}: ${o} is graded at ${objects[o]?.hands_per_seat} hands, the sweep at ${sweep.hands_per_seat}`)
      }
      return [`deep/${String(sweep.iterations).padStart(5, '0')}/${o}`, objects[o]]
    }),
  )
  return armOf(id, file, 'Modal')
}

/** Every arm trained on this game, by the run log's names. */
export const ARMS: Record<string, Arm> = {
  'mac-3x64': armOf('mac-3x64', grades64, 'Mac'),
  'modal-3x64': sweepArm('modal-3x64', '64x3'),
  'modal-4x64': sweepArm('modal-4x64', '64x4'),
  'modal-3x128': sweepArm('modal-3x128', '128x3'),
  '3x128-t200': armOf('3x128-t200', s128x3t200, 'Mac'),
  '3x128-s8000': armOf('3x128-s8000', s128x3s8000, 'Mac'),
  '4x128': armOf('4x128', s128x4, 'Mac'),
  '4x256': armOf('4x256', s256x4, 'Mac'),
  '4x256-s8000': armOf('4x256-s8000', s256x4s8000, 'Mac'),
  '4x256-s16000-lr5e-4': armOf('4x256-s16000-lr5e-4', s256x4s16000, 'Mac'),
  'rule-dcfr': armOf('rule-dcfr', ruleDcfr, 'Mac'),
  'rule-cfrplus': armOf('rule-cfrplus', ruleCfrplus, 'Mac'),
}

/** The arm run furthest: 4 × 256, 16,000-step refits at half the rate, graded to 5M hands. */
export const BEST = ARMS['4x256-s16000-lr5e-4']

export const markOf = (arm: Arm, t: number): Mark => {
  const mark = arm.marks.find((m) => m.t === t)
  if (!mark) throw new Error(`${arm.id} has no mark at t = ${t}`)
  return mark
}

/** LCFR's linear average at the same hands per seat, from rung 3's grades (chips per hand). */
export const lcfrAt = (hands: number): number => {
  const point = series('lcfr').find((p) => p.hands === hands)
  if (!point) throw new Error(`rung 3 has no LCFR grade at ${hands} hands per seat`)
  return point.exploitability
}

/** The verdict the sweep fixed before its numbers existed, and the size it picked. */
export const SWEEP_RULE: { rule: string; winner: string } = sweepData as unknown as { rule: string; winner: string }

// ---------------------------------------------------------------- ablations

/** The five nets at 1M hands per seat, all 4,000-step refits at 1e-3. */
export interface SizePoint {
  net: string
  width: number
  depth: number
  exact: number
  pi: number
  current: number
  arm: string
}

const sizePoint = (id: string): SizePoint => {
  const arm = ARMS[id]
  const m = markOf(arm, 100)
  return { net: arm.net, width: arm.width, depth: arm.depth, exact: m.exact.exploitability, pi: m.pi.exploitability, current: m.current.exploitability, arm: id }
}

export const SIZE_SWEEP: SizePoint[] = ['modal-3x64', 'modal-4x64', 'modal-3x128', '4x128', '4x256'].map(sizePoint)

/** The refit arms at 1M hands: steps and rate, same net, same hands. */
export interface RefitPoint {
  net: string
  steps: number
  lr: number
  exact: number
  arm: string
}

const refitPoint = (id: string): RefitPoint => {
  const arm = ARMS[id]
  return { net: arm.net, steps: arm.steps, lr: arm.lr, exact: markOf(arm, 100).exact.exploitability, arm: id }
}

export const REFIT_ARMS: RefitPoint[] = ['modal-3x128', '3x128-s8000', '4x256', '4x256-s8000', '4x256-s16000-lr5e-4'].map(refitPoint)

// ------------------------------------------------------- the rule screens

export interface ScreenPoint {
  t: number
  hands: number
  exact: number
}

const screen = (id: string): ScreenPoint[] => ARMS[id].marks.map((m) => ({ t: m.t, hands: m.hands, exact: m.exact.exploitability }))

/** The 4 × 256 net under three per-sample weightings, 4,000-step refits. */
export const RULE_SCREEN: Record<'linear' | 'dcfr' | 'cfrplus', ScreenPoint[]> = {
  linear: screen('4x256'),
  dcfr: screen('rule-dcfr'),
  cfrplus: screen('rule-cfrplus'),
}

export interface SignFlip {
  t: number
  seat: 0 | 1
  samples: number
  negativeKept: number
  positiveKept: number
  flipped: number
}

const FLIP_KEY = /^rule-(dcfr|cfrplus)-256x4\/(\d+)\/p([01])$/

function flipsOf(rule: 'dcfr' | 'cfrplus'): SignFlip[] {
  const out: SignFlip[] = []
  for (const [key, v] of Object.entries(signFlipsData as Record<string, Partial<Record<'samples' | 'negative_kept' | 'positive_kept' | 'flipped_share', unknown>>>)) {
    const m = FLIP_KEY.exec(key)
    if (!m || m[1] !== rule) continue
    for (const field of ['samples', 'negative_kept', 'positive_kept', 'flipped_share'] as const) {
      if (!isNumber(v[field])) throw new Error(`rule_sign_flips: ${key} has no numeric ${field}`)
    }
    const { samples, negative_kept, positive_kept, flipped_share } = v as Record<string, number>
    out.push({ t: Number(m[2]), seat: Number(m[3]) as 0 | 1, samples, negativeKept: negative_kept, positiveKept: positive_kept, flipped: flipped_share })
  }
  return out.sort((a, b) => a.t - b.t || a.seat - b.seat)
}

/** The mechanism behind the screen: what each rule keeps of the memory's regret mass, and the bad cells it turns positive. */
export const SIGN_FLIPS: Record<'dcfr' | 'cfrplus', SignFlip[]> = { dcfr: flipsOf('dcfr'), cfrplus: flipsOf('cfrplus') }

// ----------------------------------------------------- the policy-net sweep

export type Loss = 'mse' | 'ce' | 'kl' | 'is'
export type Head = 'linear' | 'softmax'

/** One refit of Π on the stored policy memory, under one loss, head, schedule and size. */
export interface PiArm extends Graded {
  id: string
  head: Head
  loss: Loss
  steps: number
  lr: number
  schedule: 'const' | 'cosine'
  width: number
  depth: number
  /** the mark whose memory was refit */
  t: number
  fitSeconds: number
  meanAbsDiff: number
  smallZeroed: number
}

interface PiRow extends Graded {
  mean_abs_diff_vs_exact: number
  small_zeroed_frac: number
  fit_s: number
  arm: { head: Head; loss: Loss; steps: number; lr: number; schedule: 'const' | 'cosine'; width: number; depth: number; t?: number }
}

const PI_FILE = piResults as unknown as Record<string, PiRow> & {
  _reference: { exact: { exploitability: number }; run_pi_exploitability: number }
}

/**
 * The mark an arm's memory was saved at, by the sweep script's own rule
 * (`mark_of` in `scripts/deepcfr/pi_loss_sweep.py`): an id suffixed `@t<n>`
 * refits the memory at mark n, any other id the memory at T = 500. The
 * sweep's first batch was written before the script recorded `t` beside each
 * arm, so five rows carry no `t`; every row that does must agree with its id.
 */
export const PI_FINAL_MARK = 500
export const piMarkOf = (id: string): number => {
  const m = /@t(\d+)$/.exec(id)
  return m ? Number(m[1]) : PI_FINAL_MARK
}

const PI_ROW_FIELDS = ['br0', 'br1', 'exploitability', 'value_p0', 'fit_s', 'mean_abs_diff_vs_exact', 'small_zeroed_frac'] as const
const PI_ARM_FIELDS = ['head', 'loss', 'steps', 'lr', 'schedule', 'width', 'depth'] as const

export const PI_SWEEP: PiArm[] = Object.entries(PI_FILE)
  .filter(([id]) => !id.startsWith('_'))
  .map(([id, row]) => {
    const r = row as Partial<PiRow>
    for (const field of PI_ROW_FIELDS) {
      if (!isNumber(r[field])) throw new Error(`pi_sweep: ${id} has no numeric ${field}`)
    }
    for (const field of PI_ARM_FIELDS) {
      if (r.arm?.[field] === undefined) throw new Error(`pi_sweep: ${id} records no arm.${field}`)
    }
    const t = piMarkOf(id)
    if (r.arm!.t !== undefined && r.arm!.t !== t) throw new Error(`pi_sweep: ${id} is recorded at mark ${r.arm!.t} but named for mark ${t}`)
    const full = r as PiRow
    return {
      id,
      ...full.arm,
      t,
      br0: full.br0,
      br1: full.br1,
      exploitability: full.exploitability,
      value_p0: full.value_p0,
      fitSeconds: full.fit_s,
      meanAbsDiff: full.mean_abs_diff_vs_exact,
      smallZeroed: full.small_zeroed_frac,
    }
  })

if (!isNumber(PI_FILE._reference?.exact?.exploitability) || !isNumber(PI_FILE._reference?.run_pi_exploitability)) {
  throw new Error('pi_sweep: the file records no _reference grades')
}

/** The exact average's own grade at mark 500, and the run's Π: what every refit is read against. */
export const PI_REFERENCE = { exact: PI_FILE._reference.exact.exploitability, runPi: PI_FILE._reference.run_pi_exploitability }

export const piArm = (id: string): PiArm => {
  const arm = PI_SWEEP.find((a) => a.id === id)
  if (!arm) throw new Error(`the sweep has no arm ${id}`)
  return arm
}

export interface ReachWeighted {
  rw_mad: number
  rw_small_mad: number
  rw_small_zeroed: number
  rw_small_mass_lost: number
}

/** A refit's reach-weighted error against the exact average; `run-pi (frozen t500)` is the run's own Π. */
export const reachWeighted = (id: string): ReachWeighted => {
  const r = (piReach as Record<string, ReachWeighted>)[id]
  if (!r) throw new Error(`no reach-weighted error for ${id}`)
  return r
}

// ------------------------------------------------------------ key numbers

/** The four numbers the masthead quotes, each computed here from the files above. */
export const KEY_NUMBERS = {
  priceAt1M: markOf(BEST, 100).exact.exploitability / lcfrAt(1_000_000),
  netAt5M: markOf(BEST, 500).exact.exploitability,
  tableAt5M: lcfrAt(5_000_000),
  ruleScreenRatio: RULE_SCREEN.dcfr.find((p) => p.t === 50)!.exact / RULE_SCREEN.linear.find((p) => p.t === 50)!.exact,
}
