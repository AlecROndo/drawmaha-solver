/**
 * Where the exploitability lives: every spot's share of it, binned by how
 * often the spot is reached when the strategy plays itself, read from
 * copies of the files `minidraw-exploit-by-reach` writes
 * (`figures/rung3/exploit_by_reach.json` for the frozen table,
 * `figures/rung4/exploit_by_reach_<release>.json` for the two Deep CFR
 * averages). Chips per hand in the files; shares are computed here.
 */

import tabular from './data/exploit_by_reach.json'
import deepExact from './data/rung4/exploit_by_reach_deep-exact.json'
import deepPi from './data/rung4/exploit_by_reach_deep-pi.json'

export const STAGES = ['round 1', 'draw', 'round 2'] as const
export type Stage = (typeof STAGES)[number]

export interface ReachBin {
  /** the lowest frequency in the bin, per hand; 0 for the rarest bin */
  floor: number
  /** the frequency the bin stops short of; null for the commonest */
  ceiling: number | null
  infosets: number
  /** this bin's share of the strategy's exploitability, chips per hand */
  exploitability: number
  /** the summed probability per hand of sitting at one of the bin's spots */
  frequency: number
  stages: Record<Stage, { infosets: number; exploitability: number }>
}

export interface ReachProfile {
  release: string
  exploitability: number
  gain: [number, number]
  check: { best_response_minus_value: [number, number] }
  infosets: number
  bins: ReachBin[]
  stages: Record<Stage, { infosets: number; exploitability: number; frequency: number }>
}

const profile = (raw: unknown, release: string): ReachProfile => {
  const p = raw as ReachProfile
  if (p.release !== release) throw new Error(`exploit_by_reach: expected the ${release} profile, found ${p.release}`)
  if (!Array.isArray(p.bins) || p.bins.length === 0) throw new Error(`exploit_by_reach ${release}: no bins`)
  const summed = p.bins.reduce((s, b) => s + b.exploitability, 0)
  if (Math.abs(summed - p.exploitability) > 1e-9) throw new Error(`exploit_by_reach ${release}: the bins do not sum to the exploitability`)
  return p
}

export const REACH: Record<'tabular' | 'deep-exact' | 'deep-pi', ReachProfile> = {
  tabular: profile(tabular, 'tabular'),
  'deep-exact': profile(deepExact, 'deep-exact'),
  'deep-pi': profile(deepPi, 'deep-pi'),
}

export const REACH_NAME: Record<keyof typeof REACH, string> = {
  tabular: 'LCFR, the table',
  'deep-exact': 'Deep CFR, exact average',
  'deep-pi': 'Deep CFR, Π',
}

/** "once in 10,000 hands" words for a frequency per hand. */
export const onceIn = (frequency: number): string => `once in ${Math.round(1 / frequency).toLocaleString()} hands`

/** 1000 → 1,000; 10000 → 10k; 1000000 → 1M: a count of hands, short enough for a row label. */
export const hands = (n: number): string => (n >= 1e6 ? `${n / 1e6}M` : n >= 1e4 ? `${n / 1e3}k` : n.toLocaleString())

/** The bin's label: a range of how often its spots come up, short enough for a row label. */
export const binLabel = (bin: ReachBin): string => {
  if (bin.ceiling === null) return `≥ 1 in ${hands(Math.round(1 / bin.floor))} hands`
  if (bin.floor === 0) return `< 1 in ${hands(Math.round(1 / bin.ceiling))} hands`
  return `1 in ${hands(Math.round(1 / bin.ceiling))} to ${hands(Math.round(1 / bin.floor))}`
}

/** What a spot in the bin leaks each time it is reached: the bin's exploitability over its summed frequency, chips per visit. */
export const perVisit = (bin: ReachBin): number => (bin.frequency > 0 ? bin.exploitability / bin.frequency : 0)

/** The bins a profile actually populates, commonest first. */
export const populated = (p: ReachProfile): ReachBin[] => p.bins.filter((b) => b.infosets > 0 && b.frequency > 0)

/** The bin whose floor is `floor`. */
export const bandAt = (p: ReachProfile, floor: number): ReachBin => {
  const bin = p.bins.find((b) => b.floor === floor)
  if (!bin) throw new Error(`exploit_by_reach ${p.release}: no band with floor ${floor}`)
  return bin
}

/** The share of a profile's exploitability in spots reached at least `floor` per hand. */
export const commonShare = (p: ReachProfile, floor: number) =>
  p.bins.filter((b) => b.floor >= floor).reduce((s, b) => s + b.exploitability, 0) / p.exploitability

/** The share of a profile's infosets reached at least `floor` per hand. */
export const commonInfosets = (p: ReachProfile, floor: number) =>
  p.bins.filter((b) => b.floor >= floor).reduce((s, b) => s + b.infosets, 0) / p.infosets

/** The share of a profile's exploitability in spots reached less often than `ceiling` per hand. */
export const rareShare = (p: ReachProfile, ceiling: number) =>
  p.bins.filter((b) => b.ceiling !== null && b.ceiling <= ceiling).reduce((s, b) => s + b.exploitability, 0) / p.exploitability

/** The share of a profile's infosets reached less often than `ceiling` per hand. */
export const rareInfosets = (p: ReachProfile, ceiling: number) =>
  p.bins.filter((b) => b.ceiling !== null && b.ceiling <= ceiling).reduce((s, b) => s + b.infosets, 0) / p.infosets
