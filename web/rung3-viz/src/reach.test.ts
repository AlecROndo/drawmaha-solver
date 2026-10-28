import { describe, expect, it } from 'vitest'

import { FROZEN } from './research'
import { BEST, markOf } from './deep'
import { REACH, REACH_NAME, REACH_SHORT, STAGES, bandAt, binLabel, crossingBand, edgeWord, perVisit, populated, rareInfosets, rareShare } from './reach'

describe('exploitability by reach', () => {
  it('each profile sums to the grade the page quotes elsewhere, and its check holds', () => {
    expect(REACH.tabular.exploitability).toBeCloseTo(FROZEN.exploitability, 5)
    expect(REACH['deep-exact'].exploitability).toBeCloseTo(markOf(BEST, 500).exact.exploitability, 5)
    expect(REACH['deep-pi'].exploitability).toBeCloseTo(markOf(BEST, 500).pi.exploitability, 5)
    for (const p of Object.values(REACH)) {
      expect(p.gain[0]).toBeCloseTo(p.check.best_response_minus_value[0], 9)
      expect(p.gain[1]).toBeCloseTo(p.check.best_response_minus_value[1], 9)
      expect(p.bins.reduce((s, b) => s + b.infosets, 0)).toBe(p.infosets)
      for (const stage of STAGES) expect(p.stages[stage].infosets).toBeGreaterThan(0)
    }
  })

  it('bins are labelled by how often a spot comes up, commonest first', () => {
    const labels = REACH.tabular.bins.map(binLabel)
    expect(labels[0]).toBe('≥ 1 in 10 hands')
    expect(labels.at(-1)).toBe('< 1 in 100M hands')
    expect(labels[4]).toBe('1 in 10k to 100k')
    expect(REACH.tabular.bins.every((b, i) => i === 0 || b.floor < REACH.tabular.bins[i - 1].floor)).toBe(true)
  })

  it('for the table the leak per visit rises as a spot gets rarer, by two orders of magnitude end to end', () => {
    const leaks = populated(REACH.tabular).map(perVisit)
    expect(leaks.every((v, i) => i === 0 || v > leaks[i - 1])).toBe(true)
    expect(leaks.at(-1)! / leaks[0]).toBeGreaterThan(100)
  })

  it('the rare tail holds most infosets and little exploitability; the middle bands hold most of it', () => {
    expect(rareInfosets(REACH.tabular, 1e-6)).toBeGreaterThan(0.8)
    expect(rareShare(REACH.tabular, 1e-6)).toBeLessThan(0.15)
    const middle = [1e-4, 1e-5, 1e-6].map((f) => bandAt(REACH.tabular, f))
    expect(middle.reduce((s, b) => s + b.exploitability, 0) / REACH.tabular.exploitability).toBeGreaterThan(0.5)
  })

  it('the net leaks more per visit than the table at the commonest spots and less at the rarest', () => {
    const first = populated(REACH.tabular)[0].floor
    const last = populated(REACH.tabular).at(-1)!.floor
    expect(perVisit(bandAt(REACH['deep-exact'], first))).toBeGreaterThan(perVisit(bandAt(REACH.tabular, first)))
    expect(perVisit(bandAt(REACH['deep-exact'], last))).toBeLessThan(perVisit(bandAt(REACH.tabular, last)))
    expect(REACH.tabular.stages['round 2'].exploitability / REACH.tabular.exploitability).toBeGreaterThan(
      REACH['deep-exact'].stages['round 2'].exploitability / REACH['deep-exact'].exploitability,
    )
  })

  it("labels a band edge as 'once in N hands', and the rarest bin's floor of zero as never, not as infinity", () => {
    expect(edgeWord(0.1)).toBe('1 in 10')
    expect(edgeWord(1e-3)).toBe('1 in 1,000')
    expect(edgeWord(1e-8)).toBe('1 in 100M')
    expect(edgeWord(REACH.tabular.bins.at(-1)!.floor)).toBe('never')
  })

  it("each strategy's bin shares partition its exploitability, and every name has a short form", () => {
    for (const p of Object.values(REACH)) {
      const shares = p.bins.map((b) => b.exploitability / p.exploitability)
      expect(shares.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 9)
    }
    for (const key of Object.keys(REACH_NAME) as (keyof typeof REACH_NAME)[]) expect(REACH_SHORT[key].length).toBeLessThan(REACH_NAME[key].length)
  })

  it('the table first leaks more per visit than the net in the 1 in 100k to 1M band', () => {
    const band = crossingBand(REACH.tabular, REACH['deep-exact'])!
    expect(binLabel(band)).toBe('1 in 100k to 1M')
    // and stays ahead in every rarer band, so the crossing the caption names is the lasting one
    const from = REACH.tabular.bins.indexOf(band)
    for (const [i, bin] of REACH.tabular.bins.entries()) {
      if (i > from && bin.frequency > 0) expect(perVisit(bin)).toBeGreaterThan(perVisit(REACH['deep-exact'].bins[i]))
    }
    expect(crossingBand(REACH['deep-exact'], REACH['deep-pi'])).toBeNull()
  })

  it('every profile bins by the same decade ladder, and the populated bands are contiguous', () => {
    const ladder = REACH.tabular.bins.map((b) => [b.floor, b.ceiling])
    for (const p of Object.values(REACH)) expect(p.bins.map((b) => [b.floor, b.ceiling])).toEqual(ladder)
    for (const p of Object.values(REACH)) {
      const idx = p.bins.map((b, i) => (b.infosets > 0 && b.frequency > 0 ? i : -1)).filter((i) => i >= 0)
      expect(idx).toEqual(idx.map((_, k) => idx[0] + k))
    }
  })
})
