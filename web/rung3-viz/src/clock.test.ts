import { describe, expect, it } from 'vitest'

import { FORENSICS, HALVINGS_TO_ZERO, VISIT_CLOCK, globalDcfrAt, visitClockAt } from './clock'

describe('the clock forensics', () => {
  it('read the paper: a 7,006 reach-weighted median gap, 36,593 plain, 73% of rows unvisited by 10k', () => {
    expect(FORENSICS.rows).toBe(6_220_050)
    expect(FORENSICS.age_quantiles_reach_weighted.p50).toBe(7006)
    expect(Math.round(FORENSICS.age_quantiles_iters.p50)).toBe(36593)
    expect(1 - FORENSICS.visited_by_10k).toBeCloseTo(0.73, 2)
    expect(FORENSICS.never_visited_by_2M * FORENSICS.rows).toBeCloseTo(8036, 0)
    expect(FORENSICS.age_ge_1075).toBeGreaterThan(0.96)
    expect(HALVINGS_TO_ZERO).toBe(1075)
  })
})

describe('the visit-clock arm', () => {
  it('has four checkpoints at W = 20, two averages each, on the race grid', () => {
    expect(VISIT_CLOCK.map((p) => p.hands)).toEqual([200_000, 200_000, 400_000, 400_000, 1_000_000, 1_000_000, 2_000_000, 2_000_000])
  })

  it('at 2M hands: 0.884 linear and 0.784 quadratic against the race DCFR at 0.981', () => {
    expect(visitClockAt(2_000_000, 'linear')).toBeCloseTo(0.8837, 3)
    expect(visitClockAt(2_000_000, 'quadratic')).toBeCloseTo(0.784, 2)
    expect(globalDcfrAt(2_000_000)).toBeCloseTo(0.981, 2)
  })
})
