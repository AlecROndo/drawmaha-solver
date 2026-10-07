import { describe, expect, it } from 'vitest'

import {
  CATEGORIES,
  CHECKPOINTS,
  DRAW,
  FINAL_COLUMNS,
  FINAL_ITERATION,
  FROZEN,
  ROUND_ONE,
  RULES,
  UNIFORM_RANDOM,
  WORKERS,
  perHundred,
  series,
  throwRate,
} from './research'

describe('the race data', () => {
  it('has every rule graded at the same eight checkpoints, in order, with hands = iterations × W', () => {
    expect(WORKERS).toBe(10)
    expect(CHECKPOINTS).toEqual([10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000])
    for (const rule of RULES) {
      const s = series(rule)
      expect(s.map((p) => p.iteration)).toEqual(CHECKPOINTS)
      expect(s.every((p) => p.hands === p.iteration * 10)).toBe(true)
      expect(s.every((p, i) => i === 0 || p.exploitability < s[i - 1].exploitability)).toBe(true)
    }
  })

  it('ends where the README says: LCFR 6.1, vanilla 8.6, CFR+ 24.8, DCFR 25.0 per hundred hands', () => {
    const at = (rule: 'vanilla' | 'lcfr' | 'cfrplus' | 'dcfr') => perHundred(series(rule).at(-1)!.exploitability)
    expect(at('lcfr')).toBeCloseTo(6.13, 1)
    expect(at('vanilla')).toBeCloseTo(8.61, 1)
    expect(at('cfrplus')).toBeCloseTo(24.85, 1)
    expect(at('dcfr')).toBeCloseTo(24.97, 1)
    expect(perHundred(UNIFORM_RANDOM.exploitability)).toBeCloseTo(484.2, 0)
  })

  it('carries the extra averaging columns at the end of the run', () => {
    expect(FINAL_ITERATION).toBe(2_000_000)
    const keys = FINAL_COLUMNS.map((p) => `${p.rule}/${p.column}`).sort()
    expect(keys).toEqual(
      ['cfrplus/linear', 'cfrplus/uniform', 'dcfr/linear', 'dcfr/quadratic', 'lcfr/linear', 'vanilla/linear', 'vanilla/uniform'].sort(),
    )
  })

  it('the frozen strategy is the LCFR winner, and P0 cannot break even against it', () => {
    expect(FROZEN.rule).toBe('lcfr')
    expect(FROZEN.exploitability).toBeCloseTo(series('lcfr').at(-1)!.exploitability, 6)
    expect(FROZEN.br0).toBeLessThan(0)
    expect(FROZEN.value_p0).toBeLessThan(0)
  })
})

describe('the answer sheet', () => {
  it('has a row per inner category in every spot, each a distribution over the spot’s columns', () => {
    for (const fig of [...ROUND_ONE, ...DRAW]) {
      expect(fig.rows.map((r) => r.category)).toEqual(CATEGORIES)
      for (const row of fig.rows) {
        expect(row.mix.length).toBe(fig.columns.length)
        expect(row.mix.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
      }
    }
  })

  it('reads the README’s draw finding: high cards throw, made flushes do not', () => {
    const p0 = DRAW[0].rows
    const rate = (c: string) => throwRate(p0.find((r) => r.category === c)!.mix)
    expect(rate('high_card')).toBeGreaterThan(0.9)
    expect(rate('flush')).toBeLessThan(0.1)
    expect(rate('trips')).toBeLessThan(0.001)
  })
})
