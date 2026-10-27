import { describe, expect, it } from 'vitest'

import { perHundred, series } from './research'
import {
  ARMS,
  BEST,
  KEY_NUMBERS,
  PI_REFERENCE,
  PI_SWEEP,
  REFIT_ARMS,
  RULE_SCREEN,
  SIGN_FLIPS,
  SIZE_SWEEP,
  lcfrAt,
  markOf,
  piArm,
  reachWeighted,
} from './deep'

describe('the best arm against the table at equal hands', () => {
  it('is graded at six marks, three objects each, on the race grid', () => {
    expect(BEST.net).toBe('4 × 256')
    expect(BEST.marks.map((m) => m.t)).toEqual([10, 20, 50, 100, 200, 500])
    for (const m of BEST.marks) {
      expect(m.hands).toBe(m.t * 10_000)
      expect(lcfrAt(m.hands)).toBeGreaterThan(0)
    }
  })

  it('reads the run log: 0.133 / 0.115 / 0.108 against LCFR 1.131 / 0.630 / 0.264 at 1M / 2M / 5M', () => {
    expect(markOf(BEST, 100).exact.exploitability).toBeCloseTo(0.1326, 3)
    expect(markOf(BEST, 200).exact.exploitability).toBeCloseTo(0.1152, 3)
    expect(markOf(BEST, 500).exact.exploitability).toBeCloseTo(0.1079, 3)
    expect(lcfrAt(1_000_000)).toBeCloseTo(1.1308, 3)
    expect(lcfrAt(2_000_000)).toBeCloseTo(0.6298, 3)
    expect(lcfrAt(5_000_000)).toBeCloseTo(0.264, 2)
    // the same number research.ts speaks in, per hundred hands
    expect(perHundred(lcfrAt(20_000_000))).toBeCloseTo(perHundred(series('lcfr').at(-1)!.exploitability), 6)
  })

  it('orders exact < Π at every mark of every linear arm', () => {
    for (const arm of Object.values(ARMS)) {
      if (arm.rule !== 'linear') continue
      for (const m of arm.marks) expect(m.exact.exploitability).toBeLessThan(m.pi.exploitability)
    }
  })
})

describe('the ablations', () => {
  it('width is the lever: five nets at 1M hands, 3 × 64 → 3 × 128 cuts 36%', () => {
    expect(SIZE_SWEEP.map((s) => s.net)).toEqual(['3 × 64', '4 × 64', '3 × 128', '4 × 128', '4 × 256'])
    const at = (net: string) => SIZE_SWEEP.find((s) => s.net === net)!.exact
    expect(at('3 × 64')).toBeCloseTo(0.3824, 3)
    expect(at('4 × 256')).toBeCloseTo(0.1524, 3)
    expect(1 - at('3 × 128') / at('3 × 64')).toBeCloseTo(0.36, 2)
  })

  it('a longer refit: 0.243 → 0.184 at 3 × 128, 0.152 → 0.134 → 0.133 at 4 × 256', () => {
    const r = (net: string, steps: number) => REFIT_ARMS.find((a) => a.net === net && a.steps === steps)!
    expect(r('3 × 128', 4000).exact).toBeCloseTo(0.2435, 3)
    expect(r('3 × 128', 8000).exact).toBeCloseTo(0.184, 3)
    expect(r('4 × 256', 16000).lr).toBe(0.0005)
    expect(r('4 × 256', 16000).exact).toBeCloseTo(0.1326, 3)
  })
})

describe('the regret-rule screen', () => {
  it('per-sample DCFR and CFR+ are about 17× linear at t = 50 and rising', () => {
    const at = (rule: 'linear' | 'dcfr' | 'cfrplus', t: number) => RULE_SCREEN[rule].find((m) => m.t === t)!.exact
    expect(at('dcfr', 50) / at('linear', 50)).toBeGreaterThan(16)
    expect(at('cfrplus', 50) / at('linear', 50)).toBeGreaterThan(17)
    expect(at('dcfr', 10)).toBeLessThan(at('dcfr', 20))
    expect(at('dcfr', 20)).toBeLessThan(at('dcfr', 50))
  })

  it('the mechanism: seat 0 bad cells flipped 8% → 36% under DCFR, negative mass kept 4%', () => {
    const d = SIGN_FLIPS.dcfr.filter((f) => f.seat === 0)
    expect(d.map((f) => f.t)).toEqual([10, 20, 50])
    expect(d[0].flipped).toBeCloseTo(0.0817, 3)
    expect(d[2].flipped).toBeCloseTo(0.3625, 3)
    expect(d[2].negativeKept).toBeCloseTo(0.0407, 3)
  })
})

describe('the policy-net loss sweep', () => {
  it('holds 22 arms at mark 500, 200 or 100, and the reference', () => {
    expect(PI_SWEEP.length).toBe(22)
    expect(PI_REFERENCE.exact).toBeCloseTo(0.1079, 4)
    expect(PI_REFERENCE.runPi).toBeCloseTo(0.1363, 4)
  })

  it('cross-entropy at 64k cosine grades 0.1093; the control reproduces the run; KL equals CE to the digit', () => {
    expect(piArm('ce-softmax-64k-cos').exploitability).toBeCloseTo(0.1093, 4)
    expect(piArm('mse-linear-16k').exploitability).toBeCloseTo(PI_REFERENCE.runPi, 3)
    expect(piArm('kl-softmax-64k-cos').exploitability).toBe(piArm('ce-softmax-64k-cos').exploitability)
  })

  it('the reach-weighted error tracks exploitability: 0.033 for the run, 0.017 for the cross-entropy refit', () => {
    expect(reachWeighted('run-pi (frozen t500)').rw_mad).toBeCloseTo(0.033, 3)
    expect(reachWeighted('ce-softmax-64k-cos').rw_mad).toBeCloseTo(0.0168, 3)
  })
})

describe('the key numbers', () => {
  it('are the four the masthead quotes, each from the files', () => {
    expect(KEY_NUMBERS.priceAt1M).toBeCloseTo(0.1326 / 1.1308, 2)
    expect(KEY_NUMBERS.netAt5M).toBeCloseTo(0.1079, 3)
    expect(KEY_NUMBERS.tableAt5M).toBeCloseTo(0.264, 2)
    expect(KEY_NUMBERS.ruleScreenRatio).toBeGreaterThan(16)
  })
})
