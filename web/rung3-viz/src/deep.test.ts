import { describe, expect, it } from 'vitest'

import { perHundred, series } from './research'
import piResults from './data/rung4/pi_sweep_results.json'
import {
  ARMS,
  BEST,
  KEY_NUMBERS,
  PI_FINAL_MARK,
  PI_REFERENCE,
  PI_SWEEP,
  POT_ARMS,
  REFIT_ARMS,
  RULE_SCREEN,
  SIGN_FLIPS,
  SIZE_SWEEP,
  SWEEP_RULE,
  armOf,
  lcfrAt,
  markOf,
  piArm,
  piArmOf,
  piMarkOf,
  reachWeighted,
  screenAt,
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
  it('width is the lever: five nets at 1M hands, 3 × 64 → 3 × 128 cuts 36%, and the sweep picked 128x3', () => {
    expect(SWEEP_RULE.winner).toBe('128x3')
    expect(SWEEP_RULE.rule).toContain('64x3 unless')
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

  it('the pot arms: ÷ √pot 0.108 → 0.086 and ÷ pot 0.114 → 0.084 against raw 0.133 → 0.115 at 1M → 2M hands', () => {
    expect(POT_ARMS.map((a) => a.potPower)).toEqual([0, 0, 0.5, 1])
    expect(POT_ARMS.map((a) => a.lr)).toEqual([0.0005, 0.001, 0.0005, 0.0005])
    for (const a of POT_ARMS) {
      expect(a.net).toBe('4 × 256')
      expect(a.steps).toBe(16_000)
    }
    const at = (potPower: number, lr: number, t: 100 | 200) => POT_ARMS.find((a) => a.potPower === potPower && a.lr === lr)!.exact[t]
    expect(at(0, 0.0005, 100)).toBeCloseTo(0.1326, 3)
    expect(at(0, 0.0005, 200)).toBeCloseTo(0.1152, 3)
    expect(at(0, 0.001, 100)).toBeCloseTo(0.1275, 3)
    expect(at(0, 0.001, 200)).toBeCloseTo(0.1036, 3)
    expect(at(0.5, 0.0005, 100)).toBeCloseTo(0.1077, 3)
    expect(at(0.5, 0.0005, 200)).toBeCloseTo(0.0859, 3)
    expect(at(1, 0.0005, 100)).toBeCloseTo(0.1143, 3)
    expect(at(1, 0.0005, 200)).toBeCloseTo(0.0841, 3)
    // the run log's finding 7: −25% at 2M hands under √pot, and the two pot arms tied there
    expect(1 - at(0.5, 0.0005, 200) / at(0, 0.0005, 200)).toBeCloseTo(0.254, 2)
    expect(Math.abs(at(0.5, 0.0005, 200) - at(1, 0.0005, 200))).toBeLessThan(0.005)
    // the current iterate at 2M hands: √pot well ahead of ÷ pot
    const cur = (potPower: number) => POT_ARMS.find((a) => a.potPower === potPower && a.lr === 0.0005)!.current[200]
    expect(cur(0.5)).toBeCloseTo(0.158, 3)
    expect(cur(1)).toBeCloseTo(0.214, 3)
  })

  it('reads pot_power from the config, and a file written before it existed as raw chips', () => {
    expect(ARMS['4x256-s16000-lr5e-4'].potPower).toBe(0)
    expect(ARMS['pot0.5-4x256-s16000-lr5e-4'].potPower).toBe(0.5)
    expect(ARMS['pot1-4x256-s16000-lr5e-4'].potPower).toBe(1)
    expect(ARMS['4x256-s16000'].potPower).toBe(0)
  })
})

describe('the regret-rule screen', () => {
  it('per-sample DCFR and CFR+ are 16.6× and 17.9× linear at t = 50 and rising', () => {
    const at = (rule: 'linear' | 'dcfr' | 'cfrplus', t: number) => RULE_SCREEN[rule].find((m) => m.t === t)!.exact
    expect(at('dcfr', 50) / at('linear', 50)).toBeCloseTo(16.62, 2)
    expect(at('cfrplus', 50) / at('linear', 50)).toBeCloseTo(17.91, 2)
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
  it('holds 22 arms, 19 at mark 500, two at 200 and one at 100, and the reference', () => {
    expect(PI_SWEEP.length).toBe(22)
    expect(PI_SWEEP.filter((a) => a.t === 500).length).toBe(19)
    expect(PI_SWEEP.filter((a) => a.t === 200).map((a) => a.id).sort()).toEqual(['ce-softmax-64k-cos@t200', 'mse-linear-64k-cos@t200'])
    expect(PI_SWEEP.filter((a) => a.t === 100).map((a) => a.id)).toEqual(['ce-softmax-64k-cos@t100'])
    expect(PI_REFERENCE.exact).toBeCloseTo(0.1079, 4)
    expect(PI_REFERENCE.runPi).toBeCloseTo(0.1363, 4)
  })

  it("reads an arm's mark the way the script fixes it: the @t suffix, else the final mark", () => {
    expect(PI_FINAL_MARK).toBe(500)
    expect(piMarkOf('ce-softmax-64k-cos@t200')).toBe(200)
    expect(piMarkOf('mse-linear-16k')).toBe(500)
    for (const a of PI_SWEEP) expect(a.t).toBe(piMarkOf(a.id))
    // the five rows the first batch wrote before the script recorded t beside each arm
    const unrecorded = Object.entries(piResults as Record<string, { arm?: { t?: number } }>)
      .filter(([id, row]) => !id.startsWith('_') && row.arm?.t === undefined)
      .map(([id]) => id)
    expect(unrecorded.sort()).toEqual(['ce-softmax-16k', 'ce-softmax-64k-cos', 'mse-linear-16k', 'mse-linear-64k-cos', 'mse-softmax-16k'])
    for (const id of unrecorded) expect(id).not.toMatch(/@t\d+$/)
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
    expect(KEY_NUMBERS.priceAt1M).toBeCloseTo(0.1173, 4)
    expect(KEY_NUMBERS.netAt5M).toBeCloseTo(0.1079, 4)
    expect(KEY_NUMBERS.tableAt5M).toBeCloseTo(0.2641, 4)
    expect(KEY_NUMBERS.ruleScreenRatio).toBeCloseTo(16.62, 2)
  })
})

describe('a grade file is checked as it is read', () => {
  type ToyConfig = { width: number; depth: number; steps: number; lr: number; workers: number; pot_power?: number }
  const grade = (t: number, hands = t * 10_000, config: ToyConfig = { width: 64, depth: 3, steps: 4000, lr: 0.001, workers: 4 }) =>
    Object.fromEntries(
      (['exact', 'pi', 'current'] as const).map((o) => [
        `deep/${String(t).padStart(5, '0')}/${o}`,
        { br0: 0.1, br1: 0.2, exploitability: 0.3, value_p0: -0.1, hands_per_seat: hands, wall_clock_s: 1, config: { ...config } },
      ]),
    )

  it('reads a well-formed file into marks in order', () => {
    const arm = armOf('toy', { ...grade(20), ...grade(10) }, 'Mac')
    expect(arm.marks.map((m) => m.t)).toEqual([10, 20])
    expect(arm.net).toBe('3 × 64')
    expect(arm.rule).toBe('linear')
  })

  it('names the arm, the mark and the object a half-graded mark lacks', () => {
    const file = grade(10)
    delete file['deep/00010/pi']
    expect(() => armOf('toy', file, 'Mac')).toThrow('toy: mark 10 is graded without pi')
  })

  it('names the field a grade or its config lacks', () => {
    const file = grade(10) as Record<string, Record<string, unknown>>
    delete file['deep/00010/exact'].exploitability
    expect(() => armOf('toy', file, 'Mac')).toThrow('toy: deep/00010/exact has no numeric exploitability')
    const bare = grade(10) as Record<string, { config: Record<string, unknown> }>
    delete bare['deep/00010/current'].config.lr
    expect(() => armOf('toy', bare, 'Mac')).toThrow('toy: deep/00010/current has no numeric config.lr')
  })

  it('refuses marks that disagree on the config, so an arm is one run', () => {
    const file = { ...grade(10), ...grade(20, 200_000, { width: 128, depth: 3, steps: 4000, lr: 0.001, workers: 4 }) }
    expect(() => armOf('toy', file, 'Mac')).toThrow('toy: the marks disagree on width (64 and 128)')
    const mixed = { ...grade(10), ...grade(20, 200_000, { width: 64, depth: 3, steps: 4000, lr: 0.001, workers: 4, pot_power: 0.5 }) }
    expect(() => armOf('toy', mixed, 'Mac')).toThrow('toy: the marks disagree on pot_power (undefined and 0.5)')
  })

  it('names the rule and the mark a screen lookup lacks', () => {
    expect(screenAt('dcfr', 50)).toBeCloseTo(2.8701, 4)
    expect(() => screenAt('dcfr', 7)).toThrow('rule screen: no mark 7 for dcfr')
  })

  it('names what a Π-sweep row lacks: a field, its arm, an arm setting, or a mark that matches its name', () => {
    const row = {
      br0: 0.1, br1: 0.1, exploitability: 0.11, value_p0: -0.1, fit_s: 1, mean_abs_diff_vs_exact: 0.01, small_zeroed_frac: 0.1,
      arm: { head: 'softmax', loss: 'ce', steps: 64000, lr: 0.001, schedule: 'cosine', width: 256, depth: 4, t: 200 },
    }
    expect(piArmOf('toy@t200', row).t).toBe(200)
    expect(piArmOf('toy', { ...row, arm: { ...row.arm, t: undefined } }).t).toBe(500)
    expect(() => piArmOf('toy', { ...row, exploitability: undefined })).toThrow('pi_sweep: toy has no numeric exploitability')
    expect(() => piArmOf('toy', { ...row, arm: undefined })).toThrow('pi_sweep: toy records no arm')
    expect(() => piArmOf('toy', { ...row, arm: { ...row.arm, lr: undefined } })).toThrow('pi_sweep: toy records no arm.lr')
    expect(() => piArmOf('toy', row)).toThrow('pi_sweep: toy is recorded at mark 200 but named for mark 500')
  })

  it('refuses a key that is not deep/<t>/<objective>, and an empty file', () => {
    expect(() => armOf('toy', { ...grade(10), 'deep/10/mean': {} }, 'Mac')).toThrow('toy: deep/10/mean is not a deep/<t>/<objective> key')
    expect(() => armOf('toy', {}, 'Mac')).toThrow('toy: the grade file has no marks')
  })
})
